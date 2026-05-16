"""
market_segmentation.py
─────────────────────────────────────────────────────────────────────────────
Market Segmentation — Goodreads Capstone (Deliverable 1)
Find the top 100 pairs of books read by the most similar groups of users
using MinHash LSH (Jaccard similarity over reader sets).

Similarity measure:
  Jaccard(A, B) = |readers(A) ∩ readers(B)| / |readers(A) ∪ readers(B)|

Scalability:
  MinHash LSH (Spark ML) — approximates Jaccard similarity without
  computing all O(n²) pairs explicitly.

Data: hdfs:///user/mg8958_nyu_edu/capstone_final/train.parquet

Usage:
  spark-submit --deploy-mode client market_segmentation.py
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinHashLSH, CountVectorizer

# ── Spark session ─────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("market_segmentation") \
    .config("spark.sql.shuffle.partitions", "300") \
    .config("spark.task.maxFailures", "8") \
    .config("spark.speculation", "true") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .config("spark.shuffle.io.retryWait", "60s") \
    .config("spark.shuffle.io.maxRetries", "10") \
    .config("spark.rpc.askTimeout", "600s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(
    "hdfs:///user/mg8958_nyu_edu/capstone_final/checkpoints"
)

# ── Configuration ─────────────────────────────────────────────────────────
BASE          = "hdfs:///user/mg8958_nyu_edu/capstone_final"
MIN_READERS   = 50     # minimum readers per book — filters noisy low-count books
NUM_HASH_TABLES     = 5      # more = better recall, slower
JACCARD_THRESHOLD   = 0.5    # distance threshold: similarity >= 0.5
# ─────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("MARKET SEGMENTATION — MinHash LSH")
print(f"  Min readers per book    : {MIN_READERS}")
print(f"  Num hash tables         : {NUM_HASH_TABLES}")
print(f"  Jaccard dist threshold  : {JACCARD_THRESHOLD} (similarity >= {1-JACCARD_THRESHOLD})")
print("=" * 60)

# ── 1. Load train interactions ────────────────────────────────────────────
print("\nSTEP 1: Loading train interactions...")
interactions = spark.read.parquet(f"{BASE}/train.parquet")
print(f"  Total train rows : {interactions.count():,}")

# ── 2. Filter to is_read=1, unique (user_id, book_id) pairs ──────────────
# Market position defined by reader set regardless of rating
print("\nSTEP 2: Filtering to is_read=1, unique pairs...")
read_interactions = interactions \
    .filter(F.col("is_read") == 1) \
    .select("user_id", "book_id") \
    .distinct() \
    .repartition(300)

n_read = read_interactions.count()
print(f"  Unique (user, book) read pairs : {n_read:,}")

# ── 3. Filter books by minimum reader count ───────────────────────────────
# Books read by fewer than MIN_READERS users are excluded:
# - too few readers = noisy similarity estimates
# - reduces number of candidate pairs significantly
print(f"\nSTEP 3: Filtering to books with >= {MIN_READERS} readers...")
book_reader_counts = read_interactions \
    .groupBy("book_id") \
    .agg(F.count("user_id").alias("num_readers"))

qualified_books = book_reader_counts.filter(F.col("num_readers") >= MIN_READERS)
n_books = qualified_books.count()
print(f"  Books with >= {MIN_READERS} readers : {n_books:,}")

read_interactions = read_interactions \
    .join(qualified_books.select("book_id"), on="book_id") \
    .repartition(300)

n_users = read_interactions.select("user_id").distinct().count()
print(f"  Unique users in filtered set : {n_users:,}")

# ── 4. Build book feature vectors ─────────────────────────────────────────
# Each book is represented as a binary sparse vector where:
#   dimensions = all user_ids
#   value = 1 if user read the book, 0 otherwise
# CountVectorizer with binary=True is memory-efficient for this.
print("\nSTEP 4: Building book feature vectors...")

read_interactions = read_interactions.withColumn(
    "user_id_str", F.col("user_id").cast("string")
)

book_user_lists = read_interactions \
    .groupBy("book_id") \
    .agg(F.collect_list("user_id_str").alias("user_tokens")) \
    .repartition(300)

# Save intermediate result
book_user_lists.write.mode("overwrite").parquet(
    f"{BASE}/market_seg/book_tokens.parquet"
)
book_user_lists = spark.read.parquet(f"{BASE}/market_seg/book_tokens.parquet")
print(f"  Book token lists saved")

# Fit CountVectorizer — binary=True ensures pure Jaccard (presence/absence)
cv = CountVectorizer(
    inputCol="user_tokens",
    outputCol="features",
    vocabSize=n_users,
    minDF=1.0,
    binary=True
)

print("  Fitting CountVectorizer...")
cv_model = cv.fit(book_user_lists)
book_vectors = cv_model.transform(book_user_lists).select("book_id", "features")
book_vectors.checkpoint()
book_vectors.cache()
n_vecs = book_vectors.count()
print(f"  Book vectors ready: {n_vecs:,} books")

# ── 5. Fit MinHash LSH ────────────────────────────────────────────────────
# MinHash LSH approximates Jaccard similarity by hashing feature vectors.
# numHashTables controls the recall/speed tradeoff:
#   more tables = better recall (fewer missed similar pairs), more memory
print("\nSTEP 5: Fitting MinHash LSH...")

mh = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=NUM_HASH_TABLES,
    seed=42
)
model = mh.fit(book_vectors)
print("  MinHash LSH model fitted.")

# ── 6. Approximate similarity join ───────────────────────────────────────
# approxSimilarityJoin finds all pairs within the Jaccard distance threshold
# without computing all O(n²) pairs — this is the key scalability step.
print(f"\nSTEP 6: Running approxSimilarityJoin (threshold={JACCARD_THRESHOLD})...")

similar_pairs = model.approxSimilarityJoin(
    book_vectors, book_vectors,
    threshold=JACCARD_THRESHOLD,
    distCol="jaccard_dist"
).filter(
    # Remove self-pairs and duplicate pairs (A,B) == (B,A)
    F.col("datasetA.book_id") < F.col("datasetB.book_id")
).withColumn(
    "jaccard_similarity", F.round(1.0 - F.col("jaccard_dist"), 6)
).select(
    F.col("datasetA.book_id").alias("book_id_1"),
    F.col("datasetB.book_id").alias("book_id_2"),
    "jaccard_similarity"
)

# Save all pairs before sorting
similar_pairs.write.mode("overwrite").parquet(
    f"{BASE}/market_seg/similar_pairs.parquet"
)
similar_pairs_saved = spark.read.parquet(f"{BASE}/market_seg/similar_pairs.parquet")
total_pairs = similar_pairs_saved.count()
print(f"  Total candidate pairs found : {total_pairs:,}")

if total_pairs < 100:
    print(f"  ⚠️  WARNING: Only {total_pairs} pairs found.")
    print(f"  Consider lowering JACCARD_THRESHOLD or MIN_READERS.")

# ── 7. Extract top 100 pairs ──────────────────────────────────────────────
print("\nSTEP 7: Extracting top 100 pairs...")
top_100 = similar_pairs_saved \
    .orderBy(F.col("jaccard_similarity").desc()) \
    .limit(100)

# ── 8. Join with book_id_map for Goodreads IDs ───────────────────────────
# book_id_map schema:
#   book_id_csv : sequential integer (0,1,2...) = internal ID used in train.parquet
#   book_id     : actual Goodreads book ID
# Join on book_id_csv to get real Goodreads IDs.
print("\nSTEP 8: Joining with book_id_map...")
book_id_map = spark.read.csv(
    "hdfs:///user/pw44_nyu_edu/book_id_map.csv",
    header=True, inferSchema=True
)
print(f"  book_id_map rows   : {book_id_map.count():,}")

map1 = book_id_map.select(
    F.col("book_id_csv").alias("book_id_1"),
    F.col("book_id").alias("goodreads_id_1")
)
map2 = book_id_map.select(
    F.col("book_id_csv").alias("book_id_2"),
    F.col("book_id").alias("goodreads_id_2")
)

result = top_100 \
    .join(map1, on="book_id_1", how="left") \
    .join(map2, on="book_id_2", how="left") \
    .select(
        "book_id_1", "book_id_2",
        "jaccard_similarity",
        "goodreads_id_1", "goodreads_id_2"
    ) \
    .orderBy(F.col("jaccard_similarity").desc())

# ── 9. Save and display results ───────────────────────────────────────────
print("\nSTEP 9: Saving results...")
result.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/market_segmentation_top100", header=True
)

print("\n" + "=" * 80)
print("TOP 100 MOST SIMILAR BOOK PAIRS (by Jaccard similarity of reader sets)")
print("=" * 80)
result.show(100, truncate=False)

# ── 10. Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Similarity measure        : Jaccard similarity")
print(f"  Definition                : |readers(A) ∩ readers(B)| / |readers(A) ∪ readers(B)|")
print(f"  Vectorization             : CountVectorizer (binary, user_id as tokens)")
print(f"  Scalability method        : MinHash LSH (Spark ML)")
print(f"  Number of hash tables     : {NUM_HASH_TABLES}")
print(f"  Jaccard dist threshold    : {JACCARD_THRESHOLD} (similarity >= {1-JACCARD_THRESHOLD})")
print(f"  Min readers per book      : {MIN_READERS}")
print(f"  Total books in model      : {n_vecs:,}")
print(f"  Total users in reader set : {n_users:,}")
print(f"  Total candidate pairs     : {total_pairs:,}")
print(f"  Results saved to          : {BASE}/results/market_segmentation_top100")
print("=" * 80)

spark.stop()
