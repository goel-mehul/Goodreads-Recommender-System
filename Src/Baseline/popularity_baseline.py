# popularity_baseline.py
# Run with: spark-submit --deploy-mode client popularity_baseline.py
#
# Purpose: Popularity baseline recommender.
#
# Popularity score formula:
#   popularity_score = avg_rating × log(read_count)
#
#   - read_count = number of users who marked is_read = 1
#   - avg_rating = average star rating among users who gave rating > 0
#   - log compresses the volume scale so quality and quantity both matter
#
# Recommendation strategy:
#   Same top-100 books recommended to every user (no personalization).
#   This is the defining characteristic of a popularity baseline.
#
# Evaluation: MAP@100 and NDCG@100 (manual implementation, no RankingEvaluator)
#
# Data: hdfs:///user/mg8958_nyu_edu/capstone_final/

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark import StorageLevel

spark = SparkSession.builder \
    .appName("popularity_baseline") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"

# ── Helper: manual MAP@100 and NDCG@100 ──────────────────────────────────────
def evaluate_popularity(top100_list, holdout_df, label=""):
    """
    Evaluate popularity baseline against holdout.
    Every user receives the same top100_list as recommendations.
    Finds rank of each holdout book in the top-100 list.
    Reports MAP@100 and NDCG@100.
    """
    # Build ranked recommendations as a lookup: book_id → rank (1-indexed)
    top100_ranked = spark.createDataFrame(
        [(book_id, rank + 1) for rank, book_id in enumerate(top100_list)],
        ["book_id", "rank"]
    )

    # Get all holdout interactions with their book_id
    # Join with top100 to find rank of each holdout book
    holdout_with_rank = holdout_df.select(
        "user_id", F.col("book_id").cast("int").alias("book_id")
    ).join(top100_ranked, on="book_id", how="left")

    # For each user, find the best rank (lowest rank number = highest in list)
    # If a user has multiple holdout books, take the one ranked highest
    best_rank_per_user = holdout_with_rank.groupBy("user_id").agg(
        F.min("rank").alias("best_rank"),
        F.count("book_id").alias("n_holdout"),
        F.sum(F.when(F.col("rank").isNotNull(), 1).otherwise(0)).alias("n_hits")
    )

    # Compute AP and NDCG per user
    eval_df = best_rank_per_user \
        .withColumn("ap",
            F.when(F.col("best_rank").isNotNull(), 1.0 / F.col("best_rank"))
            .otherwise(0.0)
        ).withColumn("ndcg",
            F.when(F.col("best_rank").isNotNull(), 1.0 / F.log2(F.col("best_rank") + 1))
            .otherwise(0.0)
        )

    results = eval_df.agg(
        F.mean("ap").alias("MAP"),
        F.mean("ndcg").alias("NDCG"),
        F.count("user_id").alias("n_users"),
        F.sum("n_hits").alias("total_hits")
    ).collect()[0]

    map_score  = results["MAP"]  or 0.0
    ndcg_score = results["NDCG"] or 0.0
    n_users    = results["n_users"]
    total_hits = results["total_hits"]

    print(f"\n── {label} Results ──")
    print(f"  Users evaluated  : {n_users:,}")
    print(f"  Total hits@100   : {total_hits:,}")
    print(f"  MAP@100          : {map_score:.6f}")
    print(f"  NDCG@100         : {ndcg_score:.6f}")

    return map_score, ndcg_score, n_users

# ── 1. Load train data ────────────────────────────────────────────────────────
print("=" * 60)
print("POPULARITY BASELINE")
print("=" * 60)
print("\nLoading train data...")
train = spark.read.parquet(f"{BASE}/train.parquet") \
    .persist(StorageLevel.MEMORY_AND_DISK)
train_count = train.count()
print(f"  Train interactions : {train_count:,}")

# ── 2. Compute popularity score ───────────────────────────────────────────────
print("\nComputing popularity scores...")

# Read count per book (is_read = 1)
read_counts = train.filter(F.col("is_read") == 1) \
    .groupBy("book_id") \
    .agg(F.count("*").alias("read_count"))

# Average rating per book (rating > 0 only, 0 = no rating given)
avg_ratings = train.filter(F.col("rating") > 0) \
    .groupBy("book_id") \
    .agg(F.avg("rating").alias("avg_rating"))

# Popularity score = avg_rating × log(read_count)
# Inner join: only books with both reads AND at least one rating
popularity = read_counts.join(avg_ratings, on="book_id", how="inner") \
    .withColumn(
        "popularity_score",
        F.col("avg_rating") * F.log(F.col("read_count"))
    )

total_scored_books = popularity.count()
print(f"  Books with popularity score : {total_scored_books:,}")

# ── 3. Get top 100 books ──────────────────────────────────────────────────────
print("\nSelecting top 100 books...")
top100 = popularity.orderBy(F.col("popularity_score").desc()).limit(100)
top100.cache()
top100.count()

print("\n── Top 10 Most Popular Books ──")
top100.select("book_id", "read_count", "avg_rating", "popularity_score") \
    .show(10, truncate=False)

# Save top 100 to HDFS
top100.select("book_id", "read_count", "avg_rating", "popularity_score") \
    .coalesce(1).write.mode("overwrite").csv(
        f"{BASE}/results/popularity_top100", header=True
    )
print(f"  Top 100 saved to {BASE}/results/popularity_top100")

# Collect top 100 book IDs as ordered list (rank 1 = most popular)
top100_list = [row["book_id"] for row in
    top100.orderBy(F.col("popularity_score").desc())
    .select("book_id").collect()
]

train.unpersist()

# ── 4. Load holdout files ─────────────────────────────────────────────────────
print("\nLoading holdout files...")
tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet")
test_holdout   = spark.read.parquet(f"{BASE}/test_holdout.parquet")

print(f"  Tuning holdout : {tuning_holdout.count():,} interactions, "
      f"{tuning_holdout.select('user_id').distinct().count():,} users")
print(f"  Test holdout   : {test_holdout.count():,} interactions, "
      f"{test_holdout.select('user_id').distinct().count():,} users")

# ── 5. Evaluate on tuning holdout ─────────────────────────────────────────────
print("\nEvaluating on tuning holdout...")
val_map, val_ndcg, val_users = evaluate_popularity(
    top100_list, tuning_holdout, label="Tuning (Validation)"
)

# ── 6. Evaluate on test holdout ───────────────────────────────────────────────
print("\nEvaluating on test holdout...")
test_map, test_ndcg, test_users = evaluate_popularity(
    top100_list, test_holdout, label="Test"
)

# ── 7. Save results ───────────────────────────────────────────────────────────
print("\nSaving results...")
results = spark.createDataFrame([
    ("popularity_formula", "avg_rating × log(read_count)"),
    ("tuning_map",         f"{val_map:.6f}"),
    ("tuning_ndcg",        f"{val_ndcg:.6f}"),
    ("tuning_users",       str(val_users)),
    ("test_map",           f"{test_map:.6f}"),
    ("test_ndcg",          f"{test_ndcg:.6f}"),
    ("test_users",         str(test_users)),
], ["metric", "value"])

results.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/popularity_baseline", header=True
)

# ── 8. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS — POPULARITY BASELINE")
print("=" * 60)
print(f"  Formula          : avg_rating × log(read_count)")
print(f"  Top 100 books    : same list recommended to all users")
print(f"")
print(f"  Tuning MAP@100   : {val_map:.6f}")
print(f"  Tuning NDCG@100  : {val_ndcg:.6f}")
print(f"")
print(f"  Test MAP@100     : {test_map:.6f}")
print(f"  Test NDCG@100    : {test_ndcg:.6f}")
print(f"")
print(f"  Results saved to : {BASE}/results/popularity_baseline")
print("=" * 60)

print("\nDone!")
spark.stop()
