# preprocess_final.py
# Run with: spark-submit --deploy-mode client preprocess_final.py
#
# Final preprocessing pipeline:
#   - Filter to users with 100+ interactions
#   - Sample 7% of filtered users (~30K users)
#   - Split 60/20/20 train/tuning/test
#   - Train users    → ALL interactions → train.parquet
#   - Tuning users   → all except last 30 → train.parquet
#                    → next 10 → tuning_history.parquet
#                    → last 20 → tuning_holdout.parquet
#   - Test users     → all except last 30 → train.parquet
#                    → next 10 → test_history.parquet
#                    → last 20 → test_holdout.parquet
#
# Key improvements over previous versions:
#   - 7% sample → ~30K users (vs 2.5% → 11K users before)
#   - history=10 (vs 5 before) → richer user context at eval time
#   - holdout=20 (vs 4/10 before) → more stable MAP/NDCG estimates
#   - Zero interactions discarded — tuning/test extras go to train
#
# Output directory: hdfs:///user/mg8958_nyu_edu/capstone_final/

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark import StorageLevel

spark = SparkSession.builder \
    .appName("preprocess_final") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

RAW_BASE = "hdfs:///user/pw44_nyu_edu"
OUT_BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_INTERACTIONS = 100   # minimum interactions per user
SAMPLE_FRACTION  = 0.07  # 7% of filtered users
HISTORY_SIZE     = 10    # interactions given to model at eval time
HOLDOUT_SIZE     = 20    # interactions hidden for evaluation
RESERVED         = HISTORY_SIZE + HOLDOUT_SIZE  # 30 total reserved per user
SEED_TUNING      = 42
SEED_TEST        = 43
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("FINAL PREPROCESSING PIPELINE")
print(f"  Min interactions : {MIN_INTERACTIONS}")
print(f"  Sample fraction  : {SAMPLE_FRACTION} (~7%)")
print(f"  History size     : {HISTORY_SIZE}")
print(f"  Holdout size     : {HOLDOUT_SIZE}")
print(f"  Total reserved   : {RESERVED}")
print(f"  Output           : {OUT_BASE}")
print("=" * 60)

# ── 1. Read raw CSV ───────────────────────────────────────────────────────────
print("\nSTEP 1: Reading raw CSV...")
interactions = spark.read.csv(
    f"{RAW_BASE}/goodreads_interactions.csv",
    header=True, inferSchema=True
)
raw_count = interactions.count()
raw_users = interactions.select("user_id").distinct().count()
print(f"  Raw interactions : {raw_count:,}")
print(f"  Raw users        : {raw_users:,}")
print(f"  Columns          : {interactions.columns}")

# ── 2. Save as Parquet and reload ─────────────────────────────────────────────
print("\nSTEP 2: Converting to Parquet...")
interactions.write.mode("overwrite").parquet(f"{OUT_BASE}/interactions_raw.parquet")
interactions = spark.read.parquet(f"{OUT_BASE}/interactions_raw.parquet")
print("  Saved and reloaded from Parquet")

# ── 3. Filter users with 100+ interactions ────────────────────────────────────
print(f"\nSTEP 3: Filtering to users with {MIN_INTERACTIONS}+ interactions...")
user_counts = interactions.groupBy("user_id").agg(
    F.count("*").alias("n_interactions")
)
valid_users = user_counts.filter(
    F.col("n_interactions") >= MIN_INTERACTIONS
).select("user_id")

interactions_filtered = interactions.join(valid_users, on="user_id", how="inner") \
    .persist(StorageLevel.MEMORY_AND_DISK)
interactions_filtered.count()

filtered_users = interactions_filtered.select("user_id").distinct().count()
filtered_count = interactions_filtered.count()
print(f"  Users after filtering        : {filtered_users:,}")
print(f"  Interactions after filtering : {filtered_count:,}")
print(f"  Avg interactions per user    : {filtered_count // filtered_users:,}")

# ── 4. Sample 7% of filtered users ───────────────────────────────────────────
print(f"\nSTEP 4: Sampling {SAMPLE_FRACTION*100:.0f}% of filtered users...")
sampled_users = valid_users.sample(fraction=SAMPLE_FRACTION, seed=42)
n_sampled_users = sampled_users.count()

interactions_sampled = interactions_filtered.join(
    sampled_users, on="user_id", how="inner"
).persist(StorageLevel.MEMORY_AND_DISK)
interactions_sampled.count()

sampled_count = interactions_sampled.count()
print(f"  Sampled users        : {n_sampled_users:,}")
print(f"  Sampled interactions : {sampled_count:,}")
print(f"  Avg per user         : {sampled_count // n_sampled_users:,}")

# ── 5. Verify signal distribution ─────────────────────────────────────────────
print("\nSTEP 5: Verifying signal distribution...")
total = sampled_count
has_rating   = interactions_sampled.filter(F.col("rating") > 0).count()
has_is_read  = interactions_sampled.filter(F.col("is_read") == 1).count()
has_reviewed = interactions_sampled.filter(F.col("is_reviewed") == 1).count()
has_all      = interactions_sampled.filter(
    (F.col("rating") > 0) & (F.col("is_read") == 1)
).count()

print(f"  Total interactions        : {total:,}")
print(f"  rating > 0  (explicit)    : {has_rating:,}  ({100*has_rating/total:.1f}%)")
print(f"  is_read = 1 (implicit)    : {has_is_read:,}  ({100*has_is_read/total:.1f}%)")
print(f"  is_reviewed = 1           : {has_reviewed:,}  ({100*has_reviewed/total:.1f}%)")
print(f"  rating>0 AND is_read=1    : {has_all:,}  ({100*has_all/total:.1f}%)")

print("\n  Combined signal breakdown:")
interactions_sampled.groupBy(
    "is_read", "is_reviewed",
    (F.col("rating") > 0).alias("has_rating")
).count().orderBy("is_read", "is_reviewed", "has_rating").show()

# ── 6. Split users 60/20/20 ───────────────────────────────────────────────────
print("\nSTEP 6: Splitting users 60/20/20 train/tuning/test...")
unique_sampled_users = interactions_sampled.select("user_id").distinct()
train_users, tuning_users, test_users = unique_sampled_users.randomSplit(
    [0.6, 0.2, 0.2], seed=42
)
n_train_users  = train_users.count()
n_tuning_users = tuning_users.count()
n_test_users   = test_users.count()
print(f"  Train users  (60%) : {n_train_users:,}")
print(f"  Tuning users (20%) : {n_tuning_users:,}")
print(f"  Test users   (20%) : {n_test_users:,}")

# ── 7. Build splits ───────────────────────────────────────────────────────────
print("\nSTEP 7: Building all splits...")
print(f"  Strategy: last {RESERVED} interactions reserved per tuning/test user")
print(f"    → last {HOLDOUT_SIZE} → holdout")
print(f"    → previous {HISTORY_SIZE} → history")
print(f"    → everything else → train")

# ── Helper function to build history/holdout/train_contrib ────────────────────
def build_splits(interactions_sampled, users, history_size, holdout_size, seed):
    """
    For each user:
      - Rank interactions randomly (no timestamp available)
      - Last holdout_size rows → holdout
      - Previous history_size rows → history
      - Everything before that → train contribution
    """
    reserved = history_size + holdout_size
    user_interactions = interactions_sampled.join(users, on="user_id", how="inner")

    # Count interactions per user
    user_n = user_interactions.groupBy("user_id").agg(F.count("*").alias("n"))

    # Rank interactions randomly
    w = Window.partitionBy("user_id").orderBy(F.rand(seed=seed))
    ranked = user_interactions \
        .withColumn("rn", F.row_number().over(w)) \
        .join(user_n, on="user_id", how="inner")

    # Train contribution: all except last reserved
    train_contrib = ranked \
        .filter(F.col("rn") <= F.col("n") - reserved) \
        .drop("rn", "n")

    # History: positions (n - reserved + 1) to (n - holdout_size)
    history = ranked \
        .filter(
            (F.col("rn") > F.col("n") - reserved) &
            (F.col("rn") <= F.col("n") - holdout_size)
        ).drop("rn", "n")

    # Holdout: last holdout_size positions
    holdout = ranked \
        .filter(F.col("rn") > F.col("n") - holdout_size) \
        .drop("rn", "n")

    return train_contrib, history, holdout

# ── Build tuning splits ───────────────────────────────────────────────────────
print("  Building tuning splits...")
tuning_train_contrib, tuning_history, tuning_holdout = build_splits(
    interactions_sampled, tuning_users, HISTORY_SIZE, HOLDOUT_SIZE, SEED_TUNING
)

tuning_train_count   = tuning_train_contrib.count()
tuning_history_count = tuning_history.count()
tuning_holdout_count = tuning_holdout.count()
tuning_holdout_users = tuning_holdout.select("user_id").distinct().count()

print(f"    Tuning train contrib : {tuning_train_count:,}")
print(f"    Tuning history       : {tuning_history_count:,} ({tuning_holdout_users:,} users × {HISTORY_SIZE})")
print(f"    Tuning holdout       : {tuning_holdout_count:,} ({tuning_holdout_users:,} users × {HOLDOUT_SIZE})")

# ── Build test splits ─────────────────────────────────────────────────────────
print("  Building test splits...")
test_train_contrib, test_history, test_holdout = build_splits(
    interactions_sampled, test_users, HISTORY_SIZE, HOLDOUT_SIZE, SEED_TEST
)

test_train_count   = test_train_contrib.count()
test_history_count = test_history.count()
test_holdout_count = test_holdout.count()
test_holdout_users = test_holdout.select("user_id").distinct().count()

print(f"    Test train contrib   : {test_train_count:,}")
print(f"    Test history         : {test_history_count:,} ({test_holdout_users:,} users × {HISTORY_SIZE})")
print(f"    Test holdout         : {test_holdout_count:,} ({test_holdout_users:,} users × {HOLDOUT_SIZE})")

# ── Build final train ─────────────────────────────────────────────────────────
print("  Building final train set...")
train_from_train_users = interactions_sampled.join(
    train_users, on="user_id", how="inner"
)
train_from_train_count = train_from_train_users.count()

train_final = train_from_train_users \
    .union(tuning_train_contrib) \
    .union(test_train_contrib)

train_final_count = train_final.count()
print(f"    From train users     : {train_from_train_count:,}")
print(f"    From tuning users    : {tuning_train_count:,}")
print(f"    From test users      : {test_train_count:,}")
print(f"    Total train          : {train_final_count:,}")

# ── 8. Save all files ─────────────────────────────────────────────────────────
print("\nSTEP 8: Saving all files to HDFS...")

print("  Saving train.parquet...")
train_final.write.mode("overwrite").parquet(f"{OUT_BASE}/train.parquet")

print("  Saving tuning_history.parquet...")
tuning_history.write.mode("overwrite").parquet(f"{OUT_BASE}/tuning_history.parquet")

print("  Saving tuning_holdout.parquet...")
tuning_holdout.write.mode("overwrite").parquet(f"{OUT_BASE}/tuning_holdout.parquet")

print("  Saving test_history.parquet...")
test_history.write.mode("overwrite").parquet(f"{OUT_BASE}/test_history.parquet")

print("  Saving test_holdout.parquet...")
test_holdout.write.mode("overwrite").parquet(f"{OUT_BASE}/test_holdout.parquet")

print("  All files saved!")

# ── 9. Signal breakdown in final train ───────────────────────────────────────
train_reload  = spark.read.parquet(f"{OUT_BASE}/train.parquet")
train_rated   = train_reload.filter(F.col("rating") > 0).count()
train_is_read = train_reload.filter(F.col("is_read") == 1).count()

# ── 10. Final summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Raw CSV                        : {raw_count:,} interactions, {raw_users:,} users")
print(f"  After {MIN_INTERACTIONS}+ filter              : {filtered_count:,} interactions, {filtered_users:,} users")
print(f"  After {SAMPLE_FRACTION*100:.0f}% sample               : {sampled_count:,} interactions, {n_sampled_users:,} users")
print(f"")
print(f"  User splits:")
print(f"    Train users  (60%)           : {n_train_users:,}")
print(f"    Tuning users (20%)           : {n_tuning_users:,}")
print(f"    Test users   (20%)           : {n_test_users:,}")
print(f"")
print(f"  Output files:")
print(f"    train.parquet                : {train_final_count:,} interactions")
print(f"      → from train users         : {train_from_train_count:,}")
print(f"      → from tuning users        : {tuning_train_count:,}")
print(f"      → from test users          : {test_train_count:,}")
print(f"    tuning_history.parquet       : {tuning_history_count:,} ({n_tuning_users:,} users × {HISTORY_SIZE})")
print(f"    tuning_holdout.parquet       : {tuning_holdout_count:,} ({tuning_holdout_users:,} users × {HOLDOUT_SIZE})")
print(f"    test_history.parquet         : {test_history_count:,} ({n_test_users:,} users × {HISTORY_SIZE})")
print(f"    test_holdout.parquet         : {test_holdout_count:,} ({test_holdout_users:,} users × {HOLDOUT_SIZE})")
print(f"")
print(f"  Interactions discarded         : 0 ✅")
print(f"")
print(f"  fit(train + tuning_history)    : ~{train_final_count + tuning_history_count:,} interactions")
print(f"")
print(f"  Signal coverage in train:")
print(f"    rating > 0  (explicit)       : {train_rated:,}  ({100*train_rated/train_final_count:.1f}%)")
print(f"    is_read = 1 (implicit)       : {train_is_read:,}  ({100*train_is_read/train_final_count:.1f}%)")
print(f"")
print(f"  All files saved to: {OUT_BASE}/")
print("=" * 60)

spark.stop()
