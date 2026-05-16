# analyze_data.py
# Run with: spark-submit --deploy-mode client analyze_data.py
#
# Purpose: Complete analysis of the capstone_final preprocessed data.
# Run after preprocess_final.py completes.
#
# Reports:
#   1. File sizes and row counts
#   2. User statistics
#   3. Signal distribution (explicit + implicit)
#   4. Train/tuning/test split verification
#   5. History and holdout size verification
#   6. Explicit ALS readiness check
#   7. Implicit ALS readiness check
#   8. Summary table

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("analyze_data") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"

print("=" * 70)
print("CAPSTONE FINAL — COMPLETE DATA ANALYSIS")
print("=" * 70)

# ── Load all files ────────────────────────────────────────────────────────────
print("\nLoading all files...")
train          = spark.read.parquet(f"{BASE}/train.parquet")
tuning_history = spark.read.parquet(f"{BASE}/tuning_history.parquet")
tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet")
test_history   = spark.read.parquet(f"{BASE}/test_history.parquet")
test_holdout   = spark.read.parquet(f"{BASE}/test_holdout.parquet")
print("  All files loaded!")

# ── 1. File sizes ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. FILE SIZES")
print("=" * 70)

files = {
    "train.parquet"          : train,
    "tuning_history.parquet" : tuning_history,
    "tuning_holdout.parquet" : tuning_holdout,
    "test_history.parquet"   : test_history,
    "test_holdout.parquet"   : test_holdout,
}

total_interactions = 0
for name, df in files.items():
    n_rows  = df.count()
    n_users = df.select("user_id").distinct().count()
    n_books = df.select("book_id").distinct().count()
    total_interactions += n_rows
    print(f"\n  {name}")
    print(f"    Rows          : {n_rows:,}")
    print(f"    Unique users  : {n_users:,}")
    print(f"    Unique books  : {n_books:,}")

print(f"\n  Total interactions across all files : {total_interactions:,}")

# ── 2. User statistics ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. USER STATISTICS")
print("=" * 70)

train_users   = train.select("user_id").distinct()
tuning_users  = tuning_history.select("user_id").distinct()
test_users    = test_history.select("user_id").distinct()

n_train_users  = train_users.count()
n_tuning_users = tuning_users.count()
n_test_users   = test_users.count()
total_users    = n_train_users + n_tuning_users + n_test_users

print(f"\n  Train users  (60%) : {n_train_users:,}  ({100*n_train_users/total_users:.1f}%)")
print(f"  Tuning users (20%) : {n_tuning_users:,}  ({100*n_tuning_users/total_users:.1f}%)")
print(f"  Test users   (20%) : {n_test_users:,}  ({100*n_test_users/total_users:.1f}%)")
print(f"  Total users        : {total_users:,}")

# Verify no user overlap between splits
tuning_in_train = tuning_users.join(train_users, on="user_id", how="inner").count()
test_in_train   = test_users.join(train_users, on="user_id", how="inner").count()
tuning_in_test  = tuning_users.join(test_users, on="user_id", how="inner").count()

print(f"\n  User overlap checks:")
print(f"    Tuning users in train : {tuning_in_train}  {'✅' if tuning_in_train == 0 else '❌ OVERLAP!'}")
print(f"    Test users in train   : {test_in_train}  {'✅' if test_in_train == 0 else '❌ OVERLAP!'}")
print(f"    Tuning users in test  : {tuning_in_test}  {'✅' if tuning_in_test == 0 else '❌ OVERLAP!'}")

# Interactions per user in train
train_user_counts = train.groupBy("user_id").agg(F.count("*").alias("n"))
train_stats = train_user_counts.agg(
    F.min("n").alias("min"),
    F.max("n").alias("max"),
    F.mean("n").alias("mean"),
    F.expr("percentile_approx(n, 0.5)").alias("median"),
    F.expr("percentile_approx(n, 0.25)").alias("p25"),
    F.expr("percentile_approx(n, 0.75)").alias("p75"),
).collect()[0]

print(f"\n  Interactions per user in train:")
print(f"    Min    : {train_stats['min']:,}")
print(f"    p25    : {train_stats['p25']:,}")
print(f"    Median : {train_stats['median']:,}")
print(f"    Mean   : {train_stats['mean']:,.1f}")
print(f"    p75    : {train_stats['p75']:,}")
print(f"    Max    : {train_stats['max']:,}")

# ── 3. Signal distribution ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. SIGNAL DISTRIBUTION IN TRAIN")
print("=" * 70)

total_train = train.count()
rated        = train.filter(F.col("rating") > 0).count()
is_read      = train.filter(F.col("is_read") == 1).count()
is_reviewed  = train.filter(F.col("is_reviewed") == 1).count()
both         = train.filter((F.col("rating") > 0) & (F.col("is_read") == 1)).count()

print(f"\n  Total train interactions     : {total_train:,}")
print(f"  rating > 0  (explicit)       : {rated:,}  ({100*rated/total_train:.1f}%)")
print(f"  is_read = 1 (implicit)       : {is_read:,}  ({100*is_read/total_train:.1f}%)")
print(f"  is_reviewed = 1              : {is_reviewed:,}  ({100*is_reviewed/total_train:.1f}%)")
print(f"  rating>0 AND is_read=1       : {both:,}  ({100*both/total_train:.1f}%)")

print(f"\n  Rating distribution in train:")
train.filter(F.col("rating") > 0) \
    .groupBy("rating").count() \
    .orderBy("rating").show()

print(f"  Combined signal breakdown:")
train.groupBy(
    "is_read", "is_reviewed",
    (F.col("rating") > 0).alias("has_rating")
).count().orderBy("is_read", "is_reviewed", "has_rating").show()

# ── 4. History and holdout verification ──────────────────────────────────────
print("\n" + "=" * 70)
print("4. HISTORY AND HOLDOUT VERIFICATION")
print("=" * 70)

# Check history size per user
tuning_hist_per_user = tuning_history.groupBy("user_id").count()
test_hist_per_user   = test_history.groupBy("user_id").count()

tuning_hist_stats = tuning_hist_per_user.agg(
    F.min("count").alias("min"),
    F.max("count").alias("max"),
    F.mean("count").alias("mean")
).collect()[0]

test_hist_stats = test_hist_per_user.agg(
    F.min("count").alias("min"),
    F.max("count").alias("max"),
    F.mean("count").alias("mean")
).collect()[0]

print(f"\n  Tuning history interactions per user:")
print(f"    Min : {tuning_hist_stats['min']}  Max : {tuning_hist_stats['max']}  Mean : {tuning_hist_stats['mean']:.1f}")
print(f"    {'✅ Correct (all users have 10)' if tuning_hist_stats['min'] == tuning_hist_stats['max'] == 10 else '⚠️  Check history size'}")

print(f"\n  Test history interactions per user:")
print(f"    Min : {test_hist_stats['min']}  Max : {test_hist_stats['max']}  Mean : {test_hist_stats['mean']:.1f}")
print(f"    {'✅ Correct (all users have 10)' if test_hist_stats['min'] == test_hist_stats['max'] == 10 else '⚠️  Check history size'}")

# Check holdout size per user
tuning_hold_per_user = tuning_holdout.groupBy("user_id").count()
test_hold_per_user   = test_holdout.groupBy("user_id").count()

tuning_hold_stats = tuning_hold_per_user.agg(
    F.min("count").alias("min"),
    F.max("count").alias("max"),
    F.mean("count").alias("mean")
).collect()[0]

test_hold_stats = test_hold_per_user.agg(
    F.min("count").alias("min"),
    F.max("count").alias("max"),
    F.mean("count").alias("mean")
).collect()[0]

print(f"\n  Tuning holdout interactions per user:")
print(f"    Min : {tuning_hold_stats['min']}  Max : {tuning_hold_stats['max']}  Mean : {tuning_hold_stats['mean']:.1f}")
print(f"    {'✅ Correct (all users have 20)' if tuning_hold_stats['min'] == tuning_hold_stats['max'] == 20 else '⚠️  Check holdout size'}")

print(f"\n  Test holdout interactions per user:")
print(f"    Min : {test_hold_stats['min']}  Max : {test_hold_stats['max']}  Mean : {test_hold_stats['mean']:.1f}")
print(f"    {'✅ Correct (all users have 20)' if test_hold_stats['min'] == test_hold_stats['max'] == 20 else '⚠️  Check holdout size'}")

# ── 5. Explicit ALS readiness ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. EXPLICIT ALS READINESS CHECK")
print("=" * 70)

# Train
train_explicit = train.filter(F.col("rating") > 0)
n_train_explicit = train_explicit.count()
n_train_explicit_users = train_explicit.select("user_id").distinct().count()

# Tuning history
tuning_hist_explicit = tuning_history.filter(F.col("rating") > 0)
n_tuning_hist_explicit = tuning_hist_explicit.count()

# Tuning holdout
tuning_hold_explicit = tuning_holdout.filter(F.col("rating") > 0)
n_tuning_hold_explicit = tuning_hold_explicit.count()
n_tuning_hold_explicit_users = tuning_hold_explicit.select("user_id").distinct().count()

# Test holdout
test_hold_explicit = test_holdout.filter(F.col("rating") > 0)
n_test_hold_explicit = test_hold_explicit.count()
n_test_hold_explicit_users = test_hold_explicit.select("user_id").distinct().count()

print(f"\n  Filter applied: rating > 0")
print(f"  train (rating>0)               : {n_train_explicit:,} interactions, {n_train_explicit_users:,} users")
print(f"  tuning_history (rating>0)      : {n_tuning_hist_explicit:,} interactions")
print(f"  tuning_holdout (rating>0)      : {n_tuning_hold_explicit:,} interactions, {n_tuning_hold_explicit_users:,} users")
print(f"  test_holdout (rating>0)        : {n_test_hold_explicit:,} interactions, {n_test_hold_explicit_users:,} users")
print(f"  fit size (train + tuning_hist) : ~{n_train_explicit + n_tuning_hist_explicit:,} interactions")
print(f"  {'✅ Ready for Explicit ALS' if n_train_explicit > 0 and n_tuning_hold_explicit > 0 else '❌ Check data'}")

# ── 6. Implicit ALS readiness ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. IMPLICIT ALS READINESS CHECK")
print("=" * 70)

# Train
train_implicit = train.filter(F.col("is_read") == 1)
n_train_implicit = train_implicit.count()
n_train_implicit_users = train_implicit.select("user_id").distinct().count()

# Confidence breakdown
read_only     = train.filter((F.col("is_read")==1) & (F.col("is_reviewed")==0)).count()
read_reviewed = train.filter((F.col("is_read")==1) & (F.col("is_reviewed")==1)).count()

# Tuning history
tuning_hist_implicit = tuning_history.filter(F.col("is_read") == 1)
n_tuning_hist_implicit = tuning_hist_implicit.count()

# Tuning holdout
tuning_hold_implicit = tuning_holdout.filter(F.col("is_read") == 1)
n_tuning_hold_implicit = tuning_hold_implicit.count()
n_tuning_hold_implicit_users = tuning_hold_implicit.select("user_id").distinct().count()

# Test holdout
test_hold_implicit = test_holdout.filter(F.col("is_read") == 1)
n_test_hold_implicit = test_hold_implicit.count()
n_test_hold_implicit_users = test_hold_implicit.select("user_id").distinct().count()

print(f"\n  Filter applied: is_read = 1")
print(f"  train (is_read=1)              : {n_train_implicit:,} interactions, {n_train_implicit_users:,} users")
print(f"    → read only (conf=11)        : {read_only:,}  ({100*read_only/n_train_implicit:.1f}%)")
print(f"    → read+reviewed (conf=51)    : {read_reviewed:,}  ({100*read_reviewed/n_train_implicit:.1f}%)")
print(f"  tuning_history (is_read=1)     : {n_tuning_hist_implicit:,} interactions")
print(f"  tuning_holdout (is_read=1)     : {n_tuning_hold_implicit:,} interactions, {n_tuning_hold_implicit_users:,} users")
print(f"  test_holdout (is_read=1)       : {n_test_hold_implicit:,} interactions, {n_test_hold_implicit_users:,} users")
print(f"  fit size (train + tuning_hist) : ~{n_train_implicit + n_tuning_hist_implicit:,} interactions")
print(f"  {'✅ Ready for Implicit ALS' if n_train_implicit > 0 and n_tuning_hold_implicit > 0 else '❌ Check data'}")

# ── 7. Final summary table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. FINAL SUMMARY TABLE")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                    CAPSTONE FINAL DATASET                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  Preprocessing                                                  │
  │    Min interactions filter : 100+                               │
  │    Sample fraction         : 7%                                 │
  │    Total users             : {total_users:,}                         │
  │    History per user        : 10                                 │
  │    Holdout per user        : 20                                 │
  │    Interactions discarded  : 0                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │  Files                                                          │
  │    train.parquet           : {train_final_count:>10,} interactions         │
  │    tuning_history.parquet  : {tuning_hist_count:>10,} interactions         │
  │    tuning_holdout.parquet  : {tuning_hold_count:>10,} interactions         │
  │    test_history.parquet    : {test_hist_count:>10,} interactions         │
  │    test_holdout.parquet    : {test_hold_count:>10,} interactions         │
  ├─────────────────────────────────────────────────────────────────┤
  │  Explicit ALS                                                   │
  │    Train (rating>0)        : {n_train_explicit:>10,} interactions         │
  │    Fit size                : {n_train_explicit + n_tuning_hist_explicit:>10,} interactions         │
  │    Eval users (tuning)     : {n_tuning_hold_explicit_users:>10,} users                │
  │    Eval users (test)       : {n_test_hold_explicit_users:>10,} users                │
  ├─────────────────────────────────────────────────────────────────┤
  │  Implicit ALS                                                   │
  │    Train (is_read=1)       : {n_train_implicit:>10,} interactions         │
  │    Fit size                : {n_train_implicit + n_tuning_hist_implicit:>10,} interactions         │
  │    Eval users (tuning)     : {n_tuning_hold_implicit_users:>10,} users                │
  │    Eval users (test)       : {n_test_hold_implicit_users:>10,} users                │
  └─────────────────────────────────────────────────────────────────┘
""")

print("Analysis complete!")
spark.stop()
