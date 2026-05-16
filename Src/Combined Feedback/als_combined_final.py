"""
als_combined_final.py
─────────────────────────────────────────────────────────────────────────────
Combined Explicit + Implicit ALS — Final Evaluation Script
Runs two models:
  Model 1: fit(train + tuning_history) → evaluate tuning_holdout
  Model 2: fit(train + test_history)   → evaluate test_holdout
Metrics: MAP@100 | NDCG@100 (manual, no RankingEvaluator)

Combined confidence formula:
  combined_confidence = 1 + alpha*is_read + beta*is_reviewed + w*rating

Where:
  alpha = 10  (is_read weight, from implicit ALS tuning)
  beta  = 20  (is_reviewed weight, from implicit ALS tuning)
  w     = 5   (rating weight, justified as alpha / avg_rating ≈ 10/3.9 ≈ 2.5 → 5)

Inclusion rule: any row where is_read=1 OR rating>0 OR is_reviewed=1
This captures all meaningful interactions across both explicit and implicit signals.

ALS mode: implicitPrefs=True
  - Input treated as confidence c_ui, not a literal rating
  - Missing interactions treated as weak negatives (confidence=1)
  - Model learns: how confident are we that user u prefers item i?

Usage:
  spark-submit --deploy-mode client als_combined_final.py
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark import StorageLevel

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────
RANK      = 20      # best from implicit ALS tuning
REG_PARAM = 0.1     # best from implicit ALS tuning
MAX_ITER  = 10
TOP_K     = 100

# Combined confidence weights
ALPHA = 10    # is_read weight
BETA  = 20    # is_reviewed weight
W     = 5     # rating weight (justified: alpha / avg_rating ≈ 10/3.9 → 5)
# ─────────────────────────────────────────────────────────────────────────

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"
# ─────────────────────────────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName(f"als_combined_final_rank{RANK}_reg{REG_PARAM}") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(f"{BASE}/checkpoints")

print(f"\n{'='*60}")
print(f"  ALS COMBINED FINAL | rank={RANK} reg={REG_PARAM} iter={MAX_ITER}")
print(f"  alpha={ALPHA} beta={BETA} w={W}")
print(f"  Formula: 1 + {ALPHA}*is_read + {BETA}*is_reviewed + {W}*rating")
print(f"{'='*60}\n")

# ── Helper: build combined confidence ─────────────────────────────────────
def build_combined_confidence(df, alpha, beta, w):
    """
    Combined confidence formula:
      confidence = 1 + alpha*is_read + beta*is_reviewed + w*rating

    Inclusion: any row with is_read=1 OR rating>0 OR is_reviewed=1
    This captures all meaningful signals from both explicit and implicit feedback.

    For rows with rating=0 (no explicit rating):
      confidence = 1 + alpha*is_read + beta*is_reviewed
      (same as implicit-only formula)

    For rows with rating>0 and is_read=1:
      confidence = 1 + alpha + beta*is_reviewed + w*rating
      (explicit rating boosts confidence on top of implicit signals)

    A 5-star rating adds w*5=25 confidence — strong positive signal.
    A 1-star rating adds w*1=5 confidence — weak, almost like not reading.
    """
    return df.filter(
        (F.col("is_read") == 1) |
        (F.col("rating") > 0) |
        (F.col("is_reviewed") == 1)
    ).withColumn(
        "confidence",
        F.lit(1.0)
        + F.lit(float(alpha)) * F.col("is_read").cast("double")
        + F.lit(float(beta))  * F.col("is_reviewed").cast("double")
        + F.lit(float(w))     * F.col("rating").cast("double")
    ).select(
        F.col("user_id").cast("int"),
        F.col("book_id").cast("int"),
        F.col("confidence").cast("float")
    )

# ── Helper: manual MAP@100 and NDCG@100 ──────────────────────────────────
def evaluate_ranking(model, holdout_df, label=""):
    eval_users = holdout_df.select("user_id").distinct()
    recs = model.recommendForUserSubset(eval_users, TOP_K)

    recs_exploded = recs.select(
        "user_id",
        F.posexplode("recommendations").alias("rank_0indexed", "rec")
    ).select(
        "user_id",
        (F.col("rank_0indexed") + 1).alias("rank"),
        F.col("rec.book_id").cast("int").alias("book_id")
    )

    hit_df = recs_exploded.join(
        holdout_df.select("user_id", F.col("book_id").cast("int").alias("book_id")),
        on=["user_id", "book_id"], how="inner"
    ).groupBy("user_id").agg(F.min("rank").alias("best_rank"))

    labels = holdout_df.groupBy("user_id").agg(
        F.collect_list(F.col("book_id").cast("int")).alias("holdout_books")
    )

    eval_df = labels.join(hit_df, on="user_id", how="left") \
        .withColumn("ap",
            F.when(F.col("best_rank").isNotNull(), 1.0 / F.col("best_rank"))
            .otherwise(0.0)
        ).withColumn("ndcg",
            F.when(F.col("best_rank").isNotNull(), 1.0 / F.log2(F.col("best_rank") + 1))
            .otherwise(0.0)
        )

    row = eval_df.agg(
        F.mean("ap").alias("MAP"),
        F.mean("ndcg").alias("NDCG"),
        F.count("user_id").alias("n_users"),
        F.sum(F.when(F.col("best_rank").isNotNull(), 1).otherwise(0)).alias("n_hits")
    ).collect()[0]

    map_score  = row["MAP"]  or 0.0
    ndcg_score = row["NDCG"] or 0.0

    print(f"\n── {label} ──")
    print(f"  Users evaluated : {row['n_users']:,}")
    print(f"  Users with hit  : {row['n_hits']:,}")
    print(f"  MAP@100         : {map_score:.6f}")
    print(f"  NDCG@100        : {ndcg_score:.6f}")

    return map_score, ndcg_score, row["n_users"], row["n_hits"]

# ── 1. Load base train ────────────────────────────────────────────────────
print("Loading train data...")
train_raw = spark.read.parquet(f"{BASE}/train.parquet") \
    .persist(StorageLevel.MEMORY_AND_DISK)
train_raw.count()

train_conf = build_combined_confidence(train_raw, ALPHA, BETA, W)
train_conf.cache()
n_train = train_conf.count()
print(f"  Train combined interactions : {n_train:,}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Tuning evaluation
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  MODEL 1: TUNING EVALUATION")
print(f"{'='*60}")

tuning_hist = build_combined_confidence(
    spark.read.parquet(f"{BASE}/tuning_history.parquet"), ALPHA, BETA, W
)
tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet")

fit_data_tuning = train_conf.union(tuning_hist)
fit_data_tuning.cache()
fit_data_tuning.count()
print(f"  Fit interactions : {fit_data_tuning.count():,}")

print(f"Fitting combined ALS...")
model_tuning = ALS(
    rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM,
    userCol="user_id", itemCol="book_id", ratingCol="confidence",
    implicitPrefs=True, coldStartStrategy="drop",
    nonnegative=False, checkpointInterval=2,
).fit(fit_data_tuning)
print("  Training complete!")

val_map, val_ndcg, val_users, val_hits = evaluate_ranking(
    model_tuning, tuning_holdout, label="Tuning Evaluation"
)

tuning_model_path = f"{BASE}/models/als_combined_model_tuning"
print(f"\nSaving tuning model to {tuning_model_path}...")
model_tuning.write().overwrite().save(tuning_model_path)
print("  Model saved!")

fit_data_tuning.unpersist()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: Test evaluation
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  MODEL 2: TEST EVALUATION")
print(f"{'='*60}")

test_hist = build_combined_confidence(
    spark.read.parquet(f"{BASE}/test_history.parquet"), ALPHA, BETA, W
)
test_holdout = spark.read.parquet(f"{BASE}/test_holdout.parquet")

fit_data_test = train_conf.union(test_hist)
fit_data_test.cache()
fit_data_test.count()
print(f"  Fit interactions : {fit_data_test.count():,}")

print(f"Fitting combined ALS...")
model_test = ALS(
    rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM,
    userCol="user_id", itemCol="book_id", ratingCol="confidence",
    implicitPrefs=True, coldStartStrategy="drop",
    nonnegative=False, checkpointInterval=2,
).fit(fit_data_test)
print("  Training complete!")

test_map, test_ndcg, test_users, test_hits = evaluate_ranking(
    model_test, test_holdout, label="Test Evaluation"
)

test_model_path = f"{BASE}/models/als_combined_model_test"
print(f"\nSaving test model to {test_model_path}...")
model_test.write().overwrite().save(test_model_path)
print("  Model saved!")

fit_data_test.unpersist()
train_conf.unpersist()
train_raw.unpersist()

# ── Save results ──────────────────────────────────────────────────────────
print("\nSaving results...")
results = spark.createDataFrame([
    ("rank",         str(RANK)),
    ("reg",          str(REG_PARAM)),
    ("alpha",        str(ALPHA)),
    ("beta",         str(BETA)),
    ("w",            str(W)),
    ("max_iter",     str(MAX_ITER)),
    ("formula",      f"1 + {ALPHA}*is_read + {BETA}*is_reviewed + {W}*rating"),
    ("tuning_map",   f"{val_map:.6f}"),
    ("tuning_ndcg",  f"{val_ndcg:.6f}"),
    ("tuning_users", str(val_users)),
    ("tuning_hits",  str(val_hits)),
    ("test_map",     f"{test_map:.6f}"),
    ("test_ndcg",    f"{test_ndcg:.6f}"),
    ("test_users",   str(test_users)),
    ("test_hits",    str(test_hits)),
], ["metric", "value"])

results.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/als_combined_final", header=True
)

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL RESULTS — COMBINED ALS")
print(f"  rank={RANK} | reg={REG_PARAM} | maxIter={MAX_ITER}")
print(f"  Formula: 1 + {ALPHA}*is_read + {BETA}*is_reviewed + {W}*rating")
print(f"{'='*60}")
print(f"  Tuning MAP@100  : {val_map:.6f}")
print(f"  Tuning NDCG@100 : {val_ndcg:.6f}")
print(f"")
print(f"  Test MAP@100    : {test_map:.6f}")
print(f"  Test NDCG@100   : {test_ndcg:.6f}")
print(f"")
print(f"  Models saved to : {BASE}/models/")
print(f"  Results saved   : {BASE}/results/als_combined_final")
print(f"{'='*60}\n")

spark.stop()
