"""
als_explicit_final.py
─────────────────────────────────────────────────────────────────────────────
Explicit ALS — Final Evaluation Script
Runs two models:
  Model 1: fit(train + tuning_history) → evaluate tuning_holdout
  Model 2: fit(train + test_history)   → evaluate test_holdout
Metrics: RMSE | MAP@100 | NDCG@100

Update RANK and REG_PARAM below with best values from als_explicit_tune.py

Usage:
  spark-submit --deploy-mode client als_explicit_final.py
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import StorageLevel

# ── BEST HYPERPARAMETERS — update after tuning ────────────────────────────
RANK      = 10      # best rank from tuning
REG_PARAM = 0.1     # best reg from tuning
MAX_ITER  = 10      # higher than tuning for final model
TOP_K     = 100
# ─────────────────────────────────────────────────────────────────────────

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"
# ─────────────────────────────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName(f"als_explicit_final_rank{RANK}_reg{REG_PARAM}") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(f"{BASE}/checkpoints")

print(f"\n{'='*60}")
print(f"  ALS EXPLICIT FINAL | rank={RANK} reg={REG_PARAM} iter={MAX_ITER}")
print(f"{'='*60}\n")

# ── Helper: manual MAP@100 and NDCG@100 ──────────────────────────────────
def evaluate_ranking(model, holdout_df, label=""):
    """
    Manual MAP@100 and NDCG@100 using posexplode.
    Finds best rank of any holdout book in top-100 per user.
    AP   = 1/best_rank if hit, else 0
    NDCG = 1/log2(best_rank+1) if hit, else 0
    """
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

    labels = holdout_df.groupBy("user_id").agg(
        F.collect_list(F.col("book_id").cast("int")).alias("holdout_books")
    )

    hit_df = recs_exploded.join(
        holdout_df.select("user_id", F.col("book_id").cast("int").alias("book_id")),
        on=["user_id", "book_id"], how="inner"
    ).groupBy("user_id").agg(F.min("rank").alias("best_rank"))

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
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating") \
    .persist(StorageLevel.MEMORY_AND_DISK)
train_raw.count()
print(f"  Train (rating>0) : {train_raw.count():,}")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: Tuning evaluation
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  MODEL 1: TUNING EVALUATION")
print(f"{'='*60}")

tuning_hist = spark.read.parquet(f"{BASE}/tuning_history.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

fit_data_tuning = train_raw.union(tuning_hist)
fit_data_tuning.cache()
fit_data_tuning.count()
print(f"  Fit interactions : {fit_data_tuning.count():,}")

print(f"Fitting ALS...")
model_tuning = ALS(
    rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM,
    userCol="user_id", itemCol="book_id", ratingCol="rating",
    implicitPrefs=False, coldStartStrategy="drop",
    nonnegative=False, checkpointInterval=2,
).fit(fit_data_tuning)
print("  Training complete!")

# RMSE
preds_tuning = model_tuning.transform(tuning_holdout) \
    .filter(F.col("prediction").isNotNull())
rmse_tuning = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
).evaluate(preds_tuning)
print(f"  RMSE: {rmse_tuning:.6f}")

# Ranking metrics
val_map, val_ndcg, val_users, val_hits = evaluate_ranking(
    model_tuning, tuning_holdout, label="Tuning Evaluation"
)

# Save tuning model
tuning_model_path = f"{BASE}/models/als_explicit_model_tuning"
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

test_hist = spark.read.parquet(f"{BASE}/test_history.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

test_holdout = spark.read.parquet(f"{BASE}/test_holdout.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

fit_data_test = train_raw.union(test_hist)
fit_data_test.cache()
fit_data_test.count()
print(f"  Fit interactions : {fit_data_test.count():,}")

print(f"Fitting ALS...")
model_test = ALS(
    rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM,
    userCol="user_id", itemCol="book_id", ratingCol="rating",
    implicitPrefs=False, coldStartStrategy="drop",
    nonnegative=False, checkpointInterval=2,
).fit(fit_data_test)
print("  Training complete!")

# RMSE
preds_test = model_test.transform(test_holdout) \
    .filter(F.col("prediction").isNotNull())
rmse_test = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
).evaluate(preds_test)
print(f"  RMSE: {rmse_test:.6f}")

# Ranking metrics
test_map, test_ndcg, test_users, test_hits = evaluate_ranking(
    model_test, test_holdout, label="Test Evaluation"
)

# Save test model
test_model_path = f"{BASE}/models/als_explicit_model_test"
print(f"\nSaving test model to {test_model_path}...")
model_test.write().overwrite().save(test_model_path)
print("  Model saved!")

fit_data_test.unpersist()
train_raw.unpersist()

# ── Save all results ──────────────────────────────────────────────────────
print("\nSaving results...")
results = spark.createDataFrame([
    ("rank",           str(RANK)),
    ("reg",            str(REG_PARAM)),
    ("max_iter",       str(MAX_ITER)),
    ("tuning_rmse",    f"{rmse_tuning:.6f}"),
    ("tuning_map",     f"{val_map:.6f}"),
    ("tuning_ndcg",    f"{val_ndcg:.6f}"),
    ("tuning_users",   str(val_users)),
    ("tuning_hits",    str(val_hits)),
    ("test_rmse",      f"{rmse_test:.6f}"),
    ("test_map",       f"{test_map:.6f}"),
    ("test_ndcg",      f"{test_ndcg:.6f}"),
    ("test_users",     str(test_users)),
    ("test_hits",      str(test_hits)),
], ["metric", "value"])

results.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/als_explicit_final", header=True
)

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL RESULTS — EXPLICIT ALS")
print(f"  rank={RANK} | reg={REG_PARAM} | maxIter={MAX_ITER}")
print(f"{'='*60}")
print(f"  Tuning RMSE    : {rmse_tuning:.6f}")
print(f"  Tuning MAP@100 : {val_map:.6f}")
print(f"  Tuning NDCG@100: {val_ndcg:.6f}")
print(f"")
print(f"  Test RMSE      : {rmse_test:.6f}")
print(f"  Test MAP@100   : {test_map:.6f}")
print(f"  Test NDCG@100  : {test_ndcg:.6f}")
print(f"")
print(f"  Models saved to: {BASE}/models/")
print(f"  Results saved  : {BASE}/results/als_explicit_final")
print(f"{'='*60}\n")

spark.stop()
