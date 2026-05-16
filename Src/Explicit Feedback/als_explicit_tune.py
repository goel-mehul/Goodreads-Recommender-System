"""
als_explicit_tune.py
─────────────────────────────────────────────────────────────────────────────
Explicit ALS — Hyperparameter Tuning Script (RMSE only)
Fits on  : train + tuning_history  (rating > 0 only)
Evaluates: tuning_holdout          (rating > 0 only)
Metric   : RMSE only (faster than ranking metrics, sufficient for tuning)

Run grid (6 runs total):
  rank     : [10, 20]
  regParam : [0.01, 0.1, 1.0]

Usage:
  spark-submit --deploy-mode client als_explicit_tune.py

Edit RANK and REG_PARAM below between runs.
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# ── HYPERPARAMETERS — edit between runs ───────────────────────────────────
RANK      = 10      # try: 10, 20
REG_PARAM = 0.1     # try: 0.01, 0.1, 1.0
MAX_ITER  = 5       # keep low for tuning speed
# ─────────────────────────────────────────────────────────────────────────

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"
# ─────────────────────────────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName(f"als_explicit_tune_rank{RANK}_reg{REG_PARAM}") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(f"{BASE}/checkpoints")

print(f"\n{'='*60}")
print(f"  ALS EXPLICIT TUNING | rank={RANK} reg={REG_PARAM} iter={MAX_ITER}")
print(f"{'='*60}\n")

# ── 1. Load and filter ────────────────────────────────────────────────────
# rating=0 means no rating given on Goodreads — not a zero-star review.
print("Loading data...")
train_raw = spark.read.parquet(f"{BASE}/train.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

tuning_hist = spark.read.parquet(f"{BASE}/tuning_history.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet") \
    .filter(F.col("rating") > 0) \
    .select("user_id", "book_id", "rating")

# ── 2. Build training matrix ──────────────────────────────────────────────
# Union train + tuning_history so model learns embeddings for tuning users.
fit_data = train_raw.union(tuning_hist)
fit_data.cache()
fit_data.count()

n_fit     = fit_data.count()
n_holdout = tuning_holdout.count()
print(f"  Fit interactions     : {n_fit:,}")
print(f"  Holdout interactions : {n_holdout:,}")

# ── 3. Fit ALS ────────────────────────────────────────────────────────────
print(f"\nFitting ALS (rank={RANK}, reg={REG_PARAM}, maxIter={MAX_ITER})...")
model = ALS(
    rank=RANK,
    maxIter=MAX_ITER,
    regParam=REG_PARAM,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="rating",
    implicitPrefs=False,
    coldStartStrategy="drop",
    nonnegative=False,
    checkpointInterval=2,
).fit(fit_data)
print("  Training complete!")

# ── 4. RMSE on tuning holdout ─────────────────────────────────────────────
# RMSE is sufficient for hyperparameter selection in explicit ALS.
# It directly measures prediction error on ratings which is what ALS optimizes.
# Ranking metrics are computed in the final evaluation script only.
print("Computing RMSE...")
predictions = model.transform(tuning_holdout) \
    .filter(F.col("prediction").isNotNull())
rmse = RegressionEvaluator(
    metricName="rmse", labelCol="rating", predictionCol="prediction"
).evaluate(predictions)

# ── 5. Save results ───────────────────────────────────────────────────────
result_df = spark.createDataFrame([
    ("rank",     str(RANK)),
    ("reg",      str(REG_PARAM)),
    ("max_iter", str(MAX_ITER)),
    ("rmse",     f"{rmse:.6f}"),
], ["metric", "value"])

result_df.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/als_explicit_tuning/rank{RANK}_reg{REG_PARAM}",
    header=True
)

# ── 6. Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TUNING RESULTS | rank={RANK} reg={REG_PARAM} iter={MAX_ITER}")
print(f"{'='*60}")
print(f"  RMSE     : {rmse:.6f}")
print(f"  Saved to : {BASE}/results/als_explicit_tuning/rank{RANK}_reg{REG_PARAM}")
print(f"{'='*60}\n")

fit_data.unpersist()
spark.stop()
