"""
als_combined_tune.py
─────────────────────────────────────────────────────────────────────────────
Combined Explicit + Implicit ALS — Hyperparameter Tuning Script
Fits on  : train + tuning_history  (combined confidence signal)
Evaluates: tuning_holdout
Metrics  : MAP@100 | NDCG@100 (manual, no RankingEvaluator)

Combined confidence formula:
  confidence = 1 + alpha*is_read + beta*is_reviewed + w*rating

Tuning (4 runs): tune rating weight w only
  rank=20, reg=0.1, alpha=10, beta=20 fixed from prior ALS tuning
  w: [1, 5, 10, 20]

Justification for fixing rank/reg:
  Both explicit ALS (RMSE-based) and implicit ALS (MAP-based) tuning
  independently selected rank=20, reg=0.1 as best on this dataset.

Usage:
  spark-submit --deploy-mode client als_combined_tune.py
  Edit W below between runs.
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

# ── TUNE THIS — edit between runs ─────────────────────────────────────────
W = 5       # try: 1, 5, 10, 20
# ─────────────────────────────────────────────────────────────────────────

# ── FIXED HYPERPARAMETERS ─────────────────────────────────────────────────
RANK      = 20      # best from both explicit + implicit ALS tuning
REG_PARAM = 0.1     # best from both explicit + implicit ALS tuning
ALPHA     = 10      # is_read weight, from implicit ALS tuning
BETA      = 20      # is_reviewed weight, from implicit ALS tuning
MAX_ITER  = 5       # keep low for tuning speed
TOP_K     = 100
# ─────────────────────────────────────────────────────────────────────────

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"
# ─────────────────────────────────────────────────────────────────────────

spark = SparkSession.builder \
    .appName(f"als_combined_tune_w{W}") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(f"{BASE}/checkpoints")

print(f"\n{'='*60}")
print(f"  ALS COMBINED TUNING | w={W}")
print(f"  rank={RANK} reg={REG_PARAM} alpha={ALPHA} beta={BETA} iter={MAX_ITER}")
print(f"  Formula: 1 + {ALPHA}*is_read + {BETA}*is_reviewed + {W}*rating")
print(f"{'='*60}\n")

# ── Helper: build combined confidence ─────────────────────────────────────
def build_combined_confidence(df, alpha, beta, w):
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

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading data...")
train_conf = build_combined_confidence(
    spark.read.parquet(f"{BASE}/train.parquet"), ALPHA, BETA, W
)
tuning_hist = build_combined_confidence(
    spark.read.parquet(f"{BASE}/tuning_history.parquet"), ALPHA, BETA, W
)
tuning_holdout = spark.read.parquet(f"{BASE}/tuning_holdout.parquet")

# ── 2. Build training matrix ──────────────────────────────────────────────
fit_data = train_conf.union(tuning_hist)
fit_data.cache()
fit_data.count()

n_fit         = fit_data.count()
n_holdout_usr = tuning_holdout.select("user_id").distinct().count()
print(f"  Fit interactions : {n_fit:,}")
print(f"  Holdout users    : {n_holdout_usr:,}")

# ── 3. Fit combined ALS ───────────────────────────────────────────────────
print(f"\nFitting combined ALS...")
model = ALS(
    rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM,
    userCol="user_id", itemCol="book_id", ratingCol="confidence",
    implicitPrefs=True, coldStartStrategy="drop",
    nonnegative=False, checkpointInterval=2,
).fit(fit_data)
print("  Training complete!")

# ── 4. Ranking metrics (manual) ───────────────────────────────────────────
print(f"Generating top-{TOP_K} recommendations...")
eval_users = tuning_holdout.select("user_id").distinct()
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
    tuning_holdout.select("user_id", F.col("book_id").cast("int").alias("book_id")),
    on=["user_id", "book_id"], how="inner"
).groupBy("user_id").agg(F.min("rank").alias("best_rank"))

labels = tuning_holdout.groupBy("user_id").agg(
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

results_row = eval_df.agg(
    F.mean("ap").alias("MAP"),
    F.mean("ndcg").alias("NDCG"),
    F.count("user_id").alias("n_users"),
    F.sum(F.when(F.col("best_rank").isNotNull(), 1).otherwise(0)).alias("n_hits")
).collect()[0]

map_score  = results_row["MAP"]  or 0.0
ndcg_score = results_row["NDCG"] or 0.0

# ── 5. Save results ───────────────────────────────────────────────────────
result_df = spark.createDataFrame([
    ("w",        str(W)),
    ("rank",     str(RANK)),
    ("reg",      str(REG_PARAM)),
    ("alpha",    str(ALPHA)),
    ("beta",     str(BETA)),
    ("max_iter", str(MAX_ITER)),
    ("MAP@100",  f"{map_score:.6f}"),
    ("NDCG@100", f"{ndcg_score:.6f}"),
    ("n_users",  str(results_row["n_users"])),
    ("n_hits",   str(results_row["n_hits"])),
], ["metric", "value"])

result_df.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/als_combined_tuning/w{W}",
    header=True
)

# ── 6. Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TUNING RESULTS | w={W}")
print(f"  rank={RANK} reg={REG_PARAM} alpha={ALPHA} beta={BETA}")
print(f"{'='*60}")
print(f"  MAP@100  : {map_score:.6f}")
print(f"  NDCG@100 : {ndcg_score:.6f}")
print(f"  Users    : {results_row['n_users']:,}  Hits: {results_row['n_hits']:,}")
print(f"  Saved to : {BASE}/results/als_combined_tuning/w{W}")
print(f"{'='*60}\n")

fit_data.unpersist()
spark.stop()
