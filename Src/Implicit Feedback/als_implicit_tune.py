"""
als_implicit_tune.py
─────────────────────────────────────────────────────────────────────────────
Implicit ALS — Hyperparameter Tuning Script
Fits on  : train + tuning_history  (is_read = 1 only)
Evaluates: tuning_holdout          (is_read = 1 only)
Metrics  : MAP@100 | NDCG@100 (manual, no RankingEvaluator)

Confidence formula (Hu et al. 2008, dual-alpha):
  confidence = 1 + alpha * is_read + beta * is_reviewed

Two-stage tuning (10 runs total):
  Stage 1 (6 runs): Tune rank + regParam, fix alpha=10, beta=40
    rank     : [10, 20]
    regParam : [0.01, 0.1, 1.0]

  Stage 2 (4 runs): Fix best rank+reg, tune alpha + beta
    alpha : [1, 10]
    beta  : [20, 40]

Usage:
  spark-submit --deploy-mode client als_implicit_tune.py
  Edit STAGE and params below between runs.
─────────────────────────────────────────────────────────────────────────────
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS

# ── STAGE — set to 1 or 2 ────────────────────────────────────────────────
STAGE = 1
# ─────────────────────────────────────────────────────────────────────────

# ── STAGE 1 PARAMS — edit between runs ───────────────────────────────────
RANK      = 10      # try: 10, 20
REG_PARAM = 0.1     # try: 0.01, 0.1, 1.0

# ── STAGE 2 PARAMS — set best rank+reg from Stage 1, edit alpha+beta ─────
BEST_RANK = 10      # fill in after Stage 1
BEST_REG  = 0.1     # fill in after Stage 1
ALPHA     = 10      # try: 1, 10
BETA      = 40      # try: 20, 40
# ─────────────────────────────────────────────────────────────────────────

MAX_ITER = 5
TOP_K    = 100

# ── PATHS ─────────────────────────────────────────────────────────────────
BASE = "hdfs:///user/mg8958_nyu_edu/capstone_final"
# ─────────────────────────────────────────────────────────────────────────

# Resolve active params based on stage
active_rank  = RANK      if STAGE == 1 else BEST_RANK
active_reg   = REG_PARAM if STAGE == 1 else BEST_REG
active_alpha = 10        if STAGE == 1 else ALPHA
active_beta  = 40        if STAGE == 1 else BETA

spark = SparkSession.builder \
    .appName(f"als_implicit_tune_s{STAGE}_rank{active_rank}_reg{active_reg}_a{active_alpha}_b{active_beta}") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.network.timeout", "800s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(f"{BASE}/checkpoints")

print(f"\n{'='*60}")
print(f"  ALS IMPLICIT TUNING | Stage {STAGE}")
print(f"  rank={active_rank} reg={active_reg} alpha={active_alpha} beta={active_beta} iter={MAX_ITER}")
print(f"{'='*60}\n")

# ── 1. Load and build confidence ──────────────────────────────────────────
def add_confidence(df, alpha, beta):
    return df.filter(F.col("is_read") == 1) \
        .select("user_id", "book_id", "is_read", "is_reviewed") \
        .withColumn("confidence",
            1 + alpha * F.col("is_read") + beta * F.col("is_reviewed")
        ).select("user_id", "book_id", "confidence")

print("Loading data...")
train_raw      = add_confidence(spark.read.parquet(f"{BASE}/train.parquet"), active_alpha, active_beta)
tuning_hist    = add_confidence(spark.read.parquet(f"{BASE}/tuning_history.parquet"), active_alpha, active_beta)
tuning_holdout = add_confidence(spark.read.parquet(f"{BASE}/tuning_holdout.parquet"), active_alpha, active_beta)

# ── 2. Build training matrix ──────────────────────────────────────────────
fit_data = train_raw.union(tuning_hist)
fit_data.cache()
fit_data.count()

n_fit         = fit_data.count()
n_holdout_usr = tuning_holdout.select("user_id").distinct().count()
print(f"  Fit interactions : {n_fit:,}")
print(f"  Holdout users    : {n_holdout_usr:,}")

# ── 3. Fit implicit ALS ───────────────────────────────────────────────────
print(f"\nFitting implicit ALS...")
model = ALS(
    rank=active_rank,
    maxIter=MAX_ITER,
    regParam=active_reg,
    userCol="user_id",
    itemCol="book_id",
    ratingCol="confidence",
    implicitPrefs=True,
    coldStartStrategy="drop",
    nonnegative=False,
    checkpointInterval=2,
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
    ("stage",    str(STAGE)),
    ("rank",     str(active_rank)),
    ("reg",      str(active_reg)),
    ("alpha",    str(active_alpha)),
    ("beta",     str(active_beta)),
    ("max_iter", str(MAX_ITER)),
    ("MAP@100",  f"{map_score:.6f}"),
    ("NDCG@100", f"{ndcg_score:.6f}"),
    ("n_users",  str(results_row["n_users"])),
    ("n_hits",   str(results_row["n_hits"])),
], ["metric", "value"])

save_key = f"s{STAGE}_rank{active_rank}_reg{active_reg}_a{active_alpha}_b{active_beta}"
result_df.coalesce(1).write.mode("overwrite").csv(
    f"{BASE}/results/als_implicit_tuning/{save_key}",
    header=True
)

# ── 6. Summary ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TUNING RESULTS | Stage {STAGE}")
print(f"  rank={active_rank} reg={active_reg} alpha={active_alpha} beta={active_beta}")
print(f"{'='*60}")
print(f"  MAP@100  : {map_score:.6f}")
print(f"  NDCG@100 : {ndcg_score:.6f}")
print(f"  Users    : {results_row['n_users']:,}  Hits: {results_row['n_hits']:,}")
print(f"  Saved to : {BASE}/results/als_implicit_tuning/{save_key}")
print(f"{'='*60}\n")

fit_data.unpersist()
spark.stop()
