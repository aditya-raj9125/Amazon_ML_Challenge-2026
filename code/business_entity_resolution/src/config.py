# =============================================================
# config.py — All tuneable hyper-parameters & path constants
# =============================================================
# Edit this file before every run instead of hunting through the notebook.
# Every other module imports ONLY from here — no magic numbers elsewhere.

import os

# ─── Data Paths (Auto-detects Local / SageMaker vs Colab Drive) ───────────────
def _find_data_root() -> str:
    # 1. Check environment variable override
    if "AMAZON_ML_DATA" in os.environ:
        env_path = os.environ["AMAZON_ML_DATA"]
        if os.path.exists(os.path.join(env_path, "train", "train_source1.tsv")):
            return env_path
        if os.path.basename(env_path) == "train" and os.path.exists(os.path.join(env_path, "train_source1.tsv")):
            return os.path.dirname(env_path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 2. Search for the actual train_source1.tsv file inside the repository
    for root, dirs, files in os.walk(repo_root):
        # Skip hidden cache directories like .cache
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if "train_source1.tsv" in files:
            if os.path.basename(root) == "train":
                return os.path.dirname(root)
            return root

    # 3. Fallbacks
    candidate_paths = [
        os.path.join(repo_root, "dataset"),
        os.path.join(repo_root, "dataset", "student_resource", "dataset"),
        os.path.join(repo_root, "dataset", "student_resource"),
        os.path.join(repo_root, "dataset", "dataset"),
        "/content/drive/MyDrive/Amazon_ML_Challenge_2026/Datasets/student_resource/dataset",
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, "train", "train_source1.tsv")):
            return p
    return candidate_paths[0]

DATA_ROOT = _find_data_root()

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

TRAIN_S1  = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2  = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3  = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT  = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1   = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_S2   = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_S3   = os.path.join(TEST_DIR, "test_source3.tsv")

# ─── Output Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR          = os.path.join(os.path.dirname(__file__), "..", "..", "output")
MATCHING_OUT        = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT       = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ─── Model Artifacts ─────────────────────────────────────────────────────────
ARTIFACTS_DIR       = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH          = os.path.join(ARTIFACTS_DIR, "lgbm_matcher.pkl")
TFIDF_NAME_PATH     = os.path.join(ARTIFACTS_DIR, "tfidf_name.pkl")
TFIDF_ADDR_PATH     = os.path.join(ARTIFACTS_DIR, "tfidf_addr.pkl")
EMBED_S1_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "embed_s1_train.npy")
EMBED_S2_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "embed_s2_train.npy")
EMBED_S3_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "embed_s3_train.npy")
EMBED_S1_TEST_PATH  = os.path.join(ARTIFACTS_DIR, "embed_s1_test.npy")
EMBED_S2_TEST_PATH  = os.path.join(ARTIFACTS_DIR, "embed_s2_test.npy")
EMBED_S3_TEST_PATH  = os.path.join(ARTIFACTS_DIR, "embed_s3_test.npy")

# ─── Parquet Cache ────────────────────────────────────────────────────────────
# TSV files are loaded once, converted to zstd-compressed parquet, and cached
# alongside the original TSV (same directory). Subsequent loads skip TSV parsing
# entirely — Polars reads parquet ~10x faster than TSV at this row count.
# Set to None to disable caching (not recommended).
PARQUET_CACHE_DIR   = None   # None = cache next to original TSV file

# ─── Train / Validation Split ─────────────────────────────────────────────────
# Split is on SOURCE-1 entity_ids (group-split — no leakage).
# 80% train, 20% validation, stratified by match_count bucket.
VAL_FRACTION        = 0.20
RANDOM_SEED         = 42

# ─── Blocking Parameters ──────────────────────────────────────────────────────
# ANN: top-k nearest neighbors retrieved from FAISS per S1 entity
ANN_TOP_K           = 30

# Sorted-neighborhood window size (over normalized name)
SNM_WINDOW          = 5

# Country safety-net: even with country partitioning, still accept a cross-country
# pair if BOTH name-char3-jaccard > this threshold AND address-token-jaccard > this threshold.
# Cheap insurance against France mislabeling.
CROSS_COUNTRY_NAME_THRESH = 0.95
CROSS_COUNTRY_ADDR_THRESH = 0.90

# ─── Embedding Model ──────────────────────────────────────────────────────────
# MIT-licensed, multilingual (handles Hindi-transliteration + French).
# ~117M params — well under the 8B cap.
# L6 is 2x faster than L12 with negligible quality drop for similarity search.
# Both are MIT licensed, 384-dim output, multilingual (Hindi + French).
EMBED_MODEL_NAME    = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"
EMBED_BATCH_SIZE    = 1024      # fp16 uses half memory → double the batch size vs fp32
EMBED_MAX_SEQ_LEN   = 128       # name + address is usually < 80 tokens

# ─── TF-IDF ───────────────────────────────────────────────────────────────────
TFIDF_ANALYZER      = "char_wb"
TFIDF_NGRAM_RANGE   = (2, 4)
TFIDF_MAX_FEATURES  = 200_000

# ─── LightGBM Hyper-parameters ────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       127,
    "max_depth":        -1,
    "learning_rate":    0.05,
    "n_estimators":     1000,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    # F0.5 is precision-heavy; down-weight positives to bias toward precision.
    # Start at 0.5 and tune on val F0.5. Counter-intuitively, LOWER = more precision.
    "scale_pos_weight": 0.5,
    "n_jobs":           -1,
    "random_state":     RANDOM_SEED,
    "verbose":          -1,
}

LGBM_EARLY_STOPPING_ROUNDS = 50

# ─── Negative Sampling ────────────────────────────────────────────────────────
# Ratio of hard negatives to positives in training set.
# Hard negatives = same-block non-matches (most informative).
NEG_TO_POS_RATIO    = 8

# ─── Threshold Sweep ─────────────────────────────────────────────────────────
# Range and step for threshold sweep on validation set.
THRESHOLD_LOW       = 0.30
THRESHOLD_HIGH      = 0.95
THRESHOLD_STEP      = 0.01

# ─── One-to-One Post-processing ───────────────────────────────────────────────
# For each S2/S3 entity that appears in more than one accepted pair,
# keep only the S1 partner with the highest model score.
ENABLE_ONE_TO_ONE_DEDUP = True

# ─── Graph Consistency Pruning ────────────────────────────────────────────────
# For each S1 entity with multiple accepted matches, compute pairwise
# similarity between the matched S2/S3 records. If the minimum pairwise
# similarity is below this threshold, drop the weakest edge.
ENABLE_GRAPH_PRUNING        = True
GRAPH_PRUNE_MIN_SIMILARITY  = 0.25   # name-char3-jaccard floor among matched set
