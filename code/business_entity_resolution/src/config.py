# =============================================================
# config.py — All tuneable hyper-parameters & path constants
# =============================================================
# Edit this file before every run instead of hunting through the notebook.
# Every other module imports ONLY from here — no magic numbers elsewhere.
#
# ── HOW DATA PATHS WORK ──────────────────────────────────────────────────────
# Notebook Cell 2 copies TSV files from Google Drive to the EC2 instance at:
#   /home/ec2-user/SageMaker/Amazon_ML_Challenge-2026/dataset/
#       train/ → train_source1.tsv, train_source2.tsv, train_source3.tsv,
#                 train_ground_truth.tsv
#       test/  → test_source1.tsv, test_source2.tsv, test_source3.tsv
#
# Cell 2 also sets the environment variable AMAZON_ML_DATA to point to that
# directory so config.py finds it immediately — no filesystem walking needed.
# SAGEMAKER INSTANCE: ml.g5.2xlarge (32 GB RAM, 75 GB EBS)
# → Budget: 32 GB RAM (tight — aggressive chunking & memory caps applied)
# → GPU: 1× NVIDIA A10G (24 GB VRAM) — excellent for embeddings + ANN
# → vCPUs: 8
# ─────────────────────────────────────────────────────────────────────────────

import os

# ─── Repository Root ─────────────────────────────────────────────────────────
# config.py lives at: <REPO_ROOT>/code/business_entity_resolution/src/config.py
# Going up 3 directory levels reaches the actual repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ─── EBS Cache & Temp Directories (redirect off small root OS partition) ─────
CACHE_DIR       = os.path.join(REPO_ROOT, ".cache")
HF_CACHE_DIR    = os.path.join(CACHE_DIR, "huggingface")
TORCH_CACHE_DIR = os.path.join(CACHE_DIR, "torch")
TMP_DIR         = os.path.join(CACHE_DIR, "tmp")

os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["TORCH_HOME"] = TORCH_CACHE_DIR
os.environ["TMPDIR"] = TMP_DIR

# ─── Data Root ───────────────────────────────────────────────────────────────
def _is_valid_data_root(path: str) -> bool:
    """True if the directory actually contains training TSV files."""
    if not path or not os.path.exists(path):
        return False
    train_sub = os.path.join(path, "train")
    for base in (train_sub, path):
        for fname in ("train_source1.tsv", "source1.tsv"):
            if os.path.exists(os.path.join(base, fname)):
                return True
    return False

_DATA_ROOT_CANDIDATES = [
    os.environ.get("AMAZON_ML_DATA", ""),
    os.path.join(REPO_ROOT, "dataset"),
    os.path.join(REPO_ROOT, "Datasets"),
    os.path.join(REPO_ROOT, "data"),
    "/home/ec2-user/SageMaker/Amazon_ML_Challenge-2026/dataset",
    "/home/ec2-user/SageMaker/dataset",
    "/home/ec2-user/SageMaker",
]

DATA_ROOT = next(
    (p for p in _DATA_ROOT_CANDIDATES if _is_valid_data_root(p)),
    os.path.join(REPO_ROOT, "dataset"),  # fallback — pipeline will raise a clear error
)

# ─── Train / Test directories ─────────────────────────────────────────────────
TRAIN_DIR = (
    os.path.join(DATA_ROOT, "train")
    if os.path.isdir(os.path.join(DATA_ROOT, "train"))
    else DATA_ROOT
)
TEST_DIR = (
    os.path.join(DATA_ROOT, "test")
    if os.path.isdir(os.path.join(DATA_ROOT, "test"))
    else DATA_ROOT
)

# ─── Resolve individual files ─────────────────────────────────────────────────
def _find_file(directory: str, candidates: list) -> str:
    """Return first existing path; fall back to <directory>/<candidates[0]> (non-existent → clear error)."""
    for c in candidates:
        p = os.path.join(directory, c)
        if os.path.exists(p):
            return p
    for c in candidates:
        p = os.path.join(os.path.dirname(directory), c)
        if os.path.exists(p):
            return p
    return os.path.join(directory, candidates[0])  # non-existent → FileNotFoundError in pipeline

TRAIN_S1 = _find_file(TRAIN_DIR, ["train_source1.tsv", "source1.tsv"])
TRAIN_S2 = _find_file(TRAIN_DIR, ["train_source2.tsv", "source2.tsv"])
TRAIN_S3 = _find_file(TRAIN_DIR, ["train_source3.tsv", "source3.tsv"])
TRAIN_GT = _find_file(TRAIN_DIR, ["train_ground_truth.tsv", "ground_truth.tsv"])

TEST_S1  = _find_file(TEST_DIR, ["test_source1.tsv", "source1.tsv"])
TEST_S2  = _find_file(TEST_DIR, ["test_source2.tsv", "source2.tsv"])
TEST_S3  = _find_file(TEST_DIR, ["test_source3.tsv", "source3.tsv"])

# ─── Output Paths ────────────────────────────────────────────────────────────
OUTPUT_DIR    = os.path.join(REPO_ROOT, "output")
MATCHING_OUT  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

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

# ─── Checkpoint Cache ─────────────────────────────────────────────────────────
# For resuming after kernel crashes. Saved as parquet/pickle.
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# ─── Parquet Cache ────────────────────────────────────────────────────────────
PARQUET_CACHE_DIR = None

# ─── Train / Validation Split ─────────────────────────────────────────────────
VAL_FRACTION = 0.20
RANDOM_SEED  = 42

# ─── Blocking Parameters ──────────────────────────────────────────────────────
ANN_TOP_K  = 30   # ANN top-k neighbours per S1 entity per source
SNM_WINDOW = 5    # sorted-neighborhood sliding window width (used in fallback)

# TF-IDF blocking parameters (from teammate's best config)
TFIDF_COMB_K = 50    # combined name+addr word TF-IDF top-K
TFIDF_ADDR_K = 20    # address-only TF-IDF top-K
TFIDF_CHAR_K = 15    # char 4-gram no-space name TF-IDF top-K
TFIDF_REV_K  = 5     # reverse (pool→S1) top-K

# Cross-country safety-net (insurance against France mislabelling)
CROSS_COUNTRY_NAME_THRESH = 0.95
CROSS_COUNTRY_ADDR_THRESH = 0.90

# ─── Embedding Model ──────────────────────────────────────────────────────────
# MIT-licensed, multilingual (Hindi-transliteration + French zero-shot).
# 117M params — well under 8B cap. 384-dim output.
EMBED_MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE  = 512    # Safe for 24 GB VRAM on NVIDIA A10G (consumes ~2.5 GB peak VRAM)
EMBED_MAX_SEQ_LEN = 128

# ─── TF-IDF ───────────────────────────────────────────────────────────────────
TFIDF_ANALYZER     = "char_wb"
TFIDF_NGRAM_RANGE  = (2, 4)
TFIDF_MAX_FEATURES = 200_000

# ─── LightGBM Hyper-parameters ────────────────────────────────────────────────
# Tuned for high precision (F0.5 is precision-heavy):
# - 255 leaves (more capacity to learn complex distractor patterns)
# - 1500 rounds with early stopping
# - scale_pos_weight < 1 for precision bias
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "boosting_type":     "gbdt",
    "num_leaves":        255,           # increased from 127 for more capacity
    "max_depth":         -1,
    "learning_rate":     0.05,
    "n_estimators":      1500,          # increased from 1000
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    # F0.5 is precision-heavy; scale_pos_weight < 1 → more precision.
    "scale_pos_weight":  0.5,
    "n_jobs":            8,             # exactly 8 vCPUs on ml.g5.2xlarge
    "random_state":      RANDOM_SEED,
    "verbose":           -1,
}

LGBM_EARLY_STOPPING_ROUNDS = 100    # increased from 50 for more patience

# ─── Training Pair & Memory Bounds (32 GB RAM Safe) ───────────────────────────
# Higher ratio = more hard negatives = better distractor discrimination.
# Capped at 2M total pairs so RAM usage during training stays under 400 MB.
NEG_TO_POS_RATIO    = 4            # 4:1 negative-to-positive ratio
MAX_TRAIN_S1_GROUPS = 250_000      # Subsample S1 entities for training (teammate best practice)
MAX_TRAIN_PAIRS     = 2_000_000    # Strict ceiling on total training pairs (prevents OOM)
MAX_VAL_SWEEP_S1    = 50_000       # Subsample val S1 entities for threshold sweep
INFER_BATCH_SIZE    = 50_000       # Streaming batch size for inference

# ─── Threshold Sweep ─────────────────────────────────────────────────────────
THRESHOLD_LOW  = 0.30
THRESHOLD_HIGH = 0.95
THRESHOLD_STEP = 0.005   # finer resolution (was 0.01)

# ─── Post-processing ──────────────────────────────────────────────────────────
ENABLE_ONE_TO_ONE_DEDUP    = True
ENABLE_GRAPH_PRUNING       = True
GRAPH_PRUNE_MIN_SIMILARITY = 0.25   # name-char3-jaccard floor among matched set

