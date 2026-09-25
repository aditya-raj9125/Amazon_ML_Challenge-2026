# Amazon ML Challenge 2026 — Business Entity Resolution

## Quick Start (Google Colab)

### Step 1 — Upload the repo to Google Drive
```
Amazon_ML_Challenge_2026/
├── code/
│   └── business_entity_resolution/
│       ├── Amazon_ML_Challenge_2026.py   ← open this in Colab as notebook
│       ├── src/
│       └── artifacts/                    ← auto-created, holds model + embeddings
├── output/                               ← auto-created, holds TSV outputs
├── utils/
│   └── validate_submission.py
└── dataset/                              ← place the competition TSV files here
    ├── train/
    │   ├── train_source1.tsv
    │   ├── train_source2.tsv
    │   ├── train_source3.tsv
    │   └── train_ground_truth.tsv
    └── test/
        ├── test_source1.tsv
        ├── test_source2.tsv
        └── test_source3.tsv
```

### Step 2 — Open notebook in Colab

1. In Google Drive, right-click `Amazon_ML_Challenge_2026.py`
2. Open With → Google Colaboratory
   *(If not available: Colab → File → Open notebook → Upload)*
3. Or: use **File → Upload notebook** and select the `.py` file — Colab reads
   `# %%` cell markers natively via the Jupytext extension.

**Alternatively**, convert to `.ipynb` first:
```bash
pip install jupytext
jupytext --to notebook Amazon_ML_Challenge_2026.py
```

### Step 3 — Configure paths

Edit `src/config.py`, line `DATA_ROOT`:
```python
DATA_ROOT = "/content/drive/MyDrive/Amazon_ML_Challenge_2026/dataset"
```
Match wherever your dataset folder lives in Drive.

### Step 4 — Run cells top to bottom

| Cell | Stage | Estimated Time (Colab T4) |
|---|---|---|
| 1 | Install packages | 3 min |
| 2 | Mount Drive + imports | 30 sec |
| 3 | Load train data | 2 min |
| 4 | Normalise text | 25 min |
| 5 | Encode train (GPU) | 90-120 min |
| 6 | Train blocking candidates | 15 min |
| 7 | Measure blocking recall | 5 min |
| 8 | Train/val split | <1 min |
| 9 | Build record lookups | 10 min |
| 10 | Build training pairs | 20 min |
| 11 | Train LightGBM | 10 min |
| 12 | Threshold sweep | 15 min |
| 13 | Validation evaluation | 10 min |
| 14 | Load & encode test | 90-120 min |
| 15 | Test blocking | 15 min |
| 16 | Test lookup | 10 min |
| 17 | Inference + write output | 30 min |
| 18 | Validate submission | 5 min |
| 19 | Preview output | <1 min |
| 20 | Download for upload | <1 min |

**Total: ~8-9 hours end-to-end.** The encoding steps (Cells 5 & 14) dominate.
They are **cached** — restarting the runtime skips them on re-run.

---

## SageMaker Migration

If Colab GPU quota runs out, switch to a SageMaker `ml.g4dn.xlarge` (T4, 16GB):

```bash
# In a SageMaker terminal:
cd /home/ec2-user/SageMaker/Amazon_ML_Challenge_2026

# Set the data path
export AMAZON_ML_DATA=/home/ec2-user/SageMaker/dataset

# Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# Run the pipeline as a script
# (pipeline.py stages can be called from a __main__ block or Jupyter kernel)
```

---

## Repository Structure

```
code/business_entity_resolution/
├── Amazon_ML_Challenge_2026.py   Main notebook (20 cells, run top-to-bottom)
├── requirements.txt              Pinned dependencies
├── artifacts/                    Auto-created: model, TF-IDF, embeddings
└── src/
    ├── config.py         All hyper-parameters and paths (edit before each run)
    ├── normalize.py      Text normalisation (NFKC, legal suffix, addr abbrev)
    ├── features.py       Pairwise feature engineering (~24 features)
    ├── blocking.py       Multi-strategy candidate generation + FAISS ANN
    ├── train_eval.py     LightGBM training, threshold sweep, F0.5 evaluation
    ├── postprocess.py    One-to-one dedup + graph consistency pruning
    ├── predict.py        Batched test-set inference + TSV output writing
    └── pipeline.py       Orchestrator (called from notebook cells)

output/
├── matching_results.tsv          ← UPLOAD THIS to the leaderboard
└── candidate_pairs.tsv           ← Include in final submission zip

utils/
└── validate_submission.py        Stdlib-only output validator
```

---

## Key Design Decisions

### Why LightGBM, not a Transformer end-to-end?

1. **Label efficiency**: GBDTs beat fine-tuned transformers on tabular feature
   vectors at typical ER label budgets (< 10M pairs). Ditto (VLDB 2020) showed
   this on company-matching benchmarks.
2. **Open-set generalisation (France)**: Every feature is a *similarity number*,
   not raw tokens. A 0.82 cosine similarity means the same thing for French,
   Hindi-transliterated, or English text. A transformer's token embeddings are
   shaped by its training vocabulary — France introduces new subwords it
   has no labeled examples of.
3. **Precision calibration**: F0.5 weights precision 2×. LightGBM with a
   tuned `scale_pos_weight < 1.0` gives well-calibrated probabilities. Cross-
   encoders fine-tuned on limited positive pairs are often poorly calibrated.
4. **Speed**: Sub-millisecond inference per pair. A cross-encoder requires a
   full transformer forward pass per pair — 100× slower on tens of millions
   of candidate pairs.

### Why the bi-encoder (multilingual MiniLM) is still used:

- As the **FAISS ANN blocking layer**: the single highest-recall blocking
  strategy, especially for transliteration and French records where lexical
  keys fail.
- As a **single feature** (embed_cosine) into LightGBM: semantic similarity
  that char-ngram / Levenshtein cannot compute.

### One-to-one dedup

EDA finding: 0 multi-link S2/S3 records in 7.64M training GT pairs.
Every S2/S3 entity belongs to at most one S1 entity.
If the model scores the same S2/S3 as matching two S1 entities, we keep
only the highest-scoring pair. This is a **pure precision gain**.

### Graph consistency pruning

For S1 entities with multiple accepted matches, we verify that the
matched S2/S3 records are mutually similar (they should describe the
same business). Inconsistent sets get the weakest-scored edge removed.
Again: **only removes false positives, never introduces new ones**.

---

## Troubleshooting

**Blocking recall < 0.97**
→ Increase `ANN_TOP_K` (default 30) to 50 in `config.py`.

**OOM during encoding**
→ Reduce `EMBED_BATCH_SIZE` to 256 in `config.py`.

**Val F0.5 stuck below 0.85**
→ Check blocking recall first (ceiling). Then inspect threshold sweep output.

**Validator FAIL on duplicate IDs**
→ A bug in postprocess.py — run `stage_validate_submission()` and read the error list.

**France records all predicted as singletons**
→ Verify `paraphrase-multilingual-MiniLM-L6-v2` is loading correctly.
   Try `intfloat/multilingual-e5-small` as an alternative in `config.py`.

---

## Reproducing Results

```bash
# Full reproduction from scratch:
# 1. Place TSV files in dataset/train/ and dataset/test/
# 2. Edit DATA_ROOT in src/config.py
# 3. pip install -r requirements.txt
# 4. Open Amazon_ML_Challenge_2026.py in Colab and run all cells
# Output: output/matching_results.tsv + output/candidate_pairs.tsv
```
