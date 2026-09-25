# Documentation Template — Amazon ML Challenge 2026
## Business Entity Resolution: Methodology Write-up

**Team Name**: [YOUR TEAM NAME]
**Submission Date**: [DATE]

---

## 1. Problem Understanding

Given business records from 3 independent data sources (S1, S2, S3) with noisy and inconsistent fields, determine which records across sources refer to the same real-world business entity.

**Key constraints driving design decisions:**
- **F₀.5 metric** (precision weighted 2× over recall): false merges cost more than missed matches
- **Open-set countries**: France appears only in test, not training — no country-specific hard-coding
- **Macro-averaged**: singletons (no-match S1 entities) count and are worth 1.0 when correctly predicted empty
- **One-to-one S2/S3 constraint** (EDA finding): each S2/S3 entity maps to at most one S1
- **≤8B params, MIT/Apache 2.0 license** for all models

---

## 2. Pipeline Architecture

```
Raw TSVs
   ↓
[Text Normalisation]
   ↓
[Multilingual Encoding — paraphrase-multilingual-MiniLM-L6-v2]
   ↓
[Blocking / Candidate Generation]
   ├── A. Token blocking (name/address prefix keys)
   ├── B. Sorted-neighborhood (window=5 over sorted norm_name)
   └── C. FAISS ANN (top-30, per-country FAISS IndexFlatIP)
            ↓ Union → candidate_pairs.tsv
[Feature Engineering — 24 language-agnostic similarity features]
   ↓
[LightGBM Binary Classifier — scale_pos_weight=0.5]
   ↓
[Threshold Sweep — macro F₀.5 on held-out 20% val set]
   ↓
[Post-processing]
   ├── One-to-one dedup (group-by-argmax on S2/S3 id)
   └── Graph consistency pruning (min pairwise char3-Jaccard floor)
            ↓
   output/matching_results.tsv
```

---

## 3. Candidate Generation / Blocking Strategy

### Country Partitioning (Pre-filter)
EDA finding: 0 cross-country GT pairs in 7.64M training matches. Applied as a hard partition before blocking — reduces the naive n×m search space by 50–70% with **zero recall loss** on training data.

Safety net: cross-country pairs with cosine similarity > 0.97 pass through regardless (insurance against France mislabeling).

### Strategy A — Token Blocking
Keys generated per record:
- `country + norm_name[:3]` (prefix-3)
- `country + norm_name[:4]` (prefix-4)
- `country + norm_addr[:4]`
- `country + norm_name` (full sorted normalised name)
- `country + norm_addr` (full sorted normalised address)

Two records are candidate pairs if they share any key. EDA: `name_normalized` keys are 98.8% singleton buckets (very high precision), but recall alone is insufficient — true matches average only 0.62 name-token Jaccard, so exact-normalised blocking misses the majority.

### Strategy B — Sorted-Neighborhood
Records sorted by (country, norm_name). Sliding window of width 5 pairs every S1 in the window with every S2/S3 in the same window. Catches near-duplicates that sort adjacently but differ by prefix characters (typos, abbreviation variation).

### Strategy C — FAISS ANN
Embeddings from `paraphrase-multilingual-MiniLM-L6-v2` (MIT license, 22M params).
Per-country FAISS `IndexFlatIP` (inner product on L2-normalised vectors = cosine).
Top-30 nearest S2/S3 retrieved per S1. This is the only strategy that handles:
- Hindi-transliteration variants (Devanagari ↔ Latin)
- French records at test time (zero labeled examples)
- Semantic paraphrases invisible to character-level features

**Union of A+B+C = candidate_pairs.tsv**

### Blocking Recall Target
≥ 0.99 (measured on held-out val split). Every GT pair not in candidates is an **unrecoverable recall ceiling** — no downstream model can invent a pair it never saw.

---

## 4. Feature Engineering

All features are **language-agnostic** — defined on similarity scores, not raw country labels. The same 24-feature vector is computed identically for US, India, and France pairs.

### Name Features (10)
| Feature | Description |
|---|---|
| `name_exact` | Raw string exact match |
| `name_norm_exact` | Sorted, suffix-stripped exact match |
| `name_token_jaccard` | Jaccard on token sets |
| `name_char3_jaccard` | Jaccard on char-3grams |
| `name_char2_jaccard` | Jaccard on char-2grams |
| `name_jaro_winkler` | RapidFuzz WRatio (typo-robust) |
| `name_edit_sim` | 1 - normalised Levenshtein |
| `name_token_sort_ratio` | fuzz.token_sort_ratio (word-order robust) |
| `name_partial_ratio` | fuzz.partial_ratio (substring) |
| `name_length_ratio` | min/max token count ratio |

### Address Features (11)
| Feature | Description |
|---|---|
| `addr_exact` | Raw string exact match |
| `addr_norm_exact` | Sorted normalised exact match |
| `addr_token_jaccard` | Jaccard on token sets |
| `addr_char3_jaccard` | Jaccard on char-3grams |
| `addr_edit_sim` | 1 - normalised Levenshtein |
| `addr_token_sort_ratio` | fuzz.token_sort_ratio |
| `addr_partial_ratio` | fuzz.partial_ratio |
| `addr_length_ratio` | min/max token count ratio |
| `addr_numeric_exact` | Any shared numeric token (house# / PIN) |
| `addr_numeric_jaccard` | Jaccard on purely numeric tokens |
| `addr_landmark_jaccard` | Jaccard on post-landmark tokens (Near, Opp.) |

### Cross Features (3)
| Feature | Description |
|---|---|
| `country_equal` | Countries match (weak signal, not a filter) |
| `embed_cosine` | Cosine of multilingual bi-encoder embeddings |
| `name_addr_product` | name_char3_jaccard × addr_char3_jaccard (interaction) |

---

## 5. Model Architecture

### Primary Model: LightGBM Binary Classifier

**Why LightGBM over a fine-tuned cross-encoder:**
1. **Label efficiency**: GBDTs match or exceed fine-tuned transformers on tabular feature vectors at ER-scale label budgets (Ditto, VLDB 2020)
2. **France generalisation**: splits like `if char3_jaccard > 0.8 and addr_jaccard > 0.6 → match` transfer without retraining; transformer token embeddings would need France examples to handle French vocabulary
3. **Calibrated probabilities**: needed for precise F₀.5 threshold optimisation
4. **Inference speed**: sub-millisecond per pair; transformer forward pass is 100× slower over tens of millions of candidates

**Hyper-parameters:**
- `scale_pos_weight = 0.5` (below 1.0 to bias toward precision — F₀.5 penalises FP 2× FN)
- `num_leaves = 127`, `learning_rate = 0.05`, `n_estimators = 1000`
- Early stopping on val binary log-loss (50 rounds)

**Negative sampling:**
- Negatives = hard negatives from same blocking bucket (not random)
- Ratio: 8 negatives per positive (hard negatives most informative for the boundary)

### Bi-encoder Role (paraphrase-multilingual-MiniLM-L6-v2)
Used in **two** ways — never as the final classifier:
1. Blocking recall layer (FAISS ANN, top-30)
2. `embed_cosine` feature into LightGBM

---

## 6. Train / Validation Split

**Group-split by source1_entity_id** (critical — splitting an S1 entity's matches across folds is label leakage).

**Stratified by match-count bucket** (0, 1, 2, 3, 4, 5+) to preserve:
- Proportionate singleton representation (5.58% in training S1)
- Multi-match entity distribution (most S1 have 2–5 matches per EDA)

**Split**: 80% train / 20% validation. Both splits remain as training data — the 20% is used only for threshold tuning and final evaluation, not withheld from blocking or lookup construction.

---

## 7. Threshold Optimisation

Default threshold of 0.5 is **wrong** for F₀.5. The optimal threshold is almost always higher (0.60–0.80) because precision is weighted 2×.

**Method**: sweep 0.30 → 0.95 in 0.01 steps on the val set. For each threshold, compute per-S1-entity F₀.5 then macro-average. Pick the argmax.

This matches **exactly** how the leaderboard scores — not micro-F₀.5, not global precision/recall.

---

## 8. Post-processing

### One-to-one Dedup
EDA: 0 multi-link S2/S3 records in 7.64M training GT pairs. Enforced as:
> For each S2/S3 entity_id appearing in more than one accepted pair, keep only the pair with the highest model probability. Drop all others.

This is a pure O(n log n) group-by-argmax. Can only remove false positives.

### Graph Consistency Pruning
For each S1 entity with ≥ 2 accepted matches:
1. Compute pairwise char3-Jaccard between matched S2/S3 norm_names.
2. If minimum pairwise similarity < 0.25, the set is internally inconsistent.
3. Remove the matched record with the lowest model score.
4. Repeat until consistent or set has 1 member.

Motivated by: matched S2/S3 records should describe the same real business.

---

## 9. Experiments and Results

| Stage | Val Blocking Recall | Val F₀.5 | Notes |
|---|---|---|---|
| Token blocking only | ~0.72 | — | Insufficient alone (name_normalized 98.8% singleton) |
| + Sorted-neighborhood | ~0.81 | — | +9% recall on near-duplicates |
| + FAISS ANN (k=30) | ~0.993 | — | +12% recall on transliteration/paraphrase |
| LightGBM (threshold=0.5) | — | ~0.81 | Suboptimal threshold |
| + Threshold sweep | — | ~0.91 | Optimal threshold ~0.65 |
| + One-to-one dedup | — | ~0.93 | Pure precision gain |
| + Graph pruning | — | ~0.94 | Further precision gain |

*(Exact numbers from your own run — update this table after training)*

---

## 10. Conclusion

The hybrid architecture — multilingual bi-encoder for recall-maximising blocking, LightGBM over language-agnostic similarity features for precision-calibrated matching — is motivated by three core constraints:

1. **F₀.5 precision-heavy metric** → conservative matching, tuned threshold, precision-biased `scale_pos_weight`
2. **Open-set France country** → all features are similarity numbers (not raw tokens), multilingual encoder handles unseen scripts
3. **One-to-one S2/S3 constraint** → one-to-one dedup as a free, guaranteed precision improvement

The pipeline is fully reproducible from the training/test TSVs using only `requirements.txt` dependencies. No external databases, APIs, or internet lookups are used.

---

*Fill in exact metric values after training. Prioritise clarity over brevity in the final document.*
