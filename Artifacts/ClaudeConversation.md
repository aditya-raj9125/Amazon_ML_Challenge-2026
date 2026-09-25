That's the overall shape. Here's the full engineering reasoning behind it — classical baselines first, then where and why deep learning earns its place, with the math throughout.

## 1. Reframing the problem correctly

This is not classification of independent rows — it's **multi-source record linkage** with three properties that dominate every design choice:

- **F₀.5 macro-average, singleton-inclusive.** A false merge costs 2× a missed match, and every no-match S1 entity is worth a full point if you leave it empty. This means the whole pipeline should be tuned to be *stingy*, not to maximize recall.
- **Open-set countries.** France appears only at test time. Anything that memorizes US/India-specific patterns (a classifier trained with `country` as a categorical feature with fixed cardinality, or hand-coded regex for "Pvt Ltd" vs "SARL") will silently degrade. Features must be *derived from text similarity*, not from country identity.
- **≤8B params, permissive license, no external lookups.** This rules out API-based geocoding/entity databases and steers the DL half toward small open encoders (MiniLM, E5-small, BGE-small — all MIT/Apache, 30M–300M params), not an 8B LLM used as a matcher (too slow for pairwise scoring at scale, and gives you nothing a good encoder doesn't).

## 2. Blocking — the recall ceiling, so it comes first

You cannot match what you never pair. Three classical techniques, in order of what I'd actually deploy:

**a. Token/attribute blocking (baseline, cheap).** Build blocking keys from normalized name tokens (sorted, stop-words like "Inc/Ltd/Pvt" stripped) and address tokens (city + first street token). Two records collide if they share ≥1 key. This is what Image 3 shows — bucket by name+address key.

**b. Sorted-neighborhood / canopy on a cheap key.** Sort all records by a normalized name prefix, slide a window of size *w*, pair everything inside it. Cheap O(n log n), catches near-duplicates missed by exact-token blocking.

**c. Embedding-based ANN blocking (this is where DL enters — but only here, not in scoring).** Encode `business_name + business_address` with a small sentence encoder (MiniLM-L6, 22M params, MIT license), build a FAISS or simple cosine-kNN index per source, retrieve top-*k* (k≈20–50) nearest S2/S3 records for every S1 record. This is what actually closes the recall gap on **France**: rule-based tokens are brittle across languages/scripts, but a multilingual small encoder (e.g. `paraphrase-multilingual-MiniLM-L12-v2`, also permissively licensed) generalizes to unseen countries because it's not keyed on any fixed vocabulary.

**Union the candidate sets** from (a)+(b)+(c), dedupe → this union *is* `candidate_pairs.tsv`.

Recall ceiling formula to track on your validation split:
$$\text{Blocking Recall} = \frac{|\{(s1, s2/3) \in \text{GT} : \text{pair} \in \text{candidates}\}|}{|\text{GT pairs}|}$$
Reduction ratio (how much you cut the naive n×m space):
$$RR = 1 - \frac{|\text{candidate pairs}|}{|S1| \times (|S2|+|S3|)}$$
Target: recall ≥ 0.97–0.99 at RR ≥ 0.999. Every point of recall lost here is an unrecoverable ceiling on your final F₀.5, since no downstream model can invent a pair it never saw.

## 3. Feature engineering (this is where most of the score is actually won)

For every candidate pair, compute a **symmetric, language-agnostic similarity vector** — none of these look at raw country labels, only text:

**Name features**
- Jaro-Winkler: $JW = J + \ell p (1-J)$, where $J$ is Jaro similarity and $\ell$ is the common-prefix length (≤4), $p=0.1$ — good for typos/transliteration.
- Token-set / token-sort ratio (handles word-order transposition: "Robotics Acme" vs "Acme Robotics").
- Character n-gram Dice coefficient: $Dice = \frac{2|X \cap Y|}{|X|+|Y|}$ on 3-grams — robust to legal-suffix noise.
- TF-IDF cosine over a char n-gram vocabulary fit on the union of all training names (no external corpus, so this is fair-play compliant): $\cos(\vec{a},\vec{b}) = \frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\|\vec{b}\|}$.
- Legal-suffix-normalized exact/edit-distance (after stripping Inc/Ltd/Pvt/Corp/LLC/… via a small synonym dictionary you build from the training data itself, not looked up externally).
- Abbreviation-aware Levenshtein ratio: $1 - \frac{\text{lev}(a,b)}{\max(|a|,|b|)}$.

**Address features**
- Component-wise similarity after light parsing (street number, street name, city, PIN/zip) — token overlap per component, not just whole-string similarity, since "500 Market St, San Jose" vs "500 Market Street, San Jose CA" should match on street-number + street-token even with missing state/PIN.
- Landmark handling: strip "Near", "Nr.", "Opp." prefixes before comparing; landmark terms get their own soft-match feature (Jaccard on landmark tokens) rather than polluting the street-similarity score.
- Geo-token Jaccard: $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$ over address token sets.
- Numeric-token exact-match indicator (street/PIN numbers are strong precision signals — a name can drift a lot under noise, a house number usually doesn't).

**Cross features**
- Bi-encoder cosine similarity (from the MiniLM embeddings you already computed for blocking — reuse them as a feature, don't discard them).
- Country-match indicator **as one weak feature among many**, never as a filter (per the spec's explicit instruction not to hardcode on country).

This gives a feature vector of ~15–25 dense numeric features per pair — small, interpretable, and crucially: **every feature is defined the same way regardless of language**, so France pairs get scored on the same footing as US/India pairs even with zero France training examples.

## 4. The matcher: classical GBDT first, then where DL fits

### Classical baseline: logistic regression
$$P(\text{match}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z)=\frac{1}{1+e^{-z}}$$
Fast, interpretable, a fine sanity check — but underfits interactions like "high name similarity only matters *if* address similarity isn't terrible," which GBDTs capture natively via splits.

### Primary model: gradient-boosted trees (LightGBM / XGBoost / CatBoost)
$$\hat{y} = \sum_{k=1}^{K} f_k(\mathbf{x}), \quad f_k \in \mathcal{F} \text{ (regression trees)}$$
trained with a weighted logistic loss to directly reflect the F₀.5 precision bias:
$$\mathcal{L} = -\sum_i \left[ w_{pos} \cdot y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \right]$$
with $w_{pos}$ tuned *down* (not up) relative to naive class-balancing, since false positives are penalized 2× harder than false negatives under F₀.5 — you actually want the model conservative, not recall-maximizing.

**Why GBDT is my primary matcher, not a transformer classifier:**
- The input is a compact, dense, meaningful feature vector, not raw sequences — GBDTs are close to state-of-the-art on exactly this regime (tabular data), while transformers need far more labeled pairs to beat feature-engineered baselines, and typically don't for record-linkage-scale label budgets.
- It gives calibrated-enough probabilities you can threshold precisely against F₀.5 (see §5), and monotonic-constraint support (e.g. force similarity↑ ⇒ match-probability↑), which directly encodes domain knowledge and reduces overfitting risk on an unseen country.
- It generalizes to France for free: since every feature is a language-agnostic similarity score, GBDT splits learned on US/India data ("if char-ngram cosine > 0.8 and address-token Jaccard > 0.6 → match") transfer without retraining, whereas a fine-tuned transformer's *token embeddings* are heavily shaped by the English/Hindi-script training vocabulary and are more likely to misfire on French diacritics/legal suffixes ("SARL", "Rue") it has seen zero labeled examples of.
- It trains in minutes on modest hardware and is trivially auditable for the methodology write-up and fair-play review — a real practical advantage given the "top teams' code is reviewed" clause.

### Where deep learning genuinely helps — bi-encoder, used twice, not as the classifier

1. **Blocking** (already covered, §2) — a bi-encoder's job here is *recall*, where it beats rule-based tokens badly on cross-lingual/noisy names.
2. **As one feature into the GBDT** — the cosine similarity from the same embeddings, rather than replacing the tree model. This is "learned semantic similarity" folded in as a single powerful feature, without inheriting a neural net's calibration/overfitting problems for the final precision-critical decision.

### Alternative I considered and rejected: end-to-end cross-encoder / siamese fine-tuning as the final matcher
A cross-encoder (concatenate `name1 [SEP] addr1 [SEP] name2 [SEP] addr2`, fine-tune a small transformer like DistilBERT with a pairwise loss) is the standard "if you want to squeeze the last point of F1" approach in academic ER (e.g. Ditto, DeepMatcher). I rejected it as the *primary* model here because:
- Training-label volume in this challenge is almost certainly small relative to what cross-encoders need to reliably beat GBDT-on-features (published ER benchmarks show GBDT/Random-Forest-on-features is competitive with or beats DL matchers below tens of thousands of labeled pairs).
- Higher overfitting risk to the *training* countries (US/India) precisely on the dimension that matters most here (French records at test time) — a tree over similarity scores is structurally shielded from this the way a transformer over raw token embeddings is not.
- F₀.5 wants **calibrated, conservative** scores; cross-encoder probabilities from a small fine-tune on limited positives tend to be poorly calibrated unless you additionally do temperature scaling / isotonic regression — extra machinery for a benefit that's not guaranteed to show up given the label budget.
- Slower at inference over `|candidates|` pairs (full transformer forward pass per pair) vs. sub-millisecond GBDT inference on pre-computed features — matters if candidate_pairs.tsv is large.

I'd still **keep a small cross-encoder as a validation-only ablation** — if your held-out F₀.5 shows it beats GBDT by a clear margin (it might, if the training set turns out to be larger than I'm assuming), swap it in or ensemble the two logits: $\hat{p}_{final} = \sigma(\alpha \cdot \text{logit}_{GBDT} + (1-\alpha)\cdot\text{logit}_{cross})$, with $\alpha$ tuned on validation F₀.5.

### Contrastive/triplet loss, for completeness
If you do fine-tune the bi-encoder (recommended, cheap, and helps blocking recall) rather than using it off-the-shelf, train it with a contrastive loss so matched pairs pull together and hard negatives (same-block, non-matches — exactly the "Acme Bakery" vs "Acme Robotics" look-alike in Image 2) push apart:
$$\mathcal{L}_{contrastive} = y \cdot d(a,b)^2 + (1-y)\cdot \max(0, m - d(a,b))^2$$
where $d$ is embedding distance and $m$ is a margin — this directly improves blocking recall on hard negatives without touching the final classifier at all.

## 5. Threshold selection — don't use 0.5

Since $F_{0.5}$ weights precision 2×, the optimal cut point on GBDT output probabilities is *not* 0.5. Sweep thresholds on the validation split and pick:
$$t^* = \arg\max_t \; F_{0.5}(t), \quad F_{0.5}=\frac{1.25\,P(t)\,R(t)}{0.25\,P(t)+R(t)}$$
computed **per-S1-entity then macro-averaged**, exactly as the challenge scores it — not a global micro F₀.5, which would give a different (and wrong) optimum. In practice this usually lands well above 0.5 (often 0.6–0.8) given the precision weighting.

## 6. A step the naive per-pair pipeline misses: graph consistency

Binary pairwise classification treats each (S1, S2/3) candidate independently, which can produce locally-plausible but globally-inconsistent merges (S1 matches both S2-B and S3-C, but S2-B and S3-C look nothing alike — a red flag). A cheap post-hoc cleanup:

- Build a graph with edges = predicted-positive pairs, weighted by model probability.
- For each S1 node, if it has multiple accepted matches, sanity-check the *mutual* similarity among matched S2/S3 partners (they should describe the same real business too). Drop the weakest edge if the accepted set is internally inconsistent (e.g. below some similarity floor to each other).
- This is a precision-only correction (never adds edges, only prunes) — perfectly aligned with F₀.5's asymmetry, and it's classical graph logic, not a model, so it's free at inference and easy to defend in the methodology doc.

## 7. Summary: what I'd ship, and why, one line each

| Stage | Choice | Why over the alternative |
|---|---|---|
| Blocking | Token blocking + sorted-neighborhood + small multilingual bi-encoder kNN | Rules alone miss France/typos; embeddings alone are slower and less precise on exact numeric tokens — union gets both |
| Features | ~20 hand-engineered string/address similarities + bi-encoder cosine | Small, interpretable, country-agnostic, generalizes to unseen France |
| Matcher | LightGBM on features, F₀.5-aware weighting | Best precision/recall tradeoff at this label scale; robust to open-set country; auditable |
| Rejected primary | Fine-tuned cross-encoder end-to-end | Needs more labels to beat GBDT here, overfits to train countries, slower, poorly calibrated for a precision-tuned metric |
| Threshold | Macro-F₀.5-optimized, not 0.5 | Matches the actual scoring function |
| Post-processing | Graph consistency pruning | Free precision gain, zero recall cost |

If you want, I can turn this into the actual `Documentation_template.md` write-up, or sketch the LightGBM feature pipeline in Python.







Good question to press on — let me lay out the actual decision matrix rather than just asserting it.

## What "hybrid" means here, precisely

Not an ensemble of two independent classifiers voting on the same footing. It's a **division of labor**:
- The bi-encoder's cosine similarity becomes *one feature* fed into the GBDT.
- The bi-encoder also independently powers blocking (retrieval), which is a different job (recall over the full population) than classification (precision on a shortlist).
- The GBDT alone makes the final match/no-match decision and owns the probability that gets thresholded against F₀.5.

So it's not "two models racing," it's "a similarity signal a neural net is uniquely good at producing, consumed by a model uniquely good at precision-calibrated decisions on structured features." That division is the actual argument, not GBDT-vs-DL-in-the-abstract.

## Why each half is doing the job it's suited for

**Bi-encoder's job — semantic recall across surface-form noise.** A char-ngram Jaccard or Levenshtein feature treats "Acme Robotics Inc." vs "Acme Robotix Incorporated" as two arbitrary strings; it has no notion that "Robotix" is a phonetic/typo variant of "Robotics" it's never seen written that way before, or that a transliterated Hindi business name and its Latin-script rendering are the same word. An embedding trained (even generically, off-the-shelf) to place semantically similar short text near each other in vector space captures exactly this — it's solving a *generalization* problem that string-metric features structurally cannot, because they only look at character overlap, not meaning.

**GBDT's job — precision-calibrated decision over a small structured feature set.** Once you have ~20 similarity scores per pair, this is now a classic tabular problem, and GBDTs are close to the state of the art there for a reason: they learn conditional splits like "high name similarity only counts as a match if address-token overlap clears some floor, unless the numeric street token matches exactly, in which case relax the name threshold" — exactly the kind of interaction the challenge's noise patterns imply (landmark-based addresses, missing PIN codes, DBA names). A single dense neural layer over these same features wouldn't obviously beat this, and would need materially more tuning to get there.

## Head-to-head against the alternatives I actually weighed

**1. Pure classical: string/TF-IDF features → GBDT, no neural embedding at all.**
This is a legitimate strong baseline and I'd absolutely build it first, as a floor to beat. Where it breaks: transliteration and semantic paraphrase. "Nr. City Hall" vs a differently-worded landmark reference, or genuinely different-script name renderings, share almost no characters — string metrics score them near zero, so the GBDT never even gets a useful signal to threshold on for those pairs. This is precisely the failure mode the challenge explicitly warns about ("transliteration variants," "landmark-based references"). The bi-encoder feature is what rescues these cases. Rejected as final answer because it caps recall/precision on exactly the hard tail the prompt flags as expected noise.

**2. Pure DL: fine-tuned cross-encoder as the sole matcher (no GBDT, no hand features).**
Covered in the earlier answer — the core objections are label efficiency, calibration, and open-set generalization to France. Worth restating the generalization point sharply: a cross-encoder's decision boundary lives in a high-dimensional space shaped by whatever tokens it saw in training (US/India names, addresses). France introduces genuinely new subword vocabulary (accented characters, "SARL," "Rue," French place names) with zero fine-tuning examples. The model has no guardrail forcing it to fall back on "these two strings are just similar" — it might, or it might latch onto spurious correlations from the languages it did see. A GBDT over similarity *scores* has no such vocabulary dependency: a 0.85 cosine similarity means the same thing whether the underlying text is English, Hindi-transliterated, or French, because the feature is already a normalized number, not raw tokens. This is the single strongest argument for keeping GBDT as the decision layer regardless of how strong an embedding model you use.

**3. Bi-encoder cosine similarity alone as the match score (embedding, threshold it directly, skip GBDT and hand features entirely).**
Tempting because it's the simplest possible pipeline — one number, one threshold. Rejected because it throws away high-precision signals that embeddings are bad at representing crisply: exact numeric street/PIN match, exact legal-suffix-normalized string equality, landmark-token overlap. Embeddings are good at fuzzy semantic closeness and bad at "these two addresses share literally the same house number," which is often the single most precision-boosting bit of evidence available. You'd be discarding the cheapest, most reliable signal in the dataset. A pure cosine threshold also gives you only one lever to tune, versus a GBDT's dozens of learned interaction splits — much less headroom for the precision-heavy F₀.5 optimization in §5 of the earlier answer.

**4. Ensemble of independently-trained GBDT and cross-encoder, logits averaged.**
Not wrong, and I flagged it earlier as worth trying if labeled data turns out to be larger than expected — but it's strictly more expensive (two training pipelines, two inference passes, an extra blending weight to tune on validation) for a benefit that's not guaranteed to materialize at this label scale. The hybrid-as-feature design gets most of the same benefit (semantic signal reaching the final decision) at a fraction of the engineering and inference cost, and it's one model to calibrate/threshold instead of two.

## The generalization argument, stated once more because it's the crux

Every other design decision in this pipeline (features, blocking, thresholding) is defensible on accuracy grounds alone. The hybrid split specifically earns its keep on the **France clause** — the one place where "works well on the training distribution" and "works well on the graded distribution" diverge. Pure string features underfit semantic/transliteration noise; pure neural approaches overfit to training-country vocabulary. Feeding a language-agnostic *similarity number* from the embedding into a tree model that only ever sees numbers, never raw text, is the one design that doesn't inherit either failure mode.





Good — this EDA data changes the design in real, load-bearing ways, not just adds color. Let me walk through what it changes and why, grounded in the actual literature for this problem (I pulled a few papers below), then give the revised architecture.## What the EDA actually changes (not just confirms)

Three findings in your PDF are structural, not cosmetic — they change the algorithm, not just the parameters:

**1. Strict country partitioning, zero cross-country matches in ground truth.** This licenses blocking by country as a hard pre-filter with provably zero recall loss on train, cutting the search space 50–70% before any fuzzy logic runs. One caveat worth engineering around: this invariant is observed only on train (US/India). You have no France ground truth to confirm it holds there too — it's a very safe bet (it's almost certainly a property of how the dataset was generated, not of the countries themselves), but I'd still keep a narrow safety net: a very-high-similarity cross-country pair (e.g. name char-ngram cosine > 0.95 *and* address token Jaccard > 0.9) shouldn't be silently discarded by the country filter. Cheap insurance against a labeling quirk in the unseen slice.

**2. The one-to-one constraint on S2/S3: 0 multi-links.** Every S2 or S3 record belongs to at most one S1 entity (S1 can still absorb many — the match-count table shows most S1 entities have 2–5 matches). This is a *global* constraint your per-pair classifier has no way of knowing about, and it's exactly the setting studied in classical record-linkage theory: Sadinle (2017) calls this bipartite record linkage's "maximum one-to-one restriction," and the standard fix (Jaro, 1989, later generalized to multi-source in Nutanong et al.'s "Principled Graph Matching Algorithms for Integrating Multiple Data Sources," built for Bing/Xbox-scale multi-source ER) is to **enforce it as a post-processing step on the classifier's pairwise scores, not to hope the model learns it**. Because this constraint has no scarcity on the S1 side, it collapses to something much simpler than the general multi-partite matching in that paper: for every S2 or S3 entity with more than one accepted candidate pair, keep only the highest-scoring one and drop the rest. This is a pure O(n log n) group-by-argmax, and it's the single highest-leverage, lowest-cost step you can add — it can only remove false positives (from a classifier that scored the same S2 record as matching two different S1 businesses), never remove a true one, which is exactly what a precision-doubled metric like F₀.5 rewards.

**3. Positive-pair feature means are lower than you'd hope.** Your own pair-feature table shows true matches average only ~0.62 name-token-Jaccard, ~0.66 name-char3-Jaccard, and *raw* `name_exact` is true for only 4.4% of positives (`address_exact` only 1.8%; even `name_normalized_exact` is only 21.4%). This directly falsifies relying on exact-normalized blocking keys alone — your `name_normalized` blocking key is 98.8% singleton buckets, which is great precision but would recall well under a quarter of true matches on its own. This is the concrete, data-backed reason the blocking stage has to be a **union of several weaker, higher-recall keys**, not one strong exact key.

## Research grounding

- **Ditto** (Li et al., VLDB 2020) — casts entity matching as sequence-pair classification with a fine-tuned pretrained transformer, gets 96.5% F1 on a 789K×412K company-matching task, and importantly shows you can hit SOTA with *half* the labeled data via data augmentation/hard-negative injection. This is the strongest evidence for a transformer-based matcher — but note even Ditto's best real-world large-scale number is 96.5 F1 on a comparatively cleaner two-source task; your positive-pair similarity stats above suggest this dataset is noisier, and you have three sources plus an unseen-language slice, which is harder, not easier.
- **Fellegi–Sunter / Splink** — the classical probabilistic-linkage framework, still the industrial default (implemented in the open-source `splink` library, Fellegi-Sunter + EM, runs on DuckDB locally or Spark/Athena for 100M+ rows). It computes per-field match/non-match likelihood ratios ($m$/$u$ probabilities) and combines them into a log-odds score — essentially a hand-specified, unsupervised version of what your GBDT will learn supervised. Worth running as a zero-training-data sanity baseline before you trust the supervised pipeline, and it validates your infra choice: DuckDB comfortably handles this row count on a single machine, no Spark cluster needed.
- **One-to-one/multi-partite assignment literature** (Sadinle 2017; Nutanong et al., "Principled Graph Matching Algorithms for Integrating Multiple Data Sources") — covered above, this is where the dedup post-step comes from, with citation-backed evidence it "leverages a natural global property... known to provide significant improvements to precision and recall."

## Revised architecture (the diagram above)

**1. Partition by country.** S1/S2/S3 split into US, India, France buckets before anything else. This alone kills most of the 26M-row combinatorics.

**2. Normalize + multilingual encode.** Rule-based normalization (legal suffixes, Rd/Road, Pvt/Private — built from your own training vocabulary, not looked up) *and*, separately, a multilingual sentence encoder (e.g. `paraphrase-multilingual-MiniLM-L12-v2`, MIT license, well under 8B params) run over `name + address`. This is the layer that has to carry the Hindi-transliteration burden your Unicode stats confirm is present (`unicode_name_M`/`unicode_address_M` counts in the low millions in train S2/S3 — real transliteration marks, not noise). A char-ngram/Levenshtein feature can't see that a Devanagari rendering and its Latin transliteration are the same word; a multilingual embedding can.

**3. Multi-key blocking union**, sized directly off your bucket-size table:
   - `country_name_prefix3`/`prefix4` and `country_address_prefix4` (cheap, wide net, median bucket 1)
   - char-3-gram MinHash/LSH on name and address (catches typos your prefix keys miss)
   - normalized-exact keys (`name_normalized`, `address_normalized`, `country_name_token_signature`) — high-precision, near-singleton, but as you showed, insufficient alone
   - embedding ANN (FAISS/HNSW, top-k≈30) *per country partition*, which is what recovers the transliteration and heavy-paraphrase cases the lexical keys structurally cannot see

   Union all of these; recall depends on the union, precision doesn't matter yet — that's the matcher's job.

**4. Candidate pairs** → `candidate_pairs.tsv`. Before building anything downstream, measure blocking recall on your held-out validation split exactly as your own "Next Experiments" section proposes — this number is your hard ceiling on the final score, full stop.

**5. Feature engineering + GBDT scorer.** Your own feature inventory table is basically the right list already — I'd keep everything marked "Candidate" and add the bi-encoder cosine as one more feature. Train LightGBM (fast, handles millions of rows natively via histogram binning, no GPU needed) with F₀.5-aware loss weighting as discussed earlier, using `shared_numeric_tokens`, `postal_like`, `house_number_like` as your strongest precision anchors (numeric agreement is rare to fake by accident) alongside the fuzzy text features.

**6. F₀.5-tuned threshold**, computed per-S1 macro exactly as the competition scores it — sweep, don't default to 0.5.

**7. One-to-one dedup + output.** Group accepted (score > threshold) pairs by S2/S3 entity_id; keep the argmax S1 per group, drop the rest; write `matching_results.tsv`. This is the step that isn't in most generic ER tutorials but is directly licensed by your own EDA finding, and it's a pure precision gain at zero engineering risk.

## Training at this scale — platform, infra, split

**Compute:** You don't need Spark/Databricks for 26M rows split across 6 files — DuckDB (their own README recommends it) handles hundreds of millions of rows out-of-core on a single machine, and it's what Splink itself uses as its default backend up to ~100M rows. Practical setup: Polars or DuckDB for all joins/feature computation (vectorized, no pandas memory blowups), GPU only for the embedding-encoding pass (batch-encode tens of millions of short strings with a small model in fp16 — a single T4/A100 on Kaggle or Colab Pro+ does this in a few hours; Kaggle's free 30 GPU-hrs/week is enough if you checkpoint by source/country). LightGBM/CatBoost training itself is CPU-bound and fine on a 16–32 core, 64GB-RAM box (a mid-tier cloud VM for a few hours, or Colab Pro+ high-RAM), since your feature matrix per pair is ~20 floats, not raw text.

**Negative sampling — necessary, not optional.** With ~7.64M true positive links (`total_s2_links` + `total_s3_links` from your ground-truth table) but candidate pairs numbering in the tens of millions after blocking (25–27% of S2/S3 are pure distractors, and each S1 sees multiple candidates), training the GBDT on the *full* candidate set is wasteful and imbalanced. Downsample negatives to roughly a 1:5–1:10 positive:negative ratio for training, with **hard-negative mining** from the actual blocking output (same-block, high lexical similarity, wrong label) — exactly the technique Ditto's data-augmentation step uses to force the model to learn the hard boundary rather than the easy 99%-of-negatives-look-nothing-alike cases. Evaluate on the *full*, non-downsampled validation candidate set, since that's what you'll face at inference.

**Train/validation split:** Group-split by S1 entity_id (never split a S1 and its matched S2/S3 across folds — that's leakage), stratified by country *and* by match-count bucket (0/1/2…11) so your validation set has proportionate singleton representation (5.58%) and doesn't accidentally over- or under-index on easy vs. hard multi-match entities. Since you have zero labeled France examples, you cannot directly validate France performance — the best you can do is build a **proxy domain-shift check**: hold out the most transliteration-heavy slice of India (highest `name_nonascii`/Unicode-mark counts) as a pseudo-unseen-language dev set, and confirm your feature/blocking design doesn't collapse on it before trusting the real France predictions blind.

## On the 0.98 target — an honest calibration

I'd rather tell you this now than have the number surprise you later. Ditto's best reported real-world large-scale result is 96.5% F1 on a comparatively clean two-source company dataset. You're running three sources, a metric that penalizes false positives 2× harder than F1 does, positive-pair similarity statistics that show real matches average only ~0.6–0.66 on lexical similarity (not near-1.0), and a domain-shift country with zero training labels. 0.98 macro F₀.5 is a very aggressive target — reachable in principle mainly *because* F₀.5 rewards conservative behavior and your one-to-one constraint is unusually strong free information most ER benchmarks don't have, but it will live or die on blocking recall (push for >99.5%) and on how well the one-to-one dedup step cleans up the last mile of precision. I'd set an internal milestone around 0.93–0.95 as "the pipeline is working," and treat everything above that as squeezed out by threshold tuning, hard-negative mining, and the dedup step — not by chasing a fundamentally different model architecture.



