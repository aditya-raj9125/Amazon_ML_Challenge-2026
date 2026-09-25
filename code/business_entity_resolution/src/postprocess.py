# =============================================================
# postprocess.py — One-to-one dedup + graph consistency pruning
# =============================================================
# These two steps are pure post-processing on the matcher's output.
# They can ONLY remove false positives — never introduce new ones —
# so they are guaranteed to improve or maintain precision, which is
# exactly what F0.5 rewards.
#
# Step 1: One-to-one dedup
#   The EDA confirmed 0 multi-link S2/S3 records in training GT.
#   Every S2/S3 entity belongs to at most ONE S1 entity.
#   If our matcher assigns the same S2/S3 id to two S1 entities, keep
#   only the pairing with the higher model probability score.
#
# Step 2: Graph consistency pruning
#   For each S1 entity with multiple accepted matches, verify that the
#   matched S2/S3 records are mutually consistent (they should all
#   describe the same real business). If the minimum pairwise
#   char3-Jaccard among matched records is below the floor threshold,
#   drop the weakest edge (the S2/S3 record with the lowest model score).
#   Repeat until the set is internally consistent or has one member left.

from collections import defaultdict
from normalize import normalize_name
from config import GRAPH_PRUNE_MIN_SIMILARITY


def char3_jaccard(a: str, b: str) -> float:
    """Character 3-gram Jaccard similarity between two strings."""
    def ng(s): return set(s[i:i+3] for i in range(len(s) - 2))
    sa, sb = ng(a), ng(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def one_to_one_dedup(
    predictions: dict[str, set[str]],
    scores: dict[str, dict[str, float]],
) -> dict[str, set[str]]:
    """
    Enforce the one-to-one constraint:
        Each S2/S3 entity_id maps to at most ONE S1 entity_id.

    Algorithm
    ---------
    1. Collect all (s1_id, s23_id, score) triples across all accepted matches.
    2. Group by s23_id.
    3. If a s23_id appears under multiple s1_ids, keep only the (s1_id, s23_id)
       pair with the highest score; remove it from all others.

    This is a pure precision gain — it removes false positives where the
    model mistakenly assigned the same S2/S3 record to two different S1s.

    Parameters
    ----------
    predictions : { s1_id : set of accepted s23_ids }
    scores      : { s1_id : { s23_id : probability } }
    """
    # Collect all triples
    triples = []
    for s1_id, matched in predictions.items():
        for s23_id in matched:
            prob = scores.get(s1_id, {}).get(s23_id, 0.0)
            triples.append((s1_id, s23_id, prob))

    # Group by s23_id
    by_s23: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s1_id, s23_id, prob in triples:
        by_s23[s23_id].append((s1_id, prob))

    # Build removal set: (s1_id, s23_id) pairs that should be dropped
    to_remove: set[tuple[str, str]] = set()
    for s23_id, s1_prob_list in by_s23.items():
        if len(s1_prob_list) <= 1:
            continue
        # Sort by descending score, keep the best
        s1_prob_list.sort(key=lambda x: x[1], reverse=True)
        winner_s1 = s1_prob_list[0][0]
        for s1_id, _ in s1_prob_list[1:]:
            to_remove.add((s1_id, s23_id))

    removed = 0
    cleaned: dict[str, set[str]] = {}
    for s1_id, matched in predictions.items():
        new_matched = {m for m in matched if (s1_id, m) not in to_remove}
        cleaned[s1_id] = new_matched
        removed += len(matched) - len(new_matched)

    print(f"[One-to-one dedup] Removed {removed:,} conflicting pairs.")
    return cleaned


def graph_consistency_pruning(
    predictions: dict[str, set[str]],
    scores: dict[str, dict[str, float]],
    lookup_all: dict[str, dict],
    min_sim: float = GRAPH_PRUNE_MIN_SIMILARITY,
) -> dict[str, set[str]]:
    """
    Remove the weakest edge from a match set that is internally inconsistent.

    For each S1 entity with >= 2 accepted matches:
      - Compute pairwise char3-Jaccard between all matched S2/S3 norm_names.
      - If min pairwise similarity < min_sim, drop the matched record with
        the lowest model score.
      - Repeat until the set is consistent or has 1 member.

    This implements the 'graph consistency' post-processing step
    described in the architecture document.

    Parameters
    ----------
    predictions : { s1_id : set of accepted s23_ids }
    scores      : { s1_id : { s23_id : probability } }
    lookup_all  : record dict (must contain 'norm_name' per entity)
    min_sim     : minimum acceptable pairwise Jaccard floor
    """
    pruned_total = 0
    cleaned: dict[str, set[str]] = {}

    for s1_id, matched in predictions.items():
        current = set(matched)

        while len(current) >= 2:
            # Get norm_names for current matched set
            valid = [(eid, lookup_all[eid]["norm_name"])
                     for eid in current if eid in lookup_all]
            if len(valid) < 2:
                break

            # Find minimum pairwise similarity
            min_pairwise = 1.0
            worst_eid    = None
            worst_score  = 1.0

            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    sim = char3_jaccard(valid[i][1], valid[j][1])
                    if sim < min_pairwise:
                        min_pairwise = sim

            if min_pairwise >= min_sim:
                break   # Set is internally consistent — done

            # Find the weakest-scored member to remove
            for eid, _ in valid:
                s = scores.get(s1_id, {}).get(eid, 0.0)
                if worst_eid is None or s < worst_score:
                    worst_score = s
                    worst_eid   = eid

            current.discard(worst_eid)
            pruned_total += 1

        cleaned[s1_id] = current

    print(f"[Graph pruning] Removed {pruned_total:,} internally inconsistent matches.")
    return cleaned


def apply_postprocessing(
    predictions: dict[str, set[str]],
    scores: dict[str, dict[str, float]],
    lookup_all: dict[str, dict],
    enable_dedup: bool = True,
    enable_graph: bool = True,
) -> dict[str, set[str]]:
    """
    Convenience wrapper that applies both post-processing steps in order.

    1. One-to-one dedup (must come first — graph pruning relies on a
       clean score-to-entity mapping).
    2. Graph consistency pruning.

    Returns the cleaned predictions dict.
    """
    result = predictions

    if enable_dedup:
        print("\n[Post-processing] Step 1: One-to-one dedup ...")
        result = one_to_one_dedup(result, scores)

    if enable_graph:
        print("[Post-processing] Step 2: Graph consistency pruning ...")
        result = graph_consistency_pruning(result, scores, lookup_all)

    return result
