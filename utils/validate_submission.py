"""
validate_submission.py
======================
Validates output/matching_results.tsv and output/candidate_pairs.tsv
against every rule in the problem statement.

Usage (from the repo root):
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Exits 0 on PASS, 1 on any failure.
No third-party dependencies — stdlib only.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict


def load_tsv(path, required_cols):
    """Load a TSV and return (rows as list-of-dicts, set of header names)."""
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = set(reader.fieldnames or [])
        missing = required_cols - header
        if missing:
            print(f"  [ERR] {path}: missing columns {missing}")
            sys.exit(1)
        for row in reader:
            rows.append(row)
    return rows, header


def load_entity_ids(path):
    """Return set of entity_ids from a source TSV."""
    ids = set()
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            ids.add(row["entity_id"].strip())
    return ids


def validate(matching_path, candidate_path, test_dir):
    issues = []

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            print(f"  [ERR] Test file not found: {p}")
            sys.exit(1)

    print("Loading test source files ...")
    s1_ids = load_entity_ids(s1_path)
    s2_ids = load_entity_ids(s2_path)
    s3_ids = load_entity_ids(s3_path)
    valid_candidates = s2_ids | s3_ids

    print("Loading matching_results.tsv ...")
    match_rows, _ = load_tsv(matching_path, {"source1_entity_id", "matched_entity_ids"})

    print("Loading candidate_pairs.tsv ...")
    cand_rows, _ = load_tsv(candidate_path, {"source1_entity_id", "candidate_entity_ids"})

    match_seen_s1 = set()
    cand_seen_s1 = set()
    cand_lookup = {}

    print("Validating candidate_pairs.tsv ...")
    for row in cand_rows:
        s1 = row["source1_entity_id"].strip()
        raw = row["candidate_entity_ids"].strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

        if s1 in cand_seen_s1:
            issues.append(f"CAND: duplicate source1_entity_id row: {s1}")
        cand_seen_s1.add(s1)

        if s1 not in s1_ids:
            issues.append(f"CAND: source1_entity_id not in test S1: {s1}")

        id_set = set()
        for eid in ids:
            if eid in id_set:
                issues.append(f"CAND: duplicate candidate id {eid} for {s1}")
            id_set.add(eid)
            if eid not in valid_candidates:
                issues.append(f"CAND: candidate id {eid} not in test S2/S3 for {s1}")
        cand_lookup[s1] = id_set

    for s1 in s1_ids:
        if s1 not in cand_seen_s1:
            issues.append(f"CAND: missing row for S1 entity: {s1}")

    print("Validating matching_results.tsv ...")
    for row in match_rows:
        s1 = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

        if s1 in match_seen_s1:
            issues.append(f"MATCH: duplicate source1_entity_id row: {s1}")
        match_seen_s1.add(s1)

        if s1 not in s1_ids:
            issues.append(f"MATCH: source1_entity_id not in test S1: {s1}")

        id_set = set()
        for eid in ids:
            if eid in id_set:
                issues.append(f"MATCH: duplicate matched id {eid} for {s1}")
            id_set.add(eid)
            if eid not in valid_candidates:
                issues.append(f"MATCH: matched id {eid} not in test S2/S3 for {s1}")
            cands = cand_lookup.get(s1, set())
            if eid not in cands:
                issues.append(
                    f"MATCH: matched id {eid} for {s1} never appeared in candidate_pairs"
                )

    for s1 in s1_ids:
        if s1 not in match_seen_s1:
            issues.append(f"MATCH: missing row for S1 entity: {s1}")

    if issues:
        print(f"\nFAIL -- {len(issues)} issue(s) found:\n")
        for i, msg in enumerate(issues[:50], 1):
            print(f"  {i}. {msg}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more (truncated)")
        sys.exit(1)
    else:
        print("\nPASS -- submission files are valid and ready to upload.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission TSV files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()
    validate(args.matching, args.candidate, args.test_dir)
