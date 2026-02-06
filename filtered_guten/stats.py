#!/usr/bin/env python3
"""Summarize annotation results from JSONL output."""

import json
import sys
from collections import defaultdict
from pathlib import Path

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("annotations_sample.jsonl")
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    relevance_counts = defaultdict(int)
    relevance_tokens = defaultdict(int)
    confidence_counts = defaultdict(int)
    cross = defaultdict(lambda: defaultdict(int))  # relevance -> confidence -> count
    errors = 0
    total = 0

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            total += 1

            if "_error" in rec:
                errors += 1
                continue

            rel = rec.get("corpus_relevance", "unknown")
            conf = rec.get("confidence", "unknown")
            tokens = rec.get("tokens", 0)  # may not be present in annotation output

            relevance_counts[rel] += 1
            relevance_tokens[rel] += tokens
            confidence_counts[conf] += 1
            cross[rel][conf] += 1

    # Display
    print(f"Total records: {total}  (errors: {errors})\n")

    print("=== Corpus Relevance ===")
    for rel in ["core", "useful", "marginal", "questionable", "unknown"]:
        n = relevance_counts.get(rel, 0)
        if n == 0:
            continue
        tok = relevance_tokens.get(rel, 0)
        tok_str = f"  ({tok:,} tokens)" if tok > 0 else ""
        print(f"  {rel:15s}  {n:5d}{tok_str}")

    print(f"\n=== Confidence ===")
    for conf in ["high", "medium", "low", "unknown"]:
        n = confidence_counts.get(conf, 0)
        if n == 0:
            continue
        print(f"  {conf:15s}  {n:5d}")

    print(f"\n=== Cross-tab: Relevance x Confidence ===")
    header = f"  {'':15s}  {'high':>6s}  {'medium':>6s}  {'low':>6s}"
    print(header)
    for rel in ["core", "useful", "marginal", "questionable"]:
        if relevance_counts.get(rel, 0) == 0:
            continue
        row = f"  {rel:15s}"
        for conf in ["high", "medium", "low"]:
            row += f"  {cross[rel].get(conf, 0):6d}"
        print(row)

if __name__ == "__main__":
    main()