#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepares data.csv from corpus.json for syllable_grouping_training.ipynb.
Each row represents one syllable in a word sequence with features:
  - sequence_id: the word itself
  - position: 0-indexed syllable index within the word
  - syllable: the syllable text (akshara)
  - weight: 1 (laghu) or 2 (guru)
  - heaviness: 'intrinsic' (long/nasal), 'positional' (conjunct closed), or 'light'
  - label: 'B' (begins a grouping) or 'I' (inside/continues a grouping)
"""

import json
import pandas as pd
from hindi_scansion_algorithm import syllabify


def build_dataframe(corpus_path="corpus.json"):
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    skipped = 0

    for word_key, v in data.items():
        # Exclude corrupt or misaligned rows where 'word' chunks don't reconstruct the key
        if "".join(v.get("word", [])) != word_key:
            skipped += 1
            continue

        gold_groups = v.get("scansion", [])
        gold_flat = [w for g in gold_groups for w in g]
        syl = syllabify(word_key)

        # Ensure exact alignment between parsed syllables and gold weights
        if len(syl) != len(gold_flat):
            skipped += 1
            continue

        # Derive BIO grouping labels from scansion brackets
        labels = []
        for g in gold_groups:
            for idx in range(len(g)):
                labels.append("B" if idx == 0 else "I")

        for pos, (s, w, lbl) in enumerate(zip(syl, gold_flat, labels)):
            if w == 1:
                heaviness = "light"
            else:
                heaviness = "intrinsic" if (s["long"] or s["nasal"]) else "positional"

            records.append({
                "sequence_id": word_key,
                "position": pos,
                "syllable": s["text"],
                "weight": w,
                "heaviness": heaviness,
                "label": lbl
            })

    df = pd.DataFrame(records)
    print(f"Processed {len(data) - skipped} words into {len(df)} syllable rows (skipped {skipped} misaligned words).")
    return df


def main():
    df = build_dataframe("corpus.json")
    df.to_csv("data.csv", index=False, encoding="utf-8")
    print("Successfully created data.csv with shape:", df.shape)


if __name__ == "__main__":
    main()
