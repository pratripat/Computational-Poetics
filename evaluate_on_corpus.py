# -*- coding: utf-8 -*-
import json, sys
sys.path.insert(0, '.')
from prosody_rules import syllabify, assign_weights

with open('PATH_TO_YOUR_CORPUS.txt', encoding='utf-8') as f:
    data = json.load(f)
    print(data[:5])

total = 0
exact_match = 0
len_mismatch = 0
mismatches = []

for word_key, v in data.items():
    gold_flat = [x for group in v['scansion'] for x in group]
    syl = syllabify(word_key)
    pred = assign_weights(syl)
    total += 1
    if pred == gold_flat:
        exact_match += 1
    else:
        if len(pred) != len(gold_flat):
            len_mismatch += 1
        mismatches.append((word_key, v['word'], gold_flat, pred, [s['text'] for s in syl]))

print(f"Total words: {total}")
print(f"Exact flat-weight-sequence match: {exact_match} ({100*exact_match/total:.1f}%)")
print(f"Of the mismatches ({len(mismatches)}), syllable-count itself wrong: {len_mismatch}")
print()
print("Sample mismatches (syllable count correct, weight wrong):")
count = 0
for w, gold_word, gold, pred, mysyl in mismatches:
    if len(pred) == len(gold):
        print(f"  {w}: gold={gold} pred={pred} | syl={mysyl}")
        count += 1
        if count >= 20:
            break
