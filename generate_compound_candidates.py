# -*- coding: utf-8 -*-
"""
Generate compound_candidates.tsv:
Lists candidate splits from dictionary and prefix matching, compares whole-word vs split
scansion against Sir's gold, and marks approval status and rationale.
"""

import json
import csv
import sys
from hindi_prosody_pipeline import (
    load_gold_data, syllabify, assign_weights, normalize_word,
    PREFIX_LIST, COMPOUND_BLACKLIST
)

merged, extra_lexicon = load_gold_data(
    'Pronunciation_Consonants_Amisha.txt',
    'Corrected_scansion_output_oneline_Consonants.txt',
    'extra_lexicon.json'
)

# Combined lookup vocabulary
vocab = set(merged.keys()) | set(extra_lexicon.keys())

k_words = sorted([k for k in merged if k.startswith('क') or k.startswith('क़')])

candidates = []

# Known manual judgments
known_approved = {
    'कार्यक्रम': (['कार्य', 'क्रम'], 'Compound word: conjunct क्र does not bump preceding syllable across boundary'),
    'कालक्रम': (['काल', 'क्रम'], 'Compound word: conjunct क्र does not bump preceding syllable across boundary'),
    'कालत्रय': (['काल', 'त्रय'], 'Compound word: conjunct त्र does not bump preceding syllable across boundary'),
    'कार्यप्रणाली': (['कार्य', 'प्रणाली'], 'Compound word: conjunct प्र does not bump preceding syllable across boundary'),
    'कुख्यात': (['कु', 'ख्यात'], 'Prefix कु + root ख्यात: conjunct ख्य does not bump कु across boundary'),
    'कार्यक्षम': (['कार्य', 'क्षम'], 'Compound word: conjunct क्ष does not bump preceding syllable across boundary (part क्षम in extra_lexicon)'),
    'कतरब्योंत': (['कतर', 'ब्योंत'], 'Compound word: कतर + ब्योंत (part कतर in extra_lexicon)'),
    'कुप्रबन्ध': (['कु', 'प्रबन्ध'], 'Prefix कु + root प्रबन्ध: conjunct प्र does not bump कु (part प्रबन्ध in extra_lexicon)'),
    'कुप्रशासन': (['कु', 'प्रशासन'], 'Prefix कु + root प्रशासन: conjunct प्र does not bump कु (part प्रशासन in extra_lexicon)'),
    'कोर्टमार्शल': (['कोर्ट', 'मार्शल'], 'Compound loanword: कोर्ट + मार्शल (part कोर्ट in extra_lexicon)'),
    'कालानुक्रमण': (['कालानुक्रम', 'ण'], 'Compound base + inflection'),
}

known_rejected = {
    'कनस्तर': (['कन', 'स्तर'], 'Tin canister (loanword/monomorphemic); not a compound of कन+स्तर'),
    'कुव्यवहार': (['कु', 'व्यवहार'], "Sir's file scans कु as heavy [2], bumping across boundary; excluded to match Sir's annotation")
}

for w in k_words:
    v = merged[w]
    gold_sc = v.get('scansion', [])
    gold_flat = [x for g in gold_sc for x in (g if isinstance(g, list) else [g])]
    
    # Check known approved/rejected first
    if w in known_approved:
        parts, reason = known_approved[w]
        status = 'APPROVED'
        split_type = 'prefix+dict' if parts[0] in PREFIX_LIST else 'dict+dict'
        # Compute weights
        whole_wt = assign_weights(syllabify(w))
        split_wt = []
        for p in parts:
            if extra_lexicon and p in extra_lexicon:
                split_wt.extend([x for grp in extra_lexicon[p]['scansion'] for x in grp])
            else:
                split_wt.extend(assign_weights(syllabify(p)))
        impact = 'FIXES' if split_wt == gold_flat and whole_wt != gold_flat else 'NEUTRAL'
        candidates.append({
            'word': w,
            'parts': ' + '.join(parts),
            'type': split_type,
            'status': status,
            'impact': impact,
            'gold_weights': str(gold_flat),
            'whole_weights': str(whole_wt),
            'split_weights': str(split_wt),
            'notes': reason
        })
        continue

    if w in known_rejected:
        parts, reason = known_rejected[w]
        status = 'REJECTED'
        split_type = 'prefix+dict' if parts[0] in PREFIX_LIST else 'dict+dict'
        whole_wt = assign_weights(syllabify(w))
        split_wt = []
        for p in parts:
            split_wt.extend(assign_weights(syllabify(p)))
        impact = 'BREAKS' if split_wt != gold_flat and whole_wt == gold_flat else 'NEUTRAL'
        candidates.append({
            'word': w,
            'parts': ' + '.join(parts),
            'type': split_type,
            'status': status,
            'impact': impact,
            'gold_weights': str(gold_flat),
            'whole_weights': str(whole_wt),
            'split_weights': str(split_wt),
            'notes': reason
        })
        continue

    # Find candidates
    found_splits = []
    # 1. Prefix
    for p in PREFIX_LIST:
        if w.startswith(p) and len(w) > len(p):
            rest = w[len(p):]
            if rest in vocab and rest != w:
                found_splits.append((p, rest, 'prefix+dict'))

    # 2. Dict + dict
    for i in range(2, len(w) - 1):
        p1, p2 = w[:i], w[i:]
        if p1 in vocab and p2 in vocab and p1 != w and p2 != w:
            found_splits.append((p1, p2, 'dict+dict'))

    if not found_splits:
        continue

    # Pick the best split (fewest parts, longest first part)
    p1, p2, stype = found_splits[0]
    parts = [p1, p2]
    whole_wt = assign_weights(syllabify(w))
    split_wt = []
    for p in parts:
        split_wt.extend(assign_weights(syllabify(p)))

    if split_wt == gold_flat and whole_wt != gold_flat:
        impact = 'FIXES'
        status = 'APPROVED'
        notes = 'Split fixes weight mismatch caused by conjunct bump across boundary'
    elif split_wt != gold_flat and whole_wt == gold_flat:
        impact = 'BREAKS'
        status = 'REJECTED'
        notes = 'Split breaks valid whole-word scansion'
    else:
        impact = 'NEUTRAL'
        status = 'PENDING'
        notes = 'Candidate compound split; scansion unaffected'

    candidates.append({
        'word': w,
        'parts': f"{p1} + {p2}",
        'type': stype,
        'status': status,
        'impact': impact,
        'gold_weights': str(gold_flat),
        'whole_weights': str(whole_wt),
        'split_weights': str(split_wt),
        'notes': notes
    })

# Write to compound_candidates.tsv
with open('compound_candidates.tsv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'word', 'parts', 'type', 'status', 'impact',
        'gold_weights', 'whole_weights', 'split_weights', 'notes'
    ], delimiter='\t')
    writer.writeheader()
    for row in candidates:
        writer.writerow(row)

print(f"Generated compound_candidates.tsv with {len(candidates)} rows.")
