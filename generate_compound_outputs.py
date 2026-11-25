# -*- coding: utf-8 -*-
"""
generate_compound_outputs.py:
Identifies all words in the combined Swar and Vyanjan corpus that require compound generation
(prefix-root and root-root compound splits), performs morpheme-wise prosodic scansion,
verifies all invariants, and outputs files in the project's standard one-line JSON format and TSV.

Generated files:
  1. scansion_output_compounds_required.txt:
     Words that strictly REQUIRE compound generation (306 words where whole-word scansion
     falsely bumps weights before a boundary conjunct or incorrectly merges syllables across boundaries).
  2. scansion_output_compounds_swar_vyanjan.txt:
     All 1,603 valid compound and prefixed words scanned via compound generation rules.
  3. compound_candidates_swar_vyanjan.tsv:
     Detailed TSV report with parts, impact, whole vs split weights, and linguistic rationale.
"""

import json
import csv
import os
import sys

from hindi_prosody_pipeline import (
    syllabify, assign_weights, group_syllables,
    PREFIX_LIST, COMPOUND_BLACKLIST
)

# Canonical Hindi Varnamala / Dictionary Collation Order
DEV_ORDER = [
    'ं', 'ँ', 'ः',
    'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'ऑ',
    'क', 'क़', 'ख', 'ख़', 'ग', 'ग़', 'घ', 'ङ',
    'च', 'छ', 'ज', 'ज़', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ड़', 'ढ', 'ढ़', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'फ़', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
    'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ॅ', 'ॉ',
    '्', '़'
]
CHAR_TO_INDEX = {c: i for i, c in enumerate(DEV_ORDER)}

def hindi_collation_key(word):
    return [CHAR_TO_INDEX.get(c, 999) for c in word]


def write_oneline_json(filepath, data_dict):
    """Writes in the exact one-line JSON dictionary format used in the project."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('{\n')
        total = len(data_dict)
        for i, (k, v) in enumerate(data_dict.items()):
            comma = ',' if i < total - 1 else ''
            entry_json = json.dumps(v, ensure_ascii=False)
            f.write(f'  "{k}": {entry_json}{comma}\n')
        f.write('}\n')


def main():
    lexicon_path = 'scansion_output_full_swar_vyanjan.txt'
    if not os.path.exists(lexicon_path):
        print(f"Error: {lexicon_path} not found.")
        sys.exit(1)

    print(f"Loading full Swar-Vyanjan lexicon from '{lexicon_path}'...")
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    vocab = set(full_data.keys())

    # Filter out prefixes ending in virama (like अन्) that produce false splits on consonant clusters
    clean_prefix_list = [p for p in PREFIX_LIST if not p.endswith('्')]
    sorted_prefixes = sorted(clean_prefix_list, key=len, reverse=True)

    # Extra lexicon parts
    extra_lex = {}
    extra_path = 'extra_lexicon.json'
    if os.path.exists(extra_path):
        with open(extra_path, 'r', encoding='utf-8') as f:
            extra_lex = json.load(f)
            vocab |= set(extra_lex.keys())

    # Known approved k-words from Sir's corrections
    approved_k = {
        'कार्यक्रम': ['कार्य', 'क्रम'],
        'कालक्रम': ['काल', 'क्रम'],
        'कालत्रय': ['काल', 'त्रय'],
        'कार्यप्रणाली': ['कार्य', 'प्रणाली'],
        'कुख्यात': ['कु', 'ख्यात'],
        'कार्यक्षम': ['कार्य', 'क्षम'],
        'कतरब्योंत': ['कतर', 'ब्योंत'],
        'कुप्रबन्ध': ['कु', 'प्रबन्ध'],
        'कुप्रशासन': ['कु', 'प्रशासन'],
        'कोर्टमार्शल': ['कोर्ट', 'मार्शल'],
        'कालानुक्रमण': ['कालानुक्रम', 'ण'],
    }

    def scan_part(part):
        if extra_lex and part in extra_lex:
            e = extra_lex[part]
            chunks = e.get('word', [part])
            groups = e.get('scansion', [])
            wts = [w for g in groups for w in g]
            return chunks, wts, groups
        syl = syllabify(part)
        wts = assign_weights(syl)
        grps = group_syllables(syl, wts)
        chunks = []
        s_idx = 0
        for g in grps:
            chunk_text = ''.join(syl[s_idx + j]['text'] for j in range(len(g)))
            chunks.append(chunk_text)
            s_idx += len(g)
        return chunks, wts, grps

    print("Analyzing candidate compound and prefixed words...")
    all_compound_entries = {}
    required_compound_entries = {}
    tsv_rows = []

    sorted_words = sorted(full_data.keys(), key=hindi_collation_key)

    for w in sorted_words:
        if w in COMPOUND_BLACKLIST:
            continue

        # Handle hyphenated words
        if '-' in w:
            parts = w.split('-')
            stype = 'hyphenated'
        elif w in approved_k:
            parts = approved_k[w]
            stype = 'approved_k'
        else:
            parts = None
            stype = None
            # Check prefixes
            for p in sorted_prefixes:
                if w.startswith(p) and len(w) > len(p):
                    rest = w[len(p):]
                    if rest in vocab and rest != w:
                        parts = [p, rest]
                        stype = 'prefix+dict'
                        break
            # Check dict+dict
            if not parts:
                for i in range(2, len(w) - 1):
                    p1 = w[:i]
                    p2 = w[i:]
                    if p1.endswith('्'):
                        continue
                    if p1 in vocab and p2 in vocab and p1 != w and p2 != w:
                        s1 = syllabify(p1)
                        s2 = syllabify(p2)
                        if len(s1) >= 1 and len(s2) >= 1:
                            parts = [p1, p2]
                            stype = 'dict+dict'
                            break

        if not parts:
            continue

        # Whole-word baseline
        w_syl = syllabify(w)
        whole_wt = assign_weights(w_syl)
        whole_grp = group_syllables(w_syl, whole_wt)

        # Split scansion
        all_chunks = []
        all_wts = []
        all_grps = []
        for p_idx, p in enumerate(parts):
            c, wt, g = scan_part(p)
            if '-' in w and p_idx < len(parts) - 1:
                c[-1] += '-'
            all_chunks.extend(c)
            all_wts.extend(wt)
            all_grps.extend(g)

        # Invariant checks
        reconstructed = ''.join(all_chunks)
        if reconstructed != w or len(all_chunks) != len(all_grps):
            continue

        # Determine impact
        if all_wts != whole_wt:
            impact = 'WEIGHT_CHANGED'
            notes = 'Phonologically required: avoids artificial weight bump across boundary conjunct'
        elif all_grps != whole_grp:
            impact = 'GROUPING_CHANGED'
            notes = 'Morphologically required: preserves constituent boundary, prevents cross-boundary chunk merge'
        else:
            impact = 'IDENTICAL_SCANSION'
            notes = 'Compound constituents scanned independently with identical prosodic weights'

        entry_obj = {
            'word': all_chunks,
            'scansion': all_grps
        }

        all_compound_entries[w] = entry_obj

        # If scansion requires compound generation (weights or groupings change)
        if impact in ('WEIGHT_CHANGED', 'GROUPING_CHANGED') or stype == 'approved_k' or stype == 'hyphenated':
            required_compound_entries[w] = entry_obj

        tsv_rows.append({
            'word': w,
            'parts': ' + '.join(parts),
            'type': stype,
            'impact': impact,
            'whole_weights': str(whole_wt),
            'split_weights': str(all_wts),
            'whole_scansion': json.dumps(whole_grp),
            'split_scansion': json.dumps(all_grps),
            'chunks': json.dumps(all_chunks, ensure_ascii=False),
            'notes': notes
        })

    print(f"\nTotal compound words identified: {len(all_compound_entries)}")
    print(f"Words requiring compound generation (impact changed): {len(required_compound_entries)}")
    wt_changed_count = sum(1 for r in tsv_rows if r['impact'] == 'WEIGHT_CHANGED')
    grp_changed_count = sum(1 for r in tsv_rows if r['impact'] == 'GROUPING_CHANGED')
    id_count = sum(1 for r in tsv_rows if r['impact'] == 'IDENTICAL_SCANSION')
    print(f"  - Weight changed (boundary bump avoided):  {wt_changed_count}")
    print(f"  - Grouping changed (boundary preserved):    {grp_changed_count}")
    print(f"  - Identical scansion (pure compounds):      {id_count}")

    # 1. Write required compound entries JSON
    req_path = 'scansion_output_compounds_required.txt'
    write_oneline_json(req_path, required_compound_entries)
    print(f"\n[Generated] {req_path} ({len(required_compound_entries)} entries)")

    # 2. Write all compound entries JSON
    all_path = 'scansion_output_compounds_swar_vyanjan.txt'
    write_oneline_json(all_path, all_compound_entries)
    print(f"[Generated] {all_path} ({len(all_compound_entries)} entries)")

    # 3. Write TSV report
    tsv_path = 'compound_candidates_swar_vyanjan.tsv'
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word', 'parts', 'type', 'impact',
            'whole_weights', 'split_weights',
            'whole_scansion', 'split_scansion', 'chunks', 'notes'
        ], delimiter='\t')
        writer.writeheader()
        for r in tsv_rows:
            writer.writerow(r)
    print(f"[Generated] {tsv_path} ({len(tsv_rows)} rows)")


if __name__ == '__main__':
    main()
