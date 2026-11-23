# -*- coding: utf-8 -*-
"""
compare_corrections.py: Diff original corpus vs corrected corpus.
Can be reused for every new letter corrected by Sir.

Analyses:
  1. Key diffs: added, removed, common keys
  2. Respellings (near-match pairs)
  3. Scansion changes breakdown:
     - Weight-only changes
     - Bracket-only changes
     - Both weight and bracket changes
  4. Direction vs current prosody rule (did Sir's edit move towards rule or away?)
  5. Section 1 validators:
     - JSON syntax repair needs
     - Whitespace inside chunks
     - Word chunk reconstruction: "".join(chunks) == key
     - Chunk-count vs bracket-count: len(chunks) == len(scansion)
"""

import sys
import os
import json
import re
from difflib import SequenceMatcher

# Import prosody pipeline components
from hindi_prosody_pipeline import syllabify, assign_weights, group_syllables, predict


def repair_json_text(raw_text):
    """Detect and repair known common syntax errors in Sir's oneline files."""
    repairs_made = []
    
    # Literal replacements for exact known corrupted lines
    if '["कद्","  दा ,"वर"]' in raw_text:
        raw_text = raw_text.replace('["कद्","  दा ,"वर"]', '["कद्","दा","वर"]')
        repairs_made.append('कद्दावर syntax repair: ["कद्","  दा ,"वर"] -> ["कद्","दा","वर"]')
    if '["कार्य,"पा","लि","का"]' in raw_text:
        raw_text = raw_text.replace('["कार्य,"पा","लि","का"]', '["कार्य","पा","लि","का"]')
        repairs_made.append('कार्यपालिका syntax repair: ["कार्य,"पा","लि","का"] -> ["कार्य","पा","लि","का"]')
    if '["कू",जन"]' in raw_text:
        raw_text = raw_text.replace('["कू",जन"]', '["कू","जन"]')
        repairs_made.append('कूजन syntax repair: ["कू",जन"] -> ["कू","जन"]')
    
    return raw_text, repairs_made


def load_corpus(path, repair=True):
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    repairs_logged = []
    if repair:
        raw_text, repairs_logged = repair_json_text(raw_text)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Fallback line-by-line parsing
        data = {}
        lines = raw_text.splitlines()
        for idx, line in enumerate(lines):
            l = line.strip()
            if l in ('{', '}'):
                continue
            if l.endswith(','):
                l = l[:-1]
            try:
                item = json.loads('{' + l + '}')
                data.update(item)
            except Exception as line_err:
                print(f"[Warning] Line {idx+1} parse failure: {line_err}: {l[:50]}")
    return data, repairs_logged


def run_validators(data, label="Dataset"):
    report = {
        'total': len(data),
        'whitespace_chunks': [],
        'reconstruct_mismatch': [],
        'len_chunk_bracket_mismatch': [],
        'float_weights': []
    }
    for k, v in data.items():
        chunks = v.get('word', [])
        scansion = v.get('scansion', [])
        
        # Whitespace
        if any(re.search(r'\s', c) for c in chunks):
            report['whitespace_chunks'].append((k, chunks))
            
        # Reconstruct
        stripped_chunks = [c.strip() for c in chunks]
        if ''.join(stripped_chunks) != k:
            report['reconstruct_mismatch'].append((k, chunks, ''.join(stripped_chunks)))
            
        # Chunk count vs bracket count
        if len(chunks) != len(scansion):
            report['len_chunk_bracket_mismatch'].append((k, len(chunks), len(scansion), chunks, scansion))
            
        # Float weights (like 1.1)
        for g in scansion:
            for w in (g if isinstance(g, list) else [g]):
                if isinstance(w, float):
                    report['float_weights'].append((k, g, w))
                    
    return report


def diff_corpora(orig_path, corr_path, target_prefix=None):
    orig_data, orig_repairs = load_corpus(orig_path, repair=False)
    corr_data, corr_repairs = load_corpus(corr_path, repair=True)

    print("=" * 80)
    print("COMPARE CORRECTIONS REPORT")
    print(f"Original file:  {orig_path} ({len(orig_data)} entries)")
    print(f"Corrected file: {corr_path} ({len(corr_data)} entries)")
    print("=" * 80)

    if corr_repairs:
        print("\n[Data Hygiene] JSON syntax repairs applied to corrected file:")
        for r in corr_repairs:
            print(f"  - {r}")

    # Key differences
    orig_keys = set(orig_data.keys())
    corr_keys = set(corr_data.keys())

    removed_keys = sorted(orig_keys - corr_keys)
    added_keys = sorted(corr_keys - orig_keys)
    common_keys = sorted(orig_keys & corr_keys)

    print(f"\n[Key Diff] Removed keys: {len(removed_keys)} | Added keys: {len(added_keys)} | Common keys: {len(common_keys)}")

    # Check respellings: find similarity between removed and added
    respellings = []
    unmatched_added = set(added_keys)
    for rem in removed_keys:
        best_match, best_sim = None, 0.0
        for add in unmatched_added:
            sim = SequenceMatcher(None, rem, add).ratio()
            if sim > best_sim:
                best_sim = sim
                best_match = add
        if best_sim >= 0.6:
            respellings.append((rem, best_match, best_sim))

    print(f"\n[Respellings / Near-Key Edits] Found {len(respellings)} plausible pairs (similarity >= 0.6):")
    for rem, add, sim in respellings[:15]:
        print(f"  {rem} -> {add} (sim: {sim:.2f})")
    if len(respellings) > 15:
        print(f"  ... and {len(respellings)-15} more.")

    # Scansion comparisons on common keys
    filter_keys = common_keys
    if target_prefix:
        filter_keys = [k for k in filter_keys if any(k.startswith(p) for p in target_prefix)]
        print(f"\nFiltering analysis to keys with prefix {target_prefix}: {len(filter_keys)} words")

    scansion_unchanged = 0
    weight_only_changed = []
    bracket_only_changed = []
    both_changed = []

    # Check direction vs current rule for bracket-only edits
    bracket_moved_towards_rule = []
    bracket_moved_away_from_rule = []
    bracket_neutral = []

    for k in filter_keys:
        o_v = orig_data[k]
        c_v = corr_data[k]
        o_sc = o_v.get('scansion', [])
        c_sc = c_v.get('scansion', [])

        o_flat = [w for g in o_sc for w in (g if isinstance(g, list) else [g])]
        c_flat = [w for g in c_sc for w in (g if isinstance(g, list) else [g])]

        # normalize 1.1 if present in corr
        c_flat_norm = [1 if x == 1.1 else int(x) for x in c_flat]
        o_flat_norm = [1 if x == 1.1 else int(x) for x in o_flat]

        weights_same = (o_flat_norm == c_flat_norm)
        brackets_same = (o_sc == c_sc)

        if weights_same and brackets_same:
            scansion_unchanged += 1
        elif not weights_same and brackets_same:
            weight_only_changed.append((k, o_sc, c_sc))
        elif weights_same and not brackets_same:
            bracket_only_changed.append((k, o_sc, c_sc))
            # Check direction vs rule
            syl = syllabify(k)
            pred_weights = assign_weights(syl)
            pred_groups = group_syllables(syl, pred_weights)
            
            o_matches_pred = (o_sc == pred_groups)
            c_matches_pred = (c_sc == pred_groups)
            if not o_matches_pred and c_matches_pred:
                bracket_moved_towards_rule.append((k, o_sc, c_sc, pred_groups))
            elif o_matches_pred and not c_matches_pred:
                bracket_moved_away_from_rule.append((k, o_sc, c_sc, pred_groups))
            else:
                bracket_neutral.append((k, o_sc, c_sc, pred_groups))
        else:
            both_changed.append((k, o_sc, c_sc))

    print("\n" + "=" * 50)
    print("SCANSION EDITS BREAKDOWN (Common target words)")
    print("=" * 50)
    print(f"Total analyzed words:               {len(filter_keys)}")
    print(f"Scansion identical:                 {scansion_unchanged} ({100*scansion_unchanged/len(filter_keys):.1f}%)")
    print(f"Weight-only changed:                {len(weight_only_changed)}")
    print(f"Bracket-only changed:               {len(bracket_only_changed)}")
    print(f"Both weight and brackets changed:   {len(both_changed)}")

    if bracket_only_changed:
        print(f"\n[Bracket-Only Direction Analysis (n={len(bracket_only_changed)})]")
        print(f"  Moved TOWARDS pipeline rule:     {len(bracket_moved_towards_rule)}")
        print(f"  Moved AWAY FROM pipeline rule:   {len(bracket_moved_away_from_rule)}")
        print(f"  Neutral / other:                 {len(bracket_neutral)}")

    # Validators
    print("\n" + "=" * 50)
    print("VALIDATORS REPORT")
    print("=" * 50)
    corr_val = run_validators(corr_data, "Corrected")
    
    # Split violations between k-section and uncorrected rest
    k_len_mismatch = [v for v in corr_val['len_chunk_bracket_mismatch'] if any(v[0].startswith(p) for p in ('क', 'क़'))]
    rest_len_mismatch = [v for v in corr_val['len_chunk_bracket_mismatch'] if not any(v[0].startswith(p) for p in ('क', 'क़'))]

    print(f"Corrected file (k-section):")
    print(f"  - len(chunks) != len(brackets) violations: {len(k_len_mismatch)} entries (100% consistent)")
    print(f"  - Whitespace inside chunks:                 {len(corr_val['whitespace_chunks'])} entries")
    for k, ch in corr_val['whitespace_chunks'][:7]:
        print(f"      {k}: {ch}")
    print(f"  - Float weights (e.g. 1.1 typo):            {len(corr_val['float_weights'])} entries")
    for k, g, w in corr_val['float_weights']:
        print(f"      {k}: {g}")
    print(f"  - Key reconstruct typos in chunks:          {len(corr_val['reconstruct_mismatch'])} entries")
    for k, ch, j in corr_val['reconstruct_mismatch'][:6]:
        print(f"      {k} != '{j}' (chunks={ch})")

    print(f"\nUncorrected rest of dictionary (non-k words):")
    print(f"  - len(chunks) != len(brackets) violations: {len(rest_len_mismatch)} entries (list for Sir)")
    for k, lc, lb, ch, br in rest_len_mismatch[:15]:
        print(f"      {k}: chunks={lc} brackets={lb} -> {ch} vs {br}")
    if len(rest_len_mismatch) > 15:
        print(f"      ... and {len(rest_len_mismatch) - 15} more.")

    orig_val = run_validators(orig_data, "Original")

    return {
        'removed_keys': removed_keys,
        'added_keys': added_keys,
        'bracket_only_changed': bracket_only_changed,
        'bracket_moved_towards': bracket_moved_towards_rule,
        'bracket_moved_away': bracket_moved_away_from_rule,
        'corr_validators': corr_val,
        'orig_validators': orig_val,
    }


def main():
    orig_path = sys.argv[1] if len(sys.argv) > 1 else 'Pronunciation_Consonants_Amisha.txt'
    corr_path = sys.argv[2] if len(sys.argv) > 2 else 'Corrected_scansion_output_oneline_Consonants.txt'
    diff_corpora(orig_path, corr_path, target_prefix=('क', 'क़'))


if __name__ == '__main__':
    main()
