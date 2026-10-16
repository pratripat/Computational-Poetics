# -*- coding: utf-8 -*-
"""
Hindi Laghu-Guru Scansion + Word-Formation Grouping - Rule-Based Algorithm
============================================================================
Pure rule-based (no ML). Two stages:

  STAGE 1 - Weight assignment (laghu=1 / guru=2 per syllable)
    Classical Sanskrit/Hindi prosody rules, refined against the corpus:
      - long vowel matra, OR anusvara/chandrabindu/visarga -> guru
      - a syllable followed by a conjunct consonant cluster is "closed"
        by it and becomes guru (saMyoga rule) -- UNLESS the conjunct's
        vowel-bearing consonant is 'ha' paired with a leading m/n/l onset
        (म्ह, न्ह, ल्ह don't close; र्ह etc. still do)
      - chandra vowel signs ऑ/ॅ count as short, not long
      - a word-final dangling consonant (no vowel ever attached) is
        itself laghu and does not retroactively close the previous syllable
    ~99.9% exact match against the 13k-word corpus.

  STAGE 2 - Bracket / word-formation grouping
    This is the harder, NOT-fully-solved part. Best rule found so far:
    "2-matra pairing counted from the right": a guru syllable is always
    its own group; laghu syllables pair up two-at-a-time starting from
    the rightmost syllable, leaving an odd leftover laghu (if any) as its
    own singleton group at the start of the word.
    This gets ~57% exact bracket-structure match on the corpus -- it is
    NOT a fully cracked rule. See the printed output for where it fails.

Usage:
    python3 hindi_scansion_algorithm.py /path/to/corpus.json

If no path is given, looks for a file named corpus.json in the same
directory, or falls back to the path baked in below.
"""

import json
import sys
import random

# ---------------------------------------------------------------------------
# STAGE 1: syllabification + laghu/guru weight assignment
# ---------------------------------------------------------------------------

SHORT_MATRAS = {'\u093F', '\u0941', '\u0943',
                 '\u0945', '\u0949'}                    # ि ु ृ ॅ ॉ (chandra e/o -> short)
LONG_MATRAS  = {'\u093E', '\u0940', '\u0942', '\u0947',
                '\u0948', '\u094B', '\u094C', '\u0944'} # ा ी ू े ै ो ौ ॄ

SHORT_INDEP_VOWELS = {'\u0905', '\u0907', '\u0909', '\u090B'}  # अ इ उ ऋ
LONG_INDEP_VOWELS  = {'\u0906', '\u0908', '\u090A', '\u090F',
                      '\u0910', '\u0913', '\u0914', '\u0960'}  # आ ई ऊ ए ऐ ओ औ ॠ

ANUSVARA = '\u0902'      # ं
CHANDRABINDU = '\u0901'  # ँ
VISARGA = '\u0903'       # ः
VIRAMA = '\u094D'        # ्
NUKTA = '\u093C'         # ़

VOWEL_SIGNS = SHORT_MATRAS | LONG_MATRAS
INDEP_VOWELS = SHORT_INDEP_VOWELS | LONG_INDEP_VOWELS

HA = '\u0939'  # ह
NASAL_LATERAL_ONSETS = {'\u092E', '\u0928', '\u0932'}  # म न ल


def is_consonant(ch):
    o = ord(ch)
    return (0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)


def syllabify(word):
    i, n = 0, len(word)
    syllables = []
    while i < n:
        ch = word[i]
        if ch in INDEP_VOWELS:
            long_ = ch in LONG_INDEP_VOWELS
            i += 1
            nasal = False
            if i < n and word[i] in (ANUSVARA, CHANDRABINDU, VISARGA):
                nasal = True
                i += 1
            syllables.append({'text': ch, 'long': long_, 'nasal': nasal, 'lead_conjunct': 0,
                               'has_own_vowel': True, 'final_cons': None, 'onset_first': None})
            continue
        if is_consonant(ch):
            lead_conjunct = 0
            start = i
            dangling = False
            final_cons = None
            onset_first = ch
            cur_base = ch
            while True:
                j = i + 1
                if j < n and word[j] == NUKTA:
                    j += 1
                if j < n and word[j] == VIRAMA:
                    i = j + 1
                    lead_conjunct += 1
                    if i < n and is_consonant(word[i]):
                        cur_base = word[i]
                        continue
                    else:
                        dangling = True
                        break
                else:
                    final_cons = cur_base
                    i = j
                    break
            long_ = False
            nasal = False
            has_own_vowel = not dangling
            if not dangling and i < n and word[i] in VOWEL_SIGNS:
                long_ = word[i] in LONG_MATRAS
                i += 1
            if not dangling and i < n and word[i] in (ANUSVARA, CHANDRABINDU, VISARGA):
                nasal = True
                i += 1
            syllables.append({'text': word[start:i], 'long': long_, 'nasal': nasal,
                               'lead_conjunct': lead_conjunct, 'has_own_vowel': has_own_vowel,
                               'final_cons': final_cons,
                               'onset_first': onset_first if lead_conjunct >= 1 else None})
            continue
        i += 1  # skip unknown chars (punctuation etc.)
    return syllables


def assign_weights(syllables):
    n = len(syllables)
    weights = []
    for s in syllables:
        if not s['has_own_vowel']:
            weights.append(1)
        else:
            weights.append(2 if (s['long'] or s['nasal']) else 1)
    for idx in range(1, n):
        s = syllables[idx]
        is_mnl_ha_exception = (s['final_cons'] == HA and s['onset_first'] in NASAL_LATERAL_ONSETS)
        if s['lead_conjunct'] >= 1 and s['has_own_vowel'] and not is_mnl_ha_exception:
            weights[idx - 1] = 2
    return weights


# ---------------------------------------------------------------------------
# STAGE 2: bracket / word-formation grouping (best rule found -- ~57% exact)
# ---------------------------------------------------------------------------

def group_weights(weights):
    """2-matra pairing counted from the right. Guru always solo; laghus
    pair up from the right; an odd leftover laghu becomes a singleton
    at the START of the word."""
    groups_rev = []
    buf = []
    for w in reversed(weights):
        if w == 2:
            if buf:
                groups_rev.append(list(reversed(buf)))
                buf = []
            groups_rev.append([2])
        else:
            buf.append(1)
            if len(buf) == 2:
                groups_rev.append(list(reversed(buf)))
                buf = []
    if buf:
        groups_rev.append(list(reversed(buf)))
    return list(reversed(groups_rev))


def analyze(word):
    """Full pipeline: word -> (syllable texts, weights, groups)"""
    syl = syllabify(word)
    weights = assign_weights(syl)
    groups = group_weights(weights)
    return [s['text'] for s in syl], weights, groups


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "corpus.json"
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Could not find corpus file at '{path}'. Pass the path as an argument:")
        print("  python3 hindi_scansion_algorithm.py /path/to/corpus.json")
        sys.exit(1)

    total = 0
    weight_correct = 0
    group_correct = 0
    correct_examples = []
    wrong_examples = []

    random.seed(42)
    items = list(data.items())
    random.shuffle(items)

    for word_key, v in items:
        # skip rows where the corpus's own 'word' field doesn't reconstruct
        # the key -- these are annotation glitches, not real test cases
        if ''.join(v.get('word', [])) != word_key:
            continue
        gold_groups = v['scansion']
        gold_flat = [x for g in gold_groups for x in g]

        syl_texts, pred_weights, pred_groups = analyze(word_key)

        total += 1
        w_ok = (pred_weights == gold_flat)
        g_ok = (pred_groups == gold_groups)
        if w_ok:
            weight_correct += 1
        if g_ok:
            group_correct += 1

        record = {
            'word': word_key, 'syllables': syl_texts,
            'gold_groups': gold_groups, 'pred_groups': pred_groups,
            'weight_ok': w_ok, 'group_ok': g_ok
        }
        if w_ok and g_ok and len(correct_examples) < 10:
            correct_examples.append(record)
        if not g_ok and len(wrong_examples) < 10:
            wrong_examples.append(record)

    print("=" * 70)
    print("CORRECT EXAMPLES (weights + grouping both right)")
    print("=" * 70)
    for r in correct_examples:
        print(f"  {r['word']:12s} syllables={r['syllables']} -> groups={r['pred_groups']}")

    print()
    print("=" * 70)
    print("WRONG EXAMPLES (grouping mismatch)")
    print("=" * 70)
    for r in wrong_examples:
        tag = "weights also wrong" if not r['weight_ok'] else "weights correct, grouping wrong"
        print(f"  {r['word']:12s} syllables={r['syllables']}")
        print(f"      gold = {r['gold_groups']}")
        print(f"      pred = {r['pred_groups']}   ({tag})")

    print()
    print("=" * 70)
    print("OVERALL ACCURACY")
    print("=" * 70)
    print(f"  Words evaluated (excluding {len(data)-total} corrupt/misaligned rows): {total}")
    print(f"  Stage 1 - flat weight sequence exact match: {weight_correct}/{total} "
          f"({100*weight_correct/total:.1f}%)")
    print(f"  Stage 2 - full bracket structure exact match: {group_correct}/{total} "
          f"({100*group_correct/total:.1f}%)")


if __name__ == "__main__":
    main()
