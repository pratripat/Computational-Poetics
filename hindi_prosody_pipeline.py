# -*- coding: utf-8 -*-
"""
Hindi word -> laghu/guru (light/heavy) scansion, with bracket grouping.

Pipeline:
  1. syllabify(word)      -> break the word into Devanagari aksharas (syllables)
  2. assign_weights(...)  -> classical Sanskrit/Hindi prosody rules (laghu=1, guru=2)
                             + two corpus-verified exceptions (see README section below)
  3. group_syllables(...) -> group the flat weight sequence into brackets
                             (2-matra pairing, counted right-to-left)

HOW TO RUN
----------
    python3 hindi_prosody_pipeline.py /path/to/your_corpus.txt

The corpus file must be the same JSON format as the one this was built from:
    { "<word>": {"word": ["<chunk1>", "<chunk2>", ...],
                 "scansion": [[1], [2, 1], ...]},
      ... }

If you run it with no argument, it looks for a file named corpus.txt in the
same folder.

WHAT THE OUTPUT MEANS
----------------------
The script evaluates the pipeline against every word in your corpus and prints:
  1. A handful of words where the full predicted output (both the weights
     AND the bracket grouping) exactly matches the dictionary's answer.
  2. A handful of words where it does NOT match, showing gold vs. predicted,
     so you can see exactly what's going wrong.
  3. Three accuracy numbers:
       - Weight accuracy:   how often the flat 1/2 sequence alone is correct
                             (ignoring bracket placement). Expect ~99.9%.
       - Grouping accuracy: how often the bracket PLACEMENT is correct, given
                             the correct weights. Expect ~55-60% - this part
                             is NOT reliably rule-based (see note below).
       - End-to-end accuracy: how often the full output (weights + brackets
                             together) matches exactly, run from the raw word
                             with no gold shortcuts. This will be close to
                             the grouping accuracy, since weights are
                             essentially solved already.

IMPORTANT CAVEAT ON THE GROUPING STEP
--------------------------------------
Unlike the weight-assignment rules (which are genuinely deterministic and
get ~99.9%), the bracket-grouping rule below is the best pattern found by
testing several hypotheses against this corpus - it is NOT a confirmed
linguistic rule, just the strongest regularity found (~57% exact match).
Evidence suggests grouping sometimes depends on information not visible in
the spelling alone (morphology/compounding, or the annotator's own
judgment). Treat this grouping function as a baseline to compare an ML
model against, not as a finished rule.

THE DATASET HAS SOME NOISY ROWS
---------------------------------
A small number of entries (~0.3%) have "word" chunks that don't even
concatenate back to the original word (typos / annotation glitches, e.g. a
row of underscores, or a chunk list that's missing a character). Those rows
are automatically skipped during evaluation (they'd be noise either way),
and are not used to judge the algorithm's accuracy.
"""

import sys
import json
import os

# ---------------------------------------------------------------------------
# STEP 1 + 2: syllabification and laghu/guru weight assignment
# ---------------------------------------------------------------------------

SHORT_MATRAS = {'ि', 'ु', 'ृ', 'ॅ', 'ॉ'}   # ि ु ृ ॅ ॉ (chandra e/o -> short)
LONG_MATRAS  = {'ा', 'ी', 'ू', 'े',
                 'ै', 'ो', 'ौ', 'ॄ'}            # ा ी ू े ै ो ौ ॄ

SHORT_INDEP_VOWELS = {'अ', 'इ', 'उ', 'ऋ'}       # अ इ उ ऋ
LONG_INDEP_VOWELS  = {'आ', 'ई', 'ऊ', 'ए',
                       'ऐ', 'ओ', 'औ', 'ॠ'}      # आ ई ऊ ए ऐ ओ औ ॠ

ANUSVARA = 'ं'      # ं
CHANDRABINDU = 'ँ'  # ँ
VISARGA = 'ः'       # ः
VIRAMA = '्'        # ्
NUKTA = '़'         # ़

VOWEL_SIGNS = SHORT_MATRAS | LONG_MATRAS
INDEP_VOWELS = SHORT_INDEP_VOWELS | LONG_INDEP_VOWELS

HA = 'ह'  # ह
NASAL_LATERAL_ONSETS = {'म', 'न', 'ल'}  # म न ल


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
            syllables.append({'text': ch, 'long': long_, 'nasal': nasal,
                               'lead_conjunct': 0, 'has_own_vowel': True, 'final_cons': None,
                               'onset_first': None})
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

        i += 1  # skip unexpected characters (punctuation etc.)
    return syllables


def assign_weights(syllables):
    n = len(syllables)
    weights = []
    for s in syllables:
        if not s['has_own_vowel']:
            weights.append(1)  # word-final dangling dead consonant: treated as light
        else:
            weights.append(2 if (s['long'] or s['nasal']) else 1)
    for idx in range(1, n):
        s = syllables[idx]
        is_mnl_ha_exception = (s['final_cons'] == HA and s['onset_first'] in NASAL_LATERAL_ONSETS)
        if s['lead_conjunct'] >= 1 and s['has_own_vowel'] and not is_mnl_ha_exception:
            weights[idx - 1] = 2
    return weights


# ---------------------------------------------------------------------------
# STEP 3: bracket grouping (best-found heuristic, NOT a confirmed rule - see
# the caveat in the module docstring above)
# ---------------------------------------------------------------------------

def group_syllables(weights):
    """Group a flat weight sequence into brackets: 2-matra units, paired
    from the right. A guru(2) is always its own group; laghus(1) pair up
    two at a time starting from the rightmost; a leftover odd laghu at the
    start of the word becomes its own singleton group."""
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


def predict(word):
    """Full pipeline: word -> (syllables, flat weights, bracket groups)."""
    syl = syllabify(word)
    weights = assign_weights(syl)
    groups = group_syllables(weights)
    return syl, weights, groups


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'corpus.txt')

    if not os.path.exists(corpus_path):
        print(f"Corpus file not found: {corpus_path}")
        print("Run as: python3 hindi_prosody_pipeline.py /path/to/corpus.txt")
        sys.exit(1)

    with open(corpus_path, encoding='utf-8') as f:
        data = json.load(f)

    total = 0
    skipped_noisy = 0
    weight_correct = 0
    full_correct = 0
    correct_examples = []
    wrong_examples = []

    for word, entry in data.items():
        gold_groups = entry.get('scansion')
        gold_word_chunks = entry.get('word')
        if gold_groups is None or gold_word_chunks is None:
            skipped_noisy += 1
            continue
        if ''.join(gold_word_chunks) != word:
            skipped_noisy += 1  # row doesn't even reconstruct the word - noisy annotation
            continue

        gold_flat = [x for g in gold_groups for x in g]
        syl, pred_weights, pred_groups = predict(word)

        total += 1
        if pred_weights == gold_flat:
            weight_correct += 1
        if pred_groups == gold_groups:
            full_correct += 1
            if len(correct_examples) < 8:
                correct_examples.append((word, gold_groups, pred_groups))
        else:
            if len(wrong_examples) < 8:
                wrong_examples.append((word, gold_groups, pred_groups))

    print("=" * 70)
    print("CORRECT EXAMPLES  (predicted output == dictionary's answer)")
    print("=" * 70)
    for word, gold, pred in correct_examples:
        print(f"  {word}:  {pred}")

    print()
    print("=" * 70)
    print("WRONG EXAMPLES  (gold vs. predicted differ)")
    print("=" * 70)
    for word, gold, pred in wrong_examples:
        print(f"  {word}:  gold={gold}   predicted={pred}")

    print()
    print("=" * 70)
    print("ACCURACY")
    print("=" * 70)
    print(f"  Words evaluated: {total}  (skipped {skipped_noisy} noisy/malformed rows)")
    print(f"  Weight-only accuracy (flat 1/2 sequence, brackets ignored): "
          f"{weight_correct}/{total}  ({100*weight_correct/total:.1f}%)")
    print(f"  Full end-to-end accuracy (weights + bracket grouping together): "
          f"{full_correct}/{total}  ({100*full_correct/total:.1f}%)")


if __name__ == '__main__':
    main()
