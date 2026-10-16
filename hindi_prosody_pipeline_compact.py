import sys
import json
import os

SHORT_MATRAS = {'ि', 'ु', 'ृ', 'ॅ', 'ॉ'}
LONG_MATRAS  = {'ा', 'ी', 'ू', 'े',
                 'ै', 'ो', 'ौ', 'ॄ'}
SHORT_INDEP_VOWELS = {'अ', 'इ', 'उ', 'ऋ'}
LONG_INDEP_VOWELS  = {'आ', 'ई', 'ऊ', 'ए',
                       'ऐ', 'ओ', 'औ', 'ॠ'}

ANUSVARA = 'ं'
CHANDRABINDU = 'ँ'
VISARGA = 'ः'
VIRAMA = '्'
NUKTA = '़'

VOWEL_SIGNS = SHORT_MATRAS | LONG_MATRAS
INDEP_VOWELS = SHORT_INDEP_VOWELS | LONG_INDEP_VOWELS

HA = 'ह'
NASAL_LATERAL_ONSETS = {'म', 'न', 'ल'}

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
        i += 1
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

def group_syllables(weights):
    groups = []
    i = 0
    n = len(weights)
    while i < n:
        if weights[i] == 2 and i + 1 < n and weights[i + 1] == 1:
            groups.append([2, 1])
            i += 2
        elif weights[i] == 1 and i + 1 < n and weights[i + 1] == 1:
            groups.append([1, 1])
            i += 2
        else:
            groups.append([weights[i]])
            i += 1
    return groups

def predict(word):
    syl = syllabify(word)
    weights = assign_weights(syl)
    groups = group_syllables(weights)
    return syl, weights, groups

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
            skipped_noisy += 1
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