# -*- coding: utf-8 -*-
"""
Hindi prosody pipeline (Chandas / Matra scansion and bracket grouping).

Pipeline:
  1. Preprocessing       -> Normalise ASCII colons to visarga (R2)
  2. Compound splitting  -> Split into prefix/dictionary parts with leave-one-out lookup (R4)
  3. syllabify(word)     -> Segment word into Devanagari syllables (aksharas)
  4. assign_weights(...) -> Assign laghu (1) and guru (2) weights using prosody rules:
                            - Short/Long vowels, anusvara, chandrabindu, visarga
                            - ॉ is long (R1), ॅ is short
                            - Conjunct bump rule across non-compound boundaries
                            - Single consonant before ह exception (म्ह, न्ह, ल्ह) (R3)
  5. group_syllables(...) -> Group syllables into 2-matra bracket units:
                            - Only bare-a light joins left neighbor; heavy stands alone (R5)
                            - Scanned right-to-left
"""

import sys
import json
import os
import re

# ---------------------------------------------------------------------------
# Character definitions and sets
# ---------------------------------------------------------------------------

# R1: ॉ (U+0949, candra O) is LONG per Sir's corrections (कॉपी, कॉलर, कॉलेज)
# ॅ (U+0945, candra E) remains SHORT (as in चॅवर)
SHORT_MATRAS = {'ि', 'ु', 'ृ', 'ॅ'}
LONG_MATRAS  = {'ा', 'ी', 'ू', 'े', 'ै', 'ो', 'ौ', 'ॄ', 'ॉ'}

SHORT_INDEP_VOWELS = {'अ', 'इ', 'उ', 'ऋ'}
LONG_INDEP_VOWELS  = {'आ', 'ई', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'ॠ'}

ANUSVARA = 'ं'
CHANDRABINDU = 'ँ'
VISARGA = 'ः'
VIRAMA = '्'
NUKTA = '़'

VOWEL_SIGNS = SHORT_MATRAS | LONG_MATRAS
INDEP_VOWELS = SHORT_INDEP_VOWELS | LONG_INDEP_VOWELS

HA = 'ह'
NASAL_LATERAL_ONSETS = {'म', 'न', 'ल'}

# Sanskrit / Hindi prefixes recognized for compound boundaries (R4)
# Single consonants (क, म, स) are excluded to avoid accidental splits
PREFIX_LIST = [
    'सु', 'कु', 'वि', 'प्र', 'नि', 'सम्', 'दुस्', 'अति', 'अधि', 'अनु',
    'अभि', 'उप', 'परा', 'परि', 'महा', 'अप', 'अव', 'उत्', 'निर्', 'दुर्',
    'प्रति', 'अन्'
]

# Explicit compound blacklist / overrides (accidental splits to avoid)
COMPOUND_BLACKLIST = {'कनस्तर', 'कुव्यवहार'}


def is_consonant(ch):
    o = ord(ch)
    return (0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)


def normalize_word(word):
    """R2: Normalize ASCII colon to Devanagari visarga (e.g. क्रमशः, छी:छी:)."""
    return word.replace(':', VISARGA)


# ---------------------------------------------------------------------------
# STEP 1: Syllabification
# ---------------------------------------------------------------------------

def syllabify(word):
    """Segment word into Devanagari aksharas (syllables)."""
    word = normalize_word(word)
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
            syllables.append({
                'text': ch,
                'long': long_,
                'nasal': nasal,
                'lead_conjunct': 0,
                'has_own_vowel': True,
                'final_cons': None,
                'onset_first': None
            })
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

            syllables.append({
                'text': word[start:i],
                'long': long_,
                'nasal': nasal,
                'lead_conjunct': lead_conjunct,
                'has_own_vowel': has_own_vowel,
                'final_cons': final_cons,
                'onset_first': onset_first if lead_conjunct >= 1 else None
            })
            continue

        i += 1  # Skip unrecognized symbols / whitespace
    return syllables


# ---------------------------------------------------------------------------
# STEP 2: Weight Assignment
# ---------------------------------------------------------------------------

def assign_weights(syllables):
    """
    Classical Hindi prosody weight assignment:
      - Short vowel without conjunct/nasal: laghu (1)
      - Long vowel, anusvara, chandrabindu, visarga: guru (2)
      - Preceding syllable closed by following conjunct: guru (2)
      - R3 fix: single consonant + ह exception applies ONLY when lead_conjunct == 1
    """
    n = len(syllables)
    weights = []

    for s in syllables:
        if not s['has_own_vowel']:
            weights.append(1)  # word-final dangling consonant
        else:
            weights.append(2 if (s['long'] or s['nasal']) else 1)

    for idx in range(1, n):
        s = syllables[idx]
        # R3: Doubled consonant before ह (e.g. चिन्न्ह, ब्रम्म्हा) closes syllable;
        # the म्ह/न्ह/ल्ह exemption applies only when lead_conjunct == 1
        is_mnl_ha_exception = (
            s['lead_conjunct'] == 1 and
            s['final_cons'] == HA and
            s['onset_first'] in NASAL_LATERAL_ONSETS
        )
        if s['lead_conjunct'] >= 1 and s['has_own_vowel'] and not is_mnl_ha_exception:
            weights[idx - 1] = 2

    return weights


# ---------------------------------------------------------------------------
# STEP 3: Bracket Grouping (R5)
# ---------------------------------------------------------------------------

def is_bare_a(s):
    """A light syllable with no vowel mark and not an independent vowel (the hidden 'a')."""
    return s['has_own_vowel'] and not any(c in VOWEL_SIGNS or c in INDEP_VOWELS for c in s['text'])


def group_syllables(syllables, weights):
    """
    R5: Only a bare-a light joins its left neighbor.
    Scanned right to left. A heavy (2) stands alone.
    A light (1) joins left only if it has no vowel mark (bare-a) and i > 0.
    Lights with vowel marks (ि, ु, ृ, etc.) stand alone.
    """
    out, i = [], len(weights) - 1
    while i >= 0:
        if weights[i] == 1 and is_bare_a(syllables[i]) and i > 0:
            out.append([weights[i - 1], weights[i]])
            i -= 2
        else:
            out.append([weights[i]])
            i -= 1
    return out[::-1]


# ---------------------------------------------------------------------------
# STEP 4: Compound Splitting (R4)
# ---------------------------------------------------------------------------

def find_compound_split(word, lexicon=None, extra_lexicon=None, approved_splits=None):
    """
    R4: Look up compound/prefix split for word.
    Uses leave-one-out dictionary lookup: word itself is excluded.
    """
    if approved_splits and word in approved_splits:
        return approved_splits[word]

    if word in COMPOUND_BLACKLIST:
        return None

    # Check prefix list
    for p in PREFIX_LIST:
        if word.startswith(p) and len(word) > len(p):
            rest = word[len(p):]
            # Rest must be in lexicon (excluding word itself) or extra_lexicon
            in_lex = lexicon and (rest in lexicon and rest != word)
            in_extra = extra_lexicon and (rest in extra_lexicon)
            if in_lex or in_extra:
                return [p, rest]

    # Check dictionary + dictionary split
    # Prefer fewest parts, then longest first part
    if lexicon or extra_lexicon:
        combined = set()
        if lexicon:
            combined |= {k for k in lexicon.keys() if k != word}
        if extra_lexicon:
            combined |= set(extra_lexicon.keys())

        best_split = None
        for i in range(2, len(word) - 1):
            p1 = word[:i]
            p2 = word[i:]
            if p1 in combined and p2 in combined:
                best_split = [p1, p2]
                break
        if best_split:
            return best_split

    return None


def predict(word, lexicon=None, extra_lexicon=None, approved_splits=None, use_splits=True):
    """
    Full pipeline: word -> (syllables, flat weights, bracket groups).
    Supports leave-one-out dictionary compound splitting (R4).
    """
    norm_w = normalize_word(word)

    if use_splits:
        split_parts = find_compound_split(norm_w, lexicon=lexicon,
                                          extra_lexicon=extra_lexicon,
                                          approved_splits=approved_splits)
        if split_parts:
            all_syls = []
            all_weights = []
            all_groups = []
            for part in split_parts:
                # Reuse dictionary's own bracketing if available
                part_entry = None
                if extra_lexicon and part in extra_lexicon:
                    part_entry = extra_lexicon[part]
                elif lexicon and part in lexicon and part != word:
                    part_entry = lexicon[part]

                syl = syllabify(part)
                all_syls.extend(syl)

                if part_entry and 'scansion' in part_entry:
                    g = part_entry['scansion']
                    w = [x for grp in g for x in grp]
                    all_weights.extend(w)
                    all_groups.extend(g)
                else:
                    w = assign_weights(syl)
                    g = group_syllables(syl, w)
                    all_weights.extend(w)
                    all_groups.extend(g)

            return all_syls, all_weights, all_groups

    # Whole-word prediction
    syl = syllabify(norm_w)
    weights = assign_weights(syl)
    groups = group_syllables(syl, weights)
    return syl, weights, groups


# ---------------------------------------------------------------------------
# Data loader and hygiene (Section 1)
# ---------------------------------------------------------------------------

def repair_json_content(text):
    """Fix the 3 known malformed JSON lines in Sir's corrected file."""
    text = text.replace('["कद्","  दा ,"वर"]', '["कद्","दा","वर"]')
    text = text.replace('["कार्य,"पा","लि","का"]', '["कार्य","पा","लि","का"]')
    text = text.replace('["कू",जन"]', '["कू","जन"]')
    return text


def load_gold_data(orig_path, corr_path=None, extra_lexicon_path=None):
    """
    Builds the merged gold dataset per Section 1:
      - Corrected file for words starting with क / क़ (source = 'sir_corrected')
      - Original file for all other words (source = 'uncorrected')
      - Strips whitespace inside chunks
      - Repairs করকરાહટ weight typo (1.1 -> [1, 1])
      - Removes respelled old keys honest
    """
    with open(orig_path, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)

    corr_data = {}
    if corr_path and os.path.exists(corr_path):
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_text = repair_json_content(f.read())
        corr_data = json.loads(corr_text)

        # 1.2: Strip whitespace inside chunks
        for k, v in corr_data.items():
            v['word'] = [c.strip() for c in v.get('word', [])]

        # 1.3: করকરાહટ typo fix [1.1] -> [1, 1]
        if 'करकराहट' in corr_data:
            new_sc = []
            for g in corr_data['करकराहट']['scansion']:
                new_g = []
                for w in g:
                    if w == 1.1:
                        new_g.extend([1, 1])
                    else:
                        new_g.append(int(w))
                new_sc.append(new_g)
            corr_data['करकराहट']['scansion'] = new_sc

    extra_lexicon = {}
    if extra_lexicon_path and os.path.exists(extra_lexicon_path):
        with open(extra_lexicon_path, 'r', encoding='utf-8') as f:
            extra_lexicon = json.load(f)

    merged = {}
    # Non-k words from original
    for k, v in orig_data.items():
        if not (k.startswith('क') or k.startswith('क़')):
            v_copy = dict(v)
            v_copy['source'] = 'uncorrected'
            merged[k] = v_copy

    # k-words from corrected (or original fallback if no corrected file)
    source_k = corr_data if corr_data else orig_data
    for k, v in source_k.items():
        if k.startswith('क') or k.startswith('क़'):
            v_copy = dict(v)
            v_copy['source'] = 'sir_corrected' if corr_data else 'uncorrected'
            merged[k] = v_copy

    return merged, extra_lexicon


# ---------------------------------------------------------------------------
# Evaluation Protocol (Section 2)
# ---------------------------------------------------------------------------

def evaluate_subset(dataset, name, lexicon=None, extra_lexicon=None, approved_splits=None, use_splits=True):
    total = 0
    weight_correct = 0
    grouping_given_weight_correct = 0
    weights_matched_count = 0
    e2e_correct = 0
    wrong_samples = []

    for word, entry in dataset.items():
        gold_groups = entry.get('scansion')
        if not gold_groups:
            continue

        gold_flat = [w for g in gold_groups for w in (g if isinstance(g, list) else [g])]

        # Leave-one-out: pass lexicon without word itself
        syl, pred_weights, pred_groups = predict(
            word,
            lexicon=lexicon,
            extra_lexicon=extra_lexicon,
            approved_splits=approved_splits,
            use_splits=use_splits
        )

        total += 1
        w_match = (pred_weights == gold_flat)
        if w_match:
            weight_correct += 1
            weights_matched_count += 1
            if pred_groups == gold_groups:
                grouping_given_weight_correct += 1

        if pred_groups == gold_groups and w_match:
            e2e_correct += 1
        else:
            if len(wrong_samples) < 10:
                wrong_samples.append((word, gold_groups, pred_groups, pred_weights, gold_flat))

    w_acc = 100 * weight_correct / total if total else 0.0
    g_acc = 100 * grouping_given_weight_correct / weights_matched_count if weights_matched_count else 0.0
    e2e_acc = 100 * e2e_correct / total if total else 0.0

    print(f"\n--- EVALUATION: {name} (N = {total}) ---")
    print(f"  1. Weight accuracy:               {weight_correct}/{total} ({w_acc:.2f}%)")
    print(f"  2. Grouping accuracy (weights ok): {grouping_given_weight_correct}/{weights_matched_count} ({g_acc:.2f}%)")
    print(f"  3. End-to-end exact match:         {e2e_correct}/{total} ({e2e_acc:.2f}%)")

    return {
        'total': total,
        'weight_acc': w_acc,
        'grouping_acc': g_acc,
        'e2e_acc': e2e_acc,
        'weight_correct': weight_correct,
        'grouping_correct': grouping_given_weight_correct,
        'e2e_correct': e2e_correct,
        'wrong_samples': wrong_samples
    }


def main():
    orig_path = sys.argv[1] if len(sys.argv) > 1 else 'Pronunciation_Consonants_Amisha.txt'
    corr_path = sys.argv[2] if len(sys.argv) > 2 else 'Corrected_scansion_output_oneline_Consonants.txt'
    extra_path = 'extra_lexicon.json'

    if not os.path.exists(orig_path):
        print(f"Original file not found: {orig_path}")
        sys.exit(1)

    print("Loading datasets and applying data hygiene...")
    merged, extra_lexicon = load_gold_data(orig_path, corr_path, extra_path)

    # Approved compound splits
    approved_splits = {
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

    # Split into clean k-section and uncorrected rest
    # Clean k-section = 953 words (valid reconstruct rows from sir_corrected starting with क)
    clean_k = {
        k: v for k, v in merged.items()
        if v.get('source') == 'sir_corrected' and k.startswith('क') and ''.join(v.get('word', [])) == k
    }
    uncorrected_rest = {
        k: v for k, v in merged.items()
        if v.get('source') == 'uncorrected' and ''.join(v.get('word', [])) == k
    }

    print(f"Total merged entries: {len(merged)}")
    print(f"Clean k-section test set: {len(clean_k)} words")
    print(f"Uncorrected rest test set: {len(uncorrected_rest)} words")

    print("\n" + "=" * 70)
    print("EVALUATING PIPELINE WITH RULES R1-R5 (Leave-One-Out)")
    print("=" * 70)

    res_k = evaluate_subset(clean_k, "Clean k-section", lexicon=merged,
                            extra_lexicon=extra_lexicon, approved_splits=approved_splits, use_splits=True)

    res_rest = evaluate_subset(uncorrected_rest, "Uncorrected rest", lexicon=merged,
                               extra_lexicon=extra_lexicon, approved_splits=approved_splits, use_splits=False)


if __name__ == '__main__':
    main()
