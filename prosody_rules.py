# -*- coding: utf-8 -*-
"""
Rule-based Hindi laghu-guru (light/heavy) syllable weight assigner.
Classical Sanskrit/Hindi prosody (chandas) rules, v2.
"""

SHORT_MATRAS = {'\u093F', '\u0941', '\u0943',
                 '\u0945', '\u0949'}                    # ि  ु  ृ  ॅ  ॉ  (chandra e/o -> short, per corpus)
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

def is_consonant(ch):
    o = ord(ch)
    # standard consonant block 0915-0939, plus nukta-extended consonants 0958-095F
    return (0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)

def is_vowel_sign(ch):
    return ch in VOWEL_SIGNS

def is_indep_vowel(ch):
    return ch in INDEP_VOWELS


def syllabify(word):
    """
    Segments word into aksharas. Each syllable dict tracks:
      text, long (own vowel long?), nasal (anusvara/chandrabindu/visarga?),
      lead_conjunct_count (# of extra consonant+virama absorbed BEFORE this
      syllable's own vowel-bearing consonant - i.e. this syllable *is* a conjunct onset)
    """
    i, n = 0, len(word)
    syllables = []

    while i < n:
        ch = word[i]
        if is_indep_vowel(ch):
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
            onset_first = ch  # first (leading) consonant absorbed into this conjunct onset
            cur_base = ch
            # absorb (consonant [+nukta] + virama)* chain -> these are conjunct onsets
            while True:
                j = i + 1
                if j < n and word[j] == NUKTA:
                    j += 1
                if j < n and word[j] == VIRAMA:
                    i = j + 1
                    lead_conjunct += 1
                    if i < n and is_consonant(word[i]):
                        cur_base = word[i]
                        continue  # absorb next consonant of the cluster
                    else:
                        dangling = True
                        break  # word-final dangling virama, no more consonants
                else:
                    final_cons = cur_base  # this is the vowel-bearing consonant
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

        # unknown char (punctuation etc.) - skip
        i += 1

    return syllables


def assign_weights(syllables):
    """
    guru (2) if: own vowel is long, OR has anusvara/chandrabindu/visarga,
                 OR the *following* syllable is a conjunct onset (lead_conjunct>=1)
                 [saṃyoga / "guru by conjunction" rule]
    else laghu (1)
    """
    HA = '\u0939'  # ह
    NASAL_LATERAL_ONSETS = {'\u092E', '\u0928', '\u0932'}  # म न ल - म्ह/न्ह/ल्ह don't close; र्ह etc still do
    n = len(syllables)
    weights = []
    for s in syllables:
        if not s['has_own_vowel']:
            weights.append(1)  # word-final dangling dead consonant: light, doesn't count as "closed"
        else:
            weights.append(2 if (s['long'] or s['nasal']) else 1)
    for idx in range(1, n):
        s = syllables[idx]
        is_mnl_ha_exception = (s['final_cons'] == HA and s['onset_first'] in NASAL_LATERAL_ONSETS)
        if s['lead_conjunct'] >= 1 and s['has_own_vowel'] and not is_mnl_ha_exception:
            weights[idx - 1] = 2
    return weights
