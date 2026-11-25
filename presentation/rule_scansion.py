import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHORT_MATRAS = {'ि', 'ु', 'ृ', 'ॅ'}
LONG_MATRAS  = {'ा', 'ी', 'ू', 'े', 'ै', 'ो', 'ौ', 'ॄ', 'ॉ'}
SHORT_INDEP_VOWELS = {'अ', 'इ', 'उ', 'ऋ', 'ऍ'}
LONG_INDEP_VOWELS  = {'आ', 'ई', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ', 'ॠ', 'ऑ'}
VOWEL_SIGNS = SHORT_MATRAS | LONG_MATRAS
INDEP_VOWELS = SHORT_INDEP_VOWELS | LONG_INDEP_VOWELS

ANUSVARA, CHANDRABINDU, VISARGA, VIRAMA, NUKTA = 'ं', 'ँ', 'ः', '्', '़'
HA = 'ह'
NASAL_LATERAL = {'म', 'न', 'ल'}

PREFIX_LIST = [
    'प्रति', 'अति', 'अधि', 'अनु', 'अभि', 'परा', 'परि', 'महा',
    'सम्', 'दुस्', 'निर्', 'दुर्', 'उत्', 'अन्',
    'उप', 'अप', 'अव', 'सु', 'कु', 'वि', 'प्र', 'नि', 'दु', 'अ'
]
APPROVED_SPLITS = {
    'कार्यक्रम': ['कार्य', 'क्रम'],
    'कालक्रम': ['काल', 'क्रम'],
    'कालत्रय': ['काल', 'त्रय'],
    'कार्यप्रणाली': ['कार्य', 'प्रणाली'],
}

SHOWCASE_WORDS = [
    ("कमल",      "Simple (bare-a)"),
    ("बात",      "Simple (guru + bare-a)"),
    ("किताब",    "Marked light (ि stands alone)"),
    ("नरदेव",    "Simple 4-syl pairing"),
    ("शहसवार",   "5-syl grouping"),
    ("कक्षा",    "Conjunct closure"),
    ("कंकरीला",  "Anusvara nasal (ं=2)"),
    ("कविता",    "R5 (ि stands alone)"),
    ("कॉपी",     "R1 (ॉ is long)"),
    ("कॉलेज",    "R1 (ॉ is long)"),
    ("क्रमश:",   "R2 (ASCII ':' -> ः)"),
    ("कुम्हार",  "R3 (1 cons + ह exempt)"),
    ("चिन्न्ह",   "R3 (doubled न्ह bumps)"),
    ("ब्रम्म्हा", "R3 (doubled म्ह bumps)"),
    ("कार्यक्रम", "R4 (dict compound)"),
    ("कालत्रय",  "R4 (dict compound)"),
    ("कुख्यात",  "R4 (prefix कु + ख्यात)"),
    ("कनस्तर",   "R4 guard (do not split)"),
]


def is_consonant(ch):
    o = ord(ch)
    return (0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)


def load_merged_gold():
    with open(os.path.join(ROOT, "Pronunciation_Consonants_Amisha.txt"), encoding="utf-8") as f:
        orig = json.load(f)
    with open(os.path.join(ROOT, "Corrected_scansion_output_oneline_Consonants.txt"), encoding="utf-8") as f:
        raw = f.read()
    for a, b in [
        ('["कद्","  दा ,"वर"]', '["कद्","दा","वर"]'),
        ('["कार्य,"पा","लि","का"]', '["कार्य","पा","लि","का"]'),
        ('["कू",जन"]', '["कू","जन"]'),
    ]:
        raw = raw.replace(a, b)
    corr = json.loads(raw)

    gold = {k: v["scansion"] for k, v in orig.items() if not (k.startswith("क") or k.startswith("क़"))}
    for k, v in corr.items():
        if k.startswith("क") or k.startswith("क़"):
            gold[k] = [[1, 1] if x == [1.1] else x for x in v["scansion"]]
    return gold


def syllabify(word):
    word = word.replace(":", VISARGA)
    i, n = 0, len(word)
    syllables = []
    while i < n:
        ch = word[i]
        if ch in INDEP_VOWELS:
            start = i
            long_ = ch in LONG_INDEP_VOWELS
            i += 1
            nasal = False
            if i < n and word[i] in (ANUSVARA, CHANDRABINDU, VISARGA):
                nasal = True
                i += 1
            syllables.append({
                "text": word[start:i], "long": long_, "nasal": nasal,
                "lead_conjunct": 0, "has_own_vowel": True,
                "final_cons": None, "onset_first": None,
            })
            continue
        if is_consonant(ch):
            lc, start, dang, fc, of, cb = 0, i, False, None, ch, ch
            while True:
                j = i + 1
                if j < n and word[j] == NUKTA:
                    j += 1
                if j < n and word[j] == VIRAMA:
                    i = j + 1
                    lc += 1
                    if i < n and is_consonant(word[i]):
                        cb = word[i]
                        continue
                    dang = True
                    break
                fc = cb
                i = j
                break
            long_ = False
            nasal = False
            hov = not dang
            if not dang and i < n and word[i] in VOWEL_SIGNS:
                long_ = word[i] in LONG_MATRAS
                i += 1
            if not dang and i < n and word[i] in (ANUSVARA, CHANDRABINDU, VISARGA):
                nasal = True
                i += 1
            syllables.append({
                "text": word[start:i], "long": long_, "nasal": nasal,
                "lead_conjunct": lc, "has_own_vowel": hov,
                "final_cons": fc, "onset_first": of if lc >= 1 else None,
            })
            continue
        i += 1
    return syllables


def assign_weights(syls, is_final=True):
    ws = [1 if not s["has_own_vowel"] else (2 if (s["long"] or s["nasal"]) else 1) for s in syls]
    for idx in range(1, len(syls)):
        s = syls[idx]
        exc = (s["lead_conjunct"] == 1 and s["final_cons"] == HA and s["onset_first"] in NASAL_LATERAL)
        if s["lead_conjunct"] >= 1 and s["has_own_vowel"] and not exc:
            ws[idx - 1] = 2
    if not is_final and syls and not syls[-1]["has_own_vowel"] and len(ws) >= 2:
        ws[-2] = 2
        ws.pop()
        syls.pop()
    return ws


def bare_a(s):
    return s["has_own_vowel"] and not any(c in VOWEL_SIGNS or c in INDEP_VOWELS for c in s["text"])


def group_syllables(syl, ws):
    out, i = [], len(ws) - 1
    while i >= 0:
        if ws[i] == 1 and bare_a(syl[i]) and i > 0:
            out.append([ws[i - 1], ws[i]])
            i -= 2
        else:
            out.append([ws[i]])
            i -= 1
    return out[::-1]


def predict_rule(word, vocab_ge2):
    norm_w = word.replace(":", VISARGA)
    parts = APPROVED_SPLITS.get(norm_w)
    if parts is None:
        for p in PREFIX_LIST:
            if norm_w.startswith(p) and len(norm_w) > len(p):
                rest = norm_w[len(p):]
                if rest in vocab_ge2 and rest != norm_w:
                    parts = [p, rest]
                    break

    if parts:
        all_groups, all_weights = [], []
        for idx, part in enumerate(parts):
            syls = syllabify(part)
            ws = assign_weights(syls, is_final=(idx == len(parts) - 1))
            all_weights.extend(ws)
            all_groups.extend(group_syllables(syls, ws))
        return all_weights, all_groups

    syls = syllabify(norm_w)
    ws = assign_weights(syls, is_final=True)
    return ws, group_syllables(syls, ws)


def main():
    gold = load_merged_gold()
    vocab_ge2 = {k for k in gold if len(syllabify(k)) >= 2 and not k.endswith(VIRAMA)}

    print("RULE-BASED SCANSION (Rules R1-R5)")
    print(f"{'Word':<12} {'Rule/Case':<26} {'Predicted Brackets':<24} {'Target Brackets':<24} {'Status'}")
    print("-" * 98)

    exact = 0
    for word, tag in SHOWCASE_WORDS:
        _, pred = predict_rule(word, vocab_ge2)
        target = gold.get(word)
        ok = (pred == target)
        exact += int(ok)
        status = "MATCH" if ok else "DIFF"
        print(f"{word:<12} {tag:<26} {str(pred):<24} {str(target):<24} {status}")

    print("-" * 98)
    print(f"Exact bracket match: {exact}/{len(SHOWCASE_WORDS)}")


if __name__ == "__main__":
    main()
