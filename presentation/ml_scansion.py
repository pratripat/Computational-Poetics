import json
import os
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "trained_model", "syllable_grouping_model.joblib")

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
            long_, nasal, hov = False, False, not dang
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


def assign_weights_and_heaviness(syllables, is_final=True):
    weights, heaviness = [], []
    for s in syllables:
        if not s["has_own_vowel"]:
            weights.append(1)
            heaviness.append("light")
        elif s["long"] or s["nasal"]:
            weights.append(2)
            heaviness.append("intrinsic")
        else:
            weights.append(1)
            heaviness.append("light")
    for idx in range(1, len(syllables)):
        s = syllables[idx]
        exc = (s["lead_conjunct"] == 1 and s["final_cons"] == HA and s["onset_first"] in NASAL_LATERAL)
        if s["lead_conjunct"] >= 1 and s["has_own_vowel"] and not exc:
            weights[idx - 1] = 2
            heaviness[idx - 1] = "positional"
    if not is_final and syllables and not syllables[-1]["has_own_vowel"] and len(weights) >= 2:
        weights[-2] = 2
        heaviness[-2] = "positional"
        weights.pop()
        heaviness.pop()
        syllables.pop()
    return weights, heaviness


def get_stage1_features(word, vocab_ge2):
    norm_w = word.strip().replace(":", VISARGA)
    parts = APPROVED_SPLITS.get(norm_w)
    if parts is None:
        for p in PREFIX_LIST:
            if norm_w.startswith(p) and len(norm_w) > len(p):
                rest = norm_w[len(p):]
                if rest in vocab_ge2 and rest != norm_w:
                    parts = [p, rest]
                    break
    if not parts:
        parts = [norm_w]

    all_syl, all_w, all_h = [], [], []
    for idx, part in enumerate(parts):
        syl = syllabify(part)
        w, h = assign_weights_and_heaviness(syl, is_final=(idx == len(parts) - 1))
        all_syl.extend(syl)
        all_w.extend(w)
        all_h.extend(h)
    return all_syl, all_w, all_h


def make_features(rows, i, cols):
    syl_c, w_c, h_c, pos_c = cols
    feats = {
        "bias": 1,
        "syllable": str(rows[i][syl_c]),
        "weight": str(rows[i][w_c]),
        "heaviness": str(rows[i][h_c]),
        "position": str(rows[i][pos_c]),
        "position_numeric": float(rows[i][pos_c]),
        "is_first": int(i == 0),
        "is_last": int(i == len(rows) - 1),
        "prev_syllable": str(rows[i - 1][syl_c]) if i > 0 else "<START>",
        "prev_weight": str(rows[i - 1][w_c]) if i > 0 else "<START>",
        "prev_heaviness": str(rows[i - 1][h_c]) if i > 0 else "<START>",
        "next_syllable": str(rows[i + 1][syl_c]) if i < len(rows) - 1 else "<END>",
        "next_weight": str(rows[i + 1][w_c]) if i < len(rows) - 1 else "<END>",
        "next_heaviness": str(rows[i + 1][h_c]) if i < len(rows) - 1 else "<END>",
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        p = f"offset_{offset}_"
        if 0 <= j < len(rows):
            feats[p + "syllable"] = str(rows[j][syl_c])
            feats[p + "weight"] = str(rows[j][w_c])
            feats[p + "heaviness"] = str(rows[j][h_c])
        else:
            feats[p + "syllable"] = "<BOUNDARY>"
            feats[p + "weight"] = "<BOUNDARY>"
            feats[p + "heaviness"] = "<BOUNDARY>"
    return feats


def predict_ml(word, bundle, vocab_ge2):
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    cfg = bundle["config"]
    cols = (
        cfg.get("SYLLABLE_COL", "syllable"),
        cfg.get("WEIGHT_COL", "weight"),
        cfg.get("HEAVY_COL", "heaviness"),
        cfg.get("POSITION_COL", "position"),
    )
    syl, weights, heaviness = get_stage1_features(word, vocab_ge2)
    rows = [
        {cols[0]: s["text"], cols[1]: w, cols[2]: h, cols[3]: i}
        for i, (s, w, h) in enumerate(zip(syl, weights, heaviness))
    ]
    preds = model.predict(vectorizer.transform([make_features(rows, i, cols) for i in range(len(rows))]))

    groups, curr = [], []
    for w, lbl in zip(weights, preds):
        if lbl == "B" and curr:
            groups.append(curr)
            curr = []
        curr.append(w)
    if curr:
        groups.append(curr)
    return weights, groups


def main():
    bundle = joblib.load(MODEL_PATH)
    gold = load_merged_gold()
    vocab_ge2 = {k for k in gold if len(syllabify(k)) >= 2 and not k.endswith(VIRAMA)}

    print("ML SCANSION (Logistic Regression retrained on Corrected DB + R1-R4 Stage-1)")
    print(f"{'Word':<12} {'Rule/Case':<26} {'Predicted Brackets':<24} {'Target Brackets':<24} {'Status'}")
    print("-" * 98)

    exact = 0
    for word, tag in SHOWCASE_WORDS:
        _, pred = predict_ml(word, bundle, vocab_ge2)
        target = gold.get(word)
        ok = (pred == target)
        exact += int(ok)
        status = "MATCH" if ok else "DIFF"
        print(f"{word:<12} {tag:<26} {str(pred):<24} {str(target):<24} {status}")

    print("-" * 98)
    print(f"Exact bracket match: {exact}/{len(SHOWCASE_WORDS)}")


if __name__ == "__main__":
    main()
