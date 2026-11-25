import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

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

FEET = (2, 3, 4, 6)
PARTNER = 3
_PUNCT = re.compile(r"[।॥,.;:!?\"'“”‘’()\[\]{}<>…‌‍*_/\\|]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def is_consonant(ch):
    o = ord(ch)
    return (0x0915 <= o <= 0x0939) or (0x0958 <= o <= 0x095F)


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
    return ws


def load_vocab():
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

    keys = {k for k in orig if not (k.startswith("क") or k.startswith("क़"))}
    keys |= {k for k in corr if k.startswith("क") or k.startswith("क़")}
    return {k for k in keys if len(syllabify(k)) >= 2 and not k.endswith(VIRAMA)}


VOCAB_GE2 = load_vocab()


@lru_cache(maxsize=None)
def scan_weights(word):
    norm_w = word.replace(":", VISARGA)
    parts = APPROVED_SPLITS.get(norm_w)
    if parts is None:
        for p in PREFIX_LIST:
            if norm_w.startswith(p) and len(norm_w) > len(p):
                rest = norm_w[len(p):]
                if rest in VOCAB_GE2 and rest != norm_w:
                    parts = [p, rest]
                    break
    if parts:
        out = []
        for idx, part in enumerate(parts):
            out.extend(assign_weights(syllabify(part), is_final=(idx == len(parts) - 1)))
        return tuple(out)
    return tuple(assign_weights(syllabify(norm_w), is_final=True))


@dataclass
class Event:
    kind: str
    words: list
    text: str
    borrowed: bool = False


@dataclass
class LineResult:
    words: list
    score: float
    violations: int
    pairs: int
    borrows: int
    even_feet: int
    trailing_odd: int
    events: list = field(default_factory=list)


def analyse_line(word_weights, words):
    nw = len(word_weights)
    flat, word_of, w_start, w_end = [], [], [], []
    for k, ws in enumerate(word_weights):
        w_start.append(len(flat))
        flat.extend(ws)
        word_of.extend([k] * len(ws))
        w_end.append(len(flat))
    n = len(flat)
    pre = [0]
    for x in flat:
        pre.append(pre[-1] + x)
    S = lambda a, b: pre[b] - pre[a]

    def foot_options(p):
        k = word_of[p]
        e = w_end[k]
        if p == w_start[k] and S(p, e) in FEET:
            return [(e, S(p, e))]
        opts = [(q, S(p, q)) for q in range(e, p, -1) if S(p, q) in FEET and S(q, e) != 1]
        return opts or [(e, S(p, e))]

    def partner_from(q):
        r, t = 0, q
        while t < n:
            r += flat[t]
            t += 1
            if r == PARTNER:
                kw = word_of[t - 1]
                if S(t, w_end[kw]) == 1:
                    return None
                return t
            if r > PARTNER:
                return None
        return None

    INF = (10 ** 9, 10 ** 9)
    best = [INF] * (n + 1)
    choice = [None] * (n + 1)
    best[n] = (0, 0)

    for p in range(n - 1, -1, -1):
        for q, s in foot_options(p):
            if s % 2 == 0:
                cand, nxt, info = best[q], q, ("even", p, q, None)
            elif q == n:
                cand, nxt, info = (0, 0), n, ("trailing_odd", p, q, None)
            else:
                m = partner_from(q)
                if m is not None:
                    b = 1 if m < w_end[word_of[m - 1]] else 0
                    cand = (best[m][0], best[m][1] + b)
                    nxt, info = m, ("pair", p, q, m)
                else:
                    resume = w_end[word_of[p]]
                    cand = (best[resume][0] + 1, best[resume][1])
                    nxt, info = resume, ("violation", p, resume, None)
            if cand < best[p]:
                best[p] = cand
                choice[p] = (nxt, info)

    def wspan(a, b):
        return sorted({word_of[i] for i in range(a, b)})

    def label(a, b):
        out = []
        for k in wspan(a, b):
            lo, hi = max(a, w_start[k]), min(b, w_end[k])
            piece = words[k] if (lo, hi) == (w_start[k], w_end[k]) else f"{words[k]}[{lo - w_start[k]}:{hi - w_start[k]}]"
            out.append(piece)
        return "+".join(out)

    events, p = [], 0
    while p < n:
        nxt, (kind, a, b, m) = choice[p]
        if kind == "pair":
            events.append(Event("pair", wspan(a, m), f"{label(a, b)} | {label(b, m)}",
                                borrowed=m < w_end[word_of[m - 1]]))
        elif kind == "violation":
            msg = f"unpaired odd {label(a, b)}; discarded"
            if b < n:
                msg += f"; restarting at {words[word_of[b]]}"
            events.append(Event("violation", wspan(a, b), msg))
        else:
            events.append(Event(kind, wspan(a, b), label(a, b)))
        p = nxt

    v = sum(e.kind == "violation" for e in events)
    pairs = sum(e.kind == "pair" for e in events)
    evens = sum(e.kind == "even" for e in events)
    trail = sum(e.kind == "trailing_odd" for e in events)
    score = max(0.0, min(1.0, 1.0 - v / max(1, nw)))
    return LineResult(words, score, v, pairs, best[0][1] if n else 0, evens, trail, events)


def tokenize(line):
    line = re.sub(r"(?<=[ऀ-ॿ]):", VISARGA, line)
    line = _PUNCT.sub(" ", line.replace("-", " ").replace("–", " ").replace("—", " "))
    return [t for t in line.split() if _DEVANAGARI.search(t)]


def split_poem_lines(poem_text):
    return [seg.strip() for seg in re.findall(r"[^।॥.\r\n]+[।॥.]?", poem_text) if _DEVANAGARI.search(seg)]


def score_line(line):
    toks = [t for t in tokenize(line) if len(scan_weights(t)) > 0]
    if not toks:
        return None
    return analyse_line([list(scan_weights(t)) for t in toks], toks)


def read_pasted_poem():
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Paste your Hindi poem below (press Enter on an empty line when done):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


def main():
    poem_text = read_pasted_poem()
    poem_lines = split_poem_lines(poem_text)
    if not poem_lines:
        print("No Devanagari poem lines found.")
        return

    print("\nFORMATTED POEM:")
    for idx, pline in enumerate(poem_lines, 1):
        print(f"  {idx}. {pline}")

    sym = {"even": "EVEN ", "pair": "PAIR ", "trailing_odd": "END  ", "violation": "FAIL "}
    scores = []
    for pline in poem_lines:
        r = score_line(pline)
        print("\n" + "=" * 72)
        print("Line:", pline)
        if r is None:
            print("  (no Devanagari words found)")
            continue
        print("Words:", "  ".join(f"{w}={list(scan_weights(w))}(sum {sum(scan_weights(w))})" for w in r.words))
        print("Events:")
        for e in r.events:
            tag = "  [borrow]" if e.borrowed else ""
            print(f"   {sym[e.kind]} {e.text}{tag}")
        print(f"Violations: {r.violations} | pairs: {r.pairs} | borrows: {r.borrows} | even_feet: {r.even_feet} | trailing_odd: {r.trailing_odd} | words: {len(r.words)}")
        print(f"LINE SCORE: {r.score:.3f}")
        scores.append(r.score)

    print("\n" + "=" * 72)
    print(f"POEM SCORE (mean of {len(scores)} lines): {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
