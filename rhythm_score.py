# -*- coding: utf-8 -*-
"""
Rhythm scoring for Hindi free verse (muktak kavita).

Input : a poem (text) -> words -> flat laghu/guru weights per word (from hindi_prosody_pipeline.predict)
Output: Line Rhythm Score in [0,1] per line, Poem Rhythm Score = mean of line scores.

Run:   python3 rhythm_score.py --selftest
       python3 rhythm_score.py poem.txt            (one line of verse per line)
"""
import re
import sys
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

FEET = (2, 3, 4, 6)   # legal rhythmic groups: dwikal, trikal, chaukal, shatkal
PARTNER = 3           # an odd group must be followed by a trikal (sum exactly 3)


# --------------------------------------------------------------------------- data model
@dataclass
class Event:
    kind: str                 # 'even' | 'pair' | 'trailing_odd' | 'violation'
    words: List[int]          # indices of the words this event touches
    text: str                 # human-readable, e.g. "चलो | हम+क"
    borrowed: bool = False    # a word was cut to form the partner


@dataclass
class LineResult:
    words: List[str]
    score: float
    violations: int
    pairs: int
    borrows: int
    even_feet: int
    trailing_odd: int
    discarded_words: int
    irregular_words: int      # words whose total is not 2/3/4/6
    events: List[Event] = field(default_factory=list)


@dataclass
class PoemResult:
    score: float
    lines: List[LineResult]


# --------------------------------------------------------------------------- core algorithm
def analyse_line(word_weights: Sequence[Sequence[int]], words: Optional[Sequence[str]] = None,
                 normalise: str = "words", severity: float = 1.0) -> LineResult:
    """
    word_weights: one list of syllable weights (1/2) per word.

    Representation: ONE flat stream of weights plus word-boundary metadata
    (word_of / w_start / w_end) and prefix sums, so a group is just a span [a,b)
    and a word can be cut anywhere on a syllable boundary.

    Rules (spec):
      * even group  -> stable, stands alone
      * odd group   -> must be followed IMMEDIATELY by an odd group (partner = next syllables
                       summing to exactly 3, may end inside a word = borrowing)
      * odd group at the very end of the line -> exempt
      * no partner  -> 1 violation; the broken segment (odd group .. end of the word where the
                       look-ahead overshot 3) is discarded, evaluation resumes at the next word
    Words whose total is not 2/3/4/6 (31% of dictionary words) have no fixed grouping, so the DP
    tries every legal split of them into feet and keeps the one with fewest violations,
    then fewest borrows. Words with total 2/3/4/6 are never split.
    """
    nw = len(word_weights)
    names = list(words) if words is not None else [str(i) for i in range(nw)]
    flat, word_of, w_start, w_end = [], [], [], []
    for k, ws in enumerate(word_weights):
        w_start.append(len(flat)); flat.extend(ws); word_of.extend([k] * len(ws)); w_end.append(len(flat))
    n = len(flat)
    pre = [0]
    for x in flat:
        pre.append(pre[-1] + x)
    S = lambda a, b: pre[b] - pre[a]

    def foot_options(p):
        k = word_of[p]; e = w_end[k]
        if p == w_start[k] and S(p, e) in FEET:
            return [(e, S(p, e))]                      # regular word: one whole foot
        opts = [(q, S(p, q)) for q in range(e, p, -1)
                if S(p, q) in FEET and S(q, e) != 1]   # never strand a lone laghu
        return opts or [(e, S(p, e))]                  # fallback: whole remainder, parity only

    def partner_from(q):
        """Shortest syllable run after q with sum exactly 3. -> (end, overshoot_word)"""
        r, t = 0, q
        while t < n:
            r += flat[t]; t += 1
            if r == PARTNER:
                kw = word_of[t - 1]
                if S(t, w_end[kw]) == 1:               # cut would strand a lone laghu
                    return None, kw
                return t, None
            if r > PARTNER:
                return None, word_of[t - 1]
        return None, None                              # stream exhausted

    INF = (10 ** 9, 10 ** 9)
    best = [INF] * (n + 1); choice = [None] * (n + 1)
    best[n] = (0, 0)
    for p in range(n - 1, -1, -1):
        for q, s in foot_options(p):
            if s % 2 == 0:
                cand, nxt, info = best[q], q, ('even', p, q, None)
            elif q == n:
                cand, nxt, info = (0, 0), n, ('trailing_odd', p, q, None)
            else:
                m, over = partner_from(q)
                if m is not None:
                    b = 1 if m < w_end[word_of[m - 1]] else 0
                    cand = (best[m][0], best[m][1] + b); nxt = m; info = ('pair', p, q, m)
                else:
                    resume = w_end[over] if over is not None else n
                    cand = (best[resume][0] + 1, best[resume][1]); nxt = resume
                    info = ('violation', p, resume, None)
            if cand < best[p]:
                best[p] = cand; choice[p] = (nxt, info)

    # ---- reconstruct events
    def wspan(a, b):
        return sorted({word_of[i] for i in range(a, b)})

    def label(a, b):
        out = []
        for k in wspan(a, b):
            lo, hi = max(a, w_start[k]), min(b, w_end[k])
            piece = names[k] if (lo, hi) == (w_start[k], w_end[k]) else f"{names[k]}[{lo - w_start[k]}:{hi - w_start[k]}]"
            out.append(piece)
        return "+".join(out)

    events, p = [], 0
    while p < n:
        nxt, (kind, a, b, m) = choice[p]
        if kind == 'pair':
            events.append(Event('pair', wspan(a, m), f"{label(a, b)} | {label(b, m)}",
                                borrowed=m < w_end[word_of[m - 1]]))
        elif kind == 'violation':
            events.append(Event('violation', wspan(a, b), f"discard {label(a, b)}"))
        else:
            events.append(Event(kind, wspan(a, b), label(a, b)))
        p = nxt

    V = sum(e.kind == 'violation' for e in events)
    pairs = sum(e.kind == 'pair' for e in events)
    evens = sum(e.kind == 'even' for e in events)
    trail = sum(e.kind == 'trailing_odd' for e in events)
    discarded = sum(len(e.words) for e in events if e.kind == 'violation')
    irregular = sum(1 for ws in word_weights if sum(ws) not in FEET)

    if normalise == "words":            # V <= number of words (each violation sits in a distinct word)
        denom = max(1, nw)
    elif normalise == "units":          # share of rhythmic units that failed: V <= units by construction
        denom = max(1, evens + pairs + trail + V)
    else:
        raise ValueError(normalise)
    score = max(0.0, min(1.0, 1.0 - severity * V / denom))
    return LineResult(names, score, V, pairs, best[0][1] if n else 0, evens, trail, discarded, irregular, events)


# --------------------------------------------------------------------------- text -> weights
_PUNCT = re.compile(r"[।॥,.;:!?\"'“”‘’()\[\]{}<>…‌‍*_/\\|]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def tokenize(line: str) -> List[str]:
    line = _PUNCT.sub(" ", line.replace("-", " ").replace("–", " ").replace("—", " "))
    return [t for t in line.split() if _DEVANAGARI.search(t)]


@lru_cache(maxsize=None)
def default_scan(word: str) -> tuple:
    """Flat weights of one word from the prosody pipeline (rules). Swap in a model: scan=lambda w: [...]"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hindi_prosody_pipeline import predict
    return tuple(predict(word)[1])


def score_line(line: str, scan: Callable[[str], Sequence[int]] = default_scan, **kw) -> Optional[LineResult]:
    toks = [t for t in tokenize(line) if len(scan(t)) > 0]
    if not toks:
        return None
    return analyse_line([list(scan(t)) for t in toks], toks, **kw)


def score_poem(text: str, scan: Callable[[str], Sequence[int]] = default_scan, **kw) -> PoemResult:
    lines = [r for r in (score_line(l, scan, **kw) for l in text.splitlines()) if r is not None]
    return PoemResult(sum(r.score for r in lines) / len(lines) if lines else float("nan"), lines)


# --------------------------------------------------------------------------- self-test
def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, extra)

    # spec example 1: borrowing resolves the break
    r = analyse_line([[1, 2], [1, 1], [1, 2], [2, 1], [1, 2]], ["चलो", "हम", "कहीं", "दूर", "चले"])
    check("spec ex.1 valid borrowing -> 0 violations, score 1.0", r.violations == 0 and r.score == 1.0 and r.borrows == 1,
          [e.text for e in r.events])
    # spec example 2: cannot borrow (2+2=4); discard चलो हम और, resume at कहीं
    r = analyse_line([[1, 2], [1, 1], [2, 1], [1, 2], [1, 2]], ["चलो", "हम", "और", "कहीं", "चले"])
    check("spec ex.2 invalid borrowing -> 1 violation, 3 words discarded, then कहीं+चले pair",
          r.violations == 1 and r.discarded_words == 3 and r.pairs == 1 and abs(r.score - 0.8) < 1e-9,
          [e.text for e in r.events])
    # end-of-line exception
    r = analyse_line([[2, 2], [1, 2]]); check("trailing unpaired odd is exempt", r.violations == 0 and r.trailing_odd == 1)
    # odd followed only by evens to the end is NOT at the very end
    r = analyse_line([[1, 2], [2]]); check("odd then even to line end -> violation", r.violations == 1)
    # irregular word (5): DP splits [2,2,1] -> [2][2,1]; the 3 pairs with next word
    r = analyse_line([[2, 2, 1], [1, 2]]); check("irregular 5-matra word split by DP", r.violations == 0 and r.irregular_words == 1,
                                                 [e.text for e in r.events])
    # three odds in a row: pair, then trailing odd
    r = analyse_line([[1, 2], [1, 2], [1, 2]]); check("odd,odd,odd -> one pair + trailing odd", r.pairs == 1 and r.trailing_odd == 1 and r.violations == 0)
    # real text through the prosody pipeline
    r = score_line("चलो हम कहीं दूर चले"); check("pipeline: ex.1 as Devanagari text", r and r.score == 1.0, [e.text for e in r.events] if r else "")
    r = score_line("चलो हम और कहीं चले"); check("pipeline: ex.2 as Devanagari text", r and r.violations == 1, [e.text for e in r.events] if r else "")
    # bounds fuzz: score in [0,1], violations <= words, deterministic
    import random
    rnd = random.Random(0); bad = 0
    for _ in range(20000):
        line = [[rnd.choice((1, 2)) for _ in range(rnd.randint(1, 6))] for _ in range(rnd.randint(1, 14))]
        for norm in ("words", "units"):
            a = analyse_line(line, normalise=norm); b = analyse_line(line, normalise=norm)
            if not (0.0 <= a.score <= 1.0) or a.violations > len(line) or a.score != b.score:
                bad += 1
    check("fuzz 20000 random lines x 2 normalisers: bounded, V<=words, deterministic", bad == 0, f"(bad={bad})")
    print("\nALL PASS" if ok else "\nSOME FAILED"); return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        res = score_poem(open(sys.argv[1], encoding="utf-8").read())
        for r in res.lines:
            print(f"{r.score:.2f}  V={r.violations} pairs={r.pairs} borrows={r.borrows}  {' '.join(r.words)}")
            for e in r.events:
                if e.kind == 'violation': print("      !", e.text)
        print(f"\nPoem Rhythm Score = {res.score:.3f}  ({len(res.lines)} lines)")
    else:
        print(__doc__)
