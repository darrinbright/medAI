"""Free-text option parser for MI-CXR GTS and ICR (proposal 3.10).

Unlike TEL, these options are clinical narratives. But they are not arbitrary prose: the four
non-"none" options of an item are a **minimal pair set** - near-identical text with the relation
words swapped at a few slots. Diffing the options across 600 items recovers the whole relation
vocabulary, and it is small and regular.

Parsing therefore stays rule-based, with no LLM and no error budget to report:

  1. locate every interval marker ("Between T1 and T2", "from T2 to T3", "by T3",
     "from T2 through T4" - which spans two intervals)
  2. locate every relation phrase
  3. scope each relation to the clause/segment its marker owns, which handles both word
     orders ("Between T1 and T2, X worsens" and "a slight increase ... from T4 to T5")
  4. turn the resulting per-interval relations into a mask over the 1024 ordinal trajectories

The one real trap: "partially resolves" and "nearly resolves" mean IMPROVED, not RESOLVED. The
lexicon is ordered most-specific-first so those match before bare "resolves".
"""
from __future__ import annotations

import re

import numpy as np

from .core import ABSENT_BOTH, IMPROVED, NEW, REL, RESOLVED, STABLE, T, TRAJ, WORSENED

# ---------------------------------------------------------------- interval markers
# Each yields the 1-based interval index i, meaning T_i -> T_{i+1}.

_MARKERS = [
    # "from T2 through T4" covers intervals 2 and 3
    (re.compile(r"\bfrom T(\d)\s+through\s+T(\d)", re.I), "span"),
    (re.compile(r"\bbetween T(\d)\s+and\s+T(\d)", re.I), "pair"),
    (re.compile(r"\bfrom T(\d)\s+to\s+T(\d)", re.I), "pair"),
    (re.compile(r"\bT(\d)\s*(?:→|->|to)\s*T(\d)", re.I), "pair"),
    (re.compile(r"\bby T(\d)\b(?!\s*to)", re.I), "by"),
]

# ---------------------------------------------------------------- relation lexicon
# Ordered most-specific first; the first match at a position wins.

_LEX: list[tuple[re.Pattern, int]] = [
    # improvement that merely looks like resolution
    (re.compile(r"\b(?:partially|partly|nearly|almost|near-?complete(?:ly)?)\s+resolv\w*", re.I), IMPROVED),
    (re.compile(r"\bpartial\s+resolution", re.I), IMPROVED),
    # explicit stability, including negated-change phrasings
    (re.compile(r"\bwithout\s+(?:significant\s+|substantial\s+|notable\s+|recurrent\s+)?interval\s+change", re.I), STABLE),
    (re.compile(r"\bno\s+(?:significant\s+|substantial\s+|notable\s+)?(?:interval\s+)?change", re.I), STABLE),
    (re.compile(r"\bwithout\s+resolution", re.I), STABLE),
    (re.compile(r"\bremains?\s+(?:stable|unchanged|similar|constant)", re.I), STABLE),
    (re.compile(r"\bremain\s+(?:stable|unchanged|similar)", re.I), STABLE),
    (re.compile(r"\bpersist\w*", re.I), STABLE),
    (re.compile(r"\b(?:is|are)\s+unchanged", re.I), STABLE),
    (re.compile(r"\bunchanged|\bstable\b", re.I), STABLE),
    # absence maintained
    (re.compile(r"\bremains?\s+absent|\bremain\s+absent", re.I), ABSENT_BOTH),
    (re.compile(r"\bcontinues?\s+to\s+be\s+absent", re.I), ABSENT_BOTH),
    # true resolution
    (re.compile(r"\bcomplete(?:ly)?\s+resolv\w*|\bresolv\w*\s+complete(?:ly)?", re.I), RESOLVED),
    (re.compile(r"\bcomplete\s+resolution", re.I), RESOLVED),
    (re.compile(r"(?<!partially )(?<!partly )(?<!nearly )(?<!almost )\bresolv\w*", re.I), RESOLVED),
    (re.compile(r"\bclears?\s+completely|\bhas\s+cleared", re.I), RESOLVED),
    # emergence
    (re.compile(r"\bnewly\s+appear\w*|\bnew\s+\w+\s+appear\w*", re.I), NEW),
    (re.compile(r"\bre-?appear\w*|\bdevelop\w*|\bemerg\w*", re.I), NEW),
    (re.compile(r"\bappear\w*", re.I), NEW),
    # worsening
    (re.compile(r"\bworsen\w*|\bprogress\w*|\benlarg\w*", re.I), WORSENED),
    (re.compile(r"\bincreas\w*|\bgrows?\b|\bexpand\w*", re.I), WORSENED),
    # improvement
    (re.compile(r"\bimprov\w*|\bdecreas\w*|\bregress\w*|\bdiminish\w*|\breduc\w*|\blessen\w*", re.I), IMPROVED),
    (re.compile(r"\bsmaller\b|\bless\s+pronounced", re.I), IMPROVED),
]

NONE_OPTION = re.compile(r"^\s*(none of|there (?:is|are) no)\b", re.I)


def find_markers(text: str) -> list[tuple[int, list[int]]]:
    """[(char position, [interval indices])], deduplicated by position."""
    out, seen = [], set()
    for pat, kind in _MARKERS:
        for m in pat.finditer(text):
            if m.start() in seen:
                continue
            seen.add(m.start())
            if kind == "span":
                a, b = int(m.group(1)), int(m.group(2))
                idx = [i for i in range(a, b) if 1 <= i <= T - 1]
            elif kind == "pair":
                a, b = int(m.group(1)), int(m.group(2))
                idx = [a] if b == a + 1 and 1 <= a <= T - 1 else []
            else:  # "by Tj" means the interval ending at Tj
                j = int(m.group(1))
                idx = [j - 1] if 2 <= j <= T else []
            if idx:
                out.append((m.start(), idx))
    return sorted(out)


def find_relations(text: str) -> list[tuple[int, int]]:
    """[(char position, relation)], first-matching pattern wins at each position."""
    out, claimed = [], []
    for pat, rel in _LEX:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in claimed):
                continue
            claimed.append((m.start(), m.end()))
            out.append((m.start(), rel))
    return sorted(out)


_CLAUSE = re.compile(
    r";|(?<=[a-z]),\s*(?:then|followed by|and then|before|after which|which)\b"
    r"|,\s*(?=(?:from|between|by)\s+T\d)",
    re.I,
)


def split_clauses(text: str) -> list[tuple[int, str]]:
    """[(offset, clause)] - one clause per described interval, usually."""
    out, pos = [], 0
    for m in _CLAUSE.finditer(text):
        if m.start() > pos:
            out.append((pos, text[pos:m.start()]))
        pos = m.end()
    if pos < len(text):
        out.append((pos, text[pos:]))
    return [(o, c) for o, c in out if c.strip()]


def option_relations(text: str, *, first_wins: bool = True,
                     split_multi: bool = False) -> dict[int, int]:
    """Per-interval relation for one option: {interval index -> relation}.

    Clause-scoped rather than nearest-marker. Attaching a relation to whichever marker is
    closest in characters fails on the common pattern

        "Between T1 and T2, X progressively worsens; ... from T2 to T3 ..."

    where the verb sits nearer the *second* marker and its interval is stolen. Scoping to
    the clause that contains the marker fixes that. A clause with no marker of its own
    inherits the previous clause's interval, which handles trailing continuations.

    Within a clause the FIRST relation phrase wins. These options are minimal pairs and the
    discriminative verb leads - "partially improves but persists", "develops and progresses" -
    with the rest as elaboration. Taking the last instead costs 2.3 points of exclusivity.

    `split_multi` further subdivides a two-marker clause at the midpoint between markers. It
    is OFF because it *hurts*: 68.1% exclusive versus 86.6%. Midpoints cut phrases apart, and
    giving both intervals of a clause the same relation turns out to distinguish the minimal
    pairs more reliably than splitting them badly. Kept as a switch so the sweep is
    reproducible (see docs/micxr-analysis.md).
    """
    out: dict[int, int] = {}
    current: list[int] = []
    for _, clause in split_clauses(text):
        markers = find_markers(clause)
        if not markers:
            if not current:
                continue
            rels = find_relations(clause)
            if rels:
                pick = rels[0][1] if first_wins else rels[-1][1]
                for i in current:
                    out.setdefault(i, pick)
            continue

        if split_multi:
            bounds = []
            for k, (pos, idxs) in enumerate(markers):
                lo = 0 if k == 0 else (markers[k - 1][0] + pos) // 2
                hi = len(clause) if k == len(markers) - 1 else (pos + markers[k + 1][0]) // 2
                bounds.append((lo, hi, idxs))
        else:
            allidx = sorted({i for _, idxs in markers for i in idxs})
            bounds = [(0, len(clause), allidx)]
        for lo, hi, idxs in bounds:
            rels = find_relations(clause[lo:hi])
            current = sorted(idxs)
            if rels:
                pick = rels[0][1] if first_wins else rels[-1][1]
                for i in idxs:
                    out.setdefault(i, pick)
    return out


def relations_to_mask(rel_by_interval: dict[int, int]) -> np.ndarray:
    """Trajectories consistent with every specified per-interval relation."""
    keep = np.ones(len(TRAJ), dtype=bool)
    for i, r in rel_by_interval.items():
        s, sp = TRAJ[:, i - 1], TRAJ[:, i]
        keep &= REL[s, sp] == r
    return keep


def parse_gts_item(item: dict, **kw) -> dict:
    """Parse a GTS item whose options all describe the SAME finding.

    The "none of them are true" option is the complement of the union of the others,
    which makes the five options exhaustive by construction.
    """
    masks, rels, unparsed = {}, {}, []
    none_key = None
    for key, text in item["choices"].items():
        if NONE_OPTION.match(text):
            none_key = key
            continue
        rb = option_relations(text, **kw)
        if not rb:
            unparsed.append((key, text))
            continue
        rels[key] = rb
        masks[key] = relations_to_mask(rb)
    if none_key is not None and masks:
        union = np.zeros(len(TRAJ), dtype=bool)
        for m in masks.values():
            union |= m
        masks[none_key] = ~union
    return {"masks": masks, "relations": rels, "unparsed": unparsed,
            "none_key": none_key}


# ---------------------------------------------------------------- finding extraction
#
# multi_entity_interval_summary and icr_multi_abn put a DIFFERENT finding in each option, so
# answering them needs one trajectory posterior per finding rather than one per item. The
# exclusivity test does not apply there - the options are not competing descriptions of the
# same chain - so coverage of this extractor is the validation available without images.

FINDINGS = [
    "pleural effusion", "pneumothorax", "sub-diaphragmatic air", "subdiaphragmatic air",
    "pulmonary edema", "cardiomegaly", "vascular congestion", "vascular plethora",
    "consolidation", "atelectasis", "lung opacity", "airspace opacity", "opacification",
    "pneumonia", "rib fracture", "lung nodule", "nodule", "mediastinal displacement",
    "lung hyperexpansion", "hyperinflation", "emphysema", "interstitial edema",
    "pulmonary vascular congestion", "hilar contour", "cardiomediastinal silhouette",
    "pneumoperitoneum", "effusion", "edema", "infiltrate", "fibrosis", "granuloma",
    "aortic knob", "mediastinal widening", "tracheal deviation", "lung volume",
    # second pass, added after inspecting extraction failures
    "pneumomediastinum", "hemidiaphragm", "cardiac silhouette", "cardiac shadow",
    "heart size", "lung base", "costophrenic angle", "interstitial marking",
    "linear density", "density", "bony structural change", "cortical irregularity",
    "vascular pedicle", "pulmonary vasculature", "subcutaneous emphysema",
    "aeration", "lucency", "reticular opacity", "ground-glass opacity", "scarring",
]
_FIND_RE = [(re.compile(r"\b" + re.escape(f).replace(r"\ ", r"\s+") + r"s?\b", re.I), f)
            for f in sorted(FINDINGS, key=len, reverse=True)]


def extract_finding(text: str) -> str | None:
    """The finding an option is about: the longest lexicon term matching earliest."""
    best = None
    for pat, name in _FIND_RE:
        m = pat.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def parse_multi_entity_item(item: dict) -> dict:
    """Per-option (finding, per-interval relations) for cross-finding questions."""
    out, unparsed = {}, []
    none_key = None
    for key, text in item["choices"].items():
        if NONE_OPTION.match(text):
            none_key = key
            continue
        finding = extract_finding(text)
        rels = option_relations(text)
        if finding is None or not rels:
            unparsed.append((key, text, finding, bool(rels)))
        else:
            out[key] = {"finding": finding, "relations": rels}
    return {"options": out, "unparsed": unparsed, "none_key": none_key}
