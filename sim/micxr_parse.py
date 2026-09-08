"""Option -> trajectory-predicate parser for MI-CXR (proposal 3.10).

Proposal 3.10 flagged this component as a dependency and an attack surface: if an LLM maps
multiple-choice options onto trajectory predicates, its errors become answer errors, and a
reviewer can fairly ask whether the LLM is solving the task.

Inspecting the public test set removes that concern for most of the benchmark. All 2,994 TEL
options are short structured strings - "T2 -> T3", "T1, T5", or "There is no ..." - averaging
14 characters. They parse deterministically with regex. No LLM is involved, so there is no
attack surface and no error budget to report for 56% of the benchmark.

TEL also only ever asks about *presence*: new appearance and disappearance. Severity never
appears, so TEL needs a binary latent chain (2^5 = 32 trajectories), obtained by marginalising
the ordinal model at P(s > 0). Only GTS and ICR need the four-state scale.

GTS and ICR options are free-text clinical narratives (161 and 63 characters on average) and are
not handled here; see docs/micxr-analysis.md.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

T = 5
BIN_TRAJ = np.array(list(itertools.product([0, 1], repeat=T)), dtype=int)  # (32, 5)

_ARROW = re.compile(r"^T(\d)\s*(?:→|->)\s*T(\d)$")
_PAIR = re.compile(r"^T(\d)\s*,\s*T(\d)$")
_NONE = re.compile(r"^(there is no\b|none of\b)", re.I)

EMERGENCE_QTYPES = {"single_emergence", "multiple_emergence_1", "multiple_emergence_2"}
RESOLUTION_QTYPES = {"single_resolution", "multiple_resolution_1", "multiple_resolution_2"}
PAIR_QTYPES = {"emergence_to_resolution", "resolution_to_emergence"}
TEL_QTYPES = EMERGENCE_QTYPES | RESOLUTION_QTYPES | PAIR_QTYPES

_ORDINAL = {"multiple_emergence_2": 2, "multiple_resolution_2": 2}


def _events(traj: np.ndarray, kind: str) -> list[int]:
    """Interval indices (1-based, i means T_i -> T_{i+1}) where the event occurs."""
    out = []
    for t in range(T - 1):
        a, b = traj[t], traj[t + 1]
        if kind == "emergence" and a == 0 and b == 1:
            out.append(t + 1)
        elif kind == "resolution" and a == 1 and b == 0:
            out.append(t + 1)
    return out


def _mask_interval(kind: str, ordinal: int, i: int) -> np.ndarray:
    """Trajectories whose `ordinal`-th event of `kind` falls on interval i."""
    keep = np.zeros(len(BIN_TRAJ), dtype=bool)
    for k, tr in enumerate(BIN_TRAJ):
        ev = _events(tr, kind)
        keep[k] = len(ev) >= ordinal and ev[ordinal - 1] == i
    return keep


def _mask_none(kind: str, ordinal: int) -> np.ndarray:
    keep = np.zeros(len(BIN_TRAJ), dtype=bool)
    for k, tr in enumerate(BIN_TRAJ):
        keep[k] = len(_events(tr, kind)) < ordinal
    return keep


def _first_pair(traj: np.ndarray, qtype: str):
    """The (a, b) studies for the FIRST such sequence, or None.

    The questions ask which pair captures where a finding "*first* appears and
    subsequently resolves". An *exists* reading is wrong: a recurring finding satisfies
    several pairs at once, which is what the exclusivity check exposed. Taking the first
    event and the first opposite event after it makes exactly one pair true, as a
    well-formed multiple-choice item requires.
    """
    first, second = ("emergence", "resolution") if qtype == "emergence_to_resolution" \
        else ("resolution", "emergence")
    e1 = _events(traj, first)
    if not e1:
        return None
    a = e1[0] + 1                      # interval i ends at study i+1
    later = [i + 1 for i in _events(traj, second) if i + 1 > a]
    if not later:
        return None
    return (a, later[0])


def _mask_pair(qtype: str, a: int, b: int) -> np.ndarray:
    keep = np.zeros(len(BIN_TRAJ), dtype=bool)
    for k, tr in enumerate(BIN_TRAJ):
        keep[k] = _first_pair(tr, qtype) == (a, b)
    return keep


def parse_option(qtype: str, text: str) -> np.ndarray | None:
    """Map one option to a boolean mask over the 32 binary trajectories, or None."""
    text = text.strip()
    ordinal = _ORDINAL.get(qtype, 1)

    if qtype in PAIR_QTYPES:
        if _NONE.match(text):
            keep = np.zeros(len(BIN_TRAJ), dtype=bool)
            for k, tr in enumerate(BIN_TRAJ):
                keep[k] = _first_pair(tr, qtype) is None
            return keep
        m = _PAIR.match(text) or _ARROW.match(text)
        if m:
            return _mask_pair(qtype, int(m.group(1)), int(m.group(2)))
        return None

    kind = "emergence" if qtype in EMERGENCE_QTYPES else "resolution"
    if _NONE.match(text):
        return _mask_none(kind, ordinal)
    m = _ARROW.match(text) or _PAIR.match(text)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        if j != i + 1:
            return None
        return _mask_interval(kind, ordinal, i)
    return None


def parse_item(item: dict) -> dict | None:
    """Parse all five options of a TEL item. Returns None if the qtype is out of scope."""
    qtype = item["qtype"]
    if qtype not in TEL_QTYPES:
        return None
    masks, unparsed = {}, []
    for key, text in item["choices"].items():
        m = parse_option(qtype, text)
        if m is None:
            unparsed.append((key, text))
        else:
            masks[key] = m
    return {"qtype": qtype, "masks": masks, "unparsed": unparsed}


def answer_from_posterior(masks: dict, post32: np.ndarray) -> str:
    """Score each option by the posterior mass its predicate covers, take the argmax."""
    return max(masks, key=lambda k: float(post32[masks[k]].sum()))


def check_exclusive(masks: dict) -> tuple[bool, bool]:
    """(mutually exclusive, jointly exhaustive) over the 32 binary trajectories.

    Validates the parser without needing any images: a well-formed multiple-choice item
    should have exactly one true option, so the predicates must not overlap.
    """
    keys = list(masks)
    overlap = any((masks[a] & masks[b]).any() for a, b in itertools.combinations(keys, 2))
    union = np.zeros(len(BIN_TRAJ), dtype=bool)
    for k in keys:
        union |= masks[k]
    return (not overlap), bool(union.all())
