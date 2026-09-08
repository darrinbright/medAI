"""Validate the MI-CXR option parser without needing any images.

Two checks are available with the public test set alone:
  coverage    - does every option map to a predicate?
  exclusivity - are the five predicates mutually exclusive over the 32 binary trajectories?
                A well-formed MCQ has exactly one true option, so overlap means the parser
                (or our reading of the question) is wrong.

Usage:  python run_parse_eval.py path/to/micxr_test.jsonl
"""
import collections
import json
import sys

from sim.micxr_parse import TEL_QTYPES, check_exclusive, parse_item


def main(path):
    rows = [json.loads(line) for line in open(path)]
    tel = [r for r in rows if r["qtype"] in TEL_QTYPES]

    print("=" * 78)
    print("MI-CXR OPTION PARSER  -  validation without images")
    print(f"{len(rows)} items total, {len(tel)} in TEL scope "
          f"({len(tel) / len(rows):.1%})")
    print("=" * 78)

    stat = collections.defaultdict(
        lambda: {"n": 0, "opts": 0, "unparsed": 0, "excl": 0, "exh": 0})
    examples = []

    for r in tel:
        p = parse_item(r)
        st = stat[r["qtype"]]
        st["n"] += 1
        st["opts"] += len(r["choices"])
        st["unparsed"] += len(p["unparsed"])
        if p["unparsed"] and len(examples) < 6:
            examples.append((r["qtype"], p["unparsed"][0]))
        if len(p["masks"]) == len(r["choices"]):
            excl, exh = check_exclusive(p["masks"])
            st["excl"] += excl
            st["exh"] += exh

    print(f"\n{'qtype':<26} {'items':>6} {'options':>8} {'parsed':>8} "
          f"{'exclusive':>10} {'exhaustive':>11}")
    print("-" * 78)
    tot = {"n": 0, "opts": 0, "unparsed": 0, "excl": 0, "exh": 0}
    for q in sorted(stat):
        s = stat[q]
        for k in tot:
            tot[k] += s[k]
        print(f"{q:<26} {s['n']:>6} {s['opts']:>8} "
              f"{(s['opts'] - s['unparsed']) / s['opts']:>7.1%} "
              f"{s['excl'] / s['n']:>10.1%} {s['exh'] / s['n']:>11.1%}")
    print("-" * 78)
    print(f"{'TOTAL':<26} {tot['n']:>6} {tot['opts']:>8} "
          f"{(tot['opts'] - tot['unparsed']) / tot['opts']:>7.1%} "
          f"{tot['excl'] / tot['n']:>10.1%} {tot['exh'] / tot['n']:>11.1%}")

    if examples:
        print("\nunparsed examples:")
        for q, (k, t) in examples:
            print(f"  [{q}] {k}: {t!r}")

    out = collections.Counter(r["qtype"] for r in rows if r["qtype"] not in TEL_QTYPES)
    print("\nout of parser scope (free-text options, need a different treatment):")
    for q, n in out.most_common():
        print(f"  {n:>5}  {q}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "micxr_test.jsonl")
