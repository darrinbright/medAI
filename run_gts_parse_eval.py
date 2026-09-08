"""Validate the GTS/ICR free-text parser without images.

For single_entity_interval_summary all five options describe the SAME finding, so the same
exclusivity test used for TEL applies: a well-formed item has exactly one true option, so the
option predicates must not overlap over the 1024 ordinal trajectories.

Usage:  python run_gts_parse_eval.py path/to/micxr_test.jsonl
"""
import collections
import itertools
import json
import sys

import numpy as np

from sim.core import RELATION_NAMES, TRAJ
from sim.gts_parse import parse_gts_item

QTYPE = "single_entity_interval_summary"


def main(path):
    rows = [json.loads(line) for line in open(path)]
    items = [r for r in rows if r["qtype"] == QTYPE]

    print("=" * 78)
    print("GTS FREE-TEXT PARSER  -  validation without images")
    print(f"{QTYPE}: {len(items)} items (all options describe the same finding)")
    print("=" * 78)

    n_opts = n_parsed = 0
    n_full = excl = empty_none = 0
    ivl_cov = collections.Counter()
    rel_used = collections.Counter()
    fails = []

    for r in items:
        p = parse_gts_item(r)
        content = [k for k in r["choices"] if k != p["none_key"]]
        n_opts += len(content)
        n_parsed += len(p["relations"])
        for k, rb in p["relations"].items():
            ivl_cov[len(rb)] += 1
            for rel in rb.values():
                rel_used[RELATION_NAMES[rel]] += 1
        if len(p["relations"]) == len(content):
            n_full += 1
            keys = list(p["masks"])
            ok = not any((p["masks"][a] & p["masks"][b]).any()
                         for a, b in itertools.combinations(keys, 2))
            excl += ok
            if not ok and len(fails) < 3:
                fails.append(r)
            if p["none_key"] and not p["masks"][p["none_key"]].any():
                empty_none += 1

    print(f"\noptions parsed              {n_parsed}/{n_opts}  ({n_parsed / n_opts:.1%})")
    print(f"items with all 4 parsed     {n_full}/{len(items)}  ({n_full / len(items):.1%})")
    print(f"  of those, exclusive       {excl}/{n_full}  ({excl / max(n_full, 1):.1%})")
    print(f"  'none' predicate empty    {empty_none}/{n_full}  "
          f"({empty_none / max(n_full, 1):.1%})   <- options already cover all 1024")

    print("\nintervals recovered per option")
    for k in sorted(ivl_cov):
        print(f"  {k} interval(s): {ivl_cov[k]:>5}")

    print("\nrelations assigned")
    for k, v in rel_used.most_common():
        print(f"  {v:>6}  {k}")

    if fails:
        print("\nexample non-exclusive item:")
        r = fails[0]
        p = parse_gts_item(r)
        for k, v in r["choices"].items():
            rb = p["relations"].get(k, {})
            pretty = ", ".join(f"T{i}->T{i+1}:{RELATION_NAMES[x]}" for i, x in sorted(rb.items()))
            print(f"  {k}: {v[:110]}...")
            print(f"      parsed -> {pretty or '(none)'}")


def cross_finding(rows):
    """multi_entity and ICR: options describe DIFFERENT findings, so exclusivity does not
    apply. Coverage of (finding, relations) extraction is the check available here."""
    from sim.gts_parse import parse_multi_entity_item
    print("\n" + "=" * 78)
    print("CROSS-FINDING QUESTION TYPES  -  each option is about a different finding")
    print("=" * 78)
    print(f"\n{'qtype':<32} {'items':>6} {'opts ok':>9} {'items all-4':>12}")
    print("-" * 78)
    for qt in ("multi_entity_interval_summary", "icr_multi_abn"):
        items = [r for r in rows if r["qtype"] == qt]
        n = ok = full = 0
        for r in items:
            p = parse_multi_entity_item(r)
            content = [k for k in r["choices"] if k != p["none_key"]]
            n += len(content)
            ok += len(p["options"])
            full += len(p["options"]) == len(content)
        print(f"{qt:<32} {len(items):>6} {ok / n:>8.1%} {full / len(items):>11.1%}")
    print("-" * 78)
    print("\nThe residue is compositional clinical language, not missing lexicon entries:")
    print('  "linear density along the lateral aspect of the right lung"')
    print('  "progressive bony structural changes with new cortical irregularities"')
    print("A closed finding lexicon cannot cover these; an LLM parser is required, and")
    print("proposal 3.10's error budget applies to exactly this share of the benchmark.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "micxr_test.jsonl"
    main(path)
    cross_finding([json.loads(line) for line in open(path)])
