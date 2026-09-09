"""End-to-end dry run of the full pipeline on the REAL MI-CXR questions.

Everything except the comparator now exists: the option parsers turn real questions into
trajectory predicates, and the inference turns evidence into a posterior. This wires them
together and substitutes a *simulated* comparator of known accuracy for the one that needs
images, which answers the question the whole project rests on:

    at local comparator accuracy X, what would we score on MI-CXR?

Protocol per item:
  1. parse all five options into masks over the 1024 ordinal trajectories
  2. sample a ground-truth trajectory uniformly from the GOLD option's mask
  3. simulate per-image reads and pairwise comparisons from that trajectory
  4. run exact inference and score each option by the BAYES FACTOR of its predicate
  5. compare the argmax option against the gold letter

Step 4 is where this dry run earned its keep. Scoring options by raw posterior mass is wrong
for multiple choice: "there is no second appearance" covers most of the trajectory space and
therefore wins on size alone, pinning those question types at exactly P(gold = E). Dividing by
the predicate's PRIOR mass - the Bayes factor against the prior - is the correct decoding when
exactly one option is true. It is worth about 41 accuracy points.

Two sampling schemes bracket the truth. `prior` follows the faithful generative story: nature
produces trajectories per disease dynamics and the gold option describes what nature produced.
`uniform` samples the gold predicate's satisfying set evenly, which generates clinically
implausible timelines the CTMC is right to disbelieve, so it is adversarial to the prior. The
real number lies between them.

Usage:  python run_endtoend.py path/to/micxr_test.jsonl [n_per_item]
"""
import collections
import json
import sys

import numpy as np

from sim.core import K, T, TRAJ, log_posterior, transition_matrix
from sim.experiments import RHO_LOW
from sim.gts_parse import parse_gts_item
from sim.micxr_parse import BIN_TRAJ, TEL_QTYPES, parse_item
from sim.simulate import (
    Config, log_unary_from_obs, make_generator, simulate_comparator, simulate_unary,
)

# Map each ordinal trajectory to its binary presence image, so TEL predicates (defined over
# the 32 binary trajectories) can be scored against the ordinal posterior. This is the
# "marginalise at P(s > 0)" claim from docs/micxr-analysis.md, made concrete.
_BIN_KEY = {tuple(b): i for i, b in enumerate(BIN_TRAJ)}
ORD_TO_BIN = np.array([_BIN_KEY[tuple((t > 0).astype(int))] for t in TRAJ])

GTS_QTYPE = "single_entity_interval_summary"


def item_masks(item):
    """Masks over the 1024 ordinal trajectories, or None if the item is out of scope."""
    if item["qtype"] in TEL_QTYPES:
        p = parse_item(item)
        if len(p["masks"]) != len(item["choices"]):
            return None
        return {k: m[ORD_TO_BIN] for k, m in p["masks"].items()}   # lift binary -> ordinal
    if item["qtype"] == GTS_QTYPE:
        p = parse_gts_item(item)
        if len(p["masks"]) != len(item["choices"]):
            return None
        return p["masks"]
    return None


def prior_over_trajectories(Ps, log_pi):
    """Trajectory prior with no image evidence: CTMC transitions and the initial prior only."""
    flat_u = np.zeros((T, K))
    flat_c = np.zeros((T - 1, 6))
    return log_posterior(flat_u, flat_c, np.log(np.clip(np.stack(Ps), 1e-12, None)),
                         np.zeros(T - 1), log_pi)


def score_options(masks, post, prior, decoding):
    """Rank options.

    'mass'  - raw posterior mass inside the predicate. WRONG for multiple choice: a
              predicate covering many trajectories wins by size alone, which is why
              "there is no second appearance" was selected almost always.
    'ratio' - posterior mass divided by prior mass, i.e. the Bayes factor for that
              predicate against the prior. This is the correct decoding when exactly one
              option is true and the options are a priori equally likely.
    """
    out = {}
    for k, m in masks.items():
        num = float(post[m].sum())
        if decoding == "mass":
            out[k] = num
        else:
            den = float(prior[m].sum())
            out[k] = num / den if den > 1e-12 else 0.0
    return max(out, key=out.get)


def run(items, cfg, rng, n_per_item=1, decoding="ratio", sample="prior"):
    Q = make_generator(cfg)
    log_pi = np.log(np.array(cfg.pi))
    hits = collections.Counter()
    total = collections.Counter()

    for item in items:
        masks = item_masks(item)
        if masks is None:
            continue
        gold = item["answer"]
        allowed = np.flatnonzero(masks[gold])
        if allowed.size == 0:
            continue                       # gold predicate unsatisfiable -> parser miss

        for _ in range(n_per_item):
            dts = np.exp(rng.uniform(np.log(cfg.dt_min), np.log(cfg.dt_max), size=T - 1))
            Ps = [transition_matrix(Q, float(d)) for d in dts]
            prior = prior_over_trajectories(Ps, log_pi)

            if sample == "prior":
                # Faithful generative story: nature produces trajectories following disease
                # dynamics, and the gold option describes whatever nature produced. Sampling
                # uniformly inside the mask instead would generate clinically implausible
                # timelines the CTMC prior is right to disbelieve.
                w = prior[allowed]
                w = w / w.sum() if w.sum() > 0 else None
                idx = rng.choice(allowed, p=w)
            else:
                idx = allowed[rng.integers(allowed.size)]
            s = TRAJ[idx]

            p_cmp, reliable = simulate_comparator(s, cfg, rng)
            y = simulate_unary(s, cfg, rng)
            post = log_posterior(
                log_unary_from_obs(y, cfg, missingness_aware=True),
                np.log(np.clip(p_cmp, 1e-12, None)),
                np.log(np.clip(np.stack(Ps), 1e-12, None)),
                np.where(reliable, 1.0, RHO_LOW),
                log_pi,
            )
            pred = score_options(masks, post, prior, decoding)
            total[item["qtype"]] += 1
            hits[item["qtype"]] += int(pred == gold)

    return hits, total


def report(hits, total, label):
    n = sum(total.values())
    if not n:
        print(f"{label}: no items in scope")
        return 0.0
    acc = sum(hits.values()) / n
    print(f"\n{label}   overall {acc:.3f}   ({sum(hits.values())}/{n} trials)")
    for q in sorted(total):
        print(f"    {q:<32} {hits[q] / total[q]:.3f}   n={total[q]}")
    return acc


def main(path, n_per_item=1):
    rows = [json.loads(line) for line in open(path)]
    scope = [r for r in rows if r["qtype"] in TEL_QTYPES or r["qtype"] == GTS_QTYPE]
    in_scope = sum(item_masks(r) is not None for r in scope)

    print("=" * 78)
    print("END-TO-END DRY RUN  -  real MI-CXR questions, simulated comparator")
    print(f"{len(rows)} items total, {len(scope)} of parseable qtypes, "
          f"{in_scope} fully parsed ({in_scope / len(rows):.1%} of benchmark)")
    print("=" * 78)
    print("\npublished reference points on the FULL benchmark:")
    print("    random 20.0%   majority-class 22.8%   14-VLM mean 29.3%   best 41.8%")

    print("\n" + "-" * 78)
    print(f"{'cmp acc':>9} {'mass/prior':>11} {'ratio/uniform':>14} {'ratio/prior':>12}")
    print("-" * 78)
    rows_out = []
    for acc in (0.35, 0.50, 0.60, 0.70, 0.85):
        cfg = Config(cmp_accuracy=acc)
        got = {}
        for dec, smp in (("mass", "prior"), ("ratio", "uniform"), ("ratio", "prior")):
            h, t = run(scope, cfg, np.random.default_rng(int(acc * 1000)), n_per_item,
                       dec, smp)
            got[(dec, smp)] = (sum(h.values()) / sum(t.values()), h, t)
        print(f"{acc:>9.2f} {got[('mass','prior')][0]:>11.3f} "
              f"{got[('ratio','uniform')][0]:>14.3f} {got[('ratio','prior')][0]:>12.3f}")
        rows_out.append((acc, got))
    print("-" * 78)

    for acc, got in rows_out:
        if abs(acc - 0.60) < 1e-9:
            _, h, t = got[("ratio", "prior")]
            report(h, t, f"\nper-qtype at comparator accuracy 0.60 (ratio / prior)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "micxr_test.jsonl",
         int(sys.argv[2]) if len(sys.argv) > 2 else 1)
