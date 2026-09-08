# Pilot findings — synthetic study

Run September 2026, before any data access. Code in [`../sim/`](../sim/), reproduce with
`python run_sweep.py` and `python run_gate_study.py`.

**Scope caveat, stated up front.** These are synthetic timelines generated *from* the model
being tested, with our own query definitions and label priors. The absolute numbers are **not
comparable to MI-CXR's 29.3%** and must never be quoted as if they were. What transfers is the
*ordering* of methods, the *shape* of the sensitivity curves, and the identification of which
components carry weight.

---

## Correctness

7/7 checks pass (`python -m sim.tests`), including:

- the six relations tile `S × S` exactly — 16 cells, counts `[1,3,3,3,3,3]`
- `exp(Q·Δτ)` is row-stochastic for Δτ from 0.5 to 3650 days
- the CTMC **semigroup property** `exp(Q(a+b)) = exp(Qa)exp(Qb)` holds — this is what makes
  irregular intervals compose correctly
- **enumeration over 1024 trajectories agrees with forward–backward to 1.3e-15**

That last one matters: two independent implementations of the same exact inference agree to
machine precision, so nothing downstream is resting on an inference bug.

---

## Q1 — Does global inference beat greedy composition?

**Yes, decisively, and at every comparator accuracy.** This was the experiment that could have
killed the thesis.

| Comparator accuracy | Unary only | Greedy chain | Full model | Gain over greedy | Greedy consistency |
|---|---|---|---|---|---|
| 0.35 | 0.481 | 0.486 | 0.690 | **+0.204** | 0.062 |
| 0.45 | 0.481 | 0.512 | 0.744 | **+0.232** | 0.077 |
| 0.55 | 0.481 | 0.540 | 0.778 | **+0.237** | 0.101 |
| 0.65 | 0.481 | 0.569 | 0.815 | **+0.246** | 0.142 |
| 0.75 | 0.481 | 0.599 | 0.844 | **+0.244** | 0.201 |
| 0.85 | 0.481 | 0.633 | 0.868 | **+0.236** | 0.268 |

Two things stand out.

**The margin is large and flat.** Decomposition alone (greedy) buys relatively little; the
inference is doing the work. The contingency plan in proposal §10 — "what if decomposition, not
inference, carries the gain?" — is not needed under these assumptions.

**Greedy composition is almost never self-consistent: 6–27%.** A chain of individually plausible
argmax relations is jointly *unrealisable* 73–94% of the time — e.g. "new" asserted from a state
that is already present, or "worsened" from severe. This is the Trajectory Consistency Rate
metric, and the headroom is far larger than expected. Consistency improves as the comparator
improves, which is the right sanity check.

---

## Q2 — How accurate must the comparator be?

**Much less accurate than feared.** Even at 0.35 six-way accuracy — barely above the 1/6 ≈ 0.167
chance rate — the full model reaches 0.690 against 0.481 for unary-only.

This substantially de-risks the project. The open worry was that six-way relation accuracy would
fall well below CoCa-CXR's three-way 65% and sink the method. Under these assumptions it does not:
the unary evidence and the CTMC prior carry the trajectory even when the comparator is weak.

---

## Ablation at comparator accuracy 0.60

| Configuration | TEL | ICR | GTS | Mean | Consistency |
|---|---|---|---|---|---|
| majority-class baseline | 0.461 | 0.388 | 0.367 | 0.406 | — |
| unary only (no comparator) | 0.564 | 0.487 | 0.391 | 0.481 | 1.000 |
| pairwise + greedy chain | 0.674 | 0.533 | 0.458 | 0.555 | **0.121** |
| pairwise + inference, no unary | 0.760 | 0.731 | 0.630 | 0.707 | 1.000 |
| full, uniform transition prior | 0.658 | 0.615 | 0.414 | 0.563 | 1.000 |
| full, no reliability gate | 0.842 | 0.816 | 0.729 | 0.796 | 1.000 |
| full, "not mentioned" = absent | 0.815 | 0.798 | 0.707 | 0.773 | 1.000 |
| **full model** | **0.841** | **0.822** | **0.733** | **0.799** | 1.000 |

**The CTMC is the biggest single component: +0.236** (0.799 vs 0.563 with a uniform transition
prior). Conditioning on the true inter-visit interval is not a refinement — it is load-bearing.
Given that no existing multi-visit work uses Δτ at all, this is the most promising contribution.

**Missingness is worth +0.026.** Real, consistent, but modest. Report it as a correctness issue
that also helps slightly, not as a headline.

---

## The reliability gate does not survive its first honest test

| Scenario | No gate | Gate | Gain |
|---|---|---|---|
| mismatch → less accurate, still calibrated | 0.799 | 0.799 | **+0.000** |
| mismatch → confidently wrong | 0.739 | 0.801 | +0.062 |
| confidently wrong, 40% of intervals | 0.691 | 0.782 | +0.091 |
| confidently wrong, 55% of intervals | 0.642 | 0.762 | +0.120 |
| confidently wrong, 40%, accuracy 0.22 | 0.601 | 0.746 | +0.145 |

**When acquisition mismatch merely makes the comparator less accurate, the gate is worth
nothing** — inference already discounts flat evidence, so gating is redundant. It only pays when
mismatch makes the comparator **confidently wrong**.

This converts a vague contribution into a precise, falsifiable question about real data:

> Does an AP→PA change (or rotation, or shallow inspiration) make the comparator *confidently
> wrong*, or merely *uncertain*?

Measure that in November on same-day pairs before claiming the gate as a contribution. If the
answer is "merely uncertain", drop it and reallocate the space to the CTMC.

---

## Calibration sensitivity

Comparator accuracy fixed at 0.60, reported probabilities distorted by temperature:

| Temperature | Meaning | Greedy | Full | Gain |
|---|---|---|---|---|
| 0.4 | very overconfident | 0.555 | 0.795 | +0.241 |
| 0.7 | overconfident | 0.555 | 0.806 | +0.252 |
| 1.0 | calibrated | 0.555 | 0.799 | +0.244 |
| 1.6 | underconfident | 0.555 | 0.779 | +0.224 |
| 2.5 | very underconfident | 0.555 | 0.750 | +0.195 |

Degradation is graceful — roughly 5 points across a wide miscalibration range — and
**underconfidence hurts more than overconfidence**. Greedy is invariant by construction (it uses
only argmax), so its flat column doubles as a Monte-Carlo error bar: variation there is noise.

---

## What this changes in the plan

| Finding | Action |
|---|---|
| Global ≫ greedy at all accuracies | Thesis holds. Keep §10's contingency but deprioritise it |
| Greedy consistency 6–27% | Promote Trajectory Consistency Rate — the headroom is much larger than assumed |
| CTMC is the largest component | Make Δτ-conditioning the lead technical contribution, ahead of the gate |
| Comparator accuracy requirement is low | Six-way accuracy is a smaller risk than believed; October pilot still needed for the real figure |
| Gate worth 0.000 in the base case | **Do not claim it until the confidently-wrong condition is verified on real data** |
| Calibration costs ~5 points | Worth a calibration pass, not worth a whole thread; downgrade proposal §6 |

---

## Honest limits

- Data is generated from the same model family used for inference, so the CTMC is favoured by
  construction. A misspecification study (generate from a non-Markov or semi-Markov process,
  infer with the CTMC) is the obvious next experiment.
- The gate is an **oracle** here — it knows which intervals are unreliable. A learned gate will be
  worse, so the gains above are upper bounds.
- Conditional independence (proposal §3.9) holds by construction in the simulator but will not
  hold on real data, where the same image drives unary and both adjacent pairwise terms.
- Query definitions (onset, six shape classes) are ours, not MI-CXR's.
