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

## Misspecification study — the CTMC gain is real

The pilot above generated data from the same CTMC family used for inference, so the +0.236 could
have been the simulator agreeing with itself. This test removes that possibility: ground truth
comes from processes that each break a **different** CTMC assumption, `Q` is then fitted by MLE to
that non-CTMC data (which is exactly what estimating `Q_f` from Chest ImaGenome would be), and
inference runs with the resulting — wrong — prior.

Reproduce with `python run_misspec.py`. 6,000 timelines to fit, 4,000 held out to evaluate.

| Generative process | Assumption violated | Greedy | Uniform prior | Fitted Q | **CTMC gain** |
|---|---|---|---|---|---|
| ctmc | none (control) | 0.555 | 0.564 | 0.793 | **+0.230** |
| semi_markov | memorylessness | 0.549 | 0.581 | 0.781 | **+0.201** |
| momentum | first-order Markov | 0.541 | 0.585 | 0.803 | **+0.218** |
| nonstationary | time-homogeneity | 0.547 | 0.593 | 0.788 | **+0.196** |
| mixture | patient homogeneity | 0.545 | 0.562 | 0.789 | **+0.228** |

**The gain survives every violation.** Worst case is time-homogeneity (+0.196 vs +0.230 control) —
it loses about 15% of the benefit, not the benefit itself. The CTMC works as a useful regulariser
even when it is demonstrably the wrong family.

Global-vs-greedy is equally robust, and slightly *larger* under misspecification:

| Process | Full vs greedy | Greedy consistency |
|---|---|---|
| ctmc | +0.239 | 0.136 |
| semi_markov | +0.232 | 0.116 |
| momentum | **+0.262** | 0.136 |
| nonstationary | +0.242 | 0.136 |
| mixture | +0.244 | 0.123 |

### Q is identifiable — a direct answer to proposal §3.7

Fitted birth–death rates (1/day), from discretely observed transitions at irregular intervals:

| | q01 | q12 | q23 | q10 | q21 | q32 |
|---|---|---|---|---|---|---|
| **true (control)** | 0.0060 | 0.0050 | 0.0030 | 0.0040 | 0.0060 | 0.0070 |
| recovered (ctmc) | 0.0058 | 0.0051 | 0.0029 | 0.0036 | 0.0061 | 0.0065 |
| semi_markov | 0.0101 | 0.0072 | 0.0046 | 0.0057 | 0.0090 | 0.0121 |
| momentum | 0.0062 | 0.0093 | 0.0063 | 0.0059 | 0.0082 | 0.0059 |
| nonstationary | 0.0100 | 0.0084 | 0.0052 | 0.0038 | 0.0053 | 0.0053 |
| mixture | 0.0065 | 0.0061 | 0.0040 | 0.0048 | 0.0077 | 0.0094 |

On the control, all six rates are recovered to within ~10% — so **the six-parameter generator is
estimable from panel data with irregular intervals**, which was flagged as an open risk.

Under misspecification the rates are biased in interpretable ways: `semi_markov` inflates every
rate (increasing hazard produces more transitions than an exponential dwell would, so the fitted
exponential rate compensates upward). **This matters for the paper's claims**: `Q_f` should be
presented as a *fitted prior that helps*, not as an estimate of true biological transition rates.
The interpretability claim in proposal §3.7 needs that qualification.

### Scope

Only the **dynamics** are misspecified here. The comparator model, emission model and the
conditional-independence assumption remain correct by construction, so this does not address
proposal §3.9. It answers one question — is the CTMC gain an artifact? — and the answer is no.

---

## Conditional independence study — confirmed, measured, and cheap to fix

Proposal §3.9 predicted that treating `y_t`, `z_{t-1,t}` and `z_{t,t+1}` as conditionally
independent given `s_t` would make the posterior **overconfident**, since all three read the same
image. That prediction is now tested at its real source.

Each image carries a latent nuisance factor — rotation, inspiration depth, AP magnification, an
overlying line — that shifts its *apparent* severity. The per-image read sees the apparent
severity, and so does the comparator on **both** intervals touching that image. One bad film
therefore produces three correlated errors: an inflated read at V3, a spurious "worsened" on
V2→V3, and a spurious "improved" on V3→V4. Ground truth stays the true state sequence.

Reproduce with `python run_depend.py`.

### The cost, unmitigated

| Shared nuisance | Greedy | Full | Gain | TEL ECE | Overconfidence |
|---|---|---|---|---|---|
| 0.00 | 0.556 | 0.803 | +0.247 | 0.028 | −0.026 |
| 0.10 | 0.510 | 0.744 | +0.234 | 0.020 | +0.019 |
| 0.20 | 0.479 | 0.699 | +0.220 | 0.043 | +0.042 |
| 0.35 | 0.440 | 0.654 | +0.214 | 0.075 | **+0.075** |
| 0.50 | 0.398 | 0.594 | +0.196 | 0.126 | **+0.126** |

**The prediction holds exactly.** Overconfidence is slightly *negative* when the assumption is
satisfied (−0.026) and grows monotonically with correlation, reaching +0.126 — the posterior
claims 12.6 percentage points more confidence than it earns. ECE rises 4.5× over the same range.

Accuracy also falls, from 0.803 to 0.594. That is the more expensive consequence, and it is worth
being blunt about: correlated image nuisance is costly whether or not you model it.

**The thesis survives.** Full still beats greedy by +0.196 even under severe violation, because
greedy is hurt too (0.556 → 0.398). The ordering never flips.

### The fix: one shared tempering exponent

Weight `w` applied to all image-derived evidence (unary and pairwise), with the CTMC prior left at
full weight. At shared nuisance 0.35:

| Weight w | Accuracy | TEL ECE | Overconfidence |
|---|---|---|---|
| 1.00 | 0.648 | 0.087 | +0.085 |
| 0.85 | 0.656 | 0.050 | +0.050 |
| 0.70 | 0.652 | 0.029 | +0.023 |
| **0.55** | **0.658** | **0.016** | **−0.015** |
| 0.40 | 0.654 | 0.079 | −0.079 |

**A single scalar cuts ECE by roughly 5× at no accuracy cost.** Accuracy is flat across the whole
range (0.648–0.658, within Monte-Carlo noise), which is exactly what tempering should do: it
rescales all evidence uniformly, so the argmax barely moves and only the sharpness changes. Push
too far (w = 0.40) and it overshoots into underconfidence.

So §3.9 moves from "a known weakness we will name openly" to **"a known weakness, measured, with a
one-parameter mitigation and a tuned operating point."** That is a much stronger position to
defend, and the calibration curve is a figure in its own right.

### Honest limits

- The nuisance model is ours. Real correlation structure between a radiograph's reads may be
  stronger, weaker, or differently shaped.
- `w` must be tuned on held-out real data, not transplanted from this study.
- Tempering treats the symptom. A learned joint emission over `(y_t, z_{t-1,t}, z_{t,t+1})` would
  address the cause, at the cost of a much larger parameter count and the loss of the clean
  factorisation — worth a sentence in the paper, not a reimplementation.
- Comparison across the two tables shows the Monte-Carlo error bar: the same cell (nuisance 0.35,
  w = 1.0) reads 0.654/0.075 in one and 0.648/0.087 in the other, so treat differences below about
  0.01 accuracy and 0.012 ECE as noise.

---

## What this changes in the plan

| Finding | Action |
|---|---|
| Global ≫ greedy at all accuracies | Thesis holds. Keep §10's contingency but deprioritise it |
| Greedy consistency 6–27% | Promote Trajectory Consistency Rate — the headroom is much larger than assumed |
| CTMC is the largest component, and survives misspecification | Make Δτ-conditioning the lead technical contribution, ahead of the gate |
| Q recovered to ~10% on the control | Answers §3.7's identifiability risk. But present `Q_f` as a fitted prior, **not** as true biological rates — it is biased under misspecification |
| Comparator accuracy requirement is low | Six-way accuracy is a smaller risk than believed; October pilot still needed for the real figure |
| Gate worth 0.000 in the base case | **Do not claim it until the confidently-wrong condition is verified on real data** |
| Conditional independence violation confirmed | Report posterior calibration as standard. Apply likelihood tempering with `w` tuned on held-out data; expect roughly `w ≈ 0.55` |
| Tempering costs no accuracy | §3.9 becomes a measured-and-mitigated limitation rather than an admitted flaw |
| Calibration costs ~5 points | Worth a calibration pass, not worth a whole thread; downgrade proposal §6 |

---

## Honest limits

- ~~Data is generated from the same model family used for inference~~ — addressed above; the
  gain survives all four violated assumptions.
- The gate is an **oracle** here — it knows which intervals are unreliable. A learned gate will be
  worse, so the gains above are upper bounds.
- ~~Conditional independence holds by construction in the simulator~~ — addressed above; the
  violation is simulated at its source, and the predicted overconfidence is confirmed.
- Query definitions (onset, six shape classes) are ours, not MI-CXR's.
