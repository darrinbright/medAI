# Decision: Idea 1 — local-to-global trajectory inference

**Date:** September 2026 · **Target:** MICCAI 2027

> **Name change.** Formerly written up as *TRACE*; that name is taken by
> [an existing CXR temporal model](https://arxiv.org/abs/2602.02963). Working shorthand is now
> `L2G`, a placeholder until the method is frozen. External review applied — see
> [`review-response.md`](review-response.md).

Selected from eight candidates in [`ideas.md`](ideas.md). Full method in
[`proposal.md`](proposal.md).

## Why

Every component of the claim is independently established by other people. The gap (29.3% vs 20%
random), the mechanism (MI-CXR's own stage-wise probing: 60–76% when the interval is given), the
baselines (14 VLMs, best 41.8%) and the test set (5,311 items) are all published by the MI-CXR
authors. We supply only the method. A reviewer who rejects our framing must still accept the gap.

No other candidate has this property.

## Why not the others

Filtered against the stated requirement — a published baseline to beat:

| Idea | Disqualifier |
|---|---|
| 6 — backward labels | Own metric *and* own baseline; no published number. Best as a follow-up |
| 5 — calibrated change | A wrapper; no single benchmark to top |
| 3 — pseudoprogression | 20 test patients; can beat AUC 75.3% but cannot prove it |
| 4 — WB-CT new lesions | No published fully-automatic number; we would establish the baseline, not beat it |
| 8 — failure detection | Real baseline, least distinctive contribution |
| 2 — MS new lesions | **Runner-up.** See tension below |

## Runner-up tension (Idea 2)

The paper argues the ground truth is provably incomplete, then is scored on that same incomplete
ground truth — improvements measured by the metric we argued is broken. Survivable by reporting
standard F1 alongside re-annotation evidence, but the answer must be decided before starting.
Idea 1 has no equivalent structural tension.

## Supporting factors

- **Power:** 5,311 test items, vs 60 patients (Idea 2) and 20 (Idea 3)
- **Headline:** ~50M-param comparator + exact inference beats a 38B VLM on one consumer GPU. The
  12 GB constraint becomes a selling point
- **Compute:** lightest of the eight

## Known risks

1. **"It's just a CRF."** Concede the structure, defend the contribution. Lead with the
   diagnosis, not the machinery, and prove each component matters via the ablation grid.
2. **MI-CXR labels are report-derived and noisy.** Budget a human-agreement check on a subsample;
   corroborate on expert-labelled MS-CXR-T.
3. **Greedy composition may capture most of the gain.** The single most important experiment.
   Decide before December which paper you would write if it does — see proposal §10.
4. **Report missingness.** "Not mentioned" is not "absent"; handled by the MNAR emission model
   in proposal §3.4.

## Next steps

1. PhysioNet credentialing — critical path, start immediately ([`data.md`](data.md))
2. Synthetic simulation harness while blocked — CTMC prior + forward–backward, sweeping achievable
   accuracy against local comparator accuracy and calibration. Answers "can this work?" in week two
   and becomes Figure 2

## Held for later

**Idea 6** composes with this one rather than competing, and shares the latent-state trajectory
machinery. Revisit as the follow-up paper.
