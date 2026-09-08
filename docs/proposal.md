# TRACE: Trajectory Reasoning by Aggregating Calibrated Evidence

**Multi-visit disease trajectory inference from longitudinal chest radiographs**

Target venue: MICCAI 2027 (~Feb 2027 deadline)
Compute envelope: single RTX 5070 Ti (12 GB VRAM), 64 GB RAM

---

## 1. The problem

Radiologists read a chest radiograph in the context of the patient's whole imaging history. A
lung opacity that is *absent → new → larger → stable → improved* across five visits tells a
different clinical story than the same opacity seen once. Current systems do not reason this
way: they either analyse each image independently or compare the current image against a
single prior.

The MI-CXR benchmark (ACL Findings 2026) quantified how badly this fails. Across 5,311
five-way multiple-choice items built on five-visit MIMIC-CXR timelines, **14 state-of-the-art
VLMs averaged 29.3% accuracy against a 20% random baseline**. The best models — InternVL3.5-38B
(41.8%) and GPT-5.2 (41.1%) — barely doubled chance.

### 1.1 The diagnostic that defines this project

MI-CXR's stage-wise probing contains the key result. When a model is **told which interval to
examine** ("compare Visit 2 and Visit 3"), accuracy jumps to:

| Model family | Interval given |
|---|---|
| Closed-source | 60.1 – 76.5% |
| Medical VLMs | 58.5 – 70.5% |
| Open-source | 28.4 – 66.7% |

The benchmark authors summarise it as: models "produce locally plausible interval descriptions
but fail to enforce temporal constraints or compose evidence into globally consistent decisions
over the full timeline."

**This is not a perception failure. It is an evidence-selection and global-composition failure.**
The gap between 29% (unaided) and 76% (interval handed over) is the headroom this paper targets.

### 1.2 Why this is an open lane

The longitudinal CXR literature is saturated at the **bi-temporal** level — one prior, one
current:

- BioViL-T (CVPR 2023) — temporal multi-image encoder
- CoCa-CXR (2025) — contrastive captioners, current SOTA on MS-CXR-T at 65.0%
- ProTrans (2026) — directional semantic transitions between bi-temporal states
- GRCD (2026) — grounded region change detection over CXR *pairs*
- CheXGround (2026) — anatomical region tokens for grounded longitudinal interpretation
- Transition-Aware best-of-N (2026) — training-free reranking on (prior, current) transitions

Every one of these operates on **two** images. Multi-visit (T ≥ 3) global trajectory inference
has **no method paper**. MI-CXR defined the task and demonstrated universal failure; nobody has
yet built a method for it.

Benchmarks are also proliferating (MI-CXR, TemMed-Bench, CheXTemporal, LoMeVQA, MC-CXR) — which
is a *good* sign for a method paper: the evaluation infrastructure is built and the baselines
are published and weak.

---

## 2. Core claim

> VLMs are competent **local comparators** and incompetent **global integrators**. If we use the
> VLM only for local pairwise perception and perform exact probabilistic inference for the
> global trajectory, most of the 29 → 76 gap closes — using a model small enough to train on one
> consumer GPU.

The method never asks a VLM the hard global question. It asks only well-posed local questions
that VLMs demonstrably answer well, then fuses those answers under an explicit temporal model.

---

## 3. Method

### 3.1 Notation

- Timeline of images `x_1 … x_T` (T = 5) with acquisition times `τ_1 < … < τ_T` and DICOM
  metadata `m_t`
- Findings `f ∈ F` (pleural effusion, pneumothorax, consolidation, edema, atelectasis,
  cardiomegaly, lung opacity, …)
- **Latent ordinal state** `s_t^f ∈ S = {0: absent, 1: mild, 2: moderate, 3: severe}`

The patient's course for finding `f` is the latent chain `s_1^f … s_T^f`. Everything else is
noisy observation of that chain.

### 3.2 A clean change-relation algebra

The six clinical change relations **exactly partition** `S × S`:

| Relation | Constraint on (s_t, s_t') |
|---|---|
| absent-both | `s_t = 0, s_t' = 0` |
| new | `s_t = 0, s_t' > 0` |
| resolved | `s_t > 0, s_t' = 0` |
| worsened | `s_t' > s_t ≥ 1` |
| stable | `s_t' = s_t ≥ 1` |
| improved | `1 ≤ s_t' < s_t` |

This is the hinge of the whole method: a local comparator that predicts a distribution over
these six relations is *directly* a pairwise potential over the latent state chain. No ad-hoc
translation layer.

### 3.3 Stage 1 — Local pairwise comparator (perception)

`p_θ(c_{t,t'}^f | x_t, x_t')` — a Siamese encoder with a difference-aware fusion head,
conditioned on a learned finding embedding via FiLM so a single model covers all findings.

- Backbone: ResNet-50 or ViT-B initialised from CXR-pretrained weights (BioViL-T / CoCa-CXR
  style), 512×512 input
- Fits comfortably in 12 GB with AMP + gradient checkpointing
- Training labels: Chest ImaGenome comparison relations, MS-CXR-T progression labels
  (1,326 expert-labelled pairs, 5 findings), and GRCD's publicly released cleaned 40,250-pair
  benchmark

A per-image **absolute severity head** `p(s_t^f | x_t)` supplies unary potentials.

### 3.4 Stage 2 — Acquisition-aware reliability gating  ★ novel

Apparent change on CXR is heavily confounded by projection (AP vs PA), rotation, inspiration
depth, exposure and support devices. Radiologists hedge on exactly this ("apparent increase may
reflect differences in technique"). No existing longitudinal method models it explicitly.

Learn a per-pair reliability scalar `ρ_{t,t'} = g_φ(x_t, x_t', m_t, m_t', Δτ) ∈ (0,1]` used as
an inverse temperature:

```
p̃(c) ∝ p_θ(c)^ρ
```

`ρ → 0` flattens the evidence to uninformative; `ρ → 1` trusts it fully. This is input-dependent
calibration (conditional temperature scaling), so it is principled and citable.

**Free supervision — the null-change trick.** MIMIC-CXR contains many *same-study and same-day
repeat* radiographs: pairs where true disease change is ≈ zero but acquisition change is not.
These are free negatives. Ground truth for such pairs is constrained to {stable, absent-both} —
any predicted *new / worsened / improved / resolved* is a false alarm attributable to
acquisition. Train `g_φ` to lower `ρ` precisely where the comparator errs on these pairs.

Acquisition features: ViewPosition, PatientOrientation, lung-field area ratio (inspiration
proxy), rotation estimate, support-device presence, exposure statistics, `Δτ`.

### 3.5 Stage 3 — Time-conditioned transition prior (CTMC)  ★ novel

MIMIC intervals are wildly irregular — hours to years. A one-day interval and a ninety-day
interval have completely different transition dynamics, yet no multi-visit method conditions on
the actual gap.

Model each finding's dynamics as a **continuous-time Markov chain** on the ordinal state space
with learned generator `Q_f`:

```
P(s_{t+1} | s_t, Δτ_t, f) = [exp(Q_f · Δτ_t)]_{s_t, s_{t+1}}
```

This handles irregular sampling natively and yields an interpretable per-disease dynamics
parameter (e.g. pneumothorax resolves faster than fibrosis). Estimate `Q_f` by MLE on
Chest ImaGenome-derived progression sequences.

### 3.6 Stage 4 — Exact global inference

The graph is a chain, so **forward–backward gives exact marginals and Viterbi gives the exact
MAP trajectory** — no approximation, negligible cost.

- Unary potentials: absolute severity head
- Pairwise potentials: gated comparator output × CTMC prior

*Extension (ablation):* findings co-occur (effusion / atelectasis / consolidation). A low-rank
coupling across findings with mean-field or loopy BP is a natural extension; the chain version
is the safe core.

### 3.7 Stage 5 — Answer decoding by posterior query

Every MI-CXR task family becomes a **query on one posterior trajectory**:

- **TEL** ("when did X first appear?") → `argmax_t P(s_{t-1}=0, s_t>0)`
- **ICR** ("what changed between visit i and j?") → posterior over `c_{i,j}` from marginals
- **GTS** ("overall course?") → likelihood of each candidate trajectory shape

For MCQ, score all five options by their likelihood under the posterior. Options are mapped to
trajectory predicates once, offline, by a small LLM parser — cheap and auditable.

**Consequence:** TEL, ICR and GTS answers are **cross-task consistent by construction**, because
they are read-outs of the same posterior. This is precisely the property MI-CXR shows current
models lack.

### 3.8 Stage 6 — Adaptive interval querying  ★ novel

Rather than running all `T(T-1)/2 = 10` comparisons, greedily select pairs by expected
information gain about the answer predicate, under a budget B. This directly answers the
"which visits are relevant?" difficulty and produces an accuracy-vs-compute curve — a
test-time-compute contribution, and an efficiency argument that lands well at MICCAI.

---

## 4. Contributions (as they will be listed in the paper)

1. **First method for multi-visit (T ≥ 3) trajectory reasoning on chest radiographs**, framed as
   latent ordinal state estimation rather than direct visual question answering.
2. **Acquisition-aware reliability gating** trained with free null-change supervision mined from
   same-day repeat radiographs — an explicit model of the apparent-vs-real change confound.
3. **Continuous-time transition priors** that condition on the true inter-visit interval,
   handling irregular longitudinal sampling.
4. **Exact global inference** yielding guaranteed temporal consistency and cross-task coherence.
5. **Adaptive interval selection** for test-time compute allocation.
6. **Two new evaluation metrics**: Trajectory Consistency Rate and Cross-task Coherence.

---

## 5. Evaluation plan

### 5.1 Primary benchmark — MI-CXR

5,311 test items, reported overall and split by TEL / ICR / GTS.

**Published baselines (free, already in the paper):** 14 VLMs, range 18.1 – 41.8%.
**Baselines we run ourselves:** 2–3 open VLMs runnable locally in 4-bit (Qwen2.5-VL-7B,
InternVL-8B) under zero-shot, chain-of-thought, self-consistency, and a strong
"describe-all-pairs-then-answer" prompting condition.
**Ceiling:** the oracle-interval condition (60–76%).

**Target: 29.3% → 55–65% overall.** Even the low end is a ~14-point absolute gain over the best
published model.

### 5.2 Secondary benchmarks (generality)

- **MS-CXR-T** progression classification — validates the local comparator in isolation against
  BioViL-T and CoCa-CXR (SOTA 65.0%)
- **TemMed-Bench**, **CheXTemporal** — cross-benchmark generality

### 5.3 New metrics

- **Trajectory Consistency Rate (TCR)** — fraction of predictions corresponding to a logically
  realisable trajectory. TRACE is 100% by construction; baselines will not be, and quantifying
  that is itself a result.
- **Cross-task Coherence** — agreement between TEL and ICR answers about the same
  finding/timeline.

### 5.4 Ablations

| Ablation | Tests |
|---|---|
| − global inference (pairwise argmax + heuristic) | value of Stage 4 |
| − reliability gating | value of Stage 2 |
| − time conditioning (fixed transition matrix) | value of Stage 3 |
| discrete-time vs CTMC | parameterisation choice |
| unary-only / pairwise-only | evidence source contributions |
| budget B = 1 … 10 comparisons | Stage 6 efficiency curve |
| coupled vs independent findings | value of Section 3.6 extension |
| synthetic: accuracy vs comparator quality | *when* structured inference helps |

### 5.5 Statistical rigour

5 seeds, bootstrap confidence intervals, McNemar's test against the best baseline. MICCAI
reviewers consistently penalise single-run numbers.

---

## 6. Data

All PhysioNet credentialed, all free:

| Dataset | Use |
|---|---|
| MIMIC-CXR-JPG | images, timelines, DICOM metadata, null-change pair mining |
| Chest ImaGenome | comparison relations, anatomical regions, `Q_f` estimation |
| MS-CXR-T | expert progression labels (1,326 pairs), secondary benchmark |
| MI-CXR | primary benchmark (test JSONL public on GitHub; images from MIMIC) |
| GRCD cleaned pairs | 40,250 additional training pairs, public |

---

## 7. Fit to a 12 GB GPU — and why it is a feature

The entire trainable footprint is a ResNet-50/ViT-B comparator plus small heads. The inference
layer is exact dynamic programming on a 4-state, 5-step chain — microseconds. Large VLMs appear
only as *baselines*, run in 4-bit or via API.

This inverts into one of the paper's strongest narratives:

> **A ~50M-parameter calibrated comparator plus exact inference outperforms a 38B-parameter VLM
> on multi-visit trajectory reasoning, trained end-to-end on a single consumer GPU.**

Efficiency, interpretability (an explicit inspectable trajectory) and clinical grounding are
exactly the axes MICCAI rewards.

---

## 8. Timeline — Sept 2026 → Feb 2027

| Month | Work |
|---|---|
| **Sept** | Submit PhysioNet CITI + credentialing (blocking, start day 1). Meanwhile: build synthetic simulation harness; implement CTMC + forward–backward; derive expected-gain curves as a function of comparator accuracy and calibration. Deep literature read. |
| **Oct** | Access granted → build MIMIC-CXR five-visit cohort, extract Chest ImaGenome comparison labels, mine null-change pairs. Train comparator v1. |
| **Nov** | Reliability head, `Q_f` estimation, full pipeline end-to-end. First MI-CXR numbers. |
| **Dec** | Local VLM baselines, full ablation grid, secondary benchmarks. |
| **Jan** | Adaptive querying, coupled-findings extension, new metrics, statistical tests. |
| **Feb** | Writing, figures, buffer. Submit. |

**Critical path:** PhysioNet credentialing (CITI "Data or Specimens Only Research" course,
~4–6 hours; approval typically 3–14 days). Start it immediately.

**The September synthetic study is not filler.** It tells you, before you touch real data,
exactly how accurate and how calibrated the local comparator must be for global inference to pay
off. If the answer is "comparator needs 70% pairwise accuracy and we can only reach 55%", you
learn that in week 2 rather than month 4. It also becomes Figure 2 of the paper.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **"It's just a CRF on top of a VLM."** The single most likely reviewer critique. | Depth comes from four independent pieces: CTMC time-conditioning, input-dependent calibration with null-change supervision, adaptive querying, and the efficiency result. Lead the paper with the *diagnosis* (local ✓ / global ✗), not the machinery. |
| MI-CXR labels are report-derived and noisy | Report human-agreement analysis on a subsample; corroborate on expert-labelled MS-CXR-T |
| Local comparator accuracy is the bottleneck | The synthetic study surfaces this in September. If confirmed, "structured inference cannot rescue a weak comparator, and here is the threshold" is still a publishable negative result — but you would pivot effort to Stage 1 |
| Someone scoops multi-visit before Feb | Move fast; Stages 2, 3 and 6 are defensible contributions independently of the framing |
| 12 GB VRAM ceiling | 512px, AMP, gradient checkpointing, gradient accumulation. Baselines in 4-bit or via API |
| PhysioNet delay | September work is deliberately data-independent |

---

## 10. Why this beats the alternatives considered

**Idea 3 (3D text-to-voxel grounding)** — VoxTell (CVPR 2026, DKFZ) trained on 62K+ volumes with
released checkpoints, PyPI package and napari plugin; same group owns LesionLocator (CVPR 2025)
and the autoPET IV winner. This is a compute war against the people who defined the task.
Infeasible on 12 GB.

**Idea 2 (new-lesion discovery in whole-body CT)** — genuinely open (the Tübingen
Longitudinal-CT dataset is new as of May 2026; autoPET IV supplies lesion prompts, so fully
automatic discovery is unevaluated; Rocholl et al. explicitly call for end-to-end temporal
models). But it is 3D whole-body CT — 300 patients × 600 studies — which is not trainable on
12 GB, and it means racing DKFZ on their own dataset with their own toolchain. **Worth
revisiting if compute changes.**

**Idea 1, multi-visit (this proposal)** — open method lane, published weak baselines, public
data, exact and cheap inference, and a clinical confound (acquisition variation) that nobody
models and that a single consumer GPU is entirely sufficient to attack.
