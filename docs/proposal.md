# Local-to-global trajectory inference for multi-visit chest radiography

> **Working shorthand: `L2G`. This is a placeholder, not a proposed name.**
> The name *TRACE* is taken — [TRACE: Temporal Radiology with Anatomical Change
> Explanation](https://arxiv.org/abs/2602.02963) (Feb 2026) is a chest-X-ray temporal model.
> Choose the final acronym only once the method is frozen.

Target venue: MICCAI 2027 (~Feb 2027 deadline)
Compute envelope: single RTX 5070 Ti (12 GB VRAM), 64 GB RAM

*Revised after external review. See [`review-response.md`](review-response.md) for what changed and why.*

---

## 1. The problem

Radiologists read a chest radiograph against the patient's whole imaging history. Current systems
do not: they analyse each image independently or compare the current image against a single prior.

The MI-CXR benchmark (ACL Findings 2026) quantifies the resulting failure. Across 5,311 five-way
multiple-choice items over five-visit MIMIC-CXR timelines, **14 VLMs average 29.3% against a 20%
random baseline**; the best, InternVL3.5-38B, reaches 41.8%.

### 1.1 The diagnostic experiment

MI-CXR's stage-wise probing is what makes this a tractable research problem. When the relevant
interval is explicitly specified, **10 of the 14 evaluated models reach 60.1–76.5%** — GPT-5.2 at
76.5%, Gemini 3 Pro 74.3%, Lingshu-32B and MedGemma-27B at 70.5%, down to Claude Sonnet 4.5 at
60.1%. The remaining four sit at 59%, 58.5%, 44.8% and 28.4%.

Same images, same findings, same perception. Only the composition burden was removed.

### 1.2 The defensible reading of that result

> **The MI-CXR results indicate that longitudinal failure is not explained by visual perception
> alone; a major additional bottleneck is selecting and composing interval-level evidence into a
> globally consistent trajectory.**

This is deliberately weaker than "29% is not a vision failure." MI-CXR's own experiments still show
interval-level errors and genuine ambiguity, so perception is not irrelevant — it is insufficient
to explain the gap. The benchmark authors attribute the failure to temporal decision-making,
evidence selection, ordering, and global composition.

### 1.3 Position relative to existing work

Published longitudinal CXR **methods** operate on image *pairs*: BioViL-T, CoCa-CXR, ProTrans,
GRCD, CheXGround, TRACE (2602.02963), Transition-Aware best-of-N. Multi-visit resources that do
exist — MI-CXR, LoMeVQA, CheXTemporal, LUNGUAGE — are **benchmarks and datasets**, and LUNGUAGE's
sequential component structures *report text* (80 reports, 10 patients) rather than performing
image-based trajectory inference.

We therefore do **not** claim "the first multi-visit method." We claim a specific combination that
we have not found in prior work, stated in §7.

---

## 2. Core hypothesis

> Relocating composition out of the VLM and into explicit probabilistic inference over its local
> evidence converts strong local perception into reliable global longitudinal reasoning.

The VLM is never asked the global question. It is used as a **local relation sensor**.

---

## 3. Method

### 3.1 Latent representation

Per finding `f`, the patient's course is a hidden ordinal sequence

```
s_t ∈ S = {0 absent, 1 mild, 2 moderate, 3 severe},   t = 1 … T   (T = 5)
```

### 3.2 The relation partition — an interface, not a discovery

The six clinical change relations exactly partition `S × S`:

| Relation | Transitions | Cells |
|---|---|---|
| absent-both | 0→0 | 1 |
| new | 0→1, 0→2, 0→3 | 3 |
| resolved | 1→0, 2→0, 3→0 | 3 |
| stable | 1→1, 2→2, 3→3 | 3 |
| worsened | 1→2, 1→3, 2→3 | 3 |
| improved | 2→1, 3→1, 3→2 | 3 |
| | | **16 / 16** |

The partition is trivially verifiable and is **not itself the contribution** — a reviewer would
rightly say "you defined six labels that partition sixteen cells." The contribution is what it
enables:

> The clinically meaningful change vocabulary can be represented exactly as a partition of the
> latent transition space, letting VLM relation probabilities be injected directly as pairwise
> potentials **without an additional heuristic mapping layer**.

Similar five-class vocabularies are already established in longitudinal CXR work (CheXTemporal uses
new/worse/stable/improved/resolved), which supports the clinical grounding rather than undermining
it.

### 3.3 Full factorisation

The model uses **both** per-image and pairwise evidence. Writing `y_t` for the per-image
observation at visit `t` and `z_t` for the comparator output on interval `(t, t+1)`:

```
P(s_1:T | y_1:T , z_1:T-1)  ∝  P(s_1) · ∏_t P(y_t | s_t) · ∏_t ψ_t(s_t, s_t+1)

ψ_t(s, s′)  ∝  P_cmp( r(s,s′) )^ρ_t  ×  P_CTMC( s′ | s, Δτ_t )
```

So a five-visit timeline contributes **5 unary + 4 pairwise = 9 observations** constraining 5 latent
states. Both terms are load-bearing; the ablation in §8 isolates each.

### 3.4 Emission model with explicit missingness

**A finding not mentioned in a report is not evidence of absence.** Radiology reports are
incomplete observations, and collapsing "not mentioned" into "absent" would systematically corrupt
both training labels and inference.

The observation `y_t` therefore takes three outcomes:

```
y_t ∈ { observed-present(severity), observed-absent, NOT-OBSERVED }
```

Missingness is modelled as **informative** (MNAR): a per-state reporting rate `μ(s)` with
`μ(0) ≠ μ(3)`, since a severe finding is far more likely to be mentioned than a mild one. This
makes "not observed" weak evidence *toward* absence without ever treating it as proof — which is
both clinically correct and, we expect, a measurable accuracy gain over the naive collapse.

### 3.5 Local comparator (perception)

`P_cmp(relation | x_t, x_t′, f)` — a Siamese encoder with difference-aware fusion, conditioned on a
learned finding embedding via FiLM so one model covers all findings. Backbone: ResNet-50 or ViT-B
from CXR-pretrained weights at 512×512.

A per-image severity head supplies `P(y_t | s_t)`.

**Calibration is not optional here.** The method consumes `P(relation)`, not `argmax relation`:
`{new .51, stable .49}` and `{new .99, stable .01}` must behave differently. See §6.

### 3.6 Acquisition-aware reliability gating (`ρ`)

Apparent change on CXR is confounded by projection (AP vs PA), rotation, inspiration depth and
exposure. `ρ_t ∈ (0,1]` acts as an inverse temperature: `ρ → 0` flattens the comparator's evidence
toward uninformative, so inference falls back on the prior and the per-image reads rather than
propagating an acquisition artefact into the trajectory.

**Supervision — weak, not gold.** MIMIC-CXR contains many same-day repeat studies. These are
**weak null-change supervision**, *not* ground truth: pneumothorax, flash pulmonary edema, effusion
after drainage, atelectasis and tube repositioning all genuinely change within hours. The
high-confidence subset is therefore:

```
same-day pair  AND  report states no interval change  AND  finding is not acutely labile
                                    ↓
                    high-confidence null-change subset
```

with manual validation on a sample before the subset is used, and acquisition metadata
(`ViewPosition`, inspiration proxy, rotation estimate, exposure statistics) as auxiliary input.

### 3.7 Continuous-time transition prior

Inter-visit intervals in MIMIC span hours to years, and existing multi-visit work discards them.
Model each finding as a continuous-time Markov chain with generator `Q_f`, restricted to ordinal
birth–death structure (`0↔1↔2↔3`), giving six free rates `q01 q10 q12 q21 q23 q32`:

```
P(s_t+1 | s_t, Δτ) = [ exp(Q_f · Δτ) ]
```

**Open question, stated as such.** Chest ImaGenome provides scene-graph attributes and comparison
relations extracted from reports; it does **not** hand over a clean four-level ordinal severity
scale per finding. MS-CXR-T's labels are coarser still (improving / stable / worsening). So the
plan is not "estimate `Q_f` from Chest ImaGenome" but:

> Investigate estimating the CTMC rates from available longitudinal severity annotations, and
> assess identifiability and calibration per finding — falling back to a three-state or binary
> latent scale where the four-state scale proves unidentifiable.

This is a genuine risk and is scheduled early (§9).

### 3.8 Inference

The graph is a chain over 4 states and 5 steps, so `4^5 = 1024` trajectories can be enumerated
outright; forward–backward gives marginals and Viterbi the MAP path.

**Stated precisely:** this is **exact inference in the specified finite-state model**. It is not
exact inference about the patient — the factorisation can be exact and the model still
misspecified. Every consistency claim below is likewise *consistency with the defined latent-state
trajectory model*, not a clinical guarantee.

### 3.9 A known modelling weakness: conditional independence

`y_t`, `z_{t-1,t}` and `z_{t,t+1}` all depend on the same image `x_t`. Treating them as
conditionally independent given `s_t` is false — shared visual features drive all three — and will
make the posterior **overconfident**.

We will (a) quantify the effect via posterior calibration on held-out timelines, (b) test a shared
down-weighting exponent on overlapping evidence, and (c) report it openly rather than let a
reviewer find it. Naming this ourselves is cheaper than being caught by it.

### 3.10 Answer decoding

Each question type is a query on one posterior:

- **TEL** — `argmax_t P(s_{t-1}=0, s_t>0)`
- **ICR** — posterior over the relation on `(i, j)`
- **GTS** — likelihood of each candidate trajectory shape

MCQ options are mapped to trajectory predicates offline by a small LLM parser. **This parser is a
dependency and an attack surface** ("you used an LLM to solve part of the task"); we will report
parser accuracy against manual annotation on a sample and treat parser error as part of the error
budget. The same applies to grounding a question to the finding whose chain should be run.

### 3.11 Adaptive interval querying

Rather than running all `T(T-1)/2` comparisons, select pairs greedily by expected information gain
about the target question `Q`:

```
IG(V_i, V_j ; Q)  =  H(Q | E)  −  E_z [ H(Q | E, z_ij) ]
```

```
current posterior → which pair most reduces uncertainty about Q?
                  → query the VLM on that pair → update → repeat under budget B
```

This yields an **accuracy vs. number of VLM calls** curve. It is currently the least developed
component and is scheduled last; it is a bonus contribution, not a load-bearing one.

---

## 4. Data

| Dataset | Role |
|---|---|
| MIMIC-CXR-JPG v2.1.0 | images; `ViewPosition`, `StudyDate`, `StudyTime` for timelines, Δτ and acquisition covariates |
| Chest ImaGenome v1.0.0 | localized comparison relations across sequential exams; candidate source for severity/CTMC estimation (§3.7) |
| MIMIC-Ext-CXR-QBA v1.0.0 | training/validation splits (public MI-CXR release is test-only) |
| MS-CXR-T v1.0.0 | expert progression labels, held-out evaluation |
| MI-CXR (GitHub) | primary benchmark, 5,311 test items |

See [`data.md`](data.md) for access and disk strategy.

---

## 5. Pilot experiment — before committing to the main dataset

The single quantity the whole project depends on is **local comparator accuracy and calibration on
six-way relations**, which nobody has published. CoCa-CXR's 65.0% on MS-CXR-T is a **three-class**
paired-image progression result and is *not* evidence that six-way will reach 65%.

Run as a formal pilot:

1. **Synthetic study (no data access needed).** Sweep achievable trajectory accuracy as a function
   of comparator accuracy and calibration. Establishes the required operating point in advance.
2. **Six-way comparator pilot** on a small MS-CXR-T / Chest ImaGenome slice, once credentialed.

If the pilot says the required comparator accuracy is unreachable, that is a finding worth knowing
in October rather than January.

---

## 6. Comparator calibration — its own experimental thread

The pairwise potential uses `P_cmp(r)^ρ`. That is a valid *score* regardless, but calling it a
probability-derived likelihood requires evidence. Where scores come from a VLM's token
probabilities, they are not automatically calibrated clinical probabilities.

Planned: raw scores vs. temperature scaling vs. isotonic regression vs. per-finding calibration,
each evaluated by reliability diagrams and ECE, and each propagated through to final trajectory
accuracy. Expect this to be one of the paper's more informative tables.

---

## 7. Claimed contribution

> A multi-visit CXR inference framework that converts local VLM change probabilities into explicit
> pairwise potentials over a latent ordinal disease trajectory and performs exact, interval-aware
> probabilistic inference in that model, with all downstream temporal questions derived from a
> single posterior.

Stated as a combination, since no individual ingredient is new:

```
VLM local relation probabilities
        + ordinal latent disease states
        + relation-space factorisation
        + acquisition reliability gating
        + continuous-time transition prior
        + exact multi-visit inference
        + multiple downstream question read-outs
```

We have not found this combination in prior CXR work. The related-work section will state that as a
searched claim about a *combination*, never as "nobody has done multi-visit CXR."

### Contributions as they will be listed

1. **A local-to-global formulation for longitudinal CXR reasoning** that uses VLMs as
   interval-level change comparators and delegates trajectory composition to structured
   probabilistic inference.
2. **A clinically grounded transition representation** in which the six change relations form an
   exact partition of the four-state ordinal transition space, enabling direct conversion of
   comparator outputs into pairwise potentials.
3. **Acquisition-aware evidence weighting** that down-weights unreliable comparisons arising from
   projection, positioning and image-quality differences.
4. **A continuous-time transition prior** accounting for irregular inter-visit intervals.
5. **Exact finite-state inference** producing a single trajectory posterior from which TEL, ICR and
   GTS answers are derived, giving architectural cross-task consistency.
6. **A systematic evaluation of local-to-global reasoning** — comparator calibration, trajectory
   consistency, and compute/accuracy trade-offs.

---

## 8. Evaluation

### 8.1 The headline ablation

This table *is* the paper. The critical row is **greedy vs. global**: if pairwise-then-greedy
already captures most of the gain, the central thesis is weakened and the story must change (§10).

| Configuration | What it tests |
|---|---|
| Direct VLM | published baseline (29.3% mean, 41.8% best) |
| VLM + specified interval | local upper bound (60.1–76.5% for 10/14 models) |
| Pairwise VLM + greedy/majority composition | **does decomposition alone explain the gain?** |
| + discrete-time Markov | value of a temporal prior |
| + CTMC | value of irregular-interval conditioning |
| + reliability gate | value of acquisition awareness |
| + unary observations | value of per-image evidence |
| + missingness-aware emission | value of not collapsing "not mentioned" into "absent" |
| + calibration | value of calibrated vs. raw comparator scores |
| + adaptive querying | compute efficiency |
| **Full model** | complete system |

### 8.2 Benchmarks

Primary: **MI-CXR**, overall and split by TEL / ICR / GTS.
Secondary: **MS-CXR-T** (validates the comparator in isolation against BioViL-T and CoCa-CXR),
plus TemMed-Bench / CheXTemporal for generality.

### 8.3 Metrics

- **Trajectory Consistency Rate** — fraction of predictions corresponding to a realisable
  trajectory under the defined model. The model is 100% by construction; the finding is **how often
  current VLM baselines emit logically impossible trajectories**. Note baselines *do not* enforce
  this, rather than *cannot* — constrained decoding could, and we will say so.
- **Cross-task Coherence** — agreement between TEL and ICR answers about the same finding/timeline.
- Posterior calibration (ECE, reliability diagrams), given §3.9.

### 8.4 Rigour

Five seeds, bootstrap CIs, McNemar against the best baseline. Human-agreement check on a subsample,
since MI-CXR labels are report-derived.

### 8.5 Target, labelled as hypothesis

**29.3% → 55–65%** is a *hypothesis*, not a projection from measured data. It rests on unmeasured
six-way comparator accuracy (§5). The claim is not asserted anywhere in the paper until the pilot
supports it.

---

## 9. Timeline — Sept 2026 → Feb 2027

| Month | Work |
|---|---|
| **Sept** | PhysioNet credentialing (blocking, start day 1). Synthetic study (§5.1). Implement CTMC + forward–backward + enumeration. |
| **Oct** | Access → five-visit cohort, comparison labels, **severity-scale identifiability check (§3.7)**, weak null-change subset with manual validation. Comparator v1 + six-way pilot. |
| **Nov** | Missingness-aware emission, reliability head, full pipeline, first MI-CXR numbers. |
| **Dec** | Baselines including **greedy composition**, full ablation grid, calibration thread, secondary benchmarks. |
| **Jan** | Adaptive querying, metrics, statistical tests, posterior calibration analysis. |
| **Feb** | Writing, figures, buffer. Freeze method → **choose final name**. |

---

## 10. Risks

| Risk | Response |
|---|---|
| **Greedy composition captures most of the gain** | Then the paper's finding is "decomposition matters more than inference" — still publishable, but a different framing. **Decide now which paper you would write** so the December result does not ambush you. |
| "It's a CRF on a VLM" | Concede the structure, defend the contribution: *yes, structurally it is a temporal graphical model; the contribution is treating a VLM as a calibrated local relation sensor and converting its clinical change vocabulary into an exact probabilistic interface, with acquisition-aware reliability and continuous-time transitions.* Then prove each component matters via §8.1. |
| Four-state severity scale unidentifiable | Fall back to three-state or binary; scheduled as an October check |
| Comparator accuracy is the bottleneck | Surfaced by the September/October pilot, not in January |
| Overconfident posterior from §3.9 | Measured and reported, with down-weighting tested |
| Option parser / finding grounding errors | Reported as part of the error budget |
| MI-CXR label noise | Human-agreement subsample; corroborate on MS-CXR-T |
| 12 GB VRAM | 512px, AMP, gradient checkpointing, accumulation; large VLMs in 4-bit or via API |

---

## 11. One-paragraph framing

> MI-CXR shows that current VLMs can often interpret a specified pair of chest radiographs but fail
> when they must simultaneously select relevant intervals and compose multiple changes into a
> coherent five-visit trajectory. We therefore treat the VLM not as a longitudinal reasoner, but as
> a local probabilistic comparator. Its interval-level change distributions are converted into
> pairwise potentials over a four-state latent disease trajectory, modulated by acquisition
> reliability and a continuous-time transition prior. Exact inference over this small state space
> produces a single posterior trajectory, from which temporal event localization, interval change
> and global trajectory questions are answered consistently. The key hypothesis is that relocating
> composition from the VLM into explicit probabilistic inference converts strong local perception
> into reliable global longitudinal reasoning.
