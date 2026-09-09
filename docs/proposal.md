# Local-to-global trajectory inference for multi-visit chest radiography

> **Working shorthand: `L2G`. This is a placeholder, not a proposed name.**
> The name *TRACE* is taken — [TRACE: Temporal Radiology with Anatomical Change
> Explanation](https://arxiv.org/abs/2602.02963) (Feb 2026) is a chest-X-ray temporal model.
> Choose the final acronym only once the method is frozen.

Target venue: MICCAI 2027 (~Feb 2027 deadline)
Compute envelope: single RTX 5070 Ti (12 GB VRAM), 64 GB RAM

*Revised after external review ([`review-response.md`](review-response.md)) and updated with
synthetic pilot results ([`pilot-findings.md`](pilot-findings.md)). Blocks marked **Pilot verdict**
are measured, not asserted.*

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
from CXR-pretrained weights at 512×512. A per-image severity head supplies `P(y_t | s_t)`, and a
third head emits the gate `ρ` of §3.6.

Implemented and tested in [`../model/comparator.py`](../model/comparator.py) — **25.4M parameters**
(23.5M backbone + 1.8M heads), 9/9 checks passing. Three design points that fell out of building it:

- **Fusion carries the signed difference** `f_a − f_b`, because the relations are order-sensitive:
  reversing a pair turns "new" into "resolved". That yields free supervision — a **swap-consistency
  loss** ties `p(r | a,b)` to `p(inv(r) | b,a)`, so every labelled pair constrains the model twice.
  Its inverse table is asserted against `sim.core.relation` rather than hand-typed, and it costs one
  extra MLP call rather than a second backbone pass.
- **`ρ` is trained to predict whether the comparison is correct.** Without this the reliability head
  has no objective at all and is silently untrainable. It also gives the gate the right semantics:
  the pilot showed gating pays only when the comparator is *confidently wrong*, which is exactly
  what a correctness-prediction head detects. Same-day pairs train it without relation labels.
- **FiLM starts near identity, not at identity.** Exact-zero weights give the finding embedding
  identically zero gradient on the first step.

**Calibration is not optional here.** The method consumes `P(relation)`, not `argmax relation`:
`{new .51, stable .49}` and `{new .99, stable .01}` must behave differently. See §6.

**Finding-specific polarity.** The six relations are defined over *severity of the abnormality*,
but report language often describes a measured quantity: "lung volumes have decreased" is a
*worsening*. Chest ImaGenome labels are report-derived, so training targets must be converted to
abnormality-severity polarity for volume/aeration-type findings, or the comparator learns the
inverted direction for that class. Implemented and tested for the option parser
(`sim/gts_parse.py`); see [`micxr-analysis.md`](micxr-analysis.md).

### 3.6 Acquisition-aware reliability gating (`ρ`)

Apparent change on CXR is confounded by projection (AP vs PA), rotation, inspiration depth and
exposure. `ρ_t ∈ (0,1]` acts as an inverse temperature: `ρ → 0` flattens the comparator's evidence
toward uninformative, so inference falls back on the prior and the per-image reads rather than
propagating an acquisition artefact into the trajectory.

> **Pilot verdict (conditional).** The gate is worth **+0.000** when acquisition mismatch merely
> makes the comparator *less accurate* — inference already discounts flat evidence, so gating is
> redundant. It pays **+0.06 to +0.15** only when mismatch makes the comparator *confidently
> wrong*. This is now a falsifiable question about real data, to settle in November on same-day
> pairs before the gate is claimed as a contribution at all. If mismatch merely produces
> uncertainty, **drop the gate** and give the space to §3.7.

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

> **Pilot verdict: this is the single largest component of the method.** Worth **+0.236** over a
> uniform transition prior — more than everything else combined — and the gain **survives
> misspecification**: +0.201 to +0.228 when ground truth is generated by a semi-Markov, momentum,
> non-stationary or mixture process and `Q` is fitted by MLE to that non-CTMC data. Worst case
> (time-homogeneity violated) loses ~15% of the benefit, not the benefit. Δτ-conditioning should
> therefore **lead** the technical contributions.

**Estimation: answered.** All six rates are recovered to within ~10% from discretely observed
transitions at irregular intervals (`run_misspec.py`), so the generator *is* identifiable from
panel data. This closes half the risk flagged in earlier drafts.

**Interpretation: qualified.** Under misspecification the fitted rates are biased in interpretable
ways — a semi-Markov truth inflates every rate, because increasing hazard produces more transitions
than an exponential dwell would and the fitted exponential compensates upward. So `Q_f` must be
presented as **a fitted prior that helps**, never as an estimate of true biological transition
rates. Do not write "the learned rate at which pneumothorax resolves" without first establishing
the process is Markov.

**Open question, narrowed.** What remains is a *labelling* question, not an estimation one: Chest
ImaGenome provides scene-graph attributes and comparison relations from reports, not a clean
four-level ordinal severity scale, and MS-CXR-T is coarser still. Scheduled for October, with a
documented fallback to a three-state or binary latent scale.

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

> **Pilot verdict: confirmed, quantified, mitigated.** Simulating the violation at its source — a
> per-image nuisance factor that shifts apparent severity, corrupting the unary read *and* both
> pairwise comparisons touching that image — reproduces the predicted signature exactly.
> Overconfidence is slightly *negative* (−0.026) when the assumption holds and grows monotonically
> once violated, reaching **+0.126** at strong correlation, with ECE rising 4.5×. Accuracy falls
> 0.803 → 0.594, which is the more expensive consequence.
>
> **A single shared tempering exponent on all image-derived evidence cuts ECE roughly fivefold at
> no accuracy cost** (`w ≈ 0.55`: ECE 0.087 → 0.016, accuracy 0.648 → 0.658). Accuracy is flat
> across the whole range, which is the correct behaviour — tempering rescales evidence uniformly,
> so only posterior sharpness changes. Over-tempering overshoots into underconfidence.

Therefore: report posterior calibration as standard, apply tempering with `w` tuned on held-out
real data (not transplanted from the synthetic study), and present this as a measured limitation
with a one-parameter fix. A learned joint emission over `(y_t, z_{t-1,t}, z_{t,t+1})` would address
the cause rather than the symptom, at the cost of the clean factorisation — worth a sentence, not a
reimplementation.

### 3.10 Answer decoding

Each question type is a query on one posterior:

- **TEL** — `argmax_t P(s_{t-1}=0, s_t>0)`
- **ICR** — posterior over the relation on `(i, j)`
- **GTS** — likelihood of each candidate trajectory shape

**Score options by Bayes factor, never by posterior mass.** An option's predicate covers a set of
trajectories, and set sizes differ wildly — "there is no second appearance" covers most of the
space. Ranking by raw posterior mass therefore selects the largest predicate almost regardless of
evidence, pinning those question types at `P(gold = E)`. Dividing by the predicate's prior mass is
the correct decoding when exactly one option is true. In the end-to-end dry run this is worth
**41 accuracy points** (0.363 → 0.813 at comparator accuracy 0.60), and it generalises to any
method scoring multiple-choice options against a distribution over structured objects.

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

## 5. Pilot results

The synthetic study is **done** — code in [`../sim/`](../sim/), full results in
[`pilot-findings.md`](pilot-findings.md). Absolute numbers are synthetic and are **not** comparable
to MI-CXR's 29.3%; what transfers is the ordering of methods and the shape of the sensitivity
curves.

| Question | Answer |
|---|---|
| Does global inference beat greedy composition? | **Yes, +0.20 to +0.26 at every comparator accuracy from 0.35 to 0.85.** The thesis holds |
| How accurate must the comparator be? | **Much less than feared.** Even at 0.35 six-way accuracy the full model reaches 0.690 vs 0.481 unary-only |
| Is the CTMC gain a simulator artifact? | **No.** +0.196 to +0.228 across four violated assumptions |
| Is `Q` identifiable? | **Yes**, all six rates to within ~10% |
| Does the independence violation bite? | **Yes**, up to +0.126 overconfidence — and `w ≈ 0.55` tempering fixes it free |
| Is the reliability gate worth it? | **Not in the base case (+0.000).** Only under confidently-wrong mismatch |
| How often is greedy self-consistent? | **6–27%.** A chain of plausible argmax relations is usually jointly unrealisable |

Correctness is machine-checked: 7/7 tests, with enumeration over 1024 trajectories agreeing with an
independent forward–backward implementation to 1.3e-15.

**What still needs real data.** The one quantity nobody has published remains **six-way comparator
accuracy and calibration** — CoCa-CXR's 65.0% is a *three-class* result and is not evidence for
six-way. October pilot on an MS-CXR-T / Chest ImaGenome slice.

---

## 6. Comparator calibration — a pass, not a thread

> **Pilot verdict: downgraded.** Degradation across a wide miscalibration range is graceful —
> roughly 5 points — and underconfidence hurts more than overconfidence. Worth a calibration pass
> and one table; not worth the standalone thread earlier drafts implied.

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

Ordered by measured contribution, not by narrative convenience.

1. **A local-to-global formulation for longitudinal CXR reasoning** that uses VLMs as
   interval-level change comparators and delegates trajectory composition to structured
   probabilistic inference. *Pilot: +0.20 to +0.26 over greedy composition at every comparator
   accuracy.*
2. **A continuous-time transition prior** accounting for irregular inter-visit intervals, with the
   six-rate birth–death generator estimated from panel data. *Pilot: +0.236, the largest single
   component, and robust to four kinds of model misspecification.*
3. **A clinically grounded transition representation** in which the six change relations form an
   exact partition of the four-state ordinal transition space, enabling direct conversion of
   comparator outputs into pairwise potentials without a heuristic mapping layer.
4. **Exact finite-state inference** producing a single trajectory posterior from which TEL, ICR and
   GTS answers are derived, giving architectural cross-task consistency. *Pilot: greedy composition
   is jointly self-consistent only 6–27% of the time.*
5. **A calibration treatment for correlated image evidence** — the conditional-independence
   violation measured, and a one-parameter tempering fix. *Pilot: ECE cut ~5× at no accuracy cost.*
6. **A systematic evaluation of local-to-global reasoning** — comparator calibration, trajectory
   consistency, and compute/accuracy trade-offs.

**Conditional, pending real data:** acquisition-aware evidence weighting (§3.6). Claimed only if
mismatch is shown to make the comparator confidently wrong rather than merely uncertain.

---

## 8. Evaluation

### 8.1 The headline ablation

This table *is* the paper. The critical row is **greedy vs. global**: if pairwise-then-greedy
already captures most of the gain, the central thesis is weakened and the story must change (§10).
The synthetic pilot says it does not — greedy trails by 0.20–0.26 — but that is a simulation, and
the row still has to be run on real data.

The **Expected** column records what the pilot predicts, so a real-data result that contradicts it
is visible immediately rather than being rationalised after the fact.

| Configuration | What it tests | Expected (from pilot) |
|---|---|---|
| Direct VLM | published baseline | 29.3% mean, 41.8% best |
| VLM + specified interval | local upper bound | 60.1–76.5% for 10/14 models |
| Pairwise VLM + greedy/majority composition | **does decomposition alone explain the gain?** | modest; trails full by a wide margin |
| + discrete-time Markov | value of a temporal prior | clear gain over uniform |
| + CTMC | value of irregular-interval conditioning | **largest single component** |
| + unary observations | value of per-image evidence | substantial |
| + missingness-aware emission | not collapsing "not mentioned" into "absent" | small but consistent (~+0.03) |
| + evidence tempering | correlated-image correction | calibration only, accuracy flat |
| + calibration | calibrated vs. raw comparator scores | ~5 points across a wide range |
| + reliability gate | acquisition awareness | **~0 unless mismatch is confidently wrong** |
| + adaptive querying | compute efficiency | not yet studied |
| **Full model** | complete system | — |

### 8.2 Benchmarks

Primary: **MI-CXR**, overall and split by TEL / ICR / GTS.
Secondary: **MS-CXR-T** (validates the comparator in isolation against BioViL-T and CoCa-CXR),
plus TemMed-Bench / CheXTemporal for generality.

### 8.3 Metrics

- **Trajectory Consistency Rate** — fraction of predictions corresponding to a realisable
  trajectory under the defined model. The model is 100% by construction; the finding is **how often
  current VLM baselines emit logically impossible trajectories**. The pilot suggests the headroom
  is large: greedy composition over argmax relations is jointly realisable only **6–27%** of the
  time, since a locally plausible chain routinely asserts "new" from an already-present state or
  "worsened" from severe. Note baselines *do not* enforce this, rather than *cannot* — constrained
  decoding could, and we will say so.
- **Posterior calibration** (ECE, reliability diagrams) — required, not optional, given §3.9.
- **Cross-task Coherence** — agreement between TEL and ICR answers about the same finding/timeline.

### 8.4 Rigour

Five seeds, bootstrap CIs, McNemar against the best baseline. Human-agreement check on a subsample,
since MI-CXR labels are report-derived.

### 8.5 Target, labelled as hypothesis

**29.3% → 55–65%** was a hypothesis. The end-to-end dry run on the real 5,311 questions
([`pilot-findings.md`](pilot-findings.md)) now projects **57–65%** at comparator accuracy 0.60 and
**49–54%** even at 0.35, assuming chance on the 26% of the benchmark that needs an LLM parser.
Arriving at the same range independently, from the real questions, is meaningful corroboration.

It remains a simulation. It does not include the correlated-image-nuisance effect of §3.9 (worth up
to −0.21), the comparator error structure is ours, and six-way comparator accuracy is still the one
unmeasured quantity. The claim stays hedged until the October pilot.

---

## 9. Timeline — Sept 2026 → Feb 2027

| Month | Work |
|---|---|
| **Sept** | ~~Synthetic study; CTMC + inference~~ **done** (§5). PhysioNet credentialing **submitted**. Remaining, unblocked: MI-CXR question analysis and the option→predicate parser (§3.10) — the test JSONL is public, only the images are credentialed. |
| **Oct** | Access → five-visit cohort, comparison labels, **severity-scale labelling check (§3.7)**, weak null-change subset with manual validation. Comparator v1 + **six-way accuracy pilot** — the last unmeasured quantity. |
| **Nov** | Missingness-aware emission, full pipeline, first MI-CXR numbers. **Gate decision (§3.6): measure on same-day pairs whether mismatch is confidently wrong or merely uncertain — keep or drop.** |
| **Dec** | Baselines including **greedy composition**, full ablation grid, secondary benchmarks. Tune tempering `w` on held-out data. |
| **Jan** | Adaptive querying, metrics, statistical tests, posterior calibration analysis. |
| **Feb** | Writing, figures, buffer. Freeze method → **choose final name**. |

---

## 10. Risks

Struck-through rows were live risks that the synthetic pilot has since retired. They are kept
visible rather than deleted, because each was retired *in simulation only* and would return if the
real-data result disagrees.

| Risk | Response |
|---|---|
| ~~**Greedy composition captures most of the gain**~~ | **Retired in simulation:** greedy trails by 0.20–0.26 at every comparator accuracy. Keep the contingency framing in reserve for the real-data run. |
| ~~CTMC gain is an artifact of the simulator~~ | **Retired:** survives semi-Markov, momentum, non-stationary and mixture ground truth (+0.196 to +0.228). |
| ~~`Q_f` is not identifiable~~ | **Retired:** six rates recovered to ~10% from irregular panel data. What remains is a labelling question (§3.7). |
| ~~Overconfidence from §3.9 is unquantified~~ | **Retired:** measured at up to +0.126, and `w ≈ 0.55` tempering cuts ECE ~5× free. |
| **Six-way comparator accuracy** | The last unmeasured quantity. Pilot says the requirement is low (0.35 already works in simulation), but the real figure lands in October. |
| **The reliability gate may be worth nothing** | New, from the pilot. Gate contributes +0.000 unless mismatch is confidently wrong. Decide in November; drop it if not. |
| "It's a CRF on a VLM" | Concede the structure, defend the contribution: *yes, structurally it is a temporal graphical model; the contribution is treating a VLM as a calibrated local relation sensor and converting its clinical change vocabulary into an exact probabilistic interface, with acquisition-aware reliability and continuous-time transitions.* Then prove each component matters via §8.1. |
| Four-state severity **labelling** unavailable in Chest ImaGenome | Fall back to three-state or binary; scheduled as an October check |
| Real correlation structure differs from the simulated nuisance model | `w` tuned on held-out real data, never transplanted from the synthetic study |
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
