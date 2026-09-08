# Three paper ideas

All three satisfy the same bar: a real clinical problem, public data, **published baselines with
concrete numbers to beat**, a novel method rather than a re-tuning, and trainable on a single
12 GB GPU before February.

They form one coherent arc over your stated theme — *disease evolution across time and space*:

| | Question | Modality |
|---|---|---|
| **Idea 1** | What happened across the whole timeline? | Chest X-ray, 5 visits |
| **Idea 2** | What is genuinely new right now? | Brain MRI, 2 timepoints |
| **Idea 3** | Is this change real, or is it the treatment? | Brain MRI, full trajectory |

---

# Idea 1 — TRACE: multi-visit trajectory reasoning

Full write-up in [`proposal.md`](proposal.md). Summarised here for comparison.

**Problem.** Radiologists read each film against the whole history. Models do not. MI-CXR
(ACL Findings 2026) put 14 VLMs on five-visit MIMIC-CXR timelines: **29.3% mean accuracy against
20% random**, best 41.8%. Told *which* interval to examine, the same models reach **60–76%**.

**Insight.** That gap is evidence-selection and global composition, not perception. VLMs are good
local comparators and bad global integrators.

**Method.** Never ask the VLM the global question. Treat each finding's course as a latent ordinal
state chain; use the VLM only for local pairwise comparisons; fuse with exact inference. The six
clinical change relations (absent-both / new / resolved / worsened / stable / improved) *exactly
partition* the latent state product space, so a comparator predicting them **is already** a
pairwise potential — no heuristic translation. Chain structure ⇒ forward–backward and Viterbi are
exact. Three further pieces: acquisition-aware reliability gating trained on same-day repeat
radiographs as free null-change supervision; continuous-time transition priors conditioned on the
true inter-visit gap; adaptive interval querying by information gain.

**Baselines.** 14 published VLMs (18.1–41.8%), your own CoT / self-consistency / describe-all-pairs
prompting runs, and the 60–76% oracle ceiling.

**Target.** 29.3% → **55–65%**.

**Data.** MIMIC-CXR-JPG, Chest ImaGenome, MIMIC-Ext-CXR-QBA, MS-CXR-T. See [`data.md`](data.md).

**Compute.** Lightest of the three. ~50M-param comparator; inference is DP on a 4-state, 5-step
chain.

**Risk.** Low–medium. Main exposure is the "it's just a CRF" review, and MI-CXR's report-derived
labels are noisy.

---

# Idea 2 — New-lesion detection when the ground truth is provably incomplete

**Clinical problem.** New T2/FLAIR lesions on follow-up MRI are *the* imaging biomarker of multiple
sclerosis disease activity, and they directly trigger escalation to higher-risk disease-modifying
therapies. Miss them and you under-treat active disease; over-call them and you escalate a patient
onto a drug with real harms.

### The finding this paper is built on

The MSSeg2 expert analysis (*Scientific Reports*, 2026) measured how well **human experts** do on
this task, and the numbers are remarkable:

- **40%** of ground-truth lesions were annotated by **two or fewer of four** experts
- the "average expert" **missed 43.80%** of lesions
- **39 lesions were detected by none of the four experts**, were found by top-ranked automated
  methods, and were then **confirmed real** by a three-rater check

So the benchmark's ground truth is *provably incomplete*, and models have been penalised as
false-positive for correct detections. Every method on MSSEG-2 — including the current best —
optimises against and is scored by a noisy, partial label set, and none of them models that.

### Method

Three pieces, each attacking a documented failure mode:

1. **Registration-marginalised detection.** Misregistration is the dominant false-positive source
   (independently confirmed in CT by Rocholl et al., who showed segmentation quality collapses
   under inter-scan registration error). Rather than one deterministic warp, use a probabilistic
   diffeomorphic registration and sample K plausible deformations. A candidate counts as *new*
   only if it survives the deformation posterior. Novelty probability is then marginalised over
   registration uncertainty — cheap (K forward passes), principled, and it makes "we cannot tell,
   the alignment is unreliable here" an explicit output rather than a silent false positive.

2. **Positive-unlabeled learning for incomplete annotation.** Given the evidence above, annotated
   lesions are certainly positive but unannotated voxels are **unlabeled, not negative**. Train
   with non-negative PU risk estimation, with the label-frequency prior π estimated from the
   four-rater agreement structure (per-lesion detection probability → π). This is a well-founded
   framework applied to a problem that has been *demonstrated* to have exactly that structure.

3. **Rater-aware calibration.** Predict the distribution over rater annotations rather than the
   consensus, and report a detection metric that credits candidates in proportion to expected
   rater support — alongside standard F1/Dice so the numbers stay comparable.

**Why it should win.** Current SOTA still treats consensus ground truth as truth. Registration
marginalisation should lift precision; PU learning should lift recall on the ~40% of lesions with
weak rater support. Both gains come from modelling documented properties of the data that nobody
else models.

### Baselines to beat

| Method | Result |
|---|---|
| MSSEG-2 challenge best (30 methods, 24 teams) | F1 **0.698** |
| Self-supervised + synthetic lesions (MICCAI 2024) | Dice 56.15 ± 7.06, F1 56.69 ± 9.12 |
| **Calibrated Inter-patch Blending (MICCAI 2025)** | **F1 75.80** ← the number to beat |
| Human experts (*Sci Rep* 2026) | average expert misses 43.80% |

Beating a MICCAI 2025 paper on its own benchmark, *and* being able to say something quantitative
about expert-level performance, is a strong result.

### Data and compute

MSSEG-2 (100 patients, 3D FLAIR, two timepoints, 40 train / 60 test), plus MSLesSeg and OpenMS for
generalisation. Brain FLAIR volumes are small — the whole thing is **tens of GB**, and 3D patch
training at 128³ fits comfortably in 12 GB.

**Crucially: this needs no PhysioNet access.** You can start it while credentialing runs.

**Risk.** Low. The benchmark is active, so raw incremental gains are contested — but the
incomplete-ground-truth reframing is a contribution independent of the leaderboard delta, and the
"lesions we found that no expert labelled" qualitative result is compelling on its own. Note a
lead time: MSSEG-2 access goes through the organisers' Shanoir platform.

---

# Idea 3 — Pseudoprogression: response assessment as trajectory inference

**Clinical problem.** After chemoradiation for glioblastoma, **20–30%** of patients show enlarging
contrast enhancement that is treatment effect, not tumour. Call it progression and you abandon a
therapy that is working, and may put the patient through a needless reoperation. Radiologists
often cannot tell at the timepoint itself — RANO's own answer is to wait and confirm on the next
scan.

### The insight

**Pseudoprogression is not a property of a scan. It is a property of the trajectory.** It is
*defined* retrospectively: transient enlargement that subsequently stabilises or regresses.

Yet every published deep-learning method classifies **a single follow-up timepoint** — including
the current state of the art (self-supervised multimodal, **AUC 75.3%**). The field is trying to
answer a trajectory question with a snapshot model.

### Method

1. **Generative decomposition of observed change.** Model observed enhancement as the sum of a
   latent tumour-burden component and a treatment-effect component, where the treatment-effect
   kernel is parameterised by **time since radiotherapy** — a recorded covariate with a known
   pharmacodynamic shape (effect peaks around three months post-RT, then decays). Instead of
   asking a classifier to guess, give the model the mechanism and let it explain away the
   therapy-driven part of the change. This is the same "apparent versus real change" theme as
   Idea 1, but with an actual physical driver to condition on.

2. **Early decision with abstention.** At each timepoint, output either a confident call or "the
   evidence does not yet support a call — confirm on the next scan." This mirrors real RANO
   practice and yields a clinically meaningful operating curve: **accuracy versus how long you
   wait**. A model that reaches expert accuracy one scan earlier is a real clinical result, and no
   existing method reports this axis at all.

3. **Trajectory-level inference** over the latent burden, so the retrospective definition is
   modelled rather than approximated by a snapshot.

### Baselines to beat

| Method | Result |
|---|---|
| **Self-supervised multimodal (2025, SOTA)** | **AUC 75.3%** |
| CNN-LSTM | AUC 0.64–0.81, accuracy 0.62–0.75 |
| RANO bidimensional measurement rules | clinical reference |
| Radiomics classifiers | various |

The SOTA's protocol is public and reproducible: SSL pretraining on BraTS2021 + UPenn-GBM +
UCSF-PDGM (n = 2,317 studies), training/validation on Burdenko-GBM-Progression (n = 59), external
test on GlioCMV (n = 20). Adopt the same protocol for a directly comparable number.

### Data and compute

| Dataset | Role |
|---|---|
| **LUMIERE** (91 GBM patients, 638 study dates, 2,487 images) | expert RANO ratings **with written rationale**, plus provided DeepBraTumIA / HD-GLIO-AUTO segmentations and radiomics |
| Burdenko-GBM-Progression (n = 59, TCIA) | matches SOTA protocol |
| GlioCMV (n = 20) | external test, matches SOTA |
| BraTS 2024 post-treatment glioma (1,350 train / 188 val) | pretraining |
| BraTS2021 / UPenn-GBM / UCSF-PDGM | SSL pretraining corpus |

All public. LUMIERE ships segmentations, so you work on crops and derived measurements rather than
training a 3D segmentor from scratch — very comfortable on 12 GB. **Also needs no PhysioNet
access.**

**Risk.** Highest of the three, and the reason is sample size: the directly-comparable protocol has
59 training and 20 test patients. Mitigations: use the SOTA's exact protocol for comparability, add
LUMIERE as a second independent evaluation, report confidence intervals, and be candid about
statistical power. The clinical story is the best of the three; the statistics are the weakest.

---

# Comparison

| | **Idea 1 — TRACE** | **Idea 2 — MS new lesions** | **Idea 3 — Pseudoprogression** |
|---|---|---|---|
| Clinical stakes | Moderate | High (therapy escalation) | **Highest** (abandoning effective therapy) |
| Number to beat | 41.8% acc | **F1 75.80** | AUC 75.3% |
| Headroom | **Very large** (oracle 60–76%) | Moderate | Moderate–large |
| Test set size | 5,311 items | 60 patients | 20 patients (+91 LUMIERE) |
| Data volume | ~15 GB after resize | tens of GB | tens of GB |
| **Blocked on PhysioNet?** | **Yes (1–2 wks)** | **No** | **No** |
| Other access lead time | — | Shanoir request | TCIA (fast) |
| Compute fit on 12 GB | Excellent | Good | Good |
| Novelty type | Reframing + inference | Reframing + learning theory | Mechanistic modelling |
| Statistical power | **Strong** | Adequate | **Weak** |
| Scoop risk to Feb | Moderate | Low–moderate | Low |

## Recommendation

**Idea 1 as the primary.** It has by far the largest headroom (a 5,311-item test set and a 34-point
gap between current models and the oracle condition), the strongest statistical power, and the
lightest compute. It is the one most likely to produce an unambiguous result.

**Idea 2 as the hedge, started now.** It needs no PhysioNet credentialing, so it fills exactly the
1–2 week window where Idea 1 is blocked. Its baseline is the most precisely defined of the three
(a MICCAI 2025 number on a fixed public test set), and the incomplete-ground-truth framing is
fresh — the paper it rests on is from 2026.

**Idea 3 if you want the best clinical story and can accept the variance.** Pseudoprogression is
the purest instance of your theme — apparent change that is not real change — and "no existing
method models the trajectory" is a clean, defensible gap. But 20 test patients is thin for MICCAI,
and that is a real reviewer risk, not a formality.

A practical note: Ideas 1 and 3 share machinery. Both are latent-state trajectory inference with a
confounder explained away using a recorded covariate — acquisition parameters in one case, time
since radiotherapy in the other. If Idea 1 works, Idea 3 becomes substantially cheaper as a
follow-up paper.

---
---

# Second batch — Ideas 4 to 8

Same bar as above. Two of these (4 and 6) came out of a tool discovery: **SAM / UAE / SAME++**
(Alibaba DAMO, public code) already solve registration-free dense anatomical correspondence,
reporting **91% accuracy on whole-body follow-up lesion matching**. That kills "registration-free
correspondence" as a standalone contribution — but it hands you correspondence as a *solved
component*, which makes lesion-level longitudinal work possible without training a 3D whole-body
model.

---

# Idea 4 — New-lesion discovery in whole-body CT, revived at lesion scale

**This is your original Idea 2, made feasible.** I set it aside on compute grounds; that judgement
was wrong in one specific way, and it is worth correcting.

**What I got wrong.** I assumed new-lesion discovery required training a 3D whole-body segmentor —
which is genuinely out of reach on 12 GB. But the open problem is not segmentation. It is the
**decision**: given a candidate region in the follow-up scan, is this a genuinely new lesion, or a
registration artefact, an anatomical change, a treatment effect, or a lesion that was always there
and simply unlabelled? That decision operates on **64³ crops**, which is trivial on 12 GB.

So the pipeline becomes: existing detectors propose candidates → SAM/UAE supplies correspondence →
**your contribution is the novelty-decision layer**. The remaining constraint is download size
(hundreds of GB), not the GPU.

**Why the problem is open.** autoPET IV supplies lesion prompts for follow-up lesions, so fully
automatic new-lesion discovery is *unevaluated*. The reported gap is **~0.33 Dice fully automatic
vs 0.61–0.74 prompted**. Rocholl et al. (MICCAI-LMID 2026) diagnosed why cascades fail and called
explicitly for integrated temporal models.

**Method.** A correspondence-uncertainty-aware novelty decision. SAM/UAE gives a match plus a
matching confidence; propagate that confidence into the novelty call so "new" cannot be asserted
where correspondence is unreliable. Model the four confounders explicitly rather than hoping the
network learns them: registration/correspondence failure, anatomical change, treatment effect,
and prior-scan miss. Output calibrated per-lesion new-lesion probability with a
decision-theoretic operating point, since a false new metastasis changes therapy.

**Data.** Longitudinal-CT (Tübingen, 300 patients, 600 studies, 7,182 lesions with correspondence
labels covering persistence / regression / merging / new appearance — public via FDAT), autoPET,
DeepLesion.

**Baselines.** LesionLocator (CVPR 2025, public code), autoPET IV entries, ULS23 cascade, the
0.33 automatic / 0.61–0.74 prompted gap.

**Risk.** Medium. Download volume is the real friction, and DKFZ is active here. But you would be
first to report new-lesion-specific discovery on this dataset, which is a defensible claim.

---

# Idea 5 — Calibrated change: learning the null distribution of "no change"

**Problem.** Every clinical change threshold is a fixed constant: RECIST 20% diameter, RANO 25%,
"new lesion" for MS. Those constants came from small reproducibility studies. Meanwhile deep
learning hands clinicians a volume with **no error bar at all**, so "the tumour grew 18%" is
reported with no way to know whether 18% exceeds this scan pair's measurement noise.

**Insight.** Change detection is a **hypothesis test**, not a subtraction. H₀ is "no change", and
its null distribution is *learnable* — because test-retest imaging exists, where true change is
known to be zero:

| Source | Null-change structure |
|---|---|
| **RIDER Lung CT** (TCIA, open) | same patient, two CT scans, **same scanner, same parameters, 15 minutes apart** |
| Multi-parameter test-retest lung CT (2024) | same patient reconstructed at varying parameters |
| MIMIC-CXR same-day repeats | same day, different acquisition |
| MSSEG-2 / same-session MRI rescans | zero true new lesions |

**Method.** Learn an **acquisition-conditioned null distribution** of apparent change — conditioned
on slice thickness, kernel, dose, contrast phase, positioning — then convert any existing change
detector's output into a calibrated p-value or posterior. The deliverable is a wrapper: it makes
*any* longitudinal model report statistically valid change, and it yields the first data-driven
replacements for RECIST-style constants.

**Baselines.** Fixed RECIST/RANO thresholds, ensemble variance, raw change detectors, published
repeatability coefficients.

**Novelty, honestly.** The *statistical* idea has precedent — classical neuroimaging has done
test-retest null modelling for cortical thickness and diffusion MRI, and there is an MS
"statistical detection of change" method on subtraction images. What does not exist is a **learned,
acquisition-conditioned null integrated with deep change detectors and validated across tasks**.
Frame the paper as that, not as inventing hypothesis testing.

**Why it is interesting.** It plugs into Ideas 1, 2, 3 and 4 — a general capability rather than a
single-benchmark result. Multi-dataset generality reads well at MICCAI.

**Risk.** Medium. RIDER is small (32 patients). The framing must be careful to avoid "this is just
a t-test" — lead with the learned conditional null and the cross-task evaluation.

---

# Idea 6 — Backward label propagation: learning to detect what was missed

**This is the most interesting idea in the batch, and the one I would most want to read.**

**The observation.** In longitudinal imaging, the truth at time *t* is often only knowable at time
*t+1*. Radiology has known this for decades through retrospective visibility review: when a cancer
is diagnosed, going back to the prior "negative" screen frequently shows it was **already visible
and simply not reported**.

The MSSeg2 expert analysis (*Sci Rep* 2026) is the same phenomenon measured precisely: **39 lesions
that no expert annotated** were found by automated methods and then **confirmed real** on
three-rater review.

**The method.** Turn that into supervision. A finding confirmed at visit *t+1* is propagated
**backwards** to visit *t*, where it was unlabelled. Those become positives for a "what was already
there" detector — free labels for exactly the subtle, early, easily-missed presentations that
matter most clinically and that forward-labelled training never sees.

Concretely: retrospective confirmation → backward propagation through correspondence (SAM/UAE
handles this) → a curriculum from obvious-late to subtle-early presentations → positive-unlabeled
risk estimation, because the earlier timepoint's negatives are unlabeled rather than negative.

**Why it matters.** The clinical framing is unusually strong: *teaching a model to catch findings
a year earlier by learning from what was demonstrably there but unreported.* Lead-time gain is a
result radiologists and reviewers both immediately understand.

**Data.** Longitudinal-CT (correspondence labels make backward propagation exact), MSSEG-2 (where
the phenomenon is already documented and quantified), NLST if you take on the download (interval
cancers with prior negative screens are precisely this structure).

**Baselines.** Standard detectors trained on forward labels only; the ~44% expert miss rate as a
human reference point; per-timepoint detection performance with and without backward supervision.

**Novelty, honestly.** Retrospective visibility studies are a long radiology tradition, and label
propagation exists in other guises. What is not done is using retrospective confirmation
**systematically as training supervision for earlier detection**, with the PU treatment that the
label structure demands.

**Risk.** Medium. The key threat is circularity — you must not evaluate on backward-propagated
labels you trained on. Hold out patients entirely, and evaluate lead time on prospectively
confirmed cases only. Get that protocol right in the first week or the paper is unsalvageable.

---

# Idea 7 — Radiologist hedging as free uncertainty supervision

**Problem.** Medical imaging models are badly calibrated, and calibration research is stuck because
there is no ground-truth uncertainty to calibrate *against* — only hard labels.

**Insight.** There is, and it is sitting in every radiology report. Radiologists hedge constantly
and gradationally: "possible", "cannot exclude", "questionable", "may represent", "consistent
with", "definite". That is **expert-annotated uncertainty at MIMIC scale, for free**, and it is
ordinal.

Better still, longitudinal structure gives you the *validation* nobody else has: a hedged finding
at visit *t* is often resolved at visit *t+1*. So you can ask whether hedging strength actually
predicts confirmation — and calibrate to it.

**Method.** Extract an ordinal hedging scale from reports; train the model to match the hedging
distribution rather than a hard label; validate calibration against longitudinal confirmation
rates. The claim to test: *a model calibrated to radiologist hedging is better calibrated against
outcomes than one calibrated by temperature scaling on hard labels.*

**Data.** MIMIC-CXR reports, Chest ImaGenome, MS-CXR-T. Same access as Idea 1, so the two share
setup cost.

**Baselines.** Temperature scaling, deep ensembles, evidential deep learning, MC dropout,
CheXpert/NegBio uncertain-label (−1) handling.

**Novelty, honestly.** CheXpert and MIMIC labelers already emit an "uncertain" class, and people
have used it. What is new is treating hedging as an **ordinal calibration target** and validating
it against **longitudinal confirmation**. Do not oversell it as the first use of report
uncertainty.

**Risk.** Medium. Hinges on hedging language being consistent enough across radiologists to
calibrate against — test this early on a small sample before committing.

---

# Idea 8 — Failure detection for segmentation, without the ensemble

**Problem.** Segmentation models fail silently in deployment. Nobody ships a per-case reliability
estimate, so a clinician cannot tell a good mask from a broken one.

**Benchmark.** DKFZ's `segmentation_failures_benchmark` (MedIA 2024) — five public 3D collections,
public code, realistic test-time distribution shift, risk-coverage evaluation. The established
baseline is the **pairwise Dice score between ensemble predictions**, which the benchmark found
consistently strongest.

**The opening.** That baseline needs a full ensemble — N× training and N× inference. A **single-pass
method that matches ensemble-level failure detection** wins on both performance and cost, and cost
is exactly what blocks clinical deployment.

**Baselines.** Pairwise-Dice ensemble (the one to match or beat), MC dropout, max softmax, entropy
aggregation, direct Dice regression, SegQC.

**Risk.** Low technical risk, moderate novelty risk. Everything is public and the evaluation is
pre-specified, so you will get a clean number. But it is DKFZ's benchmark and DKFZ's baseline, and
it is the least distinctive idea in this document — it is the safe one, not the interesting one.

---

# Full comparison — all eight

| # | Idea | Number to beat | PhysioNet? | Download | Novelty | Risk |
|---|---|---|---|---|---|---|
| 1 | TRACE, multi-visit CXR | 41.8% acc | **Yes** | ~15 GB | High | Low–med |
| 2 | MS new lesions, incomplete GT | F1 75.80 | No | tens of GB | High | Low |
| 3 | Pseudoprogression | AUC 75.3% | No | tens of GB | High | **High** (n=20 test) |
| 4 | WB-CT new lesions, crop-scale | 0.33→0.61 Dice gap | No | **100s of GB** | Med–high | Medium |
| 5 | Calibrated change | RECIST constants | No | small | Med (precedent) | Medium |
| 6 | Backward label propagation | forward-label detectors | No | tens of GB | **High** | Medium |
| 7 | Hedging as calibration | temp. scaling, ensembles | **Yes** | shares Idea 1 | Med–high | Medium |
| 8 | Failure detection | ensemble pairwise Dice | No | tens of GB | **Low** | Low |

## Revised recommendation

**Idea 1 still first.** Largest headroom, strongest statistical power, lightest compute, cleanest
result. Nothing in this batch displaces it.

**Idea 6 is the one I would add.** It is the most *interesting* idea here — teaching a model to
detect findings earlier by learning from what was provably there but unlabelled — and it composes
with Ideas 2 and 4 rather than competing with them. Its risk is protocol design, not feasibility,
which is the kind of risk you can eliminate in week one.

**Idea 5 is the highest-leverage if it works,** because it is a wrapper that upgrades every other
idea on this list. But it has the most prior art to navigate.

**Idea 8 only as a fallback** if you need a guaranteed publishable number rather than an
interesting paper.

Two pairings worth noting: **1 + 7** share all their data setup, so the second is cheap once the
first is running. **2 + 6** share the positive-unlabeled machinery and the incomplete-ground-truth
framing, and could plausibly be one stronger paper rather than two thinner ones.
