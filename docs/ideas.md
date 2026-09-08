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
