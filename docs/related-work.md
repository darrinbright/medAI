# Landscape and novelty audit

Survey conducted September 2026. Purpose: establish that multi-visit (T ≥ 3) trajectory
inference on chest radiographs is an unclaimed method lane, and record the competitive position
of the alternatives considered.

---

## A. Longitudinal chest X-ray — methods

**All published methods are bi-temporal (one prior, one current).**

| Work | Venue / date | Scope | Relation to TRACE |
|---|---|---|---|
| BioViL-T, "Learning to Exploit Temporal Structure for Biomedical VLP" | CVPR 2023 | CNN-Transformer hybrid multi-image encoder, temporal contrastive objective | Bi-temporal. Candidate backbone initialisation for Stage 1 |
| CoCa-CXR | 2025 | Contrastive captioners; 65.0% avg on MS-CXR-T, +4.8 over BioViL-T | Bi-temporal. Current SOTA target for the Stage 1 comparator |
| ProTrans, "Learning Directional Semantic Transitions" | 2026 | Progression as directional semantic transitions between bi-temporal states; reversed temporal learning | Bi-temporal. Closest in spirit; no global multi-visit inference |
| GRCD, "Grounded Region Change Detection for Multi-Finding CXR Pairs" | Jul 2026 | Region-Guided Change Tokens; per-finding grounding + change labels; releases cleaned 40,250-pair benchmark | Bi-temporal, **pairs**. Released data is a Stage 1 training source |
| CheXGround | Aug 2026 | Anatomical region tokens for grounded longitudinal interpretation | Bi-temporal |
| Transition-Aware best-of-N | Jun 2026 | Training-free reranking of generated reports using (prior → current) transition vectors | Bi-temporal, generation-focused, training-free |
| [TRACE](https://arxiv.org/abs/2602.02963) — Temporal Radiology with Anatomical Change Explanation | Feb 2026 | Joint temporal comparison, change classification and spatial localization; *"given a prior and current chest X-ray"* | Bi-temporal. **Also: this name is taken — do not reuse it** |

**Conclusion, stated carefully.** Every published longitudinal CXR *method* found operates on image
pairs. We do **not** claim "the first multi-visit method" — absolute negatives are unfalsifiable and
a single counterexample damages credibility. The claim made in the proposal is about a specific
*combination* (proposal §7), stated as a searched claim.

### Multi-visit resources that exist — and why they are not counterexamples

An external review cited these as refuting the bi-temporal characterisation. They do not:

| Resource | What it actually is |
|---|---|
| [LUNGUAGE](https://arxiv.org/abs/2505.21190) (also on PhysioNet) | A **benchmark + metric** (LUNGUAGESCORE), 1,473 reports / 230 patients. The sequential subset is **80 reports from 10 patients**, 3–14 studies each — and it structures *report text*, not image trajectories |
| MI-CXR, LoMeVQA, CheXTemporal, MC-CXR | **Benchmarks and datasets**, not methods |
| CheXGround, TRACE | **Bi-temporal** (prior + current) — they support the characterisation rather than refuting it |

The distinction that matters: benchmarks over T ≥ 3 exist and are proliferating; image-based
*trajectory inference methods* over T ≥ 3 were not found.

## B. Longitudinal chest X-ray — benchmarks

| Benchmark | Date | Content | SOTA |
|---|---|---|---|
| **MI-CXR** | ACL Findings 2026 | 5,311 five-way MCQ items over five-visit MIMIC-CXR timelines; TEL / ICR / GTS | **41.8%** (InternVL3.5-38B), 41.1% (GPT-5.2), 29.3% mean over 14 VLMs, 20% random |
| MS-CXR-T | 2023 | 1,326 expert-labelled progression pairs, 5 findings, {improving, stable, worsening}; + 361 sentence-similarity pairs | 65.0% (CoCa-CXR) |
| TemMed-Bench | 2025 | Change between clinical visits: VQA, report generation, image-pair selection | — |
| CheXTemporal | 2026 | Temporally grounded localisation + progression classification | — |
| LoMeVQA | 2026 | 206K longitudinal VQA pairs | — |
| MC-CXR | Aug 2026 | Context-induced disruption in VLMs | — |

Benchmarks are proliferating while methods lag — favourable conditions for a method paper.

## C. Structured / global inference over VLM evidence

Searched specifically for CRF, Viterbi, HMM, or energy-based global inference applied to
longitudinal medical imaging. **No such work found.** The field uses soft neural mechanisms
instead — pair-conditioned encoders, cross-attention fusion, structured longitudinal
representations.

Adjacent and citable, but from a different domain:

- **Joint Consistency: A Unified Test-Time Aggregation Framework via Energy Minimization** (2026)
  — constrained Ising-type energy minimisation where independent evaluation signals act as
  external fields and pairwise comparisons act as interactions. This is essentially the
  mathematical skeleton of TRACE Stage 4, but applied to aggregating LLM answers, not to
  longitudinal medical state estimation. Strong related-work anchor: it shows the formalism is
  respected, while the medical instantiation remains unclaimed.
- Bradley-Terry / Elo aggregation of pairwise comparisons into global rankings — same
  local-to-global motif, different structure (ranking, not a temporal chain).

## D. Acquisition confound in longitudinal CXR

Prior work acknowledges the confound but does not model it as a first-class quantity:

- "Deep Learning for Automated Triaging of Stable Chest Radiographs" (*Radiology*, 2023) —
  thoracic cage registration and subtraction to triage no-change pairs
- "Deep learning in chest radiography: detection of findings and presence of change"
  (*PLOS One*, 2018) — notes change can be detected "despite the confounding effect of frequent
  alteration of patient posture"
- Positioning-compliance assessment systems (2026) — score positioning quality, but as a QA
  task, not as an input to a change model

**No work learns a per-pair reliability estimate conditioned on acquisition covariates, and no
work uses same-day repeat radiographs as null-change supervision.** This is TRACE's Stage 2
contribution.

## E. Irregular sampling / time-conditioned priors

Continuous-time Markov chains are standard in disease-progression modelling from *tabular*
clinical data, but are not used to condition transition priors in longitudinal medical *imaging*
inference. Existing multi-visit imaging work treats visits as an unweighted sequence, discarding
the inter-visit interval. This is TRACE's Stage 3 contribution.

---

## F. Alternatives considered and set aside

### 3D text-to-voxel grounding — **not viable**

| Work | Status |
|---|---|
| VoxTell, "Free-Text Promptable Universal 3D Medical Image Segmentation" | CVPR 2026, DKFZ. 62K+ CT/MRI/PET volumes, 1K+ classes, multi-stage vision-language fusion. Released checkpoint (v1.1, Jan 2026), PyPI package, napari plugin |
| LesionLocator | CVPR 2025, same group. 23,262 annotated scans; first end-to-end zero-shot 3D segmentation + 4D tracking; ~10 Dice over prior promptable models |

Single group with a large compute and data moat, actively shipping. Not attackable on 12 GB.

### New-lesion discovery in longitudinal whole-body CT — **genuinely open, but wrong compute class**

The opportunity is real:

- **Longitudinal-CT** (*Scientific Data*, May 2026, University Hospital Tübingen): 300 patients,
  600 whole-body CT studies, 7,182 manually segmented lesions (4,079 baseline / 3,103 follow-up),
  each with anatomical location, volume, and longitudinal correspondence capturing persistence,
  regression, merging and new appearance. NIfTI, publicly downloadable via the FDAT repository.
- **autoPET IV** (MICCAI 2025) evaluates an interactive human-in-the-loop setting and **supplies
  lesion prompts for follow-up lesions** — so fully automatic new-lesion discovery is an
  unevaluated task. Reported gap: ~0.33 Dice fully automatic vs 0.61–0.74 Dice prompted.
- **Rocholl et al., "Unstable Prompts, Unreliable Segmentations"** (MICCAI-LMID 2026, Radboudumc)
  diagnosed the cascade failure precisely — registration error causes segmentation collapse,
  which breaks lesion correspondence — and concluded that robust tracking "requires a paradigm
  shift away from cascading single-purpose tools towards integrated, end-to-end models that are
  inherently designed for temporal analysis." An explicit invitation.
- Also in this space: "Promptable Longitudinal Lesion Segmentation in Whole-Body CT" (Sep 2025),
  "Exploiting Longitudinal Context in Clinician-Verified Interactive Lesion Tracking" (2026),
  PanTrack, LinGuinE.

**Why deferred:** 3D whole-body CT training is out of reach on 12 GB VRAM, and the space is
dominated by DKFZ, who own the dataset lineage, the tooling and nnU-Net. Revisit if cluster
compute becomes available.

A promising method sketch, preserved for later: treat new-lesion discovery as *hypothesis testing
against a patient-specific counterfactual prior* — warp the baseline to follow-up geometry,
predict the expected follow-up appearance under "no new disease", and propagate **registration
uncertainty** into the correspondence probability so that "new" cannot be asserted where the
registration is unreliable. This directly attacks the four confounders (registration error,
anatomical change, treatment effect, previously missed lesions) that make the task hard.

---

## Sources

- MI-CXR — https://aclanthology.org/2026.findings-acl.1512/ · https://github.com/AIDASLab/MI-CXR
- MS-CXR-T / BioViL-T — https://physionet.org/content/ms-cxr-t/1.0.0/ · https://arxiv.org/abs/2301.04558
- CoCa-CXR — https://arxiv.org/pdf/2502.20509
- ProTrans — https://arxiv.org/pdf/2606.15938
- GRCD — https://arxiv.org/abs/2607.02719
- CheXGround — https://arxiv.org/html/2608.30758
- Transition-Aware best-of-N — https://arxiv.org/abs/2606.28393
- Joint Consistency — https://arxiv.org/pdf/2605.06219
- VoxTell — https://arxiv.org/abs/2511.11450 · https://github.com/MIC-DKFZ/VoxTell
- LesionLocator — https://arxiv.org/abs/2502.20985 · https://github.com/MIC-DKFZ/LesionLocator
- Longitudinal-CT dataset — https://www.nature.com/articles/s41597-026-07466-y · https://fdat.uni-tuebingen.de/records/qwsry-7t837
- autoPET IV — https://autopet-iv.grand-challenge.org/
- Unstable Prompts, Unreliable Segmentations — https://arxiv.org/abs/2507.19230
- Stable-radiograph triaging — https://pubs.rsna.org/doi/full/10.1148/radiol.230606
