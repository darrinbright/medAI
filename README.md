# medAI — Multi-visit disease trajectory reasoning

Research project targeting **MICCAI 2027**.

**TRACE** — Trajectory Reasoning by Aggregating Calibrated Evidence.

> VLMs are competent local comparators and incompetent global integrators. Use the VLM only for
> local pairwise perception, then perform exact probabilistic inference over a latent disease
> state chain for the global trajectory.

Multi-visit (T ≥ 3) trajectory reasoning on chest radiographs currently has **no method paper**.
The MI-CXR benchmark (ACL Findings 2026) showed 14 state-of-the-art VLMs average **29.3%** on
five-visit timelines against a 20% random baseline — yet reach **60–76%** when told which
interval to examine. That gap is an evidence-selection and global-composition failure, not a
perception failure, and it is what this project attacks.

## Documents

| Document | Contents |
|---|---|
| [`docs/proposal.md`](docs/proposal.md) | Full research proposal: problem, method (6 stages), contributions, evaluation plan, timeline, risks |
| [`docs/related-work.md`](docs/related-work.md) | Landscape and novelty audit establishing the open lane; alternatives considered and why they were set aside |

## Constraints

- Single RTX 5070 Ti, 12 GB VRAM, 64 GB RAM
- MICCAI 2027 deadline (~Feb 2027)
- PhysioNet credentialing not yet obtained — **critical path**

## Immediate actions

1. **Start PhysioNet credentialing today.** CITI "Data or Specimens Only Research" course is
   ~4–6 hours; approval typically 3–14 days. Everything downstream blocks on this.
   - Register: https://physionet.org/register/
   - Then request access to MIMIC-CXR-JPG, Chest ImaGenome, MS-CXR-T
2. **Clone the MI-CXR test set** (public, images resolved locally from MIMIC):
   https://github.com/AIDASLab/MI-CXR
3. **Build the synthetic simulation harness** (data-independent, see proposal §8). Implement the
   CTMC prior and forward–backward, then sweep expected accuracy as a function of local
   comparator accuracy and calibration. This de-risks the entire project in week 2 and becomes
   Figure 2 of the paper.

## Target result

MI-CXR overall accuracy **29.3% → 55–65%**, against a best published baseline of 41.8%, using a
~50M-parameter comparator plus exact inference on a single consumer GPU.
