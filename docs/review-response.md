# Response to external review

Review received September 2026 on the pre-revision proposal. This records what changed, what did
not, and why. Roughly 80% of the review is correct; the errors cluster in the related-work section.

---

## Accepted — substantive changes made

### 1. Missingness (the review's best catch)

**Nothing in the earlier draft handled it.** A finding not mentioned in a report is not evidence of
absence — radiology reports are incomplete observations. Collapsing "not mentioned" into "absent"
corrupts both training labels and the emission model.

Now §3.4: `y_t ∈ {observed-present, observed-absent, NOT-OBSERVED}` with **informative** (MNAR)
missingness — a per-state reporting rate `μ(s)`, `μ(0) ≠ μ(3)`, since severe findings are far more
likely to be mentioned. Makes "not observed" weak evidence toward absence without treating it as
proof. Added to the ablation grid.

### 2. Name collision — verified, real

[TRACE: Temporal Radiology with Anatomical Change Explanation](https://arxiv.org/abs/2602.02963)
(Feb 2026) is a chest-X-ray temporal model. Working shorthand is now `L2G`, explicitly a
placeholder; final name chosen in February once the method is frozen.

Note the irony: that paper is *"given a prior and current chest X-ray"* — bi-temporal. Its existence
kills the acronym while supporting the bi-temporal characterisation of prior methods.

### 3. Same-day repeats are weak supervision, not ground truth

The earlier draft overstated this. Pneumothorax, flash edema, effusion post-drainage, atelectasis
and tube repositioning genuinely change within hours. Now framed as **weak null-change supervision**
with a high-confidence subset (same-day + report states no interval change + finding not acutely
labile) and manual validation before use.

### 4. Precision on "exact" and "guaranteed"

- "exact inference" → **exact inference in the specified finite-state model**
- "guaranteed temporal consistency" → **consistency with the defined latent-state trajectory model**

The factorisation can be exact while the model is misspecified. Cheap fix, removes an easy attack.

### 5. Chest ImaGenome does not supply a clean 0–3 severity scale

The earlier draft asserted `Q_f` would be estimated from it. Chest ImaGenome provides scene-graph
attributes and comparison relations from reports; MS-CXR-T is coarser still (improving / stable /
worsening). Now stated as an open question with an October identifiability check and a documented
fallback to a three-state or binary latent scale.

### 6. The 60–76% phrasing

Not "the same models." Now: **10 of the 14 evaluated models reach 60.1–76.5%** when the interval is
specified; the other four sit at 59%, 58.5%, 44.8% and 28.4%.

### 7. The central diagnosis, softened

"29% is not a vision failure" → *"longitudinal failure is not explained by visual perception alone;
a major additional bottleneck is selecting and composing interval-level evidence."* MI-CXR still
shows interval-level errors, so perception is insufficient to explain the gap rather than
irrelevant.

### 8. The partition is an interface, not a discovery

Explicitly conceded in §3.2: a reviewer would rightly say "you defined six labels that partition
sixteen cells." The contribution is that it removes the heuristic mapping layer between comparator
output and inference potential.

### 9. Calibration promoted to its own thread (§6)

The method consumes `P(relation)`, not `argmax`. Added: raw vs. temperature vs. isotonic vs.
per-finding calibration, reliability diagrams and ECE, each propagated to final accuracy. Also
notes that VLM token probabilities are not automatically calibrated clinical probabilities.

### 10. Greedy vs. global is now the headline ablation

Promoted from a buried list item to the critical row of the main table. It is the experiment that
decides whether the thesis is true.

### 11. Adaptive querying now has a definition

`IG(V_i,V_j;Q) = H(Q|E) − E_z[H(Q|E,z_ij)]` with greedy selection under budget B, and is labelled a
bonus contribution rather than load-bearing.

### 12. "Baselines cannot achieve consistency" → "do not"

Constrained decoding could enforce it. The result is that current baselines *do not*, and we
quantify how often they emit impossible trajectories.

---

## Rejected or corrected

### A. The unary term was never missing from the method

The review reports a "major mathematical hole" — that the formula uses only pairwise evidence while
the text claims nine observations, and recommends "Option B: keep the per-image reads."

Option B was already the design. The earlier proposal specified unary potentials from a per-image
severity head in both the comparator section and the inference section. What was wrong was the **ψ
formula in the published figure**, which showed only the pairwise factor.

Fixed by writing the full factorisation explicitly (§3.3) and correcting the figure. Presentation
bug, not a modelling hole — but a fair catch on what was actually visible.

### B. "No prior multi-visit methods = FALSE" — conclusion right, evidence wrong

The review cites LUNGUAGE, MI-CXR, LoMeVQA and CheXGround as counterexamples. Checked:

| Cited | What it actually is |
|---|---|
| [LUNGUAGE](https://arxiv.org/abs/2505.21190) | A **benchmark + metric**. Sequential subset is **80 reports from 10 patients**, and it structures *report text*, not image trajectories |
| MI-CXR, LoMeVQA, CheXTemporal | **Benchmarks** — the earlier draft already separated benchmarks from methods |
| [CheXGround](https://arxiv.org/abs/2608.30758), TRACE | Both **bi-temporal** (prior + current) — they *support* the characterisation |

None of these is a multi-visit image-trajectory *method*.

**The claim was still softened** — but for the right reason: absolute negatives are unfalsifiable
and a reviewer needs only one counterexample to damage credibility. Not because these papers refute
it. Conceding LUNGUAGE as a competing multi-visit method would give away ground unnecessarily.

### C. "~50M comparator / one consumer GPU not established"

These are design constraints in a proposal, not claims about results. Not asserted as findings; no
change needed beyond not stating them as achieved.

### D. Suggested replacement names

TIDE, LIFT, TRAIL and TAP are all heavily used in ML. The underlying advice — name after freezing
the method — is followed.

---

## Added, raised by neither the review nor the original draft

### i. Conditional independence / double counting (§3.9)

`y_t`, `z_{t-1,t}` and `z_{t,t+1}` all depend on image `x_t`. Treating them as conditionally
independent given `s_t` is false, and will make the posterior **overconfident**. Technically more
serious than most of what the review raised. Now named openly, with calibration measurement and a
shared down-weighting exponent as mitigations.

### ii. Option parser and finding grounding as an error budget (§3.10)

Mapping MCQ options to trajectory predicates, and grounding a question to the right finding, are
both dependencies *and* an attack surface ("you used an LLM to solve part of the task"). Parser
accuracy will be reported against manual annotation.

### iii. Contingency if decomposition, not inference, carries the gain (§10)

If pairwise-then-greedy reaches ~58% and full inference ~61%, the paper is still publishable but
becomes "decomposition is what matters." Decide which paper to write **before** December so the
result does not force a rushed reframe.

---

## Assessment of the review itself

The review's own §25 framing paragraph is better positioning than the earlier draft and has been
adopted nearly verbatim as §11 of the proposal.

Its ratings — idea 8.5/10, novelty as written 6.5/10, technical soundness 7/10 — look fair. The
novelty score was a fair hit on the earlier phrasing, not on the underlying idea, and §7 now states
the contribution as a combination rather than a first.
