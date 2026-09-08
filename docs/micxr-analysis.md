# MI-CXR benchmark analysis

Analysis of the public test set (`micxr_test.jsonl`, 5,311 items) from
[AIDASLab/MI-CXR](https://github.com/AIDASLab/MI-CXR). **The questions and answers are public;
only the images are credentialed**, so all of this was done before PhysioNet access.

Reproduce with `python run_parse_eval.py path/to/micxr_test.jsonl`. The dataset itself is not
vendored into this repo.

---

## Composition — the benchmark is not evenly split

| Group | Items | Share |
|---|---|---|
| **TEL** — temporal event localization | 2,994 | **56.4%** |
| **GTS** — global trajectory summarization | 2,000 | 37.7% |
| **ICR** — interval change reasoning | **317** | **6.0%** |

Earlier drafts weighted the three families equally. They are not equal: **ICR is 6% of the
benchmark**, and TEL alone is more than half. Effort should follow that distribution.

By question type:

| qtype | Group | n | Option style |
|---|---|---|---|
| single_entity_interval_summary | GTS | 1,000 | free text, ~181 chars |
| multi_entity_interval_summary | GTS | 1,000 | free text, ~161 chars |
| single_emergence | TEL | 500 | structured, ~14 chars |
| single_resolution | TEL | 500 | structured, ~14 chars |
| emergence_to_resolution | TEL | 500 | structured pairs |
| resolution_to_emergence | TEL | 500 | structured pairs |
| icr_multi_abn | ICR | 317 | free text, ~63 chars |
| multiple_emergence_1 / _2 | TEL | 250 / 250 | structured |
| multiple_resolution_1 / _2 | TEL | 250 / 244 | structured |

---

## The baseline to quote is 22.8%, not 20%

Answers are near-uniform over A–E, so the **majority-class baseline is 22.8%** overall (TEL 24.2%,
GTS 22.7%, ICR 26.2%) rather than the 20% random rate.

That sharpens the motivating gap rather than weakening it: the 14-VLM mean of 29.3% beats the
majority-class baseline by only **6.5 points**. Quote both numbers — a reviewer who computes the
majority baseline themselves should find we already did.

---

## TEL parses deterministically — no LLM, no attack surface

Proposal §3.10 flagged the option→predicate parser as a dependency *and* an attack surface: if an
LLM maps options to predicates, its errors become answer errors and a reviewer can ask whether the
LLM is solving the task.

For TEL that concern is gone. All 2,994 TEL items use short structured options — `T2 → T3`,
`T1, T5`, or `There is no ...` — averaging 14 characters. A regex parser handles them:

| qtype | Items | Options parsed | Mutually exclusive |
|---|---|---|---|
| single_emergence | 500 | 100.0% | 100.0% |
| single_resolution | 500 | 100.0% | 100.0% |
| multiple_emergence_1 / _2 | 500 | 100.0% | 100.0% |
| multiple_resolution_1 / _2 | 494 | 100.0% | 100.0% |
| emergence_to_resolution | 500 | 100.0% | 100.0% |
| resolution_to_emergence | 500 | 100.0% | 100.0% |
| **TOTAL** | **2,994** | **100.0%** | **100.0%** |

So for **56.4% of the benchmark there is no parser error budget to report**, because there is no
learned parser.

### Exclusivity checking caught a real semantic bug

Mutual exclusivity over the 32 binary trajectories is checkable **without any images**: a
well-formed multiple-choice item has exactly one true option, so the predicates must not overlap.

The first parser scored 100% coverage but only **57–59% exclusivity** on the two pair-style
qtypes. The cause was a genuine misreading: the questions ask which pair captures where a finding
"*first* appears and subsequently resolves", and an *exists* predicate is satisfied by several
pairs at once when a finding recurs. Taking the first event and the first opposite event after it
gives exactly one true pair, and exclusivity went to **100%**.

Without this check the error would have surfaced as unexplained accuracy loss on 1,000 items after
the images arrived. Worth keeping as a standing test.

*(Exhaustiveness is 0% on pair types by construction: only 4 of the 6 valid pairs are offered, so
some trajectories match no option. That is the benchmark's distractor design, not a parser defect.
Exclusivity is the property that matters for scoring.)*

---

## TEL never asks about severity — a binary chain suffices

Every TEL question concerns **presence**: new appearance, disappearance, first/second occurrence,
or a presence→absence→presence sequence. Severity qualifiers appear nowhere.

So TEL needs only a binary latent chain — `2^5 = 32` trajectories — obtained by marginalising the
ordinal model at `P(s > 0)`.

**This materially de-risks §3.7.** The open question there is whether Chest ImaGenome can support a
four-level ordinal severity scale. If it cannot, the fallback is no longer "weaken the whole
method": 56.4% of the benchmark never needed the four-state scale, and the fallback costs coverage
only on GTS and ICR.

---

## "No such event" is exactly 20% of TEL, by construction

| Group | Answer = E rate |
|---|---|
| TEL (every qtype) | **20.0%** |
| GTS | 13.8–14.6% |
| ICR | 12.9% |
| Overall | 17.4% |

Every TEL qtype lands on exactly 20.0%, which is clearly deliberate benchmark design. One fifth of
TEL requires confidently asserting that an event **never happens**.

The posterior supports this natively — option E is just `P(no emergence anywhere)` — but it makes
**absence calibration** a first-class requirement. A model biased toward finding events forfeits
20% of the largest question family. This deserves its own row in the results table.

---

## Findings are concentrated

TEL emergence/resolution questions across 20 distinct findings, but heavily skewed:

| Finding | n | | Finding | n |
|---|---|---|---|---|
| pleural effusion | 400 | | consolidation | 120 |
| pneumothorax | 400 | | lung opacity | 17 |
| sub-diaphragmatic air | 334 | | pneumonia | 8 |
| pulmonary edema | 296 | | airspace opacity | 4 |
| cardiomegaly | 237 | | *13 others* | ≤3 each |
| vascular congestion | 161 | | | |

**Seven findings cover ~98%.** The comparator needs to be good on seven findings, not on a long
tail — which makes a finding-conditioned single model clearly the right choice, and makes
per-finding CTMC rates estimable from a realistic amount of data.

---

## GTS and ICR are the hard part

Both use free-text clinical narratives, and `multi_entity_interval_summary` (1,000 items, 19% of
the benchmark) is structurally different from everything else: **each option describes a different
finding**.

> A: …the heart is mildly enlarged but decreases in size…
> B: …mild to moderate pulmonary edema that significantly improves…
> C: …small bilateral pleural effusions, which remain small…
> D: …the hilar contours are normal but become mildly prominent…
> E: None of them are true.

So the task is "which of these five statements, about five different findings, is true?" That
requires running the trajectory model **per finding** and checking each option's claim against the
corresponding posterior — plus a real parser to extract (finding, per-interval relation sequence)
from prose, with severity qualifiers.

This is where an LLM parser genuinely is needed, and where §3.10's error budget applies. It affects
44% of the benchmark and should be scheduled as its own piece of work, not folded into the TEL
path.

---

## What this changes in the plan

| Finding | Action |
|---|---|
| TEL is 56.4%, ICR only 6% | Weight effort by benchmark share; ICR is nearly a rounding error |
| TEL parses at 100% coverage and exclusivity | Drop the parser error budget for TEL; keep it for GTS/ICR |
| TEL needs only binary presence | §3.7's four-state risk now threatens 44% of the benchmark, not all of it |
| Majority baseline is 22.8% | Quote it alongside the 20% random rate; VLMs beat it by only 6.5 points |
| 20% of TEL answers are "never happens" | Absence calibration becomes a first-class requirement and a results row |
| 7 findings cover ~98% | Finding-conditioned single comparator; per-finding CTMC rates are practical |
| multi-entity GTS compares across findings | Separate work item: per-finding posteriors plus a prose parser |
