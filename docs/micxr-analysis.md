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

## The GTS and ICR path — how far rules go, and where they stop

`run_gts_parse_eval.py`. Both GTS types and ICR use free-text options, but they are **not**
arbitrary prose: the four non-"none" options of an item are a **minimal-pair set**, near-identical
text with the relation words swapped at a few slots. Diffing options across 600 items recovers the
entire relation vocabulary, and it is small and regular — `worsens / increases / enlarges /
progresses`, `improves / decreases / regresses / diminishes`, `remains stable / persists /
unchanged`, `resolves / completely resolves`, `newly appears / develops`, `remains absent`.

So the parser stays rule-based: locate interval markers, locate relation phrases, scope each
relation to the clause its marker owns, and convert per-interval relations into a mask over the
1024 ordinal trajectories.

### single-entity GTS — all options describe the same finding, so exclusivity applies

| Metric | Result |
|---|---|
| Options parsed | 3,867 / 4,000 (**96.7%**) |
| Items with all four parsed | 935 / 1,000 (**93.5%**) |
| …of those, mutually exclusive | 810 / 935 (**86.6%**) |
| End-to-end clean | **~81%** of the 1,000 items |

Two heuristics were chosen by sweep rather than intuition, and one of them was counter-intuitive:

| first relation wins | split two-marker clauses | Exclusive |
|---|---|---|
| ✅ | ❌ | **86.6%** |
| ❌ | ❌ | 84.3% |
| ✅ | ✅ | 68.1% |
| ❌ | ✅ | 66.6% |

*First* relation wins because the options are minimal pairs and the discriminative verb leads —
"partially improves **but persists**", "develops **and progresses**" — with the rest as
elaboration. And splitting a two-marker clause at the midpoint between markers **hurts by 18
points**: it cuts phrases apart, and giving both intervals of a clause the same relation
distinguishes minimal pairs more reliably than splitting them badly.

An earlier version assigned each relation to its nearest marker by character distance. That scored
61.9%, because in `"Between T1 and T2, X progressively worsens; ... from T2 to T3 ..."` the verb
sits nearer the *second* marker and its interval gets stolen. Clause scoping fixed it.

### cross-finding types — this is where rules stop

`multi_entity_interval_summary` and `icr_multi_abn` put a **different finding in each option**, so
exclusivity does not apply (the options are not competing descriptions of one chain) and answering
needs one posterior *per finding*.

| qtype | Items | Options with (finding + relations) | Items with all four |
|---|---|---|---|
| multi_entity_interval_summary | 1,000 | 74.8% | **24.6%** |
| icr_multi_abn | 317 | 63.0% | **21.1%** |

Expanding the finding lexicon from 35 to 55 terms moved options from 61.8% → 74.8% and 53.2% →
63.0%, then hit diminishing returns. The residue is not missing lexicon entries — it is
**compositional clinical language**, where the finding is an anatomical structure plus a descriptor
rather than a named entity:

> "linear density along the lateral aspect of the right lung"
> "progressive bony structural changes with new cortical irregularities"
> "the hilar contours are normal but become mildly prominent"

No closed lexicon covers that. **An LLM parser is genuinely required here**, and this is precisely
where proposal §3.10's error budget applies.

### A polarity trap the parser surfaced

For most findings, "increase" means worse. For **lung volume and aeration it is inverted**:

> C: "lung volume **decreases** from adequate to severely low aeration" → parsed IMPROVED, is WORSE
> D: "low lung volume **improves** to adequate aeration" → parsed IMPROVED, correct

The six relations are defined over *severity of the abnormality*, so the comparator and the parser
both need **finding-specific polarity** for volume/aeration-type findings. Easy to fix once known,
silently wrong if not.

### Net: the parser error budget covers ~28% of the benchmark, not all of it

| Path | Items | Share | Parser |
|---|---|---|---|
| TEL | 2,994 | 56.4% | regex, 100% exclusive — **no error budget** |
| single-entity GTS (clean) | ~810 | 15.3% | rules, 86.6% exclusive |
| multi-entity GTS + ICR | 1,317 | 24.8% | **LLM required** |
| single-entity GTS (residue) | ~190 | 3.6% | LLM or manual |

**~72% of the benchmark is answerable with deterministic parsing.** That is the number to quote
when a reviewer asks whether an LLM is solving the task.

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
| multi-entity GTS compares across findings | Per-finding posteriors plus an **LLM** parser; rules reach only 24.6% of items |
| ~72% of the benchmark parses deterministically | Quote this when asked whether an LLM is solving the task; the §3.10 error budget covers the other 28% |
| Lung volume / aeration have inverted polarity | Comparator and parser both need finding-specific polarity, or these are silently wrong |
