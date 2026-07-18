# G7 — human-written calibration subset

*Written: 2026-07-17 by claude:p4-review. Antagonistic review 2026-07-14
flagged the generator confound (all RCP calibration pairs are LLM-generated
across 14 diverse models — the 0.97 alignment could partly reflect
convergence on how LLMs write contrastive pairs, not on the underlying
concept) as P4's biggest exposed reviewer flank. G7 is the reviewer's own
suggested strengthener: a small human-written/benchmark-mined contrastive
set, spot-checked against the LLM-generated primary corpus. Gated on
C12–C14 (`tb98f327`, cleared 2026-07-16 — see `ROUND3_COMPUTE_PLAN.md`),
now unblocked.*

## What this is

Three concepts, ~220–250 contrastive "pairs" per concept (unpaired
positive/negative pools, not per-pair topic-matched the way RCP is — see
"Design difference from RCP" below), sourced from real, pre-existing,
human-authored text — zero LLM generation anywhere in this pipeline.

| Concept | Source | License | N (pos/neg) |
|---|---|---|---|
| sentiment | SST-5 (`SetFit/sst5` on HF; original Stanford Sentiment Treebank, Rotten Tomatoes reviews) | Standard NLP research benchmark, freely redistributable sentence-level data (widely used and cited without restriction; HF card lists license as unstated/unknown but this is normal for this dataset and not a redistribution blocker) | 250 / 250 |
| negation | Project Gutenberg #1661, *The Adventures of Sherlock Holmes* (Arthur Conan Doyle) | **Public domain** (pre-1930 US publication) | 250 / 250 |
| temporal_order | English Wikipedia, 40+ historical/biographical articles | **CC BY-SA 4.0** | 220 / 220 |

## Two parked decisions from `ROUND3_COMPUTE_PLAN.md`, resolved

**TimeBank licensing.** The original plan named TimeBank (LDC2006T08) for
temporal_order. Checked: TimeBank is LDC-distributed, and LDC corpora are
licensed, not freely redistributable — using it would mean the public
audit-trail repo (S1/S3's reproducibility promise) couldn't actually
include the source text, and readers without LDC access couldn't
reproduce. `NarrativeTime` (MIT-licensed) looked like an escape hatch but
re-annotates TimeBank-Dense — the underlying article text is still LDC's,
only the new relation labels are MIT. **Swapped to English Wikipedia**
instead: unambiguously CC BY-SA (freely redistributable), human-written,
richly temporal in historical/biographical articles, and — per
`MAVEN-ERE` (Wang et al., EMNLP 2022, CC0 metadata) — an already-established
academic source for large-scale temporal-relation research, so this isn't
an improvised substitution.

*SEM/ConanDoyleNeg for negation had the same shape of problem in reverse —
the *SEM 2012 shared-task release isn't trivially downloadable and its own
license terms are unclear — but its source text (Conan Doyle's Sherlock
Holmes stories) is unambiguously public domain regardless. Went straight
to the Gutenberg source rather than chase the *SEM annotation file.

**Matched-LLM-domain-control — NOT resolved, needs a decision + real work.**
The clean design: for each concept, also generate a small LLM-authored set
in the *same register* as the human source (movie-review-style sentiment,
Wikipedia-narrative-style temporal_order, Victorian-detective-narrative-style
negation) using RCP's existing 14-generator protocol, so "human vs. LLM"
is the only thing varying — domain/register held constant. Without this,
a result showing weaker alignment on the human set can't distinguish
"because it's human-written" from "because movie reviews/Wikipedia/Conan
Doyle are a different register than RCP's original prompts." **This needs
the RCP generation pipeline (14 LLM APIs) run against three new prompt
templates — not available in this session; someone with that pipeline
access needs to either run it or explicitly decide to skip the control and
disclose the confound instead.**

## Design difference from RCP — read before analyzing

RCP pairs share a topic (positive/negative exemplar on the same subject,
differing only in the target concept). This set does **not** — SST
sentences, Doyle paragraphs, and Wikipedia passages are independently
selected as belonging to the positive or negative pool, not topically
paired one-to-one. This is inherent to sourcing from found text rather
than generating matched pairs, and it's a real methodological difference
worth stating plainly in whatever P4/P3 text cites this, not glossing
over. It's closer to "two labeled corpora" than "250 pairs."

## Classification caveats — lexical-marker heuristics, not human-validated

Negation and temporal_order labels were assigned by **lexical marker
presence**, not manual annotation:
- Negation-positive = contains a negation marker (not/never/no/none/
  nothing/nobody/nowhere/neither/nor/without/n't); negation-negative =
  none present. **Known false-positive risk**: idiomatic uses like "no
  doubt" (≈ "certainly") don't semantically negate the clause — spot-check
  before trusting individual examples.
- Temporal_order-positive = 2+ explicit sequencing connectives
  (after/before/then/subsequently/following/since/until/meanwhile/etc.);
  temporal_order-negative = none present *and* no bare year/century
  pattern (to avoid mislabeling timeline-style date-only passages as
  "no temporal structure").

Sentiment (SST-5) has real human-annotated labels (very positive/positive
collapsed to positive, very negative/negative collapsed to negative,
neutral dropped) — no heuristic involved there.

**Recommend a C13-style human-validation pass** (20% sample, same
protocol as RCP's own corpus-quality check) on the negation and
temporal_order sets before treating alignment results on them as
conclusive, given the lexical-marker construction. Not done in this
session — flagged as a prerequisite, not a nice-to-have, given how much
weight this result may carry as the generator-confound response.

## Files

- `g7_sentiment_pairs.jsonl` (500 rows, 250 pos + 250 neg)
- `g7_negation_pairs.jsonl` (500 rows, 250 pos + 250 neg)
- `g7_temporal_order_pairs.jsonl` (440 rows, 220 pos + 220 neg — short of
  250 because English-language Wikipedia passages meeting both the
  length window and the 2+-connective bar are finite in the ~70-article
  pool pulled; expandable by pulling more historical/biographical
  articles if a full N=250 is wanted)

Schema: `{"pair_id", "label" (1=positive class, 0=negative class),
"domain", "source", "text", "concept"}` — same shape as RCP's own
`consensus_pairs.jsonl` files minus the `model_name`/generator field
(replaced by `source`, since there's no generator here).

## Extraction

`extract_g7.py` in this directory — same conventions as
`round3_gpu/g5_random_text_null.py` (checkpointed, per-model HF upload,
`common.py`/`forward_utils.py` reuse), adapted for this jsonl schema
instead of `rosetta_tools.dataset.load_concept_pairs`. Not yet run —
needs GPU; the round-3 host is occupied with the exfiltration rerun as of
this writing. Queue as a piggyback job once idle time is available, same
pattern as G5b.
