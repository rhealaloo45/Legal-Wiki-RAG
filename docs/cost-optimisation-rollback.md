# Cost optimisation — what changed, and how to undo it

Written September 2026, after a measurement pass over 1,720 real Ask-pipeline
queries found that **36.7% of prompt tokens were spent checking the answer
rather than producing it**.

**Status: both changes were REVERTED on 2 September 2026**, along with the
switch of the fast deployment to `gpt-5-nano-1`. A 21-case graded regression
run afterwards failed 16 of 21, and although most of those failures were
traced to the grader rather than the product, the quality signal was no longer
trustworthy enough to keep an unproven saving. Cost work is deferred, not
abandoned.

This document therefore reads in both directions: it records what the two
changes did and why they were safe, so they can be re-applied deliberately
rather than reconstructed from memory.

---

## The measurement that motivated it

Taken from `query_traces`, 1,720 queries that reached the answer pipeline:

| Call | Calls | Prompt tokens | Share | Avg |
|---|---|---|---|---|
| generate answer | 1,720 | 24,087,813 | 62.0% | 14,004 |
| grounding judge | 888 | 7,290,481 | 18.8% | 8,210 |
| citation retry | 460 | 6,985,839 | 18.0% | 15,186 |
| classify intent | 1,707 | 477,712 | 1.2% | 279 |

Per query: median 19,764, mean 22,593, p90 37,197, max 85,667.

The grounding judge fired on 56% of queries and **found nothing wrong in 92.2%
of the calls it was paid for** — 97.6% of its verdicts scored 90 or above.

---

## Change 1 — structured-block deduplication

**Commit:** `fa55e4b` · **File:** `services/wiki.py`, `get_context()`

### What it does

The structured-extraction block injects full clause text alongside the wiki
pages. Measured across 1,023 clauses on this corpus, **70.2% of that text was
already present word for word in the page prose shipping in the same prompt** —
the model was charged twice for the same sentence, up to 20,000 characters of
it per query.

Clauses whose text already appears in the page content are now skipped.

### Safety properties (both verified against live data)

1. **Compared only against each page's guaranteed-surviving 2,000-character
   prefix**, never its full text. Per-page truncation caps a page at
   `MAX_PAGE_CONTEXT_CHARS`; matching beyond that could drop a clause whose only
   other copy was then truncated away.
2. **The highest-ranked clause per document is always kept**, regardless of
   duplication, so every in-scope document still announces itself in the block.

Verified on 1,601 clauses: 1,080 deduplicated, **0 dropped that were not
already present verbatim in surviving page text**.

### Current state and how to re-apply

Reverted in `3160978`. To bring it back:

```bash
git revert 3160978
```

Or by hand — in `get_context()`, restore the clause loop to append every row
unconditionally:

```python
for _ct, _vt in sorted(
        _clause_rows,
        key=lambda r: -(_q_overlap(r[0]) * 2 + _q_overlap((r[1] or "")[:400]))):
    if _vt:
        _doc_lines.append(f'  - {_ct}: "{_vt.strip()[:500]}"')
```

and delete `_page_prefix_index`, `_norm_ws`, and the `_struct_deduped` /
`_struct_saved_chars` counters.

---

## Change 2 — grounding judge receives only cited context

**Commit:** `e689d21` · **Files:** `services/wiki.py` (`cited_context`),
`services/intent_agent.py` (`_check_grounding_hybrid`)

### What it does

The judge was sent the entire assembled context. Its actual job is comparing an
answer's claims against the passages those claims cite, and an answer's
References block names both the source filename and the page title of
everything it relies on — so the cited subset can be identified exactly rather
than guessed at.

`wiki.cited_context(context, answer)` keeps only page blocks whose title or
`[From: …]` label appears in the answer, plus the structured-extraction block
whole.

### Safety properties (unit-tested)

Returns the **full context unchanged** when:

- no `## Title` page blocks parse,
- nothing matches the answer,
- or the reduction would leave under 15% of the original context.

The structured-extraction block is always kept **with its header** (`group(0)`,
not `group(1)`) — it holds the verbatim clause text a quote is most often
checked against, and dropping it would turn correct verbatim quotes into false
ungrounded flags.

### Why this one is low-risk by construction

The grounding judge runs **after** generation and only produces a score. It
cannot change answer text. Reverting it cannot improve accuracy — it can only
change the displayed grounding number.

### Current state and how to re-apply

Reverted in `87a5908`. To bring it back:

```bash
git revert 87a5908
```

Or by hand — in `intent_agent._check_grounding_hybrid`, restore:

```python
llm_result = _check_grounding(question, context, answer, intent=intent)
```

(dropping the `_judge_ctx = wiki.cited_context(context, answer)` line). The
`cited_context` function can stay; unused it costs nothing.

---

## Change 3 — citation-retry instrumentation

**Commit:** `e689d21` · **Files:** `services/tracing.py`, `services/wiki.py`

Purely additive. Records into each trace which of the three checks triggered
the retry (`unverified_quotes`, `misattributed`, `unverified_identifiers`), the
outcome (`kept` / `discarded-shorter` / `discarded-no-body` /
`discarded-no-improvement`), and the before/after flag counts.

No behaviour change. It was reverted only because it shared a commit with
change 2; re-applying `87a5908` restores it. It is the data needed to decide
whether the 27% retry rate is reducible, so it is the first thing to bring
back when cost work resumes.

---

## Rollback points

| Commit | State |
|---|---|
| `3160978` | **current** — both changes reverted, `gpt-5-mini-1` on both deployments |
| `87a5908` | judge-context change reverted, dedup still live |
| `e689d21` | both changes live |
| `fa55e4b` | dedup only, before judge context + instrumentation |
| `ff95697` | before all cost work — 21 seed cases, no optimisation |

## The fast-deployment change, also reverted

`AZURE_FAST_DEPLOYMENT` was moved from `gpt-5-mini-1` to `gpt-5-nano-1` at the
same time. It is back on `gpt-5-mini-1`.

Nano could not reliably emit parseable JSON: five of the twenty-one graded
cases failed with `judge returned no JSON`, and the grader runs on the fast
deployment. Anything that moves work onto a smaller model has to be measured
against the graders first, because a broken grader makes every later change
unmeasurable — which is exactly what happened.

---

## What was NOT changed, and why

- **Page count / `HYBRID_FUSION_TOP_K` (23)** — left alone. Pages are the
  bulk of the generate prompt, but cutting them trades directly against recall,
  which is what holds accuracy. This is the riskiest lever, not the first one.
- **Citation retry itself** — still a full regeneration. Its 27% fire rate is
  the largest remaining target, but the cause is unknown until the
  instrumentation above has collected real data.
- **LLM reranker** — still off. It ranks `title: summary`, i.e. metadata rather
  than passage text, so enabling it would add cost without addressing the
  precision problem it looks like it should solve.

---

## An attempted measurement that failed, recorded so it is not repeated

Trying to determine *why* the citation retry fires, the context of past queries
was reconstructed from their traces and the three checks re-run. **This does
not work**: traces store the cleaned *display* title, not the raw page title, so
the context cannot be faithfully rebuilt.

The reconstruction suggested a 68% flag rate. One flagged quote was then checked
directly against source — `"During each Contract Year, Apex Zephyra Trading
Company LLC shall purchase … not less than 206,696 units"` — and proved to be a
plain substring match. The flags were the reconstruction's fault, not the
checker's.

Hence the forward instrumentation rather than a retrospective number.
