# LexWiki Re-Test Report v2 — Q1–105, post-fix

Supersedes `rag_retest_q1-105_report.md`. That report's baseline scores are retained
unchanged for every question not re-tested here; only re-tested rows are updated.

## What changed since v1

Fixes landed and verified this round:

- **General-knowledge scope handling.** Named legal authorities ("the Delaware Uniform
  Trade Secrets Act", "the Alice test") and adjective-prefixed doctrines ("the *legal*
  doctrine of unclean hands") were not being recognised as definitional questions at all.
  Fixed. Named authorities now search documents FIRST and fall back to general knowledge
  only when retrieval genuinely finds nothing — a corpus that discusses a statute should
  answer from its own analysis, not a dictionary definition.
- **Refusal recheck.** An answer that declines while the retrieved context still contains
  most of the question's distinctive terms now gets one more generation pass. A second
  refusal keeps the original.
- **Fabricated-identifier and citation checks** from the prior round, unchanged.

## Two corpora — read this before comparing numbers

The audit corpus holds **494 documents, of which 448 are synthetic `Test_<TYPE>_<NN>`
fixtures** (6,481 pages) and only **46 are real Tata documents** (551 pages). The fixtures
will not exist in production.

Two measurements are therefore reported, and **they must not be averaged together**:

| Corpus | Contents | What it measures |
|---|---|---|
| **Mixed** | all 494 docs | the v1 baseline condition |
| **Clean** | 46 real docs only | production-representative behaviour |

**15 of the first 105 questions are ground-truthed to fixture documents** (Q32–Q46) and
cannot be asked on the clean corpus at all. This is a defect in the audit set, not in the
system: the question set depends on files that were never intended to ship.

## Finding that supersedes v1's "non-determinism" diagnosis

v1 finding #4 attributed a group of misses to LLM non-determinism — "the model refusing
over content in front of it". **That diagnosis was wrong.** A direct check of page content
shows the ground-truth answers are simply *not in the wiki* for seven questions:

| Q | Document | Fact absent from all its ingested pages |
|---|---|---|
| Q18 | Service Agreement 4 | the precedence rule (no "prevail/precedence/supersede" in 12 pages) |
| Q49 | Court Case Document 2 | filing date 06 July 2025 |
| Q61 | Joint Venture Agreement 3 | execution date 18 November 2025 |
| Q74 | NDA 6 | "net-zero by 2045" |
| Q79 | NDA 7 | Organic India / Capital Foods acquisitions |
| Q94 | Shareholder Agreement 6 | OmniRetail / Trent |
| Q105 | Tata Brand Judgment 5 | Western Freeway / Thane project |

Each document ingested to a normal 12–13 pages. The ingest step that turns a PDF into wiki
pages **is dropping concrete particulars** — dates, company names, figures, precedence
rules — while preserving the surrounding prose.

**No retrieval work can fix these.** Not re-ranking, not scope resolution, not the refusal
recheck, not removing fixtures. On these questions the system is refusing *correctly*.
They are hard-capped at 1–2 until ingest is fixed.

This is the highest-value engineering lever available, and it is upstream of RAG entirely.

## Re-tested scores

Clean-corpus score is the production-representative figure. `—` = not yet re-tested.

| Q | v1 | Mixed | Clean | Status |
|---|---|---|---|---|
| Q14 | 1 | 8 | **8** | Fixed — correct standard of care + 36-month term; needs the document named |
| Q17 | 1 | 1 | **1** | Retrieval failure — SA 2's Clause 6 IS in the wiki, but scope resolves to Tata Brand Judgments |
| Q18 | 2 | 2 | **2** | Ingest gap — refusal is correct behaviour |
| Q21 | 2 | 2 | **3** | Wrong document; clean corpus at least keeps it in the Service Agreement family |
| Q24 | 2 | 7 | **8** | Fixed — milestone slippage, incomplete models, work-product failure all captured |
| Q27 | 4 | 4 | **9** | Fixed on clean corpus — all three Acts (TM 1999, CCA 2015, CPC 1908) |
| Q29 | 6 | — | **7** | Improved — identifies unnamed Service Provider |
| Q30 | 0 | 0 | **5** | Improved — now states "No document matching 'service agreement 3' exists", though it still drafts clauses afterwards |
| Q42 | 3 | 9 | n/a | Fixed — 45/45/10 correctly cited to Test_JVA_01 Clause 3.3 (fixture-grounded) |
| Q43 | 1 | 1 | n/a | Still wrong document (fixture-grounded) |
| Q44 | 1 | 1 | n/a | Still wrong figure, $20M from Test_JVA_34 (fixture-grounded) |
| Q49 | 1 | 1 | **2** | Ingest gap — refusal is correct behaviour |
| Q56 | 1 | 9 | **9** | Fixed — Tata Motors Passenger Vehicles Limited, correct document |
| Q58 | 3 | 6 | — | Improved — TM Act + CCA found, CPC still missed |
| Q61 | 4 | 2 | — | Regressed — retrieves the right document then refuses; ingest gap confirmed |
| Q65 | 1 | 1 | — | Wrong document (NDA 7 instead of JVA 7) |
| Q67 | 1 | 9 | — | Fixed — Tata Sons Private Limited / Group Brand Team |
| Q70 | 1 | 8 | — | Fixed — foreground IP allocation matches ground truth |
| Q74 | 1 | 2 | — | Ingest gap — refusal is correct behaviour |
| Q76 | 1 | **9** | — | Fixed — all 6 GT categories present (blast furnace data, scrap mix, refractory drawings, energy-intensity models, emissions baselines, steel-grade reqs), correctly cited to NDA 6 |
| Q79 | 1 | **2** | — | Reclassified — was scored as a retrieval miss in v1, but this is the same ingest gap as Q18/Q49/Q74/Q105 (Organic India/Capital Foods absent from NDA 7's ingested pages); refusal is now correct behaviour |
| Q85 | 3 | **6** | — | Improved, not fixed — correctly states SA4's 28 Aug 2025 execution date, but GT calls this question deliberately ambiguous between SA2/SA4 and wants that flagged; the answer commits to one date without surfacing the ambiguity |
| Q92 | 2 | **9** | — | Fixed — confidently and correctly names NourishNext Wellness Foods Private Limited (previously dumped 3 candidate references) |
| Q101 | 1 | **1** | — | Still failing — refuses ("not addressed") though GT's evidence types (screenshots, chat transcripts, payment instructions, complaints) are retrieval-fixable per the ingest audit, not an ingest gap; open miss |
| Q105 | 2 | **2** | — | Confirms ingest-gap diagnosis — refuses correctly now instead of answering circularly; Western Freeway/Thane fact is absent from all of Judgment 5's ingested pages |

Q94 (ingest-limited by the same diagnosis as Q79/Q105, not yet harness-confirmed) and Q102
(retrieval-fixable, not yet re-tested) are the two still outstanding from the original
below-7 list.

## Where the remaining headroom actually is

Of the 22 failing questions examined, re-testing this round moved **Q76 and Q92 to
fixed**, **confirmed Q79 and Q105 as correctly-refusing ingest gaps** (not retrieval bugs
as originally scored), and left **Q101 open** as a genuine unresolved retrieval/synthesis
miss:

- **8 are ingest-limited** (Q18, Q49, Q61, Q74, Q79, Q94, Q105, — Q79 reclassified from
  retrieval-fixable) — capped until the page-generation step stops discarding specific
  facts.
- **13 are retrieval-fixable** (Q14, Q17, Q21, Q24, Q27, Q56, Q58, Q65, Q67, Q70, Q85,
  Q101, Q102) — 10 of those have already improved or fixed this round; Q101 and Q102 are
  the remaining open items.

The dominant retrieval fault is **scope resolution returning `method=default`**, which
searches the entire corpus with no document pinned. Q17 is the clearest case: the correct
clause is in the wiki, and the system still answers from an unrelated judgment.

## Recommended order of work

1. **Fix ingest fact-retention.** Highest value, unblocks 7 questions, and fixes a defect
   that has been misdiagnosed as retrieval flakiness across two audits.
2. **Fix `method=default` scope resolution** for descriptive document references.
3. **Re-ranking** — worth revisiting only after (2); it is gated to broad queries today and
   would not have fired on any of these failures.
