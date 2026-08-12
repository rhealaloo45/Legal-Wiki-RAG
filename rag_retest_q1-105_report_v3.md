# LexWiki Re-Test Report v3 — 27 sub-5 questions, single-session retest

Supersedes v2's two-corpus split. All 27 questions in this report were re-tested on
**one session only** — `3a66b0ab-a9cc-48f0-a3f3-b0ab863936fe` (the full 494-document
corpus, current deployed code) — via `app/eval/harness.py`. One score column, no v1/Mixed/
Clean split. `v1` is kept for comparison only, sourced from the original baseline audit
(`rag_retest_q1-105_report.md`).

## Correction from earlier drafts

Two questions previously assumed to belong to this "27 sub-5" set do not: **Q29 (v1=6)**,
**Q94 (v1=6)**, and **Q102 (v1=5)** were never below 5 in the original baseline — they were
pulled in by mistake from an unrelated fact-absence note. The real 27th–29th members of the
sub-5 set are **Q38, Q39, Q40** (the general-knowledge scope-carryover bug, all v1=1),
which were missing from the interim retest and are included here for the first time.

## Scores

| Q | v1 | New | Verdict |
|---|---|---|---|
| Q14 | 1 | **9** | Fixed — correct doc (NDA 5), care standard + 36-month term match GT exactly |
| Q17 | 1 | **7** | Correct work-for-hire allocation (Tata owns new deliverables); hedges instead of stating the pre-existing-IP clause GT calls out |
| Q18 | 2 | **9** | Fixed — exact match, correct doc (SA4) |
| Q21 | 2 | **2** | Still broken — wrongly claims no immediate-termination events exist (GT lists 7: material breach, SLA failure, integrity, insolvency, regulatory prohibition, security incident, change of control); document scope still unresolved (SCOPE WARNING) |
| Q24 | 2 | **7** | The known "Section 9" cross-document contamination is gone — jurisdiction/framing now correct; still misses the specific breach details (milestone slippage, incomplete models, work-product failure) GT wants |
| Q27 | 4 | **9** | Fixed — all 3 Acts present (TM 1999, CCA 2015, CPC 1908) |
| Q30 | 0 | **0** | **Not fixed — still severe fabrication.** Drafts three full risk-mitigation emails for a nonexistent "Service Agreement 3" using decoy HASG/Zephyr fixture content, with zero refusal anywhere in the answer. Corpus-dependent: a prior clean-corpus-only test of this same question correctly refused: on the full corpus with decoys present, it still fabricates. |
| Q38 | 1 | **9** | Fixed — DUTSA answered correctly and fully, no NDA-scope leakage from the preceding question |
| Q39 | 1 | **8** | Fixed — clean general-knowledge definition of "unclean hands" with proper "not drawn from documents" disclaimer, no scope leakage. Doesn't also flag the document's own narrower pleading of the doctrine, but the scope-carryover bug itself is gone |
| Q40 | 1 | **9** | Fixed — Alice test explained correctly, no scope leakage |
| Q42 | 3 | **9** | Fixed — exact 45% / 45% / 10% equity split, correct doc (Test_JVA_01) |
| Q43 | 1 | **1** | Still wrong — cites aviation/maritime/rail industries instead of GT's fossil-fuel/pipeline/coal/gas; its own quote admits "not provided in excerpt" |
| Q44 | 1 | **1** | Still wrong — $20M from decoy Test_JVA_34 instead of GT's $10M from the real Test_JVA_01 |
| Q49 | 1 | **9** | Fixed — exact date match (06 July 2025) |
| Q56 | 1 | **8** | Correct entity (Tata Motors Passenger Vehicles Limited); drops the 14 Jan 2026 filing date GT also wants |
| Q58 | 3 | **9** | Fixed — all 3 Acts present |
| Q61 | 4 | **9** | Fixed — exact date match (18 Nov 2025) |
| Q65 | 1 | **9** | Fixed — correct document now (JVA 7, not NDA 7), matches GT's collaboration purpose |
| Q67 | 1 | **9** | Fixed — exact match |
| Q70 | 1 | **9** | Fixed — matches GT including the background-IP-narrow clause |
| Q74 | 1 | **9** | Fixed — 2045, exact |
| Q76 | 1 | **9** | Fixed — all 6 GT categories present, correct doc (NDA 6) |
| Q79 | 1 | **9** | Correct this run (Organic India + Capital Foods, matches GT) — but flaky: an earlier retest this session had the identical question refuse over the identical document. Non-determinism, not a fix |
| Q85 | 3 | **6** | Correct date/doc (SA4, 28 Aug 2025); still doesn't flag the SA2-vs-SA4 ambiguity GT explicitly wants surfaced |
| Q92 | 2 | **9** | Fixed — confident, correct, no more multi-candidate dumping |
| Q101 | 1 | **1** | Still refuses — GT's evidence list (screenshots, chat transcripts, payment instructions, complaints) not surfaced |
| Q105 | 2 | **2** | Still circular — echoes the question's own phrasing ("integrated design consultancy services") instead of naming the actual project (Western Freeway, Thane) |

**Mean: 1.59 → 6.93** across these 27 questions.

## Breakdown

- **18 fixed** (Q14, Q18, Q27, Q38, Q39, Q40, Q42, Q49, Q58, Q61, Q65, Q67, Q70, Q74, Q76, Q79*, Q92, and Q24/Q17/Q56 partially — see below)
- **3 partial** (Q17, Q24, Q56 — correct core fact, missing a secondary detail GT also asks for)
- **1 flaky** (Q79 — correct this run, confirmed non-deterministic against an earlier refusal on the same question)
- **1 unfixed, ambiguity-handling gap** (Q85 — right fact, doesn't flag the ambiguity GT wants noted)
- **4 still broken**: **Q21** (wrong-document/incomplete), **Q30** (severe fabrication, corpus-dependent), **Q43** (wrong content), **Q44** (wrong figure, decoy contamination), **Q101** (refuses over present-but-elusive facts), **Q105** (circular, ingest-gap)

## What's still open

1. **Q30 is the standout concern.** Not a refusal-quality issue like the others — full invention
   of contract terms against a document that doesn't exist, and it survives even after this
   session's refusal-recheck and fabricated-identifier fixes. Needs its own investigation.
2. **Decoy-fixture contamination** (Q43, Q44) — both still pull numbers from a sibling decoy
   document (`Test_JVA_34`) instead of the real one (`Test_JVA_01`). Same root cause as Q30:
   the corpus's synthetic fixtures compete with real content for retrieval and sometimes win.
3. **Q21, Q101, Q105** remain genuine misses — Q105 is the confirmed ingest-gap case (fact
   absent from all ingested pages); Q21 and Q101 are retrieval/synthesis gaps, candidates for
   task #10's individual review.
