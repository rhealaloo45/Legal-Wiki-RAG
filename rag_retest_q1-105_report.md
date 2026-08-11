# LexWiki Re-Test Report — Q1–105 (post-fix, testing mode enabled)

Live re-test of the first 105 ground-truth questions against the deployed app, run **after** this
session's fixes (P5-1, P5-3, 8a, 8b, 8f, 8g). 8 questions per screen batch of 20, new chat session
per batch, "Testing mode: ignore cached answers" confirmed ON for every batch. Scored against the
verified ground truth (source-document-checked, not the system's own prior answers).

## Batch averages

| Batch | Range | Mean | Notes |
|---|---|---|---|
| 1 | Q1–20 | 7.7 | Mostly clean. 2 hard misses (Q14 wrong-doc, Q17 total miss), 1 known persistent miss (Q18/SA4 precedence clause). |
| 2 | Q21–40 | 5.5 | **Disambiguation stuck-loop bug found** (Q32→40, later recovered by naming docs). 3 general-knowledge questions (Q38-40) wrongly inherited a prior document's scope. |
| 3 | Q41–60 | 6.45 | Wrong-document contamination on 2 JVA questions (Q43, Q44 — cites a decoy `Test_JVA_34` instead of the real `Test_JVA_01`). **Q56 reproduces the exact original "conversation-memory bleed" bug.** |
| 4 | Q61–80 | 6.1 | Several confirmed-content-exists-but-refused misses (Q65 wrong doc, Q74/Q76/Q79 all say "not covered" despite the answer being retrieved successfully for adjacent questions in the same session). |
| 5 | Q81–100 | 8.1 | Best batch — mostly named/dated single-document lookups. One stray `</confidence/score>` tag leaked into a visible answer (Q94) — a previously-fixed bug class resurfacing. |
| Q101–105 | — | 5.2 | **Disambiguation stuck-loop again** (Q102-105, recovered by naming judgments explicitly). Q105 answered circularly (restated the question instead of answering it). |

**Overall mean, Q1–105: 6.7 / 10** (703 total points / 105 questions)

## Full per-question scores

| # | Topic | Score | Note |
|---|---|---|---|
| Q1 | BrewSphere JV scope/exclusivity | 9 | Correct, matches GT's hedged non-specificity |
| Q2 | ReVolt JV capital contributions | 9 | Correct, correctly defers to unincluded schedules |
| Q3 | SteelLoop Reserved Matters | 9 | Correct, matches GT's "no itemized list" finding |
| Q4 | Cold Chain JV no-implied-contributions | 10 | Exact quote match |
| Q5 | DriveConnect governance/exit/call-put | 9 | Correct, correctly flags call/put as unspecified |
| Q6 | MicroFab fabrication-delay risk | 7 | Correct core finding, but verbose risk-assessment overreach for a factual question + 1 spurious citation warning |
| Q7 | VitalSpring ring-fencing | 7 | Correct facts, slightly overstated as "express" vs GT's "in passing" framing |
| Q8 | GridEdge board/quorum | 9 | Correct, matches GT |
| Q9 | VoltMetric ROFR/ROFO | 9 | Correct, matches GT |
| Q10 | SteelCircle info rights/KPI | 9 | Correct, matches GT |
| Q11 | NourishNext quorum scope | 9 | Correct, scoped to reserved matters only |
| Q12 | OmniRetail board committees | 9 | Correct, notes permissive language |
| Q13 | NDA4 LLM-training prohibition | 10 | Exact quote match |
| **Q14** | **EV battery NDA std-of-care/survival** | **1** | **Wrong document — answered from NDA 6 (NordForge/green-steel, "two years") instead of NDA 5 (Cirrus/EV battery, "36 months")** |
| Q15 | NordForge NDA permitted recipients | 9 | Correct (embedded in same turn as Q14) |
| Q16 | NDA7 oral-disclosure marking | 9 | Correct, matches GT |
| **Q17** | **SA2 IP ownership work-for-hire** | **1** | **Total miss — "the context does not contain the Services Agreement..." though SA2 is real and easily retrievable** |
| **Q18** | **SA4 main body vs SOW precedence** | **2** | **Reproduces the known persistent miss — still "not covered" though the clause exists** |
| Q19 | SA5 payment terms/TDS/GST | 9 | Correct, matches GT |
| Q20 | SA6 personnel-control clause | 8 | Correct (no clause found), slightly hedged phrasing |
| **Q21** | **SA7 termination clause** | **2** | **Wrong document — answered from SA2 (30-day notice) instead of SA7 (Tata Steel), though disclosed via SCOPE WARNING** |
| Q22 | Plaint TM Act 1957/1999 | 7 | Correctly finds TM Act 1999, correctly says 1957 not invoked, but misses noting the likely Copyright Act 1957 conflation |
| Q23 | CCD2 Order XXXIX reliefs | 9 | Correct, comprehensive |
| **Q24** | **Notice Invoking Arbitration framing** | **2** | **Reproduces known E5 contamination — wrongly claims "Section 9" jurisdiction (that's CCD5's petition, not CCD4's notice)** |
| Q25 | Section 9 Petition interim measures | 9 | Correct, matches GT |
| Q26 | Written statement manufacturer/dealer | 9 | Correct, matches GT closely |
| **Q27** | **Croma petition CCA 2015 jurisdiction** | **4** | **Reproduces known E6 — thin answer, omits TM Act 1999 + CPC joint basis** |
| Q28 | SA2 business-owner questions | 8 | Comprehensive 15-question list, flags blank counterparty, slightly less pointed than GT's ideal |
| Q29 | SA2 go/no-go | 6 | Doesn't flag that Tata's role is explicit (not assumed) or that blank fee/counterparty fields block execution |
| **Q30** | **Fictitious "SA3" risk email** | **0** | **Severe fabrication from decoy corpus (HASG/Zephyr) — known 8c issue, no action per your call** |
| Q31 | Fictitious "SA3" missing annexures | 8 | Correctly recognizes SA3 doesn't exist this time |
| Q32 | Test_NDA_38 cleanup duties | 9 | Correct once resolved by name — matches GT |
| Q33 | Test_NDA_40 permitted recipients | 8 | Correct once resolved, matches GT |
| Q34 | Test_NDA_44 compulsory disclosure | 9 | Correct once resolved, matches GT |
| Q35 | Test_NDA_50 implied license | 9 | Correct once resolved, matches GT exactly |
| Q36 | Test_NDA_56 prohibited uses | 8 | Correct once resolved, matches GT |
| Q37 | Test_NDA_66 assignment flexibility | 9 | Correct once resolved, matches GT |
| **Q38** | **DUTSA definition** | **1** | **Wrongly inherited Test_NDA_66's scope instead of a corpus search — new bug (general-knowledge scope carryover)** |
| **Q39** | **"Unclean hands" doctrine** | **1** | **Same scope-carryover bug as Q38** |
| **Q40** | **"Alice test" § 101** | **1** | **Same scope-carryover bug as Q38/39, 3rd in a row** |
| Q41 | Doctrine of prevention | 9 | Correct, matches GT exactly |
| **Q42** | **HASG equity split** | **3** | **Miss — real answer (45/45/10) exists in Test_JVA_01 but not surfaced confidently** |
| **Q43** | **Section 9.2 restricted industries** | **1** | **Wrong document — retrieved SHA-Quantum instead of the real Test_JVA_01** |
| **Q44** | **Field-of-use liquidated damages** | **1** | **Wrong figure ($20M not $10M) from a decoy document (Test_JVA_34, not Test_JVA_01)** |
| Q45 | Tech License improvements/dissolution | 9 | Correct, matches GT |
| Q46 | Brickhouse IRC section | 6 | Correct IRC sections surfaced but confusingly hedges that "Brickhouse" isn't mentioned |
| Q47 | TPSI plaint acts | 10 | Exact match, all 4 acts |
| Q48 | TPSI plaint reliefs | 9 | Correct, matches GT |
| **Q49** | **TPSI interim application date** | **1** | **Miss — "not provided" though 06 July 2025 is explicit in CCD2** |
| Q50 | Reply affidavit filer | 9 | Correct entity |
| Q51 | Tata Restart scheme type | 8 | Correct, matches GT |
| Q52 | Arbitration notice respondent | 9 | Correct, matches GT |
| Q53 | Workstream type | 9 | Correct, matches GT |
| Q54 | Arbitration Act section | 9 | Correct, matches GT |
| Q55 | Records to preserve | 7 | Correct core 4 items, some possibly-padded extras |
| **Q56** | **Written statement filer** | **1** | **Reproduces the exact original conversation-memory-bleed bug — wrong document (stale NordForge scope) instead of CCD6** |
| Q57 | TMPV written statement reliefs | 9 | Correct, matches GT (self-recovered after Q56's miss) |
| **Q58** | **Croma petition acts** | **3** | **Thin — only cites Commercial Courts Act 2015, misses TM Act 1999 + CPC** |
| Q59 | BrewSphere JV purpose | 9 | Correct, matches GT |
| Q60 | ReVolt JV lead entities | 7 | Correct facts, self-flagged citation warning on a fabricated exact quote |
| **Q61** | **SteelLoop JV execution date** | **4** | **Has the right date (18 Nov 2025) but refuses to state it over an "Effective Date" vs "execution date" technicality** |
| Q62 | SunBridge JV purpose | 9 | Correct, matches GT |
| Q63 | SunBridge lead entity | 9 | Correct, matches GT |
| Q64 | MicroFab lead entity | 7 | Correct fact, citation-integrity warning on a fabricated quote |
| **Q65** | **VitalSpring JV collaboration purpose** | **1** | **Wrong document — answered from NDA 7 (Lattice Botanicals) instead of JVA 7 (VitalSpring)** |
| Q66 | Legal Opinion 1 subject (dated) | 9 | **Resolved directly via the new date-based resolver — confirms 8a fix working** |
| **Q67** | **Legal Opinion 2 client** | **1** | **Miss — disambiguated then still answered "not addressed"; date-resolver doesn't help when no date is in the question** |
| Q68 | Legal Opinion 3 subject (dated) | 9 | **Resolved directly via date — confirms 8a fix working again** |
| Q69 | Legal Opinion 4 risk rating | 9 | Correct, matches GT exactly |
| **Q70** | **Legal Opinion 7 foreground IP** | **1** | **Miss — disambiguated then answered "not covered"; no date/unique keyword to resolve on** |
| Q71 | NDA5 subject matter | 9 | Correct, matches GT (resolved in same turn as Q70's retry) |
| Q72 | NDA5 term | 7 | Correct fact (36 months) but rendered in a confusing single-doc-forced-into-comparison-table format |
| Q73 | NDA6 purpose | 9 | Correct, matches GT |
| **Q74** | **Tata Steel net-zero year** | **1** | **Reproduces known miss — "not stated" though 2045 is explicit in the same paragraph quoted** |
| Q75 | NDA6 term | 9 | Correct, matches GT |
| **Q76** | **NDA6 confidential info categories** | **1** | **Reproduces known miss — "context does not contain the NDA" despite retrieving it successfully moments earlier in the same session** |
| Q77 | NDA7 counterparty | 9 | Correct, matches GT (resolved after 1 disambiguation) |
| Q78 | NDA7 term | 9 | Correct, matches GT |
| **Q79** | **NDA7 background acquisitions** | **1** | **Reproduces known miss — "not covered" despite Organic India/Capital Foods being in the retrieved context** |
| Q80 | Redwood Lex services | 8 | Correct, matches GT (resolved in same turn as Q79's retry) |
| Q81 | SA5 counterparty | 9 | Correct, matches GT |
| Q82 | Helios support services | 9 | Correct, matches GT |
| Q83 | SA6/Meridian deliverables | 9 | Correct, matches GT plus real extra items |
| Q84 | FerroMatrix services | 9 | Correct, matches GT exactly |
| Q85 | SA execution date (ambiguous) | 3 | GT flags this as genuinely ambiguous; system disambiguated 3x then gave up with "not covered" instead of surfacing both candidate dates |
| Q86 | GridEdge company name | 9 | Correct, matches GT |
| Q87 | GridEdge SHA date | 9 | Correct, matches GT |
| Q88 | VoltMetric company name | 9 | Correct, matches GT |
| Q89 | VoltMetric SHA date | 9 | Correct, matches GT |
| Q90 | SteelCircle lead shareholder | 9 | Correct, matches GT |
| Q91 | SteelCircle SHA date | 9 | Correct, matches GT |
| **Q92** | **SHA led by Tata Consumer Products** | **2** | **Reproduces known miss — dumps 3 candidate references instead of confidently naming NourishNext** |
| Q93 | NourishNext SHA date | 8 | Correct, matches GT, minor citation warning |
| **Q94** | **OmniRetail lead shareholder** | **6** | **Correct content but a stray `</confidence/score>` tag leaked into the visible answer — regression of a previously-fixed bug class** |
| Q95 | OmniRetail SHA date | 9 | Correct, matches GT |
| Q96 | LexPulse constituted purpose | 8 | Correct, matches GT |
| Q97 | LexPulse reserved matters | 10 | Exact match, all 6 items |
| Q98 | PrecisionLine company name | 9 | Correct, matches GT |
| Q99 | PrecisionLine constituted purpose | 9 | Correct, matches GT exactly |
| Q100 | Tata Restart case number | 8 | Correct (CS(COMM) 287/2025), but citation quote doesn't actually contain the number |
| **Q101** | **Tata Restart evidence types** | **1** | **Reproduces known miss — "not covered" though screenshots/transcripts/payment instructions/complaints are in the source** |
| Q102 | Tata Sons' role (Court description) | 5 | Partial — surfaces "principal investment holding company" but misses the fuller "trust-led group... especially pernicious" characterization |
| Q103 | Infiniti Retail's brand (Croma) | 9 | Correct once resolved by name |
| Q104 | Deepak Kumar transfer terms | 9 | Correct once resolved by name, matches GT exactly |
| **Q105** | **ANA Realty LOI project type** | **2** | **Circular — restates the question ("entered into a Letter of Intent for design consultancy services") instead of naming the actual project (a real-estate development at Western Freeway, Thane)** |

## Key findings / where improvement is needed

1. **Disambiguation stuck-loop regression (critical).** Hit twice in this run — once for a chain of purely-descriptive NDA questions (Q32–40) and once for a chain of judgment-only questions (Q102–105). Once disambiguation fires and the follow-up doesn't happen to contain a resolvable name/number, every subsequent question in the chain gets stuck re-asking "which agreement?" forever, with the frontend endlessly re-appending the growing question history. This is the same failure shape as the Phase-2 "disambiguation death spiral" bug documented as fixed earlier in this project — it's back for at least these two trigger shapes. **Needs investigation before anything else on this list.**

2. **General-legal-knowledge questions wrongly inherit document scope (new).** Three consecutive questions (Q38–40: DUTSA, unclean hands, Alice test) that should trigger a full corpus/general-knowledge search instead silently inherited the previous turn's NDA scope and answered "not addressed in the provided context" from the wrong document. This is the reverse-direction case of the already-fixed "general-knowledge breaks carryover" bug — carrying a document scope *into* a general-knowledge question was apparently never guarded.

3. **Wrong-document contamination remains common**, independent of the decoy-corpus issue (8c, already decided out of scope): Q14, Q21, Q24, Q43, Q44, Q56, Q65 all answered confidently from the wrong sibling document. This is the same root cause flagged as 8e (no re-ranker) — still on hold per your call, but this run shows it's not a rare edge case, it's a regular occurrence across unrelated document families (NDAs, JVAs, Service Agreements, Court Case Documents all affected).

4. **"Confirmed-retrievable-elsewhere-but-still-refused" misses** (Q17, Q74, Q76, Q79, Q101): the system successfully retrieves and cites a document for one question, then minutes later claims the same document "is not in the provided context" or that a fact "is not stated" when it's in the same paragraph already quoted. This looks like genuine LLM non-determinism on the retrieval/synthesis step rather than a deterministic bug — same class as the model-non-determinism documented elsewhere in this project, but the rate here (5 of 105) is high enough to be worth a closer look.

5. **A stray tag leak reappeared** (Q94: `</confidence/score>` rendered directly in the visible answer) — this bug class was previously fixed; this is either a regression or an edge case the original fix didn't cover.

6. **8a (date-based resolver) confirmed working**: Q66 and Q68 (Legal Opinions named only by their date, no document number) resolved directly with no disambiguation prompt — this is the fix from earlier in this session working as intended. Its limitation is also visible: Q67 and Q70 (Legal Opinions with no date and no single distinctive keyword in the question) still fail to resolve, exactly as scoped when the fix was built.

E1–E41 not yet re-tested — will follow if requested.
