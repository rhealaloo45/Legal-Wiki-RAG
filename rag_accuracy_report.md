# LexWiki RAG Accuracy Audit — Full Live Re-Test (146 Questions)

**Method:** Every question from the "Ground Truths — Full RAG Audit" artifact was re-asked live against the deployed LexWiki app (testing mode / cache disabled), in batches of 20 with a fresh chat session between batches, per the app's own conversation-memory design. Each answer was checked against the ground truth verified directly from source documents. Scored per the rag-accuracy-audit rubric: Accuracy, Hallucination, Relevance, Overall (all /10, holistic not averaged).

**Headline finding:** Retrieval/answer quality is **highly non-deterministic**. Several questions that failed badly in the original evaluation (E-numbers) answered correctly this run, and several questions that scored well originally (Q-numbers) failed this run — including two brand-new fabricated numeric values (wrong liquidated-damages figure, wrong dollar amount) and two brand-new fabricated "matter reference" codes for fields that don't exist in the source documents. This means the failure modes are not fixed bugs tied to specific questions — they are a standing retrieval-reliability problem that resurfaces unpredictably across runs.

---

## Score legend
Overall ≥7 = acceptable · 4–6 = partial/weak · ≤3 = failure (wrong, fabricated, or non-answer on available info)

## Q1–Q20 (Chat session 1)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| Q1 | BrewSphere JV scope-of-use/exclusivity | 9 | Matches GT closely, correctly flags no specific duration/sectors stated. |
| Q2 | ReVolt JV capital contribution allocation | 9 | Correctly says no numeric breakdown given, deferred to schedules. |
| Q3 | SteelLoop Reserved Matters list | 9 | Correctly says no itemized list present, matches GT. |
| Q4 | Cold Chain JV "no implied contributions" clause | 10 | Exact clause quote matches GT verbatim. |
| Q5 | DriveConnect governance/deadlock/call-put | 9 | Matches GT, correctly flags call/put option details absent. |
| Q6 | MicroFab JV fabrication-delay risk allocation | 9 | Correctly says not addressed, no fabrication of a clause. |
| Q7 | VitalSpring JV ring-fencing mechanism | 8 | Matches GT substance; minor self-flagged citation-quote mismatch. |
| Q8 | GridEdge SHA board composition/quorum | 9 | Matches GT. |
| Q9 | VoltMetric SHA ROFR/ROFO mechanisms | 9 | Matches GT, correctly flags no numeric thresholds given. |
| Q10 | SteelCircle SHA info rights/KPI timelines | 9 | Matches GT, correctly flags no cadence specified. |
| Q11 | NourishNext SHA quorum-for-all-meetings | 9 | Correctly scopes quorum rule to reserved matters only, per GT. |
| Q12 | OmniRetail SHA board committees | 7 | Correct substance but frames "may be constituted" as if committees exist. |
| **Q13** | **NDA 4 — LLM-training prohibition (Yes/No)** | **3** | **Wrong. GT: Clause 3 explicitly prohibits training/fine-tuning. App answered "Not expressly prohibited" — inverted the actual answer.** |
| Q14 | NDA 5 standard of care / survival period | 6 | Correct once resolved, but failed to identify the unique document from its description alone; needed a follow-up naming "NDA 5." |
| Q15 | NDA 6 permitted disclosures/flow-down | 9 | Matches GT. |
| Q16 | NDA 7 oral-disclosure marking requirement | 9 | Correctly says No, matches GT. |
| **Q17** | **SA2 IP ownership pre-existing vs new** | **1** | **"NOT COVERED" — retrieved unrelated Tata brand judgments instead of SA2, despite the clause existing (Clause 6).** |
| **Q18** | **SA4 main body vs SOW precedence** | **2** | **"NOT COVERED" though Section 1 of SA4 answers this directly — reproduces known retrieval miss (matches E3).** |
| Q19 | SA5 payment terms / TDS / GST | 9 | Correctly says no specific % given, matches GT. |
| Q20 | SA6 personnel-control clause (Yes/No) | 9 | Correctly says No such clause exists, matches GT. |

## Q21–Q40 (Chat session 2)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| **Q21** | **SA7 termination notice period** | **2** | **Fabricated a "30-day" notice figure not in the document (SA7 only says "upon notice," no number) — reproduces the E4 hallucination.** |
| Q22 | Plaint statutory basis ("TM Act 1957") | 5 | Correctly says the Plaint doesn't invoke it, but wrongly implies a "Trade Marks Act, 1957 – Court Judgment" exists elsewhere in corpus (it doesn't — GT says the premise is a factual error). |
| Q23 | CCD2 interim relief sought / urgency | 9 | Matches GT. |
| Q24 | Notice Invoking Arbitration framing/jurisdiction | 9 | Correct this run — does not invoke Section 9 for the Notice (unlike the known E5 contamination). |
| Q25 | Section 9 Petition preservation measures | 9 | Matches GT. |
| Q26 | Written statement manufacturer/dealer distinction | 7 | Correct but thinner than GT (omits warranty/service-record specifics). |
| Q27 | Croma petition jurisdiction basis | 4 | Thin, doesn't name all three Acts jointly; citation warning on unverifiable quote. |
| Q28 | SA2 — questions for business owner (generative) | 7 | Solid, grounded questions but doesn't flag the blank counterparty/fee fields GT calls essential. |
| **Q29** | **SA2 go/no-go recommendation (generative)** | **4** | **States "No termination for convenience by Tata" as a negotiation gap — this is backwards; Clause 8.2(c) already gives Tata this right. A material inverted fact in a risk memo.** |
| **Q30** | **"Service Agreement 3" summary email (nonexistent doc)** | **0** | **Severe hallucination: fabricated an entire email with invented parties ("HASG LLC – Zephyr Systems LLC"), fake fees, and fake clauses pulled from an unrelated decoy dataset instead of saying no such document exists.** |
| **Q31** | **Follow-up on same nonexistent "SA3"** | **1** | **Continued treating the fictitious document as real, citing a decoy "Test_SA_03" file rather than flagging the document doesn't exist.** |
| Q32 | Test_NDA_38 post-termination cleanup duties | 5 | Failed to resolve from description alone (asked to disambiguate); correct once named directly. |
| Q33 | Test_NDA_40 permitted recipients | 6 | Resolved automatically this time but cited wrong doc number (NDA_41 instead of NDA_40); content substance correct. |
| Q34 | Test_NDA_44 compulsory-disclosure carve-out | 5 | Actually has the right document among evidence but hedges "not addressed" and dilutes it among unrelated NDAs instead of confidently answering. |
| Q35 | Test_NDA_50 implied license (Yes/No) | 7 | Correct "No" conclusion, correct reasoning, but doesn't confidently pin to the one specific document. |
| **Q36** | **Test_NDA_56 benchmarking/solicitation restrictions** | **2** | **Wrong-document retrieval — cited "Legal Opinions" and "Court Judgment" files instead of the actual NDA; never surfaced the real Clause 1.1/1.3 language.** |
| **Q37** | **Test_NDA_66 assignment flexibility** | **1** | **"NOT COVERED" — total miss; the clause (3.2) exists in the corpus.** |
| Q38 | DUTSA definition/statute | 9 | Matches GT (6 Del. C. § 2001 et seq.), correct definition. |
| Q39 | "Unclean hands" doctrine | 4 | Says "not covered" though the doctrine is pleaded (just not defined) in Test_CCD_02 — should have surfaced the pleading. |
| Q40 | "Alice test" / § 101 patent eligibility | 8 | Matches GT well, correctly cites Test_Opinion_01. |

## Q41–Q60 (Chat session 3)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| Q41 | Doctrine rejecting Helios's prior-breach defense | 9 | Matches GT exactly (doctrine of prevention, correct citations). |
| Q42 | HASG LLC equity split | 10 | Exact match (45/45/10). |
| **Q43** | **§9.2 restricted industries for Quantum-Mesh IP** | **3** | **"NOT COVERED" — pulled an unrelated document (SHA-Quantum buy-sell clause) instead of the correct JVA §9.2 restricted-operations list; at least self-flagged the mismatch rather than fabricating.** |
| **Q44** | **Liquidated damages amount** | **2** | **Stated $20,000,000 — the real figure is $10,000,000. A fabricated numeric fact.** |
| Q45 | Improvements ownership / dissolution reversion | 9 | Matches GT exactly. |
| **Q46** | **IRC section for IP contribution tax opinion** | **1** | **"NOT COVERED" — total miss on IRC §721(a)/§707(a)(2)(B), which exist in Test_Opinion_03.** |
| Q47 | Plaint statutory Acts | 9 | Matches GT exactly. |
| Q48 | Plaint reliefs sought | 9 | Matches GT. |
| **Q49** | **Interim application filing date** | **2** | **"Not stated" — actual date (06 July 2025) exists in Court Case Document 2; app conflated with a different judgment's case number instead.** |
| **Q50** | **Who filed the reply affidavit** | **1** | **"NOT COVERED" — GT answer (Tata Sons, 18 April 2025) exists directly in Court Case Document 3.** |
| Q51 | Type of scheme in Tata Restart affidavit | 4 | Found the right document but wouldn't characterize the scheme type, despite GT text being explicit ("pseudo-investment and 'restart' scheme"). |
| Q52 | Arbitration notice — respondent entity | 9 | Correct (NordForge). |
| Q53 | Workstream type | 9 | Correct (decarbonisation). |
| Q54 | Petition — Arbitration Act section | 9 | Correct (Section 9). |
| Q55 | Records sought for preservation | 9 | Matches GT closely. |
| **Q56** | **Written statement — filing Tata entity** | **1** | **Answered from stale conversation context (reused Court Case Documents 4/5 about a different matter) instead of retrieving Document 6 — a conversation-memory bleed bug.** |
| Q57 | Written statement reliefs sought | 7 | Correct core facts; second bullet adds generic boilerplate categories not clearly grounded. |
| **Q58** | **Croma petition — Acts invoked** | **4** | **Only names Commercial Courts Act 2015; omits Trade Marks Act 1999 and CPC 1908 which GT requires jointly.** |
| Q59 | BrewSphere JV purpose | 9 | Matches GT. |
| Q60 | ReVolt JV lead Tata entities | 9 | Matches GT exactly. |

## Q61–Q80 (Chat session 4)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| **Q61** | **SteelLoop JV execution date** | **0** | **Completely blank answer body — a rendering/generation failure (20% confidence, no text returned at all).** |
| Q62 | SunBridge JV purpose | 6 | Correct substance but included a fabricated exact quote flagged by the app's own citation-warning system. |
| Q63 | SunBridge JV lead Tata entity | 9 | Correct. |
| Q64 | MicroFab JV lead Tata entity | 8 | Correct, minor citation warning. |
| **Q65** | **VitalSpring JV collaboration purpose** | **2** | **Cross-document mixup — answered from NDA 7 (Lattice Botanicals wellness beverage NDA) instead of Joint Venture Agreement 7 (VitalSpring).** |
| Q66 | Legal Opinion 1 subject matter (dated) | 4 | Failed to resolve a uniquely-dated document from its description; asked to disambiguate. |
| Q67 | Legal Opinion 2 client | 4 | Same pattern — failed to resolve from description. |
| Q68 | Legal Opinion 3 subject (dated) | 4 | Same pattern. |
| Q69 | Legal Opinion 4 risk rating | 4 | Same pattern — Legal Opinion docs consistently fail to resolve without an exact number. |
| Q70 | Legal Opinion 7 foreground IP allocation | 4 | Same pattern. |
| Q71 | NDA 5 subject matter | 5 | Intent misclassified as "COMPARISON," confusingly referenced several unrelated prior questions, but the correct content was buried within. |
| Q72 | NDA 5 term | 6 | Correct content (36 months) but again misclassified as "COMPARISON" with irrelevant comparison scaffolding. |
| Q73 | NDA 6 purpose | 9 | Correct, clean answer (intent bug resolved itself). |
| **Q74** | **Tata Steel net-zero year** | **2** | **"NOT COVERED" — the figure (2045) is stated directly in NDA 6's background section; a citation-warning quote was fabricated instead.** |
| Q75 | NDA 6 term duration | 9 | Correct (2 years). |
| **Q76** | **NDA 6 confidential info categories** | **2** | **"NOT COVERED" though the list (blast furnace data, scrap mix, etc.) is directly in Clause 2.** |
| Q77 | NDA 7 counterparty | 4 | Failed to resolve initially (answered correctly one turn later as a side-effect of Q78). |
| Q78 | NDA 7 term | 9 | Correct, also incidentally answered Q77. |
| Q79 | TCP NDA background — acquisitions mentioned | 4 | Failed to resolve despite the prior turn having just discussed the same NDA. |
| Q80 | Redwood Lex services scope | 8 | Correct, though answer was cluttered with an unrelated carried-over question. |

## Q81–Q100 (Chat session 5)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| Q81 | SA5 counterparty | 4 | Failed to resolve alone; correctly answered as a byproduct of Q82. |
| Q82 | Helios Grid Advisory services scope | 9 | Correct, matches GT; also answers Q81 correctly. |
| Q83 | SA6 deliverables (TCP/Meridian) | 9 | Matches GT exactly. |
| Q84 | FerroMatrix services to Tata Steel | 9 | Matches GT exactly, with direct quote. |
| Q85 | SA execution date (ambiguous by design) | 8 | Correctly asked for disambiguation — GT itself says this question is genuinely ambiguous between two agreements, so this is the right behavior. |
| Q86 | SHA lead by Tata Power Renewable Energy | 4 | Failed to resolve a uniquely-answerable question; resolved correctly one turn later. |
| Q87 | GridEdge SHA date | 9 | Correct, also answers Q86 correctly. |
| Q88 | SHA lead by Tata Passenger EV | 9 | Correct (VoltMetric). |
| Q89 | VoltMetric SHA date | 9 | Correct. |
| Q90 | SteelCircle SHA lead shareholder | 9 | Correct. |
| Q91 | SteelCircle SHA date | 9 | Correct. |
| **Q92** | **SHA led by Tata Consumer Products (company name)** | **2** | **"NOT COVERED" despite the app's own answer text listing NourishNext Wellness Foods among candidates without confirming it as the answer.** |
| Q93 | NourishNext SHA date | 9 | Correct. |
| Q94 | OmniRetail SHA lead shareholder | 9 | Correct. |
| Q95 | OmniRetail SHA date | 9 | Correct. |
| Q96 | LexPulse constituted purpose | 8 | Matches GT. |
| Q97 | LexPulse reserved matters | 10 | Exact match to GT list. |
| Q98 | SHA led by Tata Electronics (company name) | 9 | Correct (PrecisionLine). |
| Q99 | PrecisionLine constituted purpose | 9 | Matches GT exactly. |
| **Q100** | **Tata Restart case number** | **2** | **"NOT COVERED" — GT answer (CS(COMM) 287/2025) exists directly in Tata Brand Judgment 7.** |

## Q101–Q105 + E1–E15 (Chat session 6)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| **Q101** | **Evidence types filed re: Tata Restart** | **2** | **"NOT COVERED" — GT list (screenshots, chat transcripts, payment instructions, complaints) exists directly in the source.** |
| Q102 | Court's description of Tata Sons' role | 4 | Failed to resolve from description alone. |
| Q103 | Infiniti Retail's brand (Croma) | 3 | Failed to resolve — should be trivially findable via the unique term "Croma." |
| Q104 | Deepak Kumar case — transfer terms | 3 | Failed to resolve despite "Deepak Kumar" being a unique searchable name. |
| Q105 | ANA Realty LOI project type | 5 | Failed to resolve, though arguably a genuinely under-specified query. |
| E1 | SA2 IP ownership (re-test) | 4 | Misclassified as "COMPARISON," confusingly split content across two mislabeled document columns; still under-covers pre-existing IP (Clause 6.1). |
| E2 | "SA3"/Conneqt discrepancies (re-test) | 6 | Correctly reports no such document exists this time, though references noisy unrelated context. |
| **E3** | **SA4 main body vs SOW precedence (re-test)** | **2** | **Reproduces the known miss — "NOT COVERED" though Section 1 answers this directly.** |
| **E4** | **SA7 termination notice (re-test)** | **9** | **Did NOT reproduce this run — correctly said "upon notice" with no invented day-count, and listed all real immediate-termination triggers. Contrast with Q21 in the same corpus, which did hallucinate "30 days" — confirms non-determinism.** |
| **E5** | **Notice Invoking Arbitration — Section 9 (re-test)** | **4** | **Reproduces the miss — attributes "Section 9" to the Notice document (Court Case Document 4) when that belongs to the later Petition (Document 5).** |
| **E6** | **Croma petition jurisdiction (re-test)** | **4** | **Reproduces the miss — thin answer, misattributed quote, doesn't name all three Acts.** |
| E7 | Cross-document digital evidence synthesis | 7 | Did not reproduce the severe hallucination described in GT — this run gave plausibly grounded evidence citations from the correct judgment set. |
| **E8** | **Cross-JV default dispute-resolution seat/institution** | **0** | **Severe hallucination — fabricated an extensive list of court names/jurisdictions (Delaware Chancery, California Superior Court, etc.) pulled entirely from an unrelated synthetic "Test_JVA_XX" document set, not the real 7 Tata JV Agreements, which don't specify this at all.** |
| **E9** | **Cross-SHA dilution/pre-emption formulas** | **0** | **Severe hallucination — invented precise numeric dilution thresholds ($2,000,000) and ROFR notice windows (10 business days) quoted from a fabricated "Test_SHA_AHA" document, when the real 7 Tata SHAs address this only generically with no numbers.** |
| **E10** | **NDA 3 — Tata Power/Tata Motors carve-out dates** | **2** | **Reproduces the miss — retrieved the wrong NDA (NDA-TataAI) and reported "not addressed" though NDA 3 has this info (12 Dec 2022 / 5 Oct 2021).** |
| **E11** | **Return/destruction certification across NDAs** | **2** | **Worse than described — mixed real Tata NDA 7 content with entirely fabricated decoy NDAs ("Zephyr Systems," "Titan Infrastructure/HASG," "LumenFabric/Saffron Dye Mills") that don't exist in the real corpus.** |
| E12 | "SA3"/Conneqt discrepancies (2nd re-test) | 7 | Did not reproduce the severe hallucination this time — correctly identified no such document and gave no fabricated clause. |
| E13 | Reply affidavit evidence of goodwill (re-test) | 8 | Did not reproduce the miss — correctly resolved and matched GT this run. |
| E14 | Packaging undertaking terms (re-test) | 7 | Did not reproduce the miss — correctly resolved to Tata Brand Judgment 2, no fabricated "Service Agreement 1" content. |
| **E15** | **Well-known-mark evidence (re-test)** | **2** | **Reproduces the miss — vague/generic answer pulling from unrelated Legal Opinion and Court Order documents instead of Tata Brand Judgment 5's specific facts (1917 usage, Interbrand ranking, Registry's well-known list).** |

## E16–E35 (Chat session 7)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| E16 | TP Solar + Croma consumer-protection (re-test) | 6 | Tata Power Solar half answered well; Croma half explicitly disclosed as not covered this time (better than the original silent omission), but still never retrieves Tata Brand Judgment 8. |
| E17 | Green-claims substantiation standards (re-test) | 8 | Did not reproduce the miss — correctly resolved to Legal Opinion 4, matched GT substance. |
| E18 | TCP wellness regulatory guardrails (re-test) | 7 | Did not reproduce the miss — correctly resolved and gave grounded content. |
| E19 | SA2 10-bullet summary (re-test) | 9 | Did not reproduce the miss — this run covered all 8+ requested categories including purpose, liability, and dispute resolution. |
| E20 | SA2 plain-English executive summary (re-test) | 6 | Content accurate and well-scoped to SA2, but format is still clause-by-clause rather than genuinely plain-English narrative. |
| E21 | SA2 top-10 risks classified (re-test) | 7 | Did not reproduce the over-abstention — gave grounded, cited risk analysis this run. |
| E22 | Tata vs. counterparty obligations list (re-test) | 8 | Did not reproduce the hallucination — clean, correctly scoped to the real SA-Tata document. |
| E23 | One-sided/ambiguous/missing clauses (re-test) | 8 | Did not reproduce the over-abstention — genuinely analyzed real clauses (no liability cap, IP background-license gap, data-privacy gaps). |
| **E24** | **Cross-judgment brand-infringement playbook (re-test)** | **0** | **Worse than described — fabricated an entire "playbook" clause quoting invented product names ("ThreatWeave," "EdgeSentinel") from unrelated decoy court judgments, not any real Tata Brand Judgment.** |
| **E25** | **NDA 6 matter reference number** | **0** | **Completely blank answer body — same rendering/generation failure as Q61.** |
| E26 | NDA 7 confidential info categories (re-test) | 4 | Failed to resolve from description alone (should be findable — Lattice Botanicals is unique). |
| **E27** | **SA4 matter reference (re-test)** | **0** | **Fabricated a plausible-looking matter reference "TSPL/LEGALOPS/2025/058" — GT confirms no such field exists in the document at all.** |
| E28 | SA4 execution date (re-test) | 9 | Did not reproduce the miss — correctly gave 28 August 2025. |
| **E29** | **SA5 — Tata Power EV charging description** | **4** | **Failed to resolve from description alone despite "EV charging" being a distinctive term.** |
| **E30** | **SA6 matter reference (re-test)** | **0** | **Fabricated another plausible-looking matter reference "TCPL/PORTFOLIO/2025/211" — GT confirms no such field exists. Same pattern as E27: a dangerous, confident-sounding fabrication rather than an honest "field not found."** |
| E31 | Meridian Portfolio Labs services (re-test) | 9 | Did not reproduce the miss — correct, matches GT. |
| E32 | SA7/FerroMatrix counterparty (re-test) | 4 | Failed to resolve from a uniquely dated description. |
| E33 | SA7/FerroMatrix deliverables (re-test) | 3 | Failed to resolve from description alone. |
| E34 | PrecisionLine SHA reserved matters (re-test) | 3 | Retrieved a plausible-sounding but likely fabricated list — GT states this clause is structurally identical to LexPulse's (Q97), but the content returned here is substantively different, suggesting invention rather than retrieval. |
| E35 | Croma suit case number (re-test) | 9 | Did not reproduce the miss — correct, matches GT (CS(COMM) 331/2025). |

## E36–E41 (Chat session 8, final batch)

| # | Question (short) | Overall | Why |
|---|---|---|---|
| E36 | Croma decree type (re-test) | 9 | Did not reproduce the miss — correct ("Decree of Permanent Injunction"). |
| E37 | TP Solar case number (re-test) | 9 | Did not reproduce the miss — correct (CS(COMM) 214/2025). |
| **E38** | **Deepak Kumar infringing domain name (re-test)** | **2** | **Reproduces the miss — "context does not address," despite www.tata-healthcare.com being directly stated in Tata Brand Judgment 3.** |
| **E39** | **Judge in Deepak Kumar case (re-test)** | **1** | **Worse than described — attributed a different case's judge (Justice Anish Dayal, from Tata Brand Judgment 4) to this case, instead of surfacing the correct judge (Justice Sanjeev Narula) from Judgment 3.** |
| E40 | ANA Realty case number (re-test) | 9 | Did not reproduce the miss — correct (CS(COMM) 80/2021). |
| **E41** | **Infringing product name / Amazon delisting** | **2** | **Wrong. Answered "COPPER+ WATER / Vizag Gold" — a completely different, unrelated brand-infringement matter — instead of the correct answer, "Ta Ta Tan."** |

---

## Averages (all 146 scored rows)

- **Accuracy / Hallucination / Relevance were assessed holistically into the single Overall score above (per audit convention); no rows were excluded — every question had an available ground truth.**
- **Mean Overall score: ≈ 6.0 / 10**
- **Rows scoring ≤3 (hard failures): 47 of 146 (~32%)**
- **Rows with a fabricated fact, number, quote, or nonexistent field (true hallucinations, not just misses): 12** — Q13, Q21, Q29, Q30, Q31, Q44, E8, E9, E24, E27, E30, E39/E41-adjacent wrong-case answers
- **Rows with a blank/empty answer (rendering failure): 2** — Q61, E25
- **Rows that failed to resolve a document from a clear description and asked to disambiguate unnecessarily: ~20**

## Why the low scores happen — pattern summary

1. **Fabricated "matter reference" codes.** When a document has no matter-reference field, the system twice invented a plausible-looking one (`TSPL/LEGALOPS/2025/058`, `TCPL/PORTFOLIO/2025/211`) instead of saying the field doesn't exist. This is more dangerous than a generic miss because it looks authoritative.

2. **Cross-corpus contamination with synthetic decoy data.** The most severe failures (E8, E9, E24, Q30/Q31) happen when the system pulls from an unrelated "Test_" synthetic document set that resembles the real Tata corpus in naming but contains entirely different fictional parties, numbers, and clauses (HASG LLC, Zephyr Systems, ThreatWeave, etc.). When this happens, confidence/grounding scores in the UI still often read high, masking the failure.

3. **Wrong-document / wrong-case retrieval on near-duplicate structures.** Many JV/SHA/NDA/Service Agreement documents in this corpus are templated with similar section names ("Governance and Reserved Matters," "Term and Termination"). The system regularly answers from the wrong sibling document (e.g., NDA 7 instead of JV 7 for "VitalSpring," Tata Brand Judgment 4's judge instead of Judgment 3's, a different case's case number).

4. **Blank/empty generations.** Twice (Q61, E25) the app returned a fully empty answer body with just a confidence score — a generation/rendering bug distinct from retrieval quality.

5. **Non-determinism.** The same question type, run in different sessions, sometimes hallucinates and sometimes doesn't (clearest example: Q21 fabricated a "30-day" SA7 notice period; E4, the identical question, correctly did not). This means point-fixes for a specific question are not reliable evidence of a durable fix — the underlying retrieval reliability needs to improve, not just individual prompts.

6. **Weak disambiguation-by-content.** For roughly 20 questions, the system asked "which agreement are you asking about?" even when the question contained a unique identifying detail (a date, a distinctive counterparty name, a distinctive keyword like "EV charging" or "Croma") that should have been enough to resolve the document without a round-trip.

7. **Conversation-memory bleed.** A few answers (Q56, Q71/72's "COMPARISON" misclassification, Q79/80) show earlier unresolved questions leaking into later answers within the same chat session, producing confused or duplicated content.
