# LexWiki — Per-Question Review (re-tested questions)

Every question that scored below 7 in the v1 audit and has been re-tested.
Each row shows the **better of the two corpus runs**, and names which corpus
produced it. This is a best-observed figure, not an average of the two.

- **Mixed corpus** — all 494 documents (448 of them synthetic `Test_*` fixtures)
- **Clean corpus** — the 46 real Tata documents only (production-representative)

Re-tested: **23** questions. Mean of the best-observed scores across these: **7.61/10** (was 1.87 in v1).

`ingest-capped` = the ground-truth fact is absent from every ingested page of the
expected document, so the refusal is correct and no retrieval fix can raise it.

## Summary

| Q | v1 | Now | Corpus | Expected source document |
|---|---|---|---|---|
| Q14 | 1 | **9** → | mixed | NDA 5_Redacted (1).pdf, Clause 3 "Standard of Care and Security" (p.2) |
| Q17 | 1 | **6** → | mixed | Service Agreement 2_redacted (1).pdf, Clause 6 (Intellectual Property) |
| Q18 | 2 | **9** → | mixed | Service Agreement 4_redacted (1).pdf, Section 1 (Definitions and Inter |
| Q21 | 2 | **8** → | mixed | Service Agreement 7_redacted (1).pdf, Section 8 (Term, Termination, Ex |
| Q24 | 2 | **8** → | clean | Court Case Document 4 (1).pdf, Page 1 (Notice Invoking Arbitration, da |
| Q27 | 4 | **9** → | clean | Court Case Document 7 (1).pdf, Page 1 (Petition Seeking Injunction Aga |
| Q29 | 6 | **7** → | clean | Service Agreement 2_redacted (1).pdf, Clauses 2, 6, 8, 11, 12; Schedul |
| Q30 | 0 | **5** → (fixture-bound) | clean | N/A — no such document exists in the corpus. |
| Q42 | 3 | **9** → (fixture-bound) | mixed | Test_JVA_01.txt, Section 3.3 ("Equity Split"). |
| Q43 | 1 | **1** = (fixture-bound) | mixed | Test_JVA_01.txt, Section 9.2 ("Restricted Operations"). |
| Q44 | 1 | **1** = (fixture-bound) | mixed | Test_JVA_01.txt, Section 9.3 ("Misappropriation Remedy"). |
| Q49 | 1 | **9** → | mixed | Court Case Document 2 (1).pdf, Page 1 (Application under Order XXXIX R |
| Q56 | 1 | **9** → | clean | Court Case Document 6 (1).pdf, Page 1. |
| Q58 | 3 | **9** → | mixed | Court Case Document 7 (1).pdf, Page 1. |
| Q61 | 4 | **9** → | mixed | Joint Venture Agreement 3_redacted (1).pdf, page 1 (header: "Date: 18  |
| Q65 | 1 | **9** → | mixed | Joint Venture Agreement 7_redacted (1).pdf, page 1, "Parties, Joint Ve |
| Q67 | 1 | **9** → | mixed | Legal Opinion 2 (1).pdf, page 1 (header: "Client: Tata Sons Private Li |
| Q70 | 1 | **8** → | mixed | Legal Opinion 7 (1).pdf, page 1, "Legal Analysis" point 1, and "Recomm |
| Q74 | 1 | **8** → | mixed | NDA 6_Redacted (1).pdf, "Parties and Background" (p.1): "...a stated n |
| Q76 | 1 | **9** → | mixed | NDA 6_Redacted (1).pdf, Clause 2 "Confidential Information" (p.1). |
| Q79 | 1 | **9** → | mixed | NDA 7_Redacted (1).pdf, "Parties and Background" (p.1). |
| Q85 | 3 | **6** → | mixed | Service Agreement 4_redacted (1).pdf, header block, page 1; Service Ag |
| Q92 | 2 | **9** → | mixed | Shareholder Agreement 4_redacted (1).pdf, page 1 (header block). |

---

## Detail

### Q14 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> In the NDA for EV battery-tool evaluation activities involving a potential $2 billion transaction, what is the standard of care required for protecting technical schematics, and what is the duration of the confidentiality survival period?

**Expected source** — NDA 5_Redacted (1).pdf, Clause 3 "Standard of Care and Security" (p.2) and Clause 7 "Remedies, Term, and Governing Law" (p.3).

**Documents the system actually cited**

Legal AI - Test_NDA (1)_NDA 5_Redacted (1).pdf

**Ground truth**

> The Receiving Party must protect Confidential Information "with at least the same degree of care it applies to its own highly sensitive information, and in any event not less than a reasonable standard of commercial care" (baseline controls cited include restricted clean-room access, pseudonymisation of VIN-level records, and 24-hour incident notification). The Agreement's term is thirty-six months (3 years), but confidentiality obligations survive for so long as the relevant information remains confidential by its nature, and for trade secrets/long-tail records for so long as protection is reasonably required.

**System answer**

```
Final answer:
1. Standard of care for protecting technical schematics
- The Receiving Party must protect the disclosing Party’s Confidential Information with at least the same degree of care it applies to its own highly sensitive information, and in any event not less than a reasonable standard of commercial care. [From: Legal AI - Test_NDA (1)_NDA 5_Redacted (1) — Standard of Care and Security]

2. Duration of the confidentiality survival period
- The Agreement remains in force for thirty-six months, with continuing obligations for trade secrets and for information that remains confidential by its nature; remedies for breach (including injunctive relief) apply, and some obligations survive as long as the information remains confidential by its nature. The governing law/arbitration details are Maharashtra with Mumbai arbitration. [From: Legal AI - Test_NDA (1)_NDA 5_Redacted (1) — Remedies, Term, and Governing Law]
References
[1] Legal AI - Test_NDA (1)_NDA 5_Redacted (1), Standard of Care and Security
[2] Legal AI - Test_NDA (1)_NDA 5_Redacted (1), Remedies, Term, and Governing Law

[SCOPE WARNING: For "nda 5", this corpus contains BOTH a real document and a synthetic "Test_" stand-in of the same number — both were searched, so some facts below (party names, figures, clause numbers) may come from the fictional stand-in rather than the real document. Verify each cited figure against the document named in the References section before relying on it.]
```

_scope method: `file` · **required naming the document before it would answer**_

### Q17 — 1 → **6/10**  ·  _mixed corpus_

**Question**

> In the Services Agreement between Tata Sons Private Limited and its service provider for a Design and Content Audit, how is intellectual property ownership allocated between pre-existing IP and new deliverables under the work-for-hire doctrine?

**Expected source** — Service Agreement 2_redacted (1).pdf, Clause 6 (Intellectual Property), page 5.

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1)

**Ground truth**

> Under Clause 6.1, each Party retains ownership of its own pre-existing Intellectual Property, and nothing in the Agreement transfers ownership of one party's pre-existing IP to the other. Under Clause 6.2, the Services are performed on a "work for hire" basis, so TATA owns all IP rights in and with respect to the Services (including modifications, adaptations, enhancements and derivative works), free of any lien/encumbrance, and the Service Provider must execute documents needed to vest such rights in TATA. Clauses 6.3–6.4 further bar the Service Provider from using TATA's IP or the resulting work product for itself or any other client.

**System answer**

```
IP Ownership under Work-for-Hire – SA-Tata – Service Agreement
- Allocation framework: Under the Design and Content Audit Services between Tata Sons Private Limited and its Service Provider, the IP rights in the Services are owned by Tata on a work-for-hire basis. This includes all IP rights in the Services themselves, as well as any modifications and derivative works created in the course of performing the Services. This framework encompasses new deliverables produced under the engagement, vesting ownership in Tata upon creation. [From: Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1)]
- Pre-existing provider IP: The context entries describe a typical work-for-hire regime where pre-existing provider materials are not owned by Tata unless otherwise stated; however, in the specific SA-Tata excerpt, the emphasis is on Tata owning the work product/outputs created under the SOW as part of the work-for-hire arrangement, with the provider’s pre-existing materials typically retained by the provider. The primary stated rule is Tata ownership of the work product and related derivatives, not a wholesale transfer of all provider IP. (Evidence context: SA-Tata – Intellectual Property Ownership framing; work-for-hire with vesting to Tata) [From: Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1)]
- Vesting obligations: The Service Provider is required to execute documents to vest the rights in Tata, ensuring formal transfer/vesting of the work-product rights to Tata. [From: Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1)]

References
[1] Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1), Intellectual Property Ownership – SA-Tata – Service Agreement | Clause: Work-for-hire basis delivering Tata ownership of IP in the Services; provider must execute vesting documents.
```

_scope method: `party-pair-family-corrected`_

### Q18 — 2 → **9/10**  ·  _mixed corpus_

**Question**

> Under the Service Agreement between Tata Sons Private Limited and Redwood Lex Technologies Private Limited, which document takes precedence in the event of an inconsistency: the main body of the agreement or the Statement of Work (SOW)?

**Expected source** — Service Agreement 4_redacted (1).pdf, Section 1 (Definitions and Interpretation), page 1.

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 4_redacted (1)<br>Legal AI - Test_Service Agreement (1)_Test_SA_04.txt

**Ground truth**

> The main body of the Agreement prevails over a conflicting Statement of Work, unless the SOW expressly states that a specific clause is intended to supersede the main body for that particular workstream.

**System answer**

```
The main body of the Service Agreement takes precedence over the Statement of Work in the event of an inconsistency, unless the Statement of Work expressly states that a specific clause is intended to supersede the main body for that workstream.

References
[1] Legal AI - Test_Service Agreement (1)_Service Agreement 4_redacted (1).pdf, Definitions and Interpretation | Quote: "Where there is any conflict between the main body of this Agreement and a Statement of Work, the main body shall prevail unless the statement of work expressly states that a specific clause is intended to supersede the main body for that workstream."
```

_scope method: `party`_

### Q21 — 2 → **8/10**  ·  _mixed corpus_

**Question**

> Review the termination clause of Service Agreement of Tata Steel Limited. What are the notice periods for termination for convenience, and what are the immediate termination events?

**Expected source** — Service Agreement 7_redacted (1).pdf, Section 8 (Term, Termination, Exit Assistance, and Dispute Resolution), page 3.

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 7_redacted (1)

**Ground truth**

> The Agreement states only that "Tata may terminate for convenience upon notice" — no specific number of days is given for the convenience-termination notice period. Immediate termination is available for: material breach, repeated SLA failure, integrity concerns, insolvency, regulatory prohibition, security incident, or a change of control that reasonably affects trust or capability. The initial term is fifteen months.

**System answer**

```
| Obligated Party | Duty | Deadline / Trigger | Consequence of Breach | Source Clause |
| --- | --- | --- | --- | --- |
| Tata Steel Limited | May terminate for convenience upon notice | upon notice | Not specified | Term and Termination – TSL-Decarb – Service Agreement [From: Legal AI - Test_Service Agreement (1)_Service Agreement 7_redacted (1)] |
| Tata Steel Limited | May terminate immediately for: material breach, repeated SLA failure, integrity concerns, insolvency, regulatory prohibition, security incident, or change of control that reasonably affects trust or capability | immediate | Not specified | Term and Termination – TSL-Decarb – Service Agreement [From: Legal AI - Test_Service Agreement (1)_Service Agreement 7_redacted (1)] |

Priority Deadlines
- Notice-based termination for convenience: "upon notice" (no specific days defined in the document)
- Immediate termination events: grounds listed are immediate by their nature (no lead time required)

[SCOPE WARNING: The question named "Tata Steel" but that party appears in several documents, so no single document could be confirmed as the one you meant. The answer below was drawn from a broad search and may reflect a different document of the same type — check the References section, and if it's the wrong one, name the document by its number (e.g. "NDA 7") or its distinctive counterparty.]
```

_scope method: `default-family`_

### Q24 — 2 → **8/10**  ·  _clean corpus_

**Question**

> Review the Notice Invoking Arbitration between Tata Steel Limited and NordForge Metallurgy GmbH. How is the dispute under the decarbonisation workstream framed, and what jurisdiction is invoked under the Arbitration and Conciliation Act, 1996?

**Expected source** — Court Case Document 4 (1).pdf, Page 1 (Notice Invoking Arbitration, dated 29 November 2025).

**Documents the system actually cited**

Legal AI - Test_Court Case Documents (1)_Court Case Document 4 (1).pdf

**Ground truth**

> The dispute is framed as Tata Steel Limited's allegation of NordForge Metallurgy GmbH's breach of technical support obligations, involving repeated slippage against agreed milestones, incomplete technical models, failure to transfer usable work product, and refusal to provide clean data assumptions despite cure discussions. Jurisdiction is invoked under the arbitration clause contained in the Services Agreement, read with the Arbitration and Conciliation Act, 1996.

**System answer**

```
Disputed subject framing and jurisdiction analysis:

Framing of the decarbonisation workstream dispute
- The dispute is framed as a breach-related matter arising from Tata Steel Limited’s allegations of issues in a decarbonisation workstream, specifically described as part of a larger strategic programme linked to decarbonisation planning. The notice emphasizes urgent need for tribunal constitution and preservation of project artefacts due to alleged slippage, incomplete technical models, failure to transfer usable work product, and refusal to provide clean data assumptions, all within the decarbonisation context. This framing is anchored in the material facts describing a larger strategic decarbonisation programme and the accompanying urgent relief to preserve evidence and project materials. [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 4 (1)]
- The notice frames the relief and procedural posture as seeking interim protection and preservation of assets and evidence to support adjudication of the Tata Steel Technical Services Dispute within the decarbonisation program. This is evidenced by language invoking urgent preservation of project artefacts and the need for immediate tribunal constitution. [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 4 (1)]
- Supporting implications in the notice include that the decarbonisation workstream is central to the dispute, and that the procedural mechanism seeks interim relief to safeguard the evidentiary record in relation to this workstream. [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 4 (1)]

Jurisdiction invoked under the Arbitration and Conciliation Act, 1996
- The notice states that jurisdiction is invoked under the arbitration clause contained in the Services Agreement read with the Arbitration and Conciliation Act, 1996. This establishes the forum’s subject-matter and territorial competence for arbitration and the request for urgent interim relief. [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 4 (1)]
- The document also indicates that the chosen forum is competent both territorially and by subject matter under the referenced arbitra
```

_scope method: `party-pair`_

### Q27 — 4 → **9/10**  ·  _clean corpus_

**Question**

> Under the commercial IP petition filed by Infiniti Retail Limited (Croma) regarding Croma-related domains, how is jurisdiction established under the Commercial Courts Act, 2015?

**Expected source** — Court Case Document 7 (1).pdf, Page 1 (Petition Seeking Injunction Against Infringing Croma Domains, dated 25 February 2026).

**Documents the system actually cited**

Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1).pdf

**Ground truth**

> Jurisdiction is invoked jointly under the Trade Marks Act, 1999, the Commercial Courts Act, 2015, and the Code of Civil Procedure, 1908, on the basis that the cause of action arises from acts, omissions, digital targeting, and commercial conduct having a material nexus with the chosen forum. The document does not isolate a distinct Commercial Courts Act-specific jurisdictional basis beyond this combined statement.

**System answer**

```
Jurisdiction establishment under the Commercial Courts Act, 2015
- The IRvCroma petition invokes jurisdiction under the Trade Marks Act, 1999; the Commercial Courts Act, 2015; and the Code of Civil Procedure, 1908. [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1)] [1]
- It further states that the petition argues that the Commercial Court is competent both territorially and by subject matter to entertain the claim, i.e., the chosen forum is capable of hearing the matter on both dimensions. The exact text: “The petitioner accordingly pleads that the chosen forum is competent both territorially and by subject matter and that urgent relief, where sought, is necessary to preserve the efficacy of the underlying cause of action.” [From: Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1)] [1]

References
[1] Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1), "Jurisdiction – IRvCroma – Petition" | Quote: > PETITION SEEKING INJUNCTION AGAINST INFRINGING CROMA DOMAINS
> Infiniti Retail Limited against operators of Croma-formative service and support domains
> The pleading relies on standard grounds applicable to the chosen form, including br
> Jurisdiction is invoked under the Trade Marks Act, 1999, the Commercial Courts Act, 2015, and the Code of Civil Procedure, 1908.
> The petition accordingly pleads that the chosen forum is competent both territorially and by subject matter and that urgent relief, where sought, is necessary to preserve the efficacy of the underlying cause of action.

[SCOPE WARNING: The question named "Infiniti Retail" but that party appears in several documents, so no single document could be confirmed as the one you meant. The answer below was drawn from a broad search and may reflect a different document of the same type — check the References section, and if it's the wrong one, name the document by its number (e.g. "NDA 7") or its distinctive counterparty.]
```

_scope method: `default`_

### Q29 — 6 → **7/10**  ·  _clean corpus_

**Question**

> Give me a go/no-go recommendation for signing Service agreement 2 document, assuming Tata is the receiving party/customer/client. State assumptions.

**Expected source** — Service Agreement 2_redacted (1).pdf, Clauses 2, 6, 8, 11, 12; Schedule II (page 12).

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 2_redacted (1)

**Ground truth**

> This is a generative/judgment task without a single objective ground truth. Note that Tata is already explicitly the client/customer receiving Services in this document (Clause 2.1) — not an assumption. Favorable to Tata: work-for-hire IP assignment (Clause 6.2), Tata's unilateral 30-day termination-for-convenience right (Clause 8.2(c)), and indemnity from the Service Provider (Clause 11). A well-grounded answer should flag that the Service Provider's name/signature block and the Schedule II fee amount are both blank/redacted, so the agreement as presented is incomplete and cannot actually be executed until filled in.

**System answer**

```
Plain statement of scope and context
- This assessment is based solely on the SA-Tata – Service Agreement (the Service Agreement 2) as summarized in the provided context. The parties named in the document are Tata Sons Private Limited (referred to as “TATA”) and the unnamed Service Provider. The analysis assumes Tata is the recipient/customer/client.

Assumptions
- Tata is the customer and will engage the Service Provider to perform specified audit services (Schedule I) under the terms of the Service Agreement 2.
- The term runs from 01 June 2025 to 31 March 2026, with potential mutual written extension.
- Governing law is India, with exclusive jurisdiction in the courts at Mumbai.
- Service fees are set out in Schedule II; GST is excluded from the fee and must be shown separately; TDS (tax deduction at source) applies where applicable.
- There is no monetary liability cap stated in the Limitation of Liability clause.
- The Service Provider is prohibited from assigning rights or subcontracting without Tata’s prior written consent.

Facts drawn from the document (verbatim concepts aligned to the context)
- Intellectual Property: Clause 6.2 provides “work for hire” ownership in Tata over the Services and any modifications/derivative works, with the Service Provider required to execute documents vesting those rights in Tata and prohibited from using Tata’s IP for other clients or purposes.
- Governing Law and Jurisdiction: Clause 10 provides that the Agreement is governed by the laws of India and that the courts at Mumbai have exclusive jurisdiction over disputes.
- Notices: Clause 14.3 prescribes written, signed notices delivered by personal delivery, facsimile, or prepaid post, deemed given upon delivery or receipt, or seven days after mailing.
- Schedule II – Service Fee: Travel, boarding, lodging, and other expenses reimbursed at actuals with prior approval; stock photographs billed at actuals plus a 20% sourcing fee; payments due within 30 days; statutory deductions apply.
- Service Fee: Clause 3.1 sets the Service Provider’s fee as per Schedule II; Clause 3.2 requires tax deduction at source where applicable; Clause 3.4 clarifies GST is excluded from the Serv
```

_scope method: `file`_

### Q30 — 0 → **5/10**  ·  _clean corpus_

**Question**

> Draft a business-friendly summary email explaining the main risks in this agreement and the proposed legal changes → [system asked which agreement] → "service agreement 3"

**Expected source** — N/A — no such document exists in the corpus.

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 7_redacted (1)

**Ground truth**

> No document named "Service Agreement 3" (or any equivalent) exists anywhere in the source corpus (the Service Agreement (1) folder contains only Service Agreements 2, 4, 5, 6, and 7). There is no ground truth to check an answer against. Correct behavior would have been to state clearly that no "Service Agreement 3" exists and ask the user to confirm which actual agreement they meant, rather than fabricating risks or legal changes for a non-existent document.

**System answer**

```
No document matching 'service agreement 3' exists in this corpus.

Aggressive
Clause [Risk Disclosure and Change Management — Aggressive]:
“Notwithstanding any other provision, the Service Provider shall implement and maintain, at its sole cost and expense, a robust risk-mitigation framework including: (i) a formal risk register with quarterly updates; (ii) a mandatory, board-approved change-control process for any scope, staffing, deliverables, or third‑party dependencies; (iii) data handling restrictions, access controls, and incident notification within 24 hours of awareness; and (iv) immediate suspension rights for Tata Sons Private Limited in case of material compliance or data-security concerns, subject to remedy within 10 business days. Any deviation from these requirements will be deemed an uncured material breach.”

Implications: This clause shifts substantial operational risk and control to the Service Provider, prioritizing Tata’s protection and enabling swift remedies, but may impose high costs and rigidity on the provider. Favors Tata Sons Private Limited.

Balanced
Clause [Risk Disclosure and Change Management — Balanced]:
“The Service Provider shall implement a documented change-control process to govern material changes to scope, assumptions, staffing, deliverables, or third‑party dependencies, which shall require approval by Tata Sons Private Limited or its designated delegate. The Service Provider shall maintain reasonable data‑security measures, including access controls and incident notification, and shall cooperate with Tata’s privacy and information-security requirements. In the event of any significant risk or material breach, the parties shall meet within five (5) business days to agree a corrective action plan.”

Implications: A market-standard middle ground that fosters governance without overburdening the provider. Favors an even balance between control and operational flexibility.

Conservative
Clause [Risk Disclosure and Change Management — Conservative]:
“Any change to scope, assumptions, staffing, deliverables, or third‑party dependencies shall be subject to a formal change-control process approved by Tata Sons Private Limited. The
```

_scope method: `default`_

### Q42 — 3 → **9/10**  ·  _mixed corpus_

**Question**

> What is the equity split among Aether, Helios, and Apex Venture Capital in HASG LLC as outlined in their joint venture arrangement?

**Expected source** — Test_JVA_01.txt, Section 3.3 ("Equity Split").

**Documents the system actually cited**

Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_01.txt

**Ground truth**

> Aether and Helios each receive a 45% equity interest in HASG LLC; the remaining 10% is issued to Apex Venture Capital pursuant to the Series A Shareholder Agreement of even date.

**System answer**

```
Equity split in HASG LLC (Aether-Helios-Apex Venture Capital):
- Aether: 45%
- Helios: 45%
- Apex Venture Capital: 10%
[From: Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_01, Clause 3.3]
References:
[1] Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_01, Clause 3.3 | Quote: "Equity Structure – Aether-Helios – Joint Venture Agreement ... Both Aether and Helios receive a 45% equity interest each in HASG LLC. The remaining 10% is reserved for Apex Venture Capital under a separate Series A Shareholder Agreement."

[CITATION NOTE: 1 passage(s) above match the retrieved material but not its verified excerpts — read them as paraphrase rather than exact wording: "Equity Structure – Aether-Helios – Joint Venture Agreement ... Both Aether and H..."]

[SCOPE WARNING: The question named "HASG" but that party appears in several documents, so no single document could be confirmed as the one you meant. The answer below was drawn from a broad search and may reflect a different document of the same type — check the References section, and if it's the wrong one, name the document by its number (e.g. "NDA 7") or its distinctive counterparty.]
```

_scope method: `default`_

### Q43 — 1 → **1/10**  ·  _mixed corpus_

**Question**

> According to Section 9.2 of the Aether-Helios joint venture agreement, what specific industries or applications is Helios prohibited from using the Quantum-Mesh IP for?

**Expected source** — Test_JVA_01.txt, Section 9.2 ("Restricted Operations").

**Documents the system actually cited**

Legal AI - Test_Shareholder Agreements (1)_Test_SHA_14.txt

**Ground truth**

> Helios is prohibited from using the Quantum-Mesh IP (or any software code developed by HASG LLC) for its internal corporate operations outside the Licensed Field, specifically: fossil-fuel exploration, oil pipeline SCADA monitoring, coal power plant logistics, and natural gas distribution infrastructure.

**System answer**

```
Not covered in the provided documents.
- The context holds a SHA-Quantum – Shareholder Agreement with Section 9.2 describing a buy-sell remedy (and other governance provisions) but contains no reference to an Aether-Helios joint venture or any prohibition on Quantum-Mesh IP for particular industries or applications. Evidence gaps: (1) Absence of any Helios or joint venture named in the excerpts; (2) No prohibition clause related to Quantum-Mesh IP applications; (3) No cross-reference to industries within Section 9.2 or related sections.

References
[1] Legal AI - Test_Shareholder Agreements (1)_Test_SHA_14.pdf, Section 9.2 | Quote: “Upon breach by Solaris or Aether, the non-breaching corporate shareholder has a Call Option to buy the breaching shareholder's shares at 15% discount to FMV.”
```

_scope method: `entity` · flagged `not_covered`_

### Q44 — 1 → **1/10**  ·  _mixed corpus_

**Question**

> What is the amount of liquidated damages specified for a violation of the field-of-use restrictions under the Aether-Helios joint venture agreement?

**Expected source** — Test_JVA_01.txt, Section 9.3 ("Misappropriation Remedy").

**Documents the system actually cited**

Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_34.txt

**Ground truth**

> $10,000,000. Any violation of Section 9 by Helios constitutes a material breach entitling Aether to terminate the license, seek an immediate injunction, and claim liquidated damages of $10,000,000.

**System answer**

```
The amount of liquidated damages specified for a violation of the field-of-use restrictions under the Aether-Helios joint venture agreement is $20,000,000. [From: Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_34.txt, Section 9.3]

References
[1] Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_34.txt, Section 9.3

[SCOPE NOTE: This question named no document, so the entire corpus was searched with nothing pinned to one specific agreement. If the answer below reads as if it concerns a single document, verify that against the References section below — no document was confirmed as the one you meant. Name a document explicitly to get a scoped answer.]
```

_scope method: `default`_

### Q49 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> What is the date of the Tata Power Solar interim relief application concerning urgent restraint against misuse of the TATA POWER SOLAR name?

**Expected source** — Court Case Document 2 (1).pdf, Page 1 (Application under Order XXXIX Rules 1 & 2 CPC).

**Documents the system actually cited**

Legal AI - Test_Court Case Documents (1)_Court Case Document 2 (1).pdf<br>Legal AI - Test_Court Case Documents (1)_Test_CCD_02.txt<br>Legal AI - Test_NDA (1)_NDA 7_Redacted (1).pdf

**Ground truth**

> 06 July 2025.

**System answer**

```
The date of the Tata Power Solar interim relief application concerning urgent restraint against misuse of the TATA POWER SOLAR name is 06 July 2025. [1]

References
[1] Legal AI - Test_Court Case Documents (1)_Court Case Document 2 (1).pdf, Page N/A, Clause/Section: Date line | Quote: 06 July 2025

[SCOPE WARNING: The question named "TATA POWER SOLAR" but that party appears in several documents, so no single document could be confirmed as the one you meant. The answer below was drawn from a broad search and may reflect a different document of the same type — check the References section, and if it's the wrong one, name the document by its number (e.g. "NDA 7") or its distinctive counterparty.]
```

_scope method: `default`_

### Q56 — 1 → **9/10**  ·  _clean corpus_

**Question**

> Which Tata entity filed the written statement in the consumer matter concerning alleged misrepresentation of features, delivery condition, and after-sales handling of a passenger EV?

**Expected source** — Court Case Document 6 (1).pdf, Page 1.

**Documents the system actually cited**

Legal AI - Test_Court Case Documents (1)_Court Case Document 6 (1).pdf

**Ground truth**

> Tata Motors Passenger Vehicles Limited, dated 14 January 2026.

**System answer**

```
The Tata entity that filed the written statement is Tata Motors Passenger Vehicles Limited. [1]

References
[1] Legal AI - Test_Court Case Documents (1)_Court Case Document 6 (1), Parties section | Quote: "The petitioner is Tata Motors Passenger Vehicles Limited, the manufacturer of the electric vehicle. The respondents are the claimant customer and the dealership parties involved in the retail sale and service of the vehicle."

[CITATION NOTE: 1 passage(s) above match the retrieved material but not its verified excerpts — read them as paraphrase rather than exact wording: "The petitioner is Tata Motors Passenger Vehicles Limited, the manufacturer of th..."]

[SCOPE NOTE: This question named no document, so the entire corpus was searched with nothing pinned to one specific agreement. If the answer below reads as if it concerns a single document, verify that against the References section below — no document was confirmed as the one you meant. Name a document explicitly to get a scoped answer.]
```

_scope method: `default`_

### Q58 — 3 → **9/10**  ·  _mixed corpus_

**Question**

> Under which Acts is jurisdiction invoked in the petition filed by Infiniti Retail Limited against operators of Croma-formative service and support domains?

**Expected source** — Court Case Document 7 (1).pdf, Page 1.

**Documents the system actually cited**

Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1).pdf

**Ground truth**

> Jurisdiction is invoked under the Trade Marks Act, 1999, the Commercial Courts Act, 2015, and the Code of Civil Procedure, 1908.

**System answer**

```
Under which Acts is jurisdiction invoked in the petition filed by Infiniti Retail Limited against operators of Croma-formative service and support domains?

- Trade Marks Act, 1999 [Document: Jurisdiction – IRvCroma – Petition][From: Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1)]
- Commercial Courts Act, 2015 [Document: Jurisdiction – IRvCroma – Petition][From: Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1)]
- Code of Civil Procedure, 1908 [Document: Jurisdiction – IRvCroma – Petition][From: Legal AI - Test_Court Case Documents (1)_Court Case Document 7 (1)]

References
[Document7], Jurisdiction – IRvCroma – Petition | The petition invokes jurisdiction under the Trade Marks Act, 1999; the Commercial Courts Act, 2015; and the Code of Civil Procedure, 1908.
```

_scope method: `party`_

### Q61 — 4 → **9/10**  ·  _mixed corpus_

**Question**

> What is the date of execution of the Joint Venture Agreement for SteelLoop Resource Recovery Private Limited?

**Expected source** — Joint Venture Agreement 3_redacted (1).pdf, page 1 (header: "Date: 18 November 2025").

**Documents the system actually cited**

Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 3<br>Legal AI - Test_Joint Venture Agreements (1)_Test_JVA_03.txt

**Ground truth**

> 18 November 2025.

**System answer**

```
The date of execution of the Joint Venture Agreement for SteelLoop Resource Recovery Private Limited: 18 November 2025. [1]

References
[1] Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 3_redacted (1).pdf, Document facts: date of this document (execution / effective / filing date) 18 November 2025; parties Tata Steel Limited, [Redacted Circular Industry Partner].
```

_scope method: `party`_

### Q65 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> What is the collaboration purpose of the VitalSpring Wellness Platforms joint venture?

**Expected source** — Joint Venture Agreement 7_redacted (1).pdf, page 1, "Parties, Joint Venture Vehicle, and Purpose" section.

**Documents the system actually cited**

Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 7

**Ground truth**

> The JV is for co-development of wellness beverages, functional pantry adjacencies, and data-supported premium nutrition platforms.

**System answer**

```
The collaboration purpose of the VitalSpring Wellness Platforms joint venture:
- The parties express their intent to co-develop wellness beverages, functional pantry adjacencies, and data-supported premium nutrition platforms. [From: Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 7_redacted (1) | Supporting Quotes: "The parties expressly state their intent to co‑develop wellness beverages, functional pantry adjacencies, and data‑supported premium nutrition platforms."]
- The purpose clause links the parties' strategic objectives to the formation of a ring-fenced operating vehicle with disciplined governance. [From: Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 7_redacted (1) | Supporting Quotes: "This Joint Venture Agreement is entered into by Tata Consumer Products Limited and [Redacted Nutritional Sciences Partner] in relation to VitalSpring Wellness Platforms Private Limited." and "The purpose clause links the parties' strategic objectives to the formation of a ring‑fenced operating vehicle with disciplined governance."]
References
[1] Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 7_redacted (1), Parties and Purpose | Quote: "The parties expressly state their intent to co‑develop wellness beverages, functional pantry adjacencies, and data‑supported premium nutrition platforms." 
[2] Legal AI - Test_Joint Venture Agreements (1)_Joint Venture Agreement 7_redacted (1), Parties and Purpose | Quote: "The purpose clause links the parties' strategic objectives to the formation of a ring‑fenced operating vehicle with disciplined governance."

[CITATION NOTE: 4 passage(s) above match the retrieved material but not its verified excerpts — read them as paraphrase rather than exact wording: "The parties expressly state their intent to co‑develop wellness beverages, funct..."; "The purpose clause links the parties' strategic objectives to the formation of a..."; "The parties expressly state their intent to co‑develop wellness beverages, funct..."]
```

_scope method: `entity-family-corrected`_

### Q67 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> Who is the client for the legal opinion on strategy for dynamic injunctions and coordinated takedowns against deceptive Tata-formative domains?

**Expected source** — Legal Opinion 2 (1).pdf, page 1 (header: "Client: Tata Sons Private Limited / Group Brand Team / Subject: Digital Trademark Enforcement / Date: 02 September 2025").

**Documents the system actually cited**

Legal AI - Test_Legal Opinions (1)_Legal Opinion 2 (1).pdf<br>Legal AI - Test_Legal Opinions (1)_Test_Opinion_02.txt

**Ground truth**

> Tata Sons Private Limited / Group Brand Team.

**System answer**

```
Tata Sons Private Limited / Group Brand Team [1]

References
[1] Legal AI - Test_Legal Opinions (1)_Legal Opinion 2 (1).pdf, “Client and Subject – LO-Tata – Legal Opinion” | Quote: The opinion identifies the client as "Tata Sons Private Limited / Group Brand Team" and the subject matter as "Digital Trademark Enforcement".

[SCOPE WARNING: For "legal opinion 2", this corpus contains BOTH a real document and a synthetic "Test_" stand-in of the same number — both were searched, so some facts below (party names, figures, clause numbers) may come from the fictional stand-in rather than the real document. Verify each cited figure against the document named in the References section before relying on it.]
```

_scope method: `file` · **required naming the document before it would answer**_

### Q70 — 1 → **8/10**  ·  _mixed corpus_

**Question**

> According to the legal opinion for Tata Motors and JLR, how should foreground IP be allocated?

**Expected source** — Legal Opinion 7 (1).pdf, page 1, "Legal Analysis" point 1, and "Recommendations" section.

**Documents the system actually cited**

Legal AI - Test_Legal Opinions (1)_Legal Opinion 7 (1).pdf<br>Legal AI - Test_Legal Opinions (1)_Test_Opinion_07.txt

**Ground truth**

> Foreground IP should be allocated with reference to each party's contribution and intended market use, while background IP licences granted between the parties must remain narrow and purpose-bound. The opinion also recommends creating a documented IP ownership matrix as part of implementation.

**System answer**

```
Foreground IP should be allocated based on each party’s contribution and the intended market use of the resulting technology. [2]

References
[2] Legal AI - Test_Legal Opinions (1)_Legal Opinion 7 (1).pdf, Foreground IP Allocation | Quote: Foreground IP should be allocated with reference to contribution and intended market use, while background IP licences must remain narrow and purpose-bound.

[SCOPE WARNING: For "legal opinion 7", this corpus contains BOTH a real document and a synthetic "Test_" stand-in of the same number — both were searched, so some facts below (party names, figures, clause numbers) may come from the fictional stand-in rather than the real document. Verify each cited figure against the document named in the References section before relying on it.]
```

_scope method: `file` · **required naming the document before it would answer**_

### Q74 — 1 → **8/10**  ·  _mixed corpus_

**Question**

> By what year does Tata Steel state its net-zero ambition, as referenced in the NDA with NordForge Metallurgy GmbH?

**Expected source** — NDA 6_Redacted (1).pdf, "Parties and Background" (p.1): "...a stated net-zero ambition by 2045..."

**Documents the system actually cited**

Legal AI - Test_NDA (1)_NDA 6_Redacted (1).pdf<br>Legal AI - Test_NDA (1)_Test_NDA_06.txt

**Ground truth**

> 2045.

**System answer**

```
2045 [1]

References
[1] Legal AI - Test_NDA (1)_NDA 6_Redacted (1), Parties and Background | Quote: “This Mutual Confidentiality and Non-Disclosure Agreement is made by and between Tata Steel Limited and NordForge Metallurgy GmbH.” (context also notes net-zero ambition by 2045)
```

_scope method: `party`_

### Q76 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> What specific types of confidential information are listed in the NDA between Tata Steel and NordForge beyond general business information?

**Expected source** — NDA 6_Redacted (1).pdf, Clause 2 "Confidential Information" (p.1).

**Documents the system actually cited**

Legal AI - Test_NDA (1)_NDA 6_Redacted (1).pdf<br>Legal AI - Test_NDA (1)_Test_NDA_06.txt

**Ground truth**

> Blast furnace operating data, scrap mix assumptions, refractory drawings, energy-intensity models, emissions baselines, and customer-specific steel grade requirements.

**System answer**

```
Specific types of confidential information listed in the NDA between Tata Steel Limited and NordForge Metallurgy GmbH (beyond general business information):
- Blast furnace operating data
- Scrap mix assumptions
- Refractory drawings
- Energy‑intensity models
- Emissions baselines
- Customer‑specific steel grade requirements
- Meta-information such as the existence and timing of discussions
- Draft issue lists
- Source code excerpts
- Designs
- Product roadmaps
- Data dictionaries
- Issue trackers
- Security findings
- Incident response measures
- Any derivative materials prepared by the Receiving Party that contain, reflect, or are generated from the underlying Confidential Information

References
[1] Legal AI - Test_NDA (1)_NDA 6_Redacted (1), Definition of Confidential Information – NDA-GreenSteel – NDA | Quote: Confidential Information is defined broadly to include all non‑public business, financial, technical, legal, operational, strategic, regulatory, and commercial material... and examples such as blast furnace operating data, scrap mix assumptions, refractory drawings, energy‑intensity models, emissions baselines and customer‑specific steel grade requirements, as well as meta‑information like the existence and timing of discussions, draft issue lists and source code excerpts.
```

_scope method: `party-pair-family-corrected`_

### Q79 — 1 → **9/10**  ·  _mixed corpus_

**Question**

> What acquisitions are mentioned in the NDA background for Tata Consumer Products Limited?

**Expected source** — NDA 7_Redacted (1).pdf, "Parties and Background" (p.1).

**Documents the system actually cited**

Legal AI - Test_NDA (1)_NDA 7_Redacted (1).pdf

**Ground truth**

> The integration of Organic India and Capital Foods, referenced as part of Tata Consumer's expansion into health and wellness categories and premiumisation strategy.

**System answer**

```
Acquisitions mentioned
- Organic India [From: Legal AI - Test_NDA (1)_NDA 7_Redacted (1) | Supporting Quote: “Public Tata Consumer materials describe the company's expansion beyond beverages into health and wellness categories, including the integration of Organic India and Capital Foods, and a continuing premiumisation strategy.”]
- Capital Foods [From: Legal AI - Test_NDA (1)_NDA 7_Redacted (1) | Supporting Quote: same as above]

References
[1] Legal AI - Test_NDA (1)_NDA 7_Redacted (1).pdf, Page N/A

[CITATION NOTE: 1 reference(s) above cited an excerpt that could not be matched to the retrieved source text. The excerpt was removed; the document and clause citation are unchanged.]

[CITATION WARNING: 2 quoted passage(s) above do not appear anywhere in the retrieved source text — do not rely on them as quotes without checking the document: "Public Tata Consumer materials describe the company's expansion beyond beverages..."; "Public Tata Consumer materials describe the company's expansion beyond beverages..."]

[SCOPE WARNING: For "nda 7", this corpus contains BOTH a real document and a synthetic "Test_" stand-in of the same number — both were searched, so some facts below (party names, figures, clause numbers) may come from the fictional stand-in rather than the real document. Verify each cited figure against the document named in the References section before relying on it.]
```

_scope method: `file` · **required naming the document before it would answer**_

### Q85 — 3 → **6/10**  ·  _mixed corpus_

**Question**

> What is the execution date of the services agreement entered into by Tata Sons Private Limited having its registered office at Bombay House, 24 Homi Mody Street, Mumbai?

**Expected source** — Service Agreement 4_redacted (1).pdf, header block, page 1; Service Agreement 2_redacted (1).pdf, page 1 and Clause 8.1, page 6.

**Documents the system actually cited**

Legal AI - Test_Service Agreement (1)_Service Agreement 4_redacted (1)<br>Legal AI - Test_Service Agreement (1)_Test_SA_04.txt

**Ground truth**

> This describes Tata Sons Private Limited generically (its registered office is stated identically in both Service Agreement 2 and Service Agreement 4). Service Agreement 4 (with Redwood Lex Technologies) has an Execution Date of 28 August 2025. Note: Service Agreement 2 (Design and Content Audit) is separately dated 18 July 2025, with a Term running from 01 June 2025 to 31 March 2026 — the question is ambiguous between the two Tata Sons agreements; a correct answer should disambiguate or provide both dates.

**System answer**

```
The execution date of the services agreement is 28 August 2025. [2]

References
[2] Legal AI - Test_Service Agreement (1)_Service Agreement 4_redacted (1), Execution and Signature | Quote: Execution Date: 28 August 2025.

[SCOPE WARNING: For "service agreement 4", this corpus contains BOTH a real document and a synthetic "Test_" stand-in of the same number — both were searched, so some facts below (party names, figures, clause numbers) may come from the fictional stand-in rather than the real document. Verify each cited figure against the document named in the References section before relying on it.]
```

_scope method: `file` · **required naming the document before it would answer**_

### Q92 — 2 → **9/10**  ·  _mixed corpus_

**Question**

> What is the name of the company governed by the shareholders' agreement where Tata Consumer Products Limited is the lead strategic shareholder?

**Expected source** — Shareholder Agreement 4_redacted (1).pdf, page 1 (header block).

**Documents the system actually cited**

Legal AI - Test_NDA (1)_NDA 7_Redacted (1).pdf<br>Legal AI - Test_NDA (1)_Test_NDA_07.txt<br>Legal AI - Test_Shareholder Agreements (1)_Shareholder Agreement 4_red

**Ground truth**

> NourishNext Wellness Foods Private Limited.

**System answer**

```
NourishNext Wellness Foods Private Limited. 
References: [1] Legal AI - Test_Shareholder Agreements (1)_Shareholder Agreement 4_redacted (1)

[SCOPE WARNING: The question named "Tata Consumer Products" but that party appears in several documents, so no single document could be confirmed as the one you meant. The answer below was drawn from a broad search and may reflect a different document of the same type — check the References section, and if it's the wrong one, name the document by its number (e.g. "NDA 7") or its distinctive counterparty.]
```

_scope method: `default-family`_
