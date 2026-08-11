# LexWiki Re-Test Report v4 — production-representative corpus

**This report measures the corpus that will actually ship: 46 real Tata documents,
551 pages, zero synthetic fixtures** (session `prodcorpus-46`, rebuilt from
`3a66b0ab-…` by `app/eval/build_clean_corpus.py`, verified `decoy rows remaining 0`).

v3 and everything before it measured the 494-document audit corpus, in which 448
synthetic `Test_<TYPE>_<NN>` fixtures compete with the real documents for retrieval.
Those scores understate the shipped product, and some of the failures they record
cannot occur in production at all.

## What changed in scope since v3

- **Q43 and Q44 are excluded.** Both are ground-truthed to `Test_JVA_01`, a fixture.
  They cannot be asked in production, so they cannot be scored here. v3 counted them
  as failures; that was a defect in the audit set, not in the system.
- **Q42 is excluded** for the same reason (fixture-grounded).
- **Q38, Q39, Q40 are included** — they were missing from v3's interim table.

27 questions measured. Scores are judgments against the verified ground truth in
`rag_review_105.html`, not the system's own confidence figures.

## Scores

| Q | v1 | v3 (mixed) | **v4 (production)** | Note |
|---|---|---|---|---|
| Q14 | 1 | 9 | **9** | Correct doc, care standard + 36-month term |
| Q17 | 1 | 7 | **7** | Ceiling — Clause 6.1 is absent from every ingested page of SA 2; the hedge is correct behaviour |
| Q18 | 2 | 9 | **9** | Exact match |
| Q21 | 2 | 2 | **9** | **Fixed this round** — all seven immediate-termination grounds and the "upon notice, no day count" wording |
| Q24 | 2 | 7 | **9** | **Fixed this round** — all four pleaded breaches (slippage, incomplete models, work-product failure, clean-data refusal) |
| Q27 | 4 | 9 | **7** | Says the petition "relies on three statutes" but names only the Commercial Courts Act; ground truth wants all three named |
| Q29 | 6 | — | **6** | Opens with a "Is the context about this question? Yes" preamble; substance is reasonable |
| Q30 | 0 | 0 | **6** | **Corpus-dependent, as suspected** — now states "No document matching 'Service Agreement 3' exists in this corpus", then still drafts clauses anyway |
| Q38 | 1 | 9 | **7** | **Fixed this round** — was returning a completely empty answer on this corpus (see below) |
| Q39 | 1 | 8 | **8** | Clean general-knowledge answer, no scope leakage |
| Q40 | 1 | 9 | **9** | Correct |
| Q49 | 1 | 9 | **9** | Exact date |
| Q56 | 1 | 9 | **8** | Correct entity; the 14 Jan 2026 date appears in some runs and not others |
| Q58 | 3 | 9 | **9** | All three Acts named |
| Q61 | 4 | 9 | **9** | Exact date |
| Q65 | 1 | 9 | **9** | Correct document and purpose |
| Q67 | 1 | 9 | **9** | Exact |
| Q70 | 1 | 9 | **8** | Contribution + market use correct; background-IP narrowing less explicit |
| Q74 | 1 | 9 | **9** | Exact |
| Q76 | 1 | 9 | **9** | All six ground-truth categories |
| Q79 | 1 | 9 | **9** | Exact quote, correct document |
| Q85 | 3 | 6 | **6** | Right date, still does not flag the SA2/SA4 ambiguity ground truth asks for |
| Q92 | 2 | 9 | **9** | Confident and correct |
| Q94 | 6 | — | **9** | Correct |
| Q101 | 1 | 1 | **2** | Still failing — answers "affidavits of admission and denial" instead of the screenshots / chat transcripts / payment instructions / complaints |
| Q102 | 5 | — | **3** | Describes Tata Sons procedurally (petitioner seeking relief) rather than the Court's characterisation (principal investment holding company, trust-led group) |
| Q105 | 2 | 2 | **2** | Still circular — echoes "integrated design consultancy services" instead of naming the Western Freeway / Thane project |

**Mean across these 27: 2.04 → 7.59.**

## Fixes landed this round

**1. Umbrella party names are now spent against the instrument the question names.**
`_resolve_docs_by_party` gives up on any party matching more than four documents,
because such a name cannot identify one document alone. But the question usually
supplies a second constraint the resolver never spent — the instrument type.
"Tata Steel Limited" matches 7 documents; the Service Agreement family holds 62;
the intersection is exactly **one**, Service Agreement 7, the correct document.
Previously scope fell through to the whole family flagged *broad*, retrieval
returned roughly one page per document, and the single SA 7 page that survived was
about confidentiality — so Q21 reported that no termination grounds existed when the
document lists seven. New `party-in-family` method. Q92 also resolves through it now.

**2. A party-pair cluster is narrowed by the instrument named.**
The same two parties sign several instruments of one dispute. The kind-hint
narrowing existed but ran only when the cluster exceeded the pin limit, so Q24's
two-document cluster (Arbitration Notice + Section 9 Petition) was pinned whole and
the answer LLM chose the petition — reporting preservation relief for a question
about the notice's pleaded breaches. Added `Arbitration Notice`, `Section 9 Petition`
and `Written Statement` hints, and let the narrowing run on small clusters when the
question names exactly one instrument.

**3. A named authority the corpus never discusses no longer returns a blank answer.**
Found only by moving to this corpus. General-knowledge questions naming an authority
("the Delaware Uniform Trade Secrets Act") deliberately run retrieval first, keeping
the general answer as a fallback — but the fallback was promoted only on an explicit
refusal. On a corpus that simply has nothing to say, the answer LLM returns an *empty*
body instead, which `not_covered` does not flag. Q38's entire payload was a scope
disclosure with no answer at all. The promotion now also fires on an empty body,
discounting the pipeline's own bracketed notices before testing emptiness.

**4. The vector retrieval channel was dead, and is now alive.**
Every one of the top-15 pgvector hits was a cached `Q:` answer page — a cached answer
is by construction the nearest neighbour of the question that produced it. Those were
filtered *after* the SQL `LIMIT`, so the vector channel contributed nothing and every
unpinned question silently ran BM25-only. Now excluded in SQL.

**5. Answer caching is off** (`ENABLE_ANSWER_CACHE=false`). Nothing is written to the
wiki, and the `Q:` pages earlier runs already filed are hidden from retrieval too —
otherwise the feature would keep running on everything it wrote before. Set the flag
to `true` to restore both behaviours.

## What is still open

- **Q101 (2), Q102 (3), Q105 (2)** — three genuine misses, all in the judgments /
  court-document family, all cases where the right document is retrieved and the
  specific passage is not surfaced or not used. Q105 remains the confirmed ingest gap.
- **Q85 (6)** — no mechanism detects that a question's identifying description matches
  two documents equally well. This is the ambiguity-detection work already spawned.
- **Q30 (6)** — states the document does not exist, then drafts clauses anyway. The
  fabrication is much reduced without fixtures present, but the contradiction remains.
- **Q17 (7)** and **Q56 (8)** are near their ceiling: Q17's missing clause is an ingest
  gap, and Q56's date varies run to run.

## Note on the decoy-collision work

The decoy-fixture failures that dominated v3 — Q30's invented HASG/Zephyr contract
terms, Q43/Q44 citing `Test_JVA_34` — are artefacts of the audit corpus. On the corpus
that ships they do not arise, and two of those questions cannot even be asked. Worth
weighing before investing further engineering there.
