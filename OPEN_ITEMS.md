# Open items

1. Weaker synthesis on the smaller model for a few open-ended questions
2. Drafting answers cite a generic document type rather than the named document
3. Scope resolution matches party and document names literally
4. An anomaly that is unusual in substance is still not found by a count
5. Review/Compare tested once since the fixes
6. Golden 50 not re-run since the fixes

## Closed

- Precedent counts are approximate — counted from per-sentence wording
  fingerprints built at ingest; the three measured questions now return the
  index's own figures.
- Retrieval is keyword-only in the main chat — not the case: page selection
  fuses pgvector and BM25, and keyword-only is a fallback for when a session
  has no embeddings.
- Citation checks verify against retrieved context, not the source document —
  a flagged quote is now looked for in the whole document and then in the
  original file before it is called unverified.
- Anomaly survey finds only value anomalies — it also reports a wording the
  population shares that one or two documents do not state.
- Party scan cap of 20 — a name whose first pass stops at the cap is re-scanned
  in full before its size is compared or intersected.
- Two files tracked but matched by an ignore rule.
