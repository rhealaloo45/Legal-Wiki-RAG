# System Flow: Legal Wiki RAG

Step-by-step technical walkthrough of every pipeline in the system.

For component descriptions, see `SYSTEM_OVERVIEW.md`.
For diagrams, see `ARCHITECTURE.md` and `FLOWCHART.md`.

---

## 1. Document Ingest Pipeline

### Step 1 — Upload

```
POST /upload  {file, session_id}
```

- Flask validates MIME type (PDF, DOCX, TXT, image) and size (≤256 MB)
- File saved to `data/uploads/{session_id}_{original_filename}`
- `ingest_progress` row created or reset in PostgreSQL
- `ThreadPoolExecutor.submit(wiki.ingest, file_path, session_id)` — returns immediately; ingest runs in background

---

### Step 2 — Text extraction

`reader.read_file(file_path)` handles:
- **PDF**: `pdfminer.six` direct extraction → if output is empty or very short, fall back to Tesseract OCR page-by-page
- **DOCX**: `python-docx` paragraph extraction
- **Images**: Tesseract OCR directly
- Returns a single plain-text string

---

### Step 3 — Adaptive segmentation

```python
if len(text) <= 100_000:        # ~25,000 words
    _ingest_single_call(text, doc_name)
else:
    _ingest_overview(text, doc_name)   # Phase 1
    _ingest_detail_*()                 # Phase 2, parallel
```

**Short document path** — `_ingest_single_call()`:

LLM called with `INGEST_PROMPT_TEMPLATE`. Returns JSON:
```json
{
  "doc_type": "Court Judgment",
  "metadata": {
    "governing_law": "...", "jurisdiction": "...", "parties": "...",
    "effective_date": null, "termination_notice": null,
    "liability_cap": null, "ip_ownership": null,
    "auto_renewal": null, "notice_period": null, "payment_terms": null
  },
  "pages": {
    "Facts – Yuvraj Kanther (Court Judgment)": {
      "content": "4-10 sentence synthesis...",
      "summary": "One-line summary.",
      "quotes": ["Verbatim quote 1", "Verbatim quote 2"]
    }
  },
  "relations": [
    {"from": "Page A", "to": "Page B", "label": "applies"}
  ]
}
```

Page title rules enforced by the prompt:
- **Case-specific pages** (facts, holding, charges, procedural history, parties, contentions, relief) → prefixed with short case identifier: `"Facts – Yuvraj Kanther (Court Judgment)"`
- **Shared legal concept pages** (statutes, precedents, doctrines) → no prefix: `"Section 319 CrPC (Court Judgment)"`

**Long document path** — two phases:

*Phase 1 — Overview (`_ingest_overview`):*
Reads first 30K + last 10K chars of document. LLM returns:
```json
{
  "doc_type": "Court Judgment",
  "overview_page": {"content": "...", "summary": "..."},
  "topics": [
    "Facts – Yuvraj Kanther",
    "Section 304 Part II IPC",
    "Procedural History – Yuvraj Kanther",
    ...
  ]
}
```
Topics list includes case-specific prefixes for narrative topics, no prefix for legal concepts.

*Phase 2 — Detail (`_ingest_detail_segment` × N segments):*
Document split into 40K-char chunks with 500-char overlap.
Each chunk processed in parallel: `DETAIL_PROMPT_TEMPLATE` instructs the LLM to fill in pages for known topics found in this segment, plus any new topics discovered. Returns `{pages, relations}` (no metadata — already captured in Phase 1).

---

### Step 4 — Atomic merge (`_atomic_merge_db`)

All segment results are merged into the wiki under a per-session lock.

For each page in the new data:

```
Quotes present?
  YES → append quote block to content:
        content += "\n\n**Supporting Quotes:**\n> Quote 1\n> Quote 2"

Page exists in DB?
  NO  → upsert_page(content, summary, doc_name)
        Schedule for embedding

  YES → C3 NER pre-filter:
          extract amounts, dates, percentages from both versions
          if structural values differ:
            contradiction_flagged = True
            append variant snapshot to variants JSONB

        Merge:
          merged = existing_content
                 + "\n\n---\n*[From: {clean_doc_name}]*\n\n"
                 + new_content

        upsert_page(merged, summary, doc_name, contradiction_flagged, variants)
        Schedule for embedding
```

**C7 metadata upsert** (once per document, not per page):
```
upsert_metadata(session_id, doc_name, metadata_dict)
  → INSERT INTO page_metadata ... ON CONFLICT DO UPDATE
      SET governing_law = COALESCE(:governing_law, page_metadata.governing_law)
      -- COALESCE preserves existing non-null values
```

**S2 cross-reference** (after all pages merged, outside lock):
For each new page title, run:
```sql
SELECT title FROM pages
WHERE session_id = :sid
  AND title != :this_title
  AND content_tsv @@ plainto_tsquery('english', :title_tokens)
```
Any pages found mention this page's title → add a bidirectional relation. Also check if this page's content contains any existing page titles (Python substring check on the in-memory title list).

---

### Step 5 — Embedding (outside lock)

```python
embed_batch([summary or content[:400] for each updated page])
  → upsert_embedding(session_id, title, vector)
```

Each updated page's summary (or first 400 chars of content) is embedded with `"search_document:"` prefix and stored in `page_embeddings`. This runs entirely outside the session lock so HTTP calls don't block parallel ingest threads.

---

### Step 6 — Page compaction (`run_compaction`)

Called once at the end of each document's ingest.

```sql
SELECT title, content, summary, variants, append_count, char_count
FROM pages
WHERE session_id = :sid
  AND (
    append_count >= :append_threshold          -- default 5
    OR (append_count >= 2 AND char_count >= :char_threshold)  -- default 8000
  )
```

The `append_count >= 2` guard prevents single-version pages (even very large ones) from being compacted immediately after first ingest.

For each due page, `_compact_page()`:
1. Build compaction prompt with all content variants
2. Full-model LLM call (4,096 tokens): *"Synthesise these N versions into one coherent wiki page. Preserve all distinct facts verbatim. Note genuine contradictions with source attribution."*
3. Parse result: `{content, summary, contradictions: [{claim, value_a, source_a, value_b, source_b}]}`
4. `reset_page_after_compaction(session_id, title, new_content, new_summary)` → `append_count = 0`, `variants = NULL`
5. Re-embed compacted content
6. For each contradiction: `upsert_contradiction(session_id, page_title, claim, value_a, source_a, value_b, source_b)`

---

### Step 7 — Completion log

```
INGEST | Doc: {doc_name} | Pages updated: {N} | Contradictions found: {N}
```
Written to `data/logs/{session_id}_log.md`.

---

## 2. Query Pipeline

### Step 1 — Request

```
POST /query  {question, session_id, target_doc?, is_followup?}
→ Response is text/event-stream (Server-Sent Events)
```

### Step 1b — LangGraph orchestration (`intent_agent.py`)

`app.py` stores the user message in `chat_messages`, then runs the query `StateGraph`
via `intent_agent.run_query_stream()`. Each node emits a custom stage event; `app.py`
relays them as SSE so the chat UI animates progress tiles.

```
START → classify_intent → disambiguation → clarification → resolve_scope → retrieve → generate → validate → END
                               │ needs            │ needs
                               ▼                  ▼
                              END                END

1. classify_intent
   - Regex fast-path (0 tokens): "compare/vs/differ" → comparison;
     "draft/redline/suggest language" → drafting; assessment patterns → risk_assessment;
     "obligation/deadline/comply" → obligation.
   - Entity+doc-type regex: "ReVolt JV Agreement" → skips disambiguation.
   - No regex match → fast LLM call (MAX_TOKENS_INTENT_CLASSIFY=150) → {intent, confidence}.
   - Any failure → defaults to "factual".

2. disambiguation (wiki.classify_query) — check order:
   a. Named file/number match (_detect_mentioned_files) → skip.
   b. Broad/collective phrasing (_BROAD_SCOPE_RE, e.g. "across all NDAs") → skip
      (added this session — without it, broad cross-document questions wrongly
      got "which one?" because the type word inside the broad phrase matched
      the vague-reference pattern below).
   c. Named party resolving to exactly ONE document via full-text content
      search (_resolve_docs_by_party) → skip (added this session — catches a
      counterparty named by its full corporate name whose identity lives only
      in the document body / redaction-masked metadata, not the page-title
      tokens the entity check mines).
   d. Vague reference (_VAGUE_DOC_PATTERN, e.g. "this NDA", "the agreement")
      with multiple matching candidates → ask.
   e. _question_names_a_document() match (numbered/entity+type pattern) → skip.
   f. _question_mentions_known_entity() finds a known party/entity name in
      page titles (e.g. "ReVolt", "Meridian", "Yuvraj Kanther") → skip.
   g. Fast LLM triage fallback.
   Also skipped upstream if target_doc or is_followup is set.

   NOTE: classify_query() (this node, "should I ask the user to disambiguate?")
   and resolve_scope() (step 4 below, "what's the actual retrieval scope?") are
   two partially-redundant scope-signal systems — both run similar detectors
   independently at different pipeline stages. Known architectural quirk, not
   a bug; a fix to one detector doesn't automatically apply to the other.

3. clarification (wiki.check_ambiguity)
   - Skipped if is_followup, question names a doc, or ENABLE_CLARIFICATION=false.

4. resolve_scope (wiki.resolve_scope) — deterministic, no LLM call. Priority
   cascade, first match wins:
   a. Named file/number (_detect_mentioned_files) → scope="single_doc".
   b. Named party resolving via full-text content search
      (_resolve_docs_by_party) → scope="single_doc".
   c. Known-entity match against page titles → scope="single_doc".
   d. Collective/broad phrasing resolving to a known document family →
      scope="family" (is_broad=True).
   e. Broad phrasing with no resolvable family → scope="corpus" (is_broad=True).
   f. Default → scope="corpus" (unfiltered whole-session search, matches
      pre-Phase-2 behaviour).
   Result feeds directly into retrieve.

5. retrieve (wiki.get_context)
   - Enhanced file detection: numbered doc-type patterns ("service agreement 3")
     match against source docs via regex, even with UUID prefixes/paths.
   - Entity-aware retrieval: when no source_doc filename matches but
     _question_mentions_known_entity() finds entity names in page titles,
     those pages are force-included (fixes the "ReVolt not covered" gap).
   - Scope-constrained: a "family" scope narrows the vector search to that
     family's embeddings; a "single_doc" scope force-pins retrieval to the
     resolved document(s).
   - Retrieval hints per intent (comparison widens small-wiki threshold to 30).

6. generate (wiki.generate_answer) — intent selects prompt template, then runs
   deterministic citation verification + one corrective retry (see Step 5 below).

7. validate — format check per intent (logged, non-blocking) + independent LLM
   grounding audit (see Step 5 below), gated by ENABLE_ANSWER_VALIDATION.
```

### Step 2 — Page selection (`get_context`)

Load all pages for the session from DB (`get_pages(session_id)`).

**Document detection** (three layers):
1. Source filename matching (`_detect_mentioned_files`): exact match, stripped prefix, spacified, no-extension, numbered doc-type pattern
2. Entity-name matching (`_question_mentions_known_entity`): extracts capitalized entity names from page titles and checks if any appear in the question
3. Explicit `target_doc` parameter from disambiguation selection

**For wikis with ≤20 pages** (or ≤30 for comparison intent): use all pages.

**For wikis with >20 pages** → `_select_relevant_pages()`:

```
Path 1 — Hybrid retrieval, fused via RRF (primary):
  1. embed(question, is_query=True)  → "search_query:" prefix
  2. pgvector cosine similarity → top 15 (VECTOR_SEARCH_TOP_K), or top 80
     (BROAD_QUESTION_VECTOR_TOP_K) for broad/family-scoped questions
  3. BM25 keyword ranking over all page summaries, sized to match the vector
     candidate count
  4. Reciprocal Rank Fusion (_rrf_fuse, RRF_K=60): score(title) =
     sum(1 / (60 + rank)) across both rankings — replaces the old
     "vector first, BM25 appended if absent" order
  5. Non-broad: cap fused list at HYBRID_FUSION_TOP_K (23 pages)
     Broad/family: fuse first, then _diversify_by_document() caps each
     document's contribution (BROAD_QUESTION_PER_DOC_CAP=4) up to
     BROAD_QUESTION_TOTAL_CAP (60 pages)
  6. Optional LLM rerank (ENABLE_RERANK, off by default) — broad/family
     queries only, refines the RRF order with a fast-model relevance pass
  → 0 LLM calls (unless rerank enabled), ~5ms DB query

Path 2 — BM25 + LLM (fallback, no embeddings)
Path 3 — BM25 keyword-only (fallback, LLM fails)
```

### Step 3 — Context building

For each selected page:
```
if title starts with "Q:":          cap at 3,000 chars
elif len(content) > 2,000:          cap at 2,000 chars + "[...truncated]"

if contradiction_flagged:
    prepend "[WARNING: This page contains conflicting claims...]"

Format: "## {title}\n{content}\n"
```

### Step 4 — Answer generation

The classified intent selects one of five prompt templates:

| Intent | Prompt | Output |
|---|---|---|
| `factual` | `ANSWER_PROMPT` | Direct answer, 20+ precision rules |
| `risk_assessment` | `ASSESSMENT_PROMPT` | Reasoned judgment, risk classification |
| `comparison` | `COMPARISON_PROMPT` | Side-by-side table + key differences |
| `obligation` | `OBLIGATION_PROMPT` | Duty/deadline table |
| `drafting` | `DRAFTING_PROMPT` | Aggressive/balanced/conservative clause formulations |

All prompts include conversation history, document metadata, wiki context, and chain-of-thought `<reasoning>` block. All five now share table-citation discipline rules (every table row needs an inline citation anchor; quotes may not be manufactured to fill a table) — extended to `ASSESSMENT_PROMPT` and `OBLIGATION_PROMPT` this phase after both were found emitting uncited clause-by-clause tables.

### Step 5 — Post-processing

```python
# Tolerant regex handles unclosed tags, whitespace variants, unicode chars
_REASON_OPEN = r'<\s*reasoning\s*>'
_REASON_CLOSE = r'<\s*/?\s*reasoning\s*>'

# Extract confidence from reasoning block
reasoning_match = re.search(rf'{_REASON_OPEN}(.*?){_REASON_CLOSE}', raw_answer, re.DOTALL)
# Fallback: grab everything after opening tag if no closing tag
if not reasoning_match:
    reasoning_match = re.search(rf'{_REASON_OPEN}(.*)', raw_answer, re.DOTALL)

# Strip reasoning from user-facing answer (both closed and unclosed)
answer = re.sub(rf'{_REASON_OPEN}.*?{_REASON_CLOSE}', '', raw_answer, re.DOTALL)
answer = re.sub(rf'{_REASON_OPEN}.*', '', answer, re.DOTALL)

# Short "not covered" answer → force confidence to 0
if "not covered" in answer.lower() and len(answer) < 150:
    confidence_score = 0
```

### Step 5b — Citation verification and corrective retry (`generate_answer`)

Two deterministic (regex/substring against the exact retrieved context, not LLM-judged) checks run on every answer:

```
_verify_answer_citations(answer, context, question)
    → every quoted span must appear verbatim (whitespace/case-insensitive) in
      the context's Supporting-Quotes-restricted text; catches paraphrase
      dressed up as a verbatim quote

_verify_citation_attribution(answer, context)
    → a verbatim-correct quote must be attributed to the right source
      document; catches misattribution to a plausible-but-wrong document
```

If either check flags issues:
```
1. One corrective-retry LLM call with an addendum prompt asking the model
   to fix the flagged quotes
2. Re-run both checks against the retry's output
3. Keep the retry ONLY if:
     (fewer combined unverified+misattributed issues than the original)
     AND (retry length >= 60% of the original answer's length)
   Otherwise keep the original answer unchanged.
4. Any issues still present after this → append banners:
     "[CITATION WARNING: N quoted passage(s) above could not be verified...]"
     "[ATTRIBUTION WARNING: N quote(s) above appear to be attributed...]"
```

### Step 5c — Grounding check (`validate` node, `intent_agent.py`)

Separate from both the citation-verification checks above AND the answer's self-reported confidence score. `_check_grounding()` makes an independent full-model LLM call that audits the finished answer's factual claims against the retrieved context:

```
- Treats "not covered"/absence findings as CORRECT behaviour, never flags them
- risk_assessment/drafting: professional judgment and risk analysis expected,
  not penalized — only fabricated facts are
- Caps at 8 flagged ungrounded_claims
- MAX_TOKENS_GROUNDING_CHECK=900, doubling retry up to 3x if the model's
  hidden reasoning consumes the whole budget with no visible JSON output
- Gated by ENABLE_ANSWER_VALIDATION
→ Returns {grounding_score: 0-100, ungrounded_claims: [...], summary: "..."}
```

`confidence_score` (self-reported, inside `<reasoning>`, zero extra LLM calls) and `grounding_score` (this separate audited call) are two different numbers — both surfaced to the user, and they can disagree.

### Step 6 — Chat storage and SSE response

```
1. Store assistant answer + intent in chat_messages
   metadata={confidence_score, grounding_score, files_used, token_total, intent, intent_label}

2. Log: QUERY + TOKEN_USAGE events to session log

3. Emit final SSE event:
   data: {type: "answer", wiki: {answer, intent, intent_label, confidence_score,
                                  validation: {valid, warning, grounding}, ...}}
```

The `validate` node also emits its own `validating` active/done stage pair (with `grounding_score` on the `done` event) before the terminal `answer` event.

The SSE stream carries progress events followed by one terminal event of type `answer`, `disambiguation`, or `clarification`. The frontend renders stage tiles, then the answer card with intent tag.

---

## 3. Review Mode Pipeline

### Step 1 — Job start
```
POST /review/start  {session_id, columns: ["Governing Law", "Liability Cap", ...], doc_names: [...]}
```
Job ID generated. Progress tracked in memory (review jobs are fast; no DB persistence needed).

### Step 2 — Document discovery
If `doc_names` not provided, infer from uploaded files in the session.

### Step 3 — Concurrent cell extraction
`ThreadPoolExecutor` over all `(doc_name, column_name)` pairs:

```python
def _extract_worker(doc_name, col_name):
    # C7: check metadata cache first
    field_key = _METADATA_FIELD_MAP.get(col_name.lower())
    if field_key and config.USE_DATABASE:
        cached = _db.get_metadata(session_id, doc_name)
        if cached.get(field_key) is not None:
            return {"value": cached[field_key], "confidence": 0.95, "quote": None}

    # Cache miss: fetch wiki text for this doc → fast LLM extraction
    doc_text = _get_wiki_text_for_doc(session_id, doc_name)
    return extract_cell(doc_text[:6000], col_name)
    # Returns {"value": str|null, "confidence": float, "quote": str|null}
```

### Step 4 — Export
Result matrix assembled and exported to Excel:
- Documents as rows, columns as columns
- Cell background colour: green (≥0.8), yellow (≥0.5), red (<0.5)

---

## 4. Compare Mode Pipeline

### Step 1 — Job start
```
POST /compare/start  {session_id, doc_names: [...], aspects: [...] or null}
```

### Step 2 — Aspect identification (if not provided)
Fast LLM call: reads summaries/overviews of selected documents → returns JSON list of comparable aspects.

### Step 3 — Concurrent extraction
`ThreadPoolExecutor` over all `(aspect, doc_name)` pairs via `extract_cell()`.

### Step 4 — Outlier detection
For each aspect, identify documents whose extracted value differs significantly from the majority.

### Step 5 — Narrative synthesis
Full-model LLM call: given the complete aspect × document table, write a structured comparative analysis highlighting key differences, agreements, and outliers.

### Step 6 — Export
Excel export: aspects as rows, documents as columns, with confidence colouring.

---

## 5. Draft Mode Pipeline

### Step 1 — Classification
Determine if request is: new draft, refinement of prior draft, or clause insertion.

### Step 2 — Stance detection
Fast LLM: determine whether the draft should favour party A, party B, or be neutral.

### Step 3 — Wiki grounding (optional)
If relevant wiki pages exist for the session, relevant pages are retrieved and appended to the prompt as grounding context.

### Step 4 — Draft generation
Full-model LLM call with stance + grounding context + user instruction.

### Step 5 — Refinement loop
User can submit refinement instructions. Each refinement: prior draft + instruction → full-model LLM → new draft.

### Step 6 — DOCX export
`python-docx` renders the final markdown draft into a `.docx` file for download.

---

## 6. Data flow summary

```
Upload
  → Text extraction with positions (reader.py → db.py: source_positions)
  → LLM synthesis (wiki.py + prompts.py)
  → Auto-prefix page titles (_auto_prefix_title: SA1, NDA3, etc.)
  → Atomic merge (db.py: pages, variants, contradiction_flagged)
  → Metadata upsert (db.py: page_metadata)
  → FTS cross-reference (db.py: relations via content_tsv GIN)
  → Embedding (embedder.py → db.py: page_embeddings)
  → Compaction if due (wiki.py → db.py: reset + contradictions table)

Query (conversational, SSE-streamed via intent_agent LangGraph, 7 nodes)
  → Store user message (db.py: chat_messages)
  → classify_intent (intent_agent.py: regex fast-path → llm fast fallback)  ──► SSE stage
  → Disambiguation check (wiki.py → llm fast → chat_messages)               ──► SSE stage / early-exit
  → Clarification check (wiki.py → llm fast → chat_messages)                ──► SSE stage / early-exit
  → resolve_scope (wiki.py: deterministic file/party/entity/family/broad cascade, no LLM call)
  → Entity-aware file detection + hybrid RRF-fused retrieval (scope-constrained) ──► SSE stage
  → Context building (wiki.py: page caps, contradiction warnings, metadata block)
  → Prompt selection by intent (ANSWER / ASSESSMENT / COMPARISON / OBLIGATION / DRAFTING)
  → Answer generation (llm.py: full model)                                  ──► SSE stage
  → Citation verification + one corrective retry (wiki.py: _verify_answer_citations,
    _verify_citation_attribution — deterministic, not LLM-judged)
  → Validate format per intent (intent_agent.py: logged, non-blocking)
  → Grounding check (intent_agent.py: independent LLM audit, ENABLE_ANSWER_VALIDATION) ──► SSE stage
  → Confidence extraction (wiki.py: tolerant regex from <reasoning>)
  → Store assistant answer + intent (db.py: chat_messages)                  ──► SSE terminal event

Review
  → Metadata cache lookup (db.py: page_metadata)
  → Wiki text fetch (advanced_modes.py)
  → Cell extraction (llm.py: fast model)
  → Excel export (openpyxl)
```
