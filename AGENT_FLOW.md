# Agent Flow: Intent Classifier (LangGraph)

Technical deep-dive into the query-pipeline agent implemented in `app/services/intent_agent.py`.

For the broader system context, see `SYSTEM_OVERVIEW.md` and `FLOWCHART.md`.

---

## 1. Overview

Every user question in the Ask tab is orchestrated by a LangGraph `StateGraph` — a directed graph of seven nodes with conditional edges. The graph classifies the query from a lawyer's perspective, runs pre-query safety checks (disambiguation, clarification), resolves retrieval scope, retrieves wiki context, generates an intent-specific answer, and validates the output format (including an independent grounding audit).

Each node emits real-time **stage events** via LangGraph's custom stream writer. Flask relays these as **Server-Sent Events (SSE)** so the chat UI can render animated progress tiles. The graph runs synchronously inside the Flask request thread — no Celery or async needed.

**File:** `app/services/intent_agent.py`
**Framework:** LangGraph 1.0+ (`langgraph.graph.StateGraph`)
**Entry point:** `run_query_stream(question, session_id, target_doc, is_followup)` — yields stage event dicts

---

## 2. Graph Topology

```mermaid
flowchart TD
    START([User submits question]) --> CI

    subgraph CLASSIFY["① classify_intent"]
        CI[Regex fast-path\n0 tokens · instant] --> CID{Regex\nmatched?}
        CID -->|No| LLM_CI[Fast LLM call\n150 tokens]
        CID -->|Yes| INTENT_OUT
        LLM_CI --> INTENT_OUT
    end

    INTENT_OUT{Intent} -->|factual| TAG_F["🔵 Factual\nANSWER_PROMPT\nDirect extraction · 20 precision rules\nNo external knowledge"]
    INTENT_OUT -->|risk_assessment| TAG_R["🔴 Risk Assessment\nASSESSMENT_PROMPT\nReasoned judgment · H/M/L risk\nGo/no-go recommendation"]
    INTENT_OUT -->|comparison| TAG_C["🟣 Comparison\nCOMPARISON_PROMPT\nSide-by-side table · Key differences\nWho each clause favors"]
    INTENT_OUT -->|obligation| TAG_O["🟠 Obligation\nOBLIGATION_PROMPT\nDuty/deadline table\nParty · Duty · Trigger · Consequence"]
    INTENT_OUT -->|drafting| TAG_D["🟢 Drafting\nDRAFTING_PROMPT\nAggressive/Balanced/Conservative\nThree clause formulations"]

    TAG_F --> DIS
    TAG_R --> DIS
    TAG_C --> DIS
    TAG_O --> DIS
    TAG_D --> DIS

    subgraph PREQUERIES["② ③ Pre-query safety gates"]
        DIS[disambiguation\nSkips if doc named,\nentity matched, broad phrasing,\nor unique party match]
        DIS -->|needs_disambiguation| END_DIS([END — document chips])
        DIS -->|pass| CLAR[clarification\nSkips if followup\nor doc named]
        CLAR -->|needs_clarification| END_CLAR([END — clarifying question])
        CLAR -->|pass| SCOPE_START
    end

    SCOPE_START --> SCOPE

    subgraph SCOPING["④ Scope resolution"]
        SCOPE["resolve_scope\nDeterministic, no LLM call\nfile → party → entity →\nfamily → broad → corpus"]
    end

    SCOPE --> RET_START
    RET_START --> RET

    subgraph PIPELINE["⑤ ⑥ ⑦ Core pipeline"]
        RET["retrieve\nHybrid pgvector + BM25 → RRF fusion\nEntity-aware page matching\nScope-constrained · Intent-tuned hints"]
        RET --> GEN["generate\nFull LLM · 4096/8192 tokens\nPrompt selected by intent\nConversation + metadata injected\n+ citation verification + retry"]
        GEN --> VAL["validate\nFormat check per intent (non-blocking)\n+ independent grounding audit (LLM)"]
    end

    VAL --> END_ANS([END — answer + intent tag\n+ confidence badge + grounding score])

    style CLASSIFY fill:#e0e7ff,stroke:#4f46e5
    style PREQUERIES fill:#fef9e7,stroke:#d97706
    style SCOPING fill:#f3e8ff,stroke:#9333ea
    style PIPELINE fill:#e8f5e9,stroke:#059669
    style TAG_F fill:#dbeafe,stroke:#2563eb
    style TAG_R fill:#fee2e2,stroke:#dc2626
    style TAG_C fill:#ede9fe,stroke:#7c3aed
    style TAG_O fill:#ffedd5,stroke:#ea580c
    style TAG_D fill:#ccfbf1,stroke:#0d9488
```

**Conditional edges:**
- After `disambiguation`: if `needs_disambiguation` is true → END (short-circuit). Otherwise → `clarification`.
- After `clarification`: if `needs_clarification` is true → END (short-circuit). Otherwise → `resolve_scope`.
- `resolve_scope → retrieve` is unconditional (fail-open: any internal error yields a default "corpus" decision).
- All other edges are unconditional.

### Intent → Prompt → Output shape (quick reference)

| Intent | Colour | Prompt template | What it tells the LLM to produce |
|---|---|---|---|
| `factual` | 🔵 Blue | `ANSWER_PROMPT` | Direct answer grounded only in context. 20 precision rules (scope restriction, no external knowledge, arithmetic prohibition, legal standard precision, etc.). IEEE citations. |
| `risk_assessment` | 🔴 Red | `ASSESSMENT_PROMPT` | Professional legal judgment permitted. Risk classified as H/M/L with basis. Gaps and missing protections flagged. Concrete recommendations (accept/reject/negotiate). |
| `comparison` | 🟣 Purple | `COMPARISON_PROMPT` | Side-by-side markdown table (rows = aspects, columns = documents). Key Differences section. Who-it-favors analysis per difference. "Not addressed" for missing clauses. |
| `obligation` | 🟠 Orange | `OBLIGATION_PROMPT` | Duty/deadline table (Obligated Party · Duty · Deadline/Trigger · Consequence · Source Clause). Priority Deadlines chronological list. Direction of obligation preserved. |
| `drafting` | 🟢 Teal | `DRAFTING_PROMPT` | Three labelled formulations: Aggressive (favors client), Balanced (market-standard), Conservative (low-risk). Legal implications per formulation. Grounded in existing contract language. |

---

## 3. Graph State (`QueryState`)

A `TypedDict` carried through every node. Each node reads what it needs and writes its outputs back.

```python
class QueryState(TypedDict, total=False):
    # ── Input (set by run_query_stream before graph starts) ──
    question: str              # The user's raw question text
    session_id: str            # Wiki session UUID
    target_doc: str            # Force-scoped document (from disambiguation chip click)
    is_followup: bool          # True when user responded to a prior clarification

    # ── Set by classify_intent node ──
    intent: str                # "factual" | "risk_assessment" | "comparison" | "obligation" | "drafting"
    intent_confidence: float   # 0.0–1.0
    intent_method: str         # "regex" | "llm" | "fallback" | "disabled"

    # ── Set by disambiguation / clarification nodes ──
    needs_disambiguation: bool
    disambiguation_data: dict  # {message, documents, raw_documents}
    needs_clarification: bool
    clarification_data: dict   # {message, options, original_question}

    # ── Set by resolve_scope node ──
    scope_decision: dict       # {scope, target_docs, target_family, is_broad, confidence, method}

    # ── Set by retrieve / generate / validate nodes ──
    conversation_context: str  # Last 3–5 chat messages, max ~2000 chars
    wiki_context: str          # Formatted "## Title\ncontent" string
    selected_titles: list      # Page titles included in context
    retrieval_meta: dict       # {bm25_count, page_selection_usage}
    answer_result: dict        # Full generate_answer() return + intent fields
    validation: dict           # {valid: bool, warning: str|None, grounding: dict}
```

---

## 4. Nodes — Detailed Walkthrough

### 4.1 `classify_intent` — Intent Classification

**Purpose:** Determine what kind of legal task the user is asking for.

**SSE events emitted:**
1. `{stage: "classifying", status: "active", message: "Classifying intent…"}`
2. `{stage: "intent_identified", status: "done", intent, intent_label, intent_confidence, intent_method}`

**Logic (two-tier):**

**Tier 1 — Regex fast-path (0 tokens, <1ms):**
Checked in priority order. First match wins.

| Priority | Pattern | Intent | Example triggers |
|---|---|---|---|
| 1 | `_RX_DRAFTING` | `drafting` | "draft", "redline", "rewrite", "suggest language", "counter-proposal", "alternative wording" |
| 2 | `_RX_COMPARISON` | `comparison` | "compare", "differ", "versus", "vs", "side by side", "contrast" |
| 2b | `_RX_BETWEEN` | `comparison` | "between X and Y" (any context) |
| 3 | `_RX_RISK` | `risk_assessment` | "go/no-go", "recommend", "should we sign", "risk assessment", "red flag", "deal-breaker", "safe to sign", "negotiation strategy" |
| 4 | `_RX_OBLIGATION` | `obligation` | "what are our obligations", "list the deadlines", "comply with", "compliance requirements", "required to", "must we" |

**Design note:** The obligation regex was tightened to avoid false positives. A phrase like "return/destruction obligation" (naming an NDA section) does NOT trigger obligation intent — only phrases where obligation is the *subject* of the query match.

**Tier 2 — LLM fallback (150 tokens, ~300ms):**
If no regex matches and `ENABLE_INTENT_CLASSIFIER` is true, a fast-model LLM call classifies the query:

```
Prompt: "You classify a lawyer's question into exactly ONE intent.
Intents: factual, risk_assessment, comparison, obligation, drafting.
Question: {question}
Respond with JSON only: {"intent": "slug", "confidence": 0.0-1.0}"
```

- Model: fast tier (`AZURE_FAST_DEPLOYMENT` / `OPENROUTER_FAST_MODEL`)
- Max tokens: `MAX_TOKENS_INTENT_CLASSIFY` (150)
- Parse: `_parse_json_safe()` → validate intent slug ∈ `VALID_INTENTS`

**Fallback:** Any failure (LLM error, parse error, invalid intent) → `{"intent": "factual", "confidence": 0.5, "method": "fallback"}`. This guarantees the graph never crashes on classification — worst case is a generic factual answer.

**State written:** `intent`, `intent_confidence`, `intent_method`

---

### 4.2 `disambiguation` — Document Scope Check

**Purpose:** If the question references no specific document, ask the user which document they mean.

**SSE events emitted (only when check runs):**
1. `{stage: "disambiguation", status: "active", message: "Checking document scope…"}`
2. If needed: `{stage: "disambiguation", status: "done", type: "disambiguation", payload: {message, documents, raw_documents}}`

**Skip conditions (any → skip):**
- `target_doc` is set (user already picked a document)
- `is_followup` is true (user responded to a prior prompt)
- `_question_names_a_document()` matches (3 layers):
  1. Numbered pattern: "service agreement 1", "NDA 3", "SA1"
  2. Entity + doc type: "ReVolt JV Agreement", "Meridian service agreement"
  3. Known entity from page titles: distinctive capitalized names extracted from wiki page titles

**When it runs:**
Calls `wiki.classify_query(question, session_id)` which:
1. Checks `_detect_mentioned_files()` (filename matching)
2. Checks `_question_mentions_known_entity()` (entity matching against page titles)
3. Falls back to a fast LLM call asking if the question uses vague references ("this document", "summarize it")

**If disambiguation needed:** Sets `needs_disambiguation = True`, emits the disambiguation event with document list, and the conditional edge routes to END. The frontend renders clickable document chips.

**State written:** `needs_disambiguation`, `disambiguation_data`

---

### 4.3 `clarification` — Ambiguity Check

**Purpose:** If the question is too ambiguous, ask one clarifying question before answering.

**SSE events emitted (only when check runs):**
1. `{stage: "clarification", status: "active", message: "Checking for ambiguity…"}`
2. If needed: `{stage: "clarification", status: "done", type: "clarification", payload: {message, options}}`

**Skip conditions (any → skip):**
- `is_followup` is true
- `_question_names_a_document()` matches
- `ENABLE_CLARIFICATION` is false

**When it runs:**
Calls `wiki.check_ambiguity(question, session_id, conversation_context)` — fast LLM call that determines if the question could mean multiple very different things.

**Hard limit:** 1 clarification per turn. After the user responds (is_followup=true), this check is permanently skipped.

**Also sets:** `conversation_context` (fetched from chat_messages table via `build_conversation_context()`)

**State written:** `needs_clarification`, `clarification_data`, `conversation_context`

---

### 4.4 `resolve_scope` — Retrieval Scope Resolution

**Purpose:** Decide, in one deterministic place, whether the question targets a single document, a document family, or the whole corpus — before retrieval runs.

**SSE events emitted:** none (no `_emit()` call — this node is fast enough to fold into the surrounding stage tiles without its own).

**Logic (`wiki.resolve_scope()`, no LLM call):** priority cascade, first match wins:
1. Named file/number match (`_detect_mentioned_files`) → `scope="single_doc"`.
2. Named party resolving to a document set via full-text content search (`_resolve_docs_by_party`) → `scope="single_doc"`. Catches counterparties named by their full corporate name whose identity lives only in the document body or redaction-masked metadata — not the page-title tokens the entity check mines. A multi-instrument question naming several of that party's documents ("across the NDA, the arbitration notice, and the Section 9 petition") pins the whole resolved cluster instead of narrowing to one.
3. Known-entity match against page titles (`_question_mentions_known_entity`) → `scope="single_doc"`, pinned to the entity's dominant document (or all matched documents for a multi-instrument question).
4. Collective/broad phrasing (`_BROAD_SCOPE_RE` or a plural family noun) resolving to a document family that actually exists in this session → `scope="family"`, `is_broad=True`.
5. Broad phrasing with no resolvable family → `scope="corpus"`, `is_broad=True` (whole-session search).
6. Default (nothing matched) → `scope="corpus"`, `is_broad=False` — identical to pre-Phase-2 behaviour.

**Fail-open:** any internal exception is caught in `resolve_scope_node` and yields a default `{"scope": "corpus", "is_broad": False, "method": "error"}` decision, so the pipeline degrades to unfiltered whole-session search rather than failing the query.

**Architectural note:** this node's detectors overlap significantly with `classify_query()` (the `disambiguation` node, §4.2) — both run named-file, party-content-search, entity-match, and broad-phrasing checks independently, at different pipeline stages, answering different questions ("should I ask the user?" vs "what's the actual scope?"). This is a known duplication, not a bug, but a fix to one detector does not automatically apply to the other.

**State written:** `scope_decision`

---

### 4.5 `retrieve` — Context Retrieval

**Purpose:** Select relevant wiki pages and format them as context for the answer LLM.

**SSE events emitted:**
1. `{stage: "retrieving", status: "active", message: "Retrieving relevant pages…"}`
2. `{stage: "pages_retrieved", status: "done", count: N, message: "Retrieved N page(s)"}`

**Logic:**
1. Calls `get_query_strategy(intent)` for retrieval hints:
   - `comparison` → `{multi_doc: true, small_wiki_threshold: 30}` (widens the net)
   - `obligation` → `{keyword_boost: ["shall", "must", "obligation", …]}`
   - `drafting` → `{keyword_boost: ["definition", "clause", "shall", …]}`
   - `factual` / `risk_assessment` → `{}` (default retrieval)

2. Calls `wiki.get_context(question, session_id, target_doc, retrieval_hints, doc_family, force_broad, force_docs)` — the last three parameters are forwarded from `scope_decision` (§4.4): a `family` scope narrows the vector search to that family's embeddings, a `single_doc` scope force-pins retrieval to the resolved document(s). It runs:
   - **Document detection** (3 layers): filename matching → entity matching → target_doc
   - **Page selection** (3-path cascade): hybrid pgvector+BM25 fused via Reciprocal Rank Fusion (`_rrf_fuse`, `RRF_K=60`) → BM25+LLM fallback → BM25-only fallback. Broad/family questions widen the vector candidate pool and diversify by document (`_diversify_by_document`) after fusion; an optional fast-model LLM rerank (`ENABLE_RERANK`, off by default) can refine the fused order further for those questions.
   - **Context formatting**: cap pages at 2000 chars, prepend contradiction warnings

**State written:** `wiki_context`, `selected_titles`, `retrieval_meta`, `conversation_context`

---

### 4.6 `generate` — Answer Generation

**Purpose:** Call the full-power LLM with an intent-specific prompt template to produce the answer.

**SSE events emitted:**
1. `{stage: "generating", status: "active", intent, prompt_type, message: "Generating {intent} answer…"}`

**Prompt selection:**

| Intent | Prompt template | Output shape |
|---|---|---|
| `factual` | `ANSWER_PROMPT` | Direct answer with 20+ precision rules, IEEE citations |
| `risk_assessment` | `ASSESSMENT_PROMPT` | Reasoned judgment, H/M/L risk classification, go/no-go recommendation |
| `comparison` | `COMPARISON_PROMPT` | Side-by-side markdown table, key differences, who-it-favors analysis |
| `obligation` | `OBLIGATION_PROMPT` | Duty/deadline table (party · duty · trigger · consequence · clause) |
| `drafting` | `DRAFTING_PROMPT` | Three formulations (aggressive/balanced/conservative) with implications |

All prompts include: `{conversation_block}`, `{metadata_block}`, `{context}`, `{question}`.
All require `<reasoning>` block with `CONFIDENCE_SCORE` and `CONFIDENCE_REASON`. All five now share table-citation discipline rules (every table row needs an inline citation anchor; no manufacturing quotes to fill a table) — extended to `ASSESSMENT_PROMPT` and `OBLIGATION_PROMPT` this phase.

**Post-LLM processing** (inside `wiki.generate_answer()`):
- Extract confidence via tolerant regex (handles unclosed tags, unicode, whitespace)
- Strip `<reasoning>` block from user-facing answer
- Short "not covered" → force confidence to 0
- **Citation verification** — `_verify_answer_citations()` checks every quoted span appears verbatim in the retrieved context; `_verify_citation_attribution()` checks a verbatim-correct quote is attributed to the right source document. Both are deterministic (regex/substring), not LLM-judged.
- **Corrective retry** — if either check flags issues, one retry LLM call attempts a fix; the retry is kept only if it has fewer combined issues AND retains ≥60% of the original answer's length. Otherwise the original is kept.
- Any issues still present after the retry step → `[CITATION WARNING: ...]` / `[ATTRIBUTION WARNING: ...]` banners appended to the answer.

**Enrichment:** After `wiki.generate_answer()` returns, the node adds `intent`, `intent_label`, `intent_confidence`, `intent_method` to the result dict.

**State written:** `answer_result`

---

### 4.7 `validate` — Response Format Check + Grounding Audit

**Purpose:** Light sanity check — does the answer match the expected format for the classified intent? Plus an independent LLM audit of whether the answer's factual claims are actually grounded in the retrieved context.

**SSE events emitted:**
1. `{stage: "validating", status: "active", message: "Checking answer grounding…"}` — only when `ENABLE_ANSWER_VALIDATION` is on
2. `{stage: "validating", status: "done", grounding_score, message: "Grounding: N%"}` — only when `ENABLE_ANSWER_VALIDATION` is on
3. `{stage: "complete", status: "done", type: "answer", payload: {full answer result}, message: "Done"}`

**Format checks per intent:**

| Intent | Validation rule | What triggers a warning |
|---|---|---|
| `comparison` | Answer contains `\|` (table pipe) | No comparison table detected |
| `obligation` | Answer contains `\|` or list markers (`-`, `*`, numbered) | No list or table detected |
| `drafting` | Answer contains `` ``` `` or `>` or "aggressive"/"balanced"/"conservative" | No clause formulations detected |
| `factual` / `risk_assessment` | No check | — |

**Non-blocking:** Warnings are logged (`logger.info`) but the answer is still returned to the user. The validation never blocks or retries.

**Grounding check (`_check_grounding()`, gated by `ENABLE_ANSWER_VALIDATION`):** a separate full-model LLM call (`fast=False`) — distinct from both the deterministic citation-verification checks in `generate` (§4.6) and the answer LLM's own self-reported `confidence_score` — that audits the finished answer's factual claims against the retrieved context:
- Treats "not covered" / absence findings as **correct** behaviour, never flags them as ungrounded
- For `risk_assessment` / `drafting` intents, professional judgment and risk analysis are expected and not penalized — only fabricated facts are
- Caps output at 8 flagged `ungrounded_claims`
- `MAX_TOKENS_GROUNDING_CHECK` = 900, with a doubling-retry (up to 3 further attempts) when the model's hidden reasoning consumes the whole budget with no visible JSON output
- Returns `{grounding_score: 0-100, ungrounded_claims: [...], summary: "..."}`, stored under `answer_result.validation.grounding`

`confidence_score` and `grounding_score` are two different numbers shown to the user on the same answer card — they measure different things (self-reported vs. independently audited) and can disagree.

**State written:** `answer_result.validation` (`{valid, warning, grounding}`), `validation`

---

## 5. SSE Event Stream

`app.py` iterates over `run_query_stream()` and wraps each event as SSE:

```
data: {"stage": "classifying", "status": "active", "message": "Classifying intent…"}

data: {"stage": "intent_identified", "status": "done", "intent": "comparison", ...}

data: {"stage": "retrieving", "status": "active", "message": "Retrieving relevant pages…"}

data: {"stage": "pages_retrieved", "status": "done", "count": 15, ...}

data: {"stage": "generating", "status": "active", "message": "Generating comparison answer…"}

data: {"stage": "validating", "status": "active", "message": "Checking answer grounding…"}

data: {"stage": "validating", "status": "done", "grounding_score": 88, "message": "Grounding: 88%"}

data: {"type": "answer", "wiki": {"answer": "...", "intent": "comparison", "intent_label": "Comparison", "confidence_score": 92, "validation": {"valid": true, "warning": null, "grounding": {"grounding_score": 88, "ungrounded_claims": [], "summary": "..."}}, ...}}
```

Note: `resolve_scope` (between `clarification` and `retrieve`) emits no SSE stage of its own — it's fast enough (no LLM call) to fold silently into the surrounding stage tiles.

**Terminal events** (exactly one per request):
- `{type: "answer", wiki: {...}}` — full answer with intent metadata
- `{type: "disambiguation", message, documents, raw_documents}` — user must pick a document
- `{type: "clarification", message, options}` — user must clarify the question
- `{type: "error", error: "..."}` — pipeline failure

The frontend reads the stream via `ReadableStream.getReader()`, renders animated stage tiles (spinner → checkmark), then replaces them with the answer card carrying a coloured intent badge.

---

## 6. Tools / Services Called by Nodes

| Node | Service calls | Model tier | Token budget |
|---|---|---|---|
| `classify_intent` | `llm.ask(fast=True)` (only if regex misses) | Fast | 150 |
| `disambiguation` | `wiki.classify_query()` → `llm.ask(fast=True)` | Fast | 200 |
| `clarification` | `wiki.check_ambiguity()` → `llm.ask(fast=True)` | Fast | 300 |
| `resolve_scope` | `wiki.resolve_scope()` — pure Python cascade, DB lookups only, no LLM call | — | 0 |
| `retrieve` | `wiki.get_context()` → `embedder.embed()` + `db.search_similar_pages()` + BM25 → RRF fusion + optional rerank | — | 0 (hybrid) or 1000 (LLM fallback), + 2048 if `ENABLE_RERANK` fires on a broad query |
| `generate` | `wiki.generate_answer()` → `llm.ask(fast=False)` + citation verification + optional corrective retry | Full | 4096 (8192 broad), + up to 4096/8192 again if a corrective retry fires |
| `validate` | Pure Python (regex checks) + `_check_grounding()` (`llm.ask(fast=False)`) if `ENABLE_ANSWER_VALIDATION` | Full (grounding only) | 0 (format check) + 900 (grounding, doubles on truncation up to 4x) |

**Total worst-case token budget per query (base pass, no retries, rerank/broad off):** 150 + 200 + 300 + 1000 + 4096 + 900 = **6,646 tokens**
**Typical token budget (regex intent, hybrid retrieval, grounding on):** 200 + 300 + 4096 + 900 = **5,496 tokens** (disambiguation + clarification may not fire; `resolve_scope` and RRF fusion add 0 LLM tokens)

---

## 7. Graph Compilation & Caching

```python
_QUERY_GRAPH = None

def get_query_graph():
    global _QUERY_GRAPH
    if _QUERY_GRAPH is None:
        _QUERY_GRAPH = build_query_graph()
    return _QUERY_GRAPH
```

The graph is compiled once on first request and cached globally. `build_query_graph()` constructs the `StateGraph`, adds all nodes and edges, and calls `.compile()`. Subsequent requests reuse the compiled graph — no re-compilation overhead.

---

## 8. Debug Logging

Every node logs entry and key decisions at `INFO` level with `[AGENT]` prefix:

```
[AGENT] classify_intent_node: question='Compare indemnity in SA1 vs SA2'
[AGENT] intent=comparison conf=0.90 method=regex
[AGENT] check_disambiguation_node
[AGENT] disambiguation skipped (doc named or followup)
[AGENT] resolve_scope_node
[AGENT] scope=single_doc family=None broad=False method=file
[AGENT] retrieve_context_node: intent=comparison
[AGENT] generate_answer_node: intent=comparison pages=14
[AGENT] validate_response_node: intent=comparison
[AGENT] Grounding score: 88 | Most claims traceable to context
```

`app.py` additionally logs each SSE event:
```
SSE stage: classifying | Classifying intent…
SSE stage: intent_identified | Intent: Comparison
SSE stage: retrieving | Retrieving relevant pages…
SSE stage: pages_retrieved | Retrieved 14 page(s)
SSE stage: generating | Generating comparison answer…
SSE stage: validating | Checking answer grounding…
SSE stage: validating | Grounding: 88%
SSE answer: intent=comparison conf=92%
```

---

## 9. Failure Modes & Graceful Degradation

| Failure | Behaviour |
|---|---|
| Intent LLM call returns 429 / error | Falls back to `factual` intent (safest default) |
| Intent LLM returns invalid JSON | Falls back to `factual` |
| Disambiguation LLM fails | Skips disambiguation, proceeds to clarification |
| Clarification LLM fails | Skips clarification, proceeds to resolve_scope |
| `resolve_scope()` throws exception | Falls back to `{"scope": "corpus", "method": "error"}` — unfiltered whole-session search, identical to pre-Phase-2 behaviour |
| `generate_answer()` throws exception | Returns error message as the answer with confidence 0 |
| Citation/attribution verification finds issues | One corrective retry; if still unresolved, warning banners appended to the answer (not blocked) |
| Grounding check LLM call fails or returns unparseable output | Returns `{grounding_score: None, ...}` — no grounding badge shown, answer still returned |
| Validate finds format mismatch | Logs warning, returns the answer anyway |
| `_emit()` fails (no stream writer) | Silently skipped — answer still produced, just no SSE tiles |

The graph never crashes. Every node has try/except with a safe fallback. The worst case is a generic factual answer with no progress tiles — functionally identical to the pre-agent system.
