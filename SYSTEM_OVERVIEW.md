# System Overview: Legal LLM Wiki

This application is a research and knowledge management tool designed to process large-scale legal documents using an **LLM Wiki Synthesis** approach. It is optimized for batch ingestion of up to 50+ documents (including nested folders) at once, extracting deep semantic relationships and generating a navigable knowledge graph.

## 1. High-Level Architecture

The system is built as a **Single-Page Flask Application**:
- **Backend**: Python/Flask utilizing a `ThreadPoolExecutor` for parallel file ingestion and query processing. Uploads are **non-blocking** — the server accepts files and returns immediately while ingestion runs in background threads. Features persistent session management with automatic naming.
  > [!NOTE]
  > While the codebase includes services for raw semantic search/vector retrieval (`rag.py` and `hybrid.py`), these pipelines are currently deactivated (commented out) in the main API server (`app.py`). All document queries and analytics run exclusively through the **LLM Wiki Synthesis** engine.
- **Frontend**: Bootstrap 5 (Light Mode), Vanilla JS, and D3.js for knowledge graph visualization. Features document-level progress bars with ETA, a centralized Wiki response view, a wiki page browser, and a file tree viewer for nested uploads.
- **LLM Engine**: Configurable to use either Azure OpenAI or OpenRouter for narrative synthesis, legal extraction, and query answering.

### Concurrency & Routing Flow

```mermaid
graph TD
    Client[Client / Web Browser] -- "1. POST /upload (Files + session_id)" --> UploadRoute["Flask: upload() [app.py]"]
    Client -- "4. Polls /progress every 1.5s" --> ProgressRoute["Flask: progress() [app.py]"]
    Client -- "5. POST /query (Question + session_id)" --> QueryRoute["Flask: query_route() [app.py]"]

    subgraph BackendPool ["Backend Concurrency and Execution Pool"]
        UploadRoute -- "2a. Save files (Sync)" --> SavedDisk["data/uploads/{session_id}_{name}"]
        UploadRoute -- "2b. Initialize PROGRESS_STORE" --> StoreInit["PROGRESS_STORE[session_id]"]
        UploadRoute -- "3. Submit tasks to ThreadPoolExecutor" --> Pool["ThreadPoolExecutor"]

        Pool -- "Async Worker Thread" --> IngestWiki["wiki._ingest_single_doc_wiki()"]

        QueryRoute -- "Execute query tasks" --> RunWikiContext["wiki.get_context()"]
        RunWikiContext --> RunAnswers["Generation: Wiki Answer"]
    end

    style BackendPool fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style IngestWiki fill:#2c5282,stroke:#3182ce,color:#fff
```

---

## 2. The Core Pipeline: LLM Wiki Synthesis

The Wiki pipeline treats documents as a **source for narrative synthesis**. The goal is to build wiki pages that explain what legal provisions *mean* and how they *interact*, rather than just retrieving raw text chunks.

### Ingest Phase (The "Compiler")

The system uses **adaptive segmentation** based on document length:

**Short documents (≤ 100K chars):**
1. The full document is sent in a **single LLM call** with a legal-synthesis-oriented prompt.
2. The LLM produces wiki pages with detailed content and one-line summaries, plus inter-page relations.

**Long documents (> 100K chars) — Two-Phase Approach:**
1. **Phase 1 — Overview**: The first 6K and last 3K characters are sent to the LLM to extract a document overview page and a list of key topics/concepts.
2. **Phase 2 — Detail**: Detailed segments of **40K characters** are sent with the Phase 1 topic list as context. To optimize speed, these segments are processed **concurrently** using a thread pool (`ThreadPoolExecutor` with 5 workers) rather than sequentially.
3. **Strict Quality Instructions**: The prompts are tuned specifically for legal documents, enforcing:
   - **Source Integrity**: Explicitly passing `{doc_name}` and requiring that only citations present in the text be used.
   - **Factual Precision**: Mandating exact verbatim figures and dates, forbidding hallucination.
   - **Legal Depth**: Instructing the extraction of precedents, statutory provisions, and judicial reasoning (ratio decidendi).

**JSON Parsing & Repair:**
- Employs robust JSON extraction (`_parse_json_safe`) and an automatic fallback repair mechanism (`_repair_json`) that asks the LLM to fix malformed JSON if parsing fails.

**Merge & Persistence:**
- **New Pages**: Added to the index with content and a one-line summary (`{"title": {"content": "...", "summary": "...", "source_doc": "..."}}`).
- **Existing Pages**: Content is appended with a separator (`---`). 
- **Contradiction Detection**: Pre-flight checks on long appending segments (`>200 chars`) trigger a micro-LLM pass (`max_tokens=300`) using the **fast/cheap model** — this is a structured boolean JSON check, not synthesis. Found factual contradictions push the page into a `contradiction_flagged` state and populate a `variants` array to track the document lineage of the conflicting values.
- **Cross-Referencing**: Automatic pass to detect mentions of page titles within other pages.
- **Thread Safety**: Per-session `threading.Lock` protects the load→merge→save cycle, preventing data loss during parallel multi-document ingestion. A separate `_log_locks` dict safeguards concurrent writes to a timestamped `log.md` file.
- **Storage**: The wiki is stored as a structured `index.json` per session.

### Query Phase — Index-Based Retrieval

To prevent hallucination at scale, the query uses a **two-step retrieval** approach:

1. **Document Mention Check**: Scans the user query string for mentions of any document names. If file mentions are detected, all wiki pages containing those filenames are marked as forced selections.
2. **Page Selection**: If the wiki grows beyond 20 pages, the **fast/cheap model** receives the list of page titles + one-line summaries and selects the 15-25 most relevant pages. Title matching does not require legal reasoning depth, so the cheap model is used here to reduce cost.
3. **Deep Answer**: Only the selected pages' full content is sent to the **full synthesis model** in the answer prompt, prepending a `[WARNING]` tag to any page flagged for contradictions.
4. **Chain of Thought, Grounding & Confidence**: The LLM generates a `<reasoning>` trace to verify facts against the context, ending with a self-assessed `CONFIDENCE_SCORE` (0–100) and `CONFIDENCE_REASON`. Confidence is extracted from the reasoning block via regex — **no second LLM call is made**. This halves query-time LLM cost compared to a separate confidence-evaluation round-trip.
5. **Answer Filing**: Answers scoring `>= 80` confidence are automatically filed back into the wiki as a `"Q: {question}"` page for future compound learning. An anti-duplication title check ensures similar questions append to existing pages rather than fragmenting the wiki.
6. **Graph Rendering**: D3.js renders the `index.json` as a force-directed graph where nodes are entities and edges are relationships.

---

## 3. Advanced Modes (Review, Compare, & Draft)

To supplement standard querying, three advanced analytical modes operate independently of the ingest pipeline.

### Review Mode
Designed for bulk data extraction across documents.
- **Workflow**: User selects multiple documents and defines arbitrary column headers.
- **Concurrency**: A background `ThreadPoolExecutor` spins up 5 workers, concurrently querying every (document, column) combination using **wiki-synthesized content** (falling back to raw document text if the document has not been ingested).
- **Export**: Generates a confidence-colored `.xlsx` matrix (green for high confidence, yellow for medium, red/flagged for low or null), built using `openpyxl`.

### Compare Mode
Designed for deep, aspect-oriented comparison between existing knowledge and new unstructured uploads.
- **Retrieval**: Uses scoped lookup to retrieve wiki content specifically tagged with the requested `source_doc`. If a selected document has no wiki pages, it automatically falls back to raw text extraction.
- **Aspect Identification**: A single LLM pass generates 4-6 specific aspects to compare based on the user's topic.
- **Extraction & Outliers**: Extracts the values concurrently across documents, then runs a secondary LLM pass to explicitly identify contradictions and generate a narrative synthesis. Temporary uploaded files are strictly wiped after the job completes to preserve storage.

### Draft Mode
Designed for ephemeral, context-aware legal drafting with automatic stance detection.
- **Stance & Classification**: Analyzes the user's prompt to detect the desired negotiating stance (e.g., `tata_favorable`, `counterparty_favorable`, `neutral`) and classifies the request type (e.g., `clause`, `full_document`, `letter`).
- **Wiki Grounding**: Optionally retrieves up to 8 relevant wiki pages as drafting precedent (`get_draft_context`) or operates in a standalone mode.
- **Generation & Refinement**: Generates the initial draft and allows the user to submit refinement instructions (e.g., "make this more aggressive") which iterates on the draft while preserving existing structure.
- **Export**: Converts the markdown draft, including tables and formatting, into a downloadable `.docx` file using `python-docx`.
### Detailed Flowchart: Review and Compare Modes

#### Review Mode Flow
```mermaid
flowchart TD
    classDef review fill:#065f46,stroke:#10b981,color:#fff

    R_Start(["User Input: Select Docs & Columns"]) --> R_Submit["Submit Background Job"]
    R_Submit --> R_ThreadPool["ThreadPoolExecutor (5 Workers)"]
    
    R_ThreadPool -->|For each Doc & Column| R_Extract["LLM Extraction Pass"]
    R_Extract --> R_FormatJSON["Parse: Value, Confidence, Quote"]
    
    R_FormatJSON --> R_Aggregate["Aggregate into Data Matrix"]
    R_Aggregate --> R_UI["Render Matrix with Confidence Colors"]
    R_UI --> R_Export["Export to .xlsx"]

    class R_Start,R_Submit,R_ThreadPool,R_Extract,R_FormatJSON,R_Aggregate,R_UI,R_Export review
```

#### Compare Mode Flow
```mermaid
flowchart TD
    classDef compare fill:#78350f,stroke:#d97706,color:#fff

    C_Start(["User Input: Docs + Topic (+ Uploads)"]) --> C_Retrieve["Scoped Wiki Retrieval / Raw Text Fallback"]
    C_Retrieve --> C_Aspects["LLM Pass 1: Identify 4-6 Compare Aspects"]
    
    C_Aspects --> C_ThreadPool["ThreadPoolExecutor"]
    C_ThreadPool -->|For each Doc & Aspect| C_Extract["LLM Pass 2: Extract Aspect Data"]
    
    C_Extract --> C_Aggregate["Aggregate Aspect Matrix"]
    C_Aggregate --> C_Outliers["LLM Pass 3: Identify Outliers & Contradictions"]
    C_Outliers --> C_Synthesis["LLM Pass 4: Narrative Synthesis"]
    
    C_Synthesis --> C_UI["Render Matrix, Narrative, Outliers"]
    C_UI --> C_Wipe["Wipe Temp Uploads"]
    C_Wipe --> C_Export["Export to .xlsx"]

    class C_Start,C_Retrieve,C_Aspects,C_ThreadPool,C_Extract,C_Aggregate,C_Outliers,C_Synthesis,C_UI,C_Wipe,C_Export compare
```

#### Draft Mode Flow
```mermaid
flowchart TD
    classDef draft fill:#4c1d95,stroke:#8b5cf6,color:#fff

    D_Start(["User Input: Prompt + Stance Keywords"]) --> D_Classify["LLM: Classify Type & Detect Stance"]
    D_Classify --> D_Context{"Ground to Wiki?"}
    
    D_Context -->|Yes| D_Retrieve["Retrieve Top ~8 Relevant Pages"]
    D_Context -->|No| D_Generate["LLM: Generate Draft"]
    D_Retrieve --> D_Generate
    
    D_Generate --> D_UI["Render Editable Markdown"]
    
    D_UI --> D_Refine["User Input: Refinement"]
    D_Refine --> D_LLMRefine["LLM: Iterate & Preserve Structure"]
    D_LLMRefine --> D_UI
    
    D_UI --> D_Export["Export to .docx"]

    class D_Start,D_Classify,D_Context,D_Retrieve,D_Generate,D_UI,D_Refine,D_LLMRefine,D_Export draft
```


### Execution Flow of System Modes

#### Ask Mode (Standard Query)
```mermaid
flowchart TD
    classDef ask fill:#1e3a8a,stroke:#3b82f6,color:#fff

    A["User Input: Question"] --> A1["Document Mention Check"]
    A1 --> B{"Wiki Pages > 20?"}
    B -->|Yes| E["LLM Page Selection: Pick top 15-25"]
    B -->|No| D["Use all Wiki pages"]
    E --> F["Assemble selected page contents"]
    D --> F
    F --> G["LLM Answer Generation: CoT reasoning + inline citations"]
    G --> H{"Confidence >= 80%?"}
    H -->|Yes| I["File back answer as 'Q: {question}' wiki page"]
    H -->|No| J["Return answer to UI only"]

    class A,A1,B,D,E,F,G,H,I,J ask
```

#### Review Mode (Bulk Extraction)
```mermaid
flowchart TD
    classDef review fill:#065f46,stroke:#10b981,color:#fff

    R1["User input: Docs + Column Headers"] --> R2["Submit background job to ThreadPoolExecutor"]
    R2 --> R3["Concurrently extract cells: doc_text + column_name"]
    R3 --> R4["LLM extracts JSON: value, confidence, quote"]
    R4 --> R5["Assemble confidence-colored grid"]
    R5 --> R6["User option: Export as OpenPyXL .xlsx file"]

    class R1,R2,R3,R4,R5,R6 review
```

#### Compare Mode (Deep Comparison)
```mermaid
flowchart TD
    classDef compare fill:#78350f,stroke:#d97706,color:#fff

    C1["User input: Docs + Uploaded File + Topic"] --> C2["Retrieve target texts: Scoped Wiki pages / raw fallback"]
    C2 --> C3["LLM Aspect Identification: Generate 4-6 comparison aspects"]
    C3 --> C4["Concurrently extract aspect values for each document"]
    C4 --> C5["Secondary LLM pass: Outlier & contradiction detection"]
    C5 --> C6["Tertiary LLM pass: Narrative synthesis"]
    C6 --> C7["Assemble aspect matrix & outliers list"]
    C7 --> C8["User option: Export comparison as .xlsx file"]

    class C1,C2,C3,C4,C5,C6,C7,C8 compare
```

#### Draft Mode (Context-Aware Drafting)
```mermaid
flowchart TD
    classDef draft fill:#4c1d95,stroke:#8b5cf6,color:#fff

    D1["User input: Prompt"] --> D2["Classify draft type & Stance Rules"]
    D2 --> D3["Retrieve context from Wiki (Optional)"]
    D3 --> D4["LLM generates drafted text"]
    D4 --> D5["User refines draft via chat"]
    D5 --> D6["LLM applies refinement iteratively"]
    D6 --> D7["User option: Export draft as .docx file"]

    class D1,D2,D3,D4,D5,D6,D7 draft
```

### LLM Wiki Pipeline Flowchart

```mermaid
flowchart TD
    classDef ingest fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef query fill:#065f46,stroke:#10b981,color:#fff
    classDef process fill:#1f2937,stroke:#4b5563,color:#fff
    classDef decision fill:#78350f,stroke:#d97706,color:#fff

    subgraph Ingestion ["Ingestion Pipeline"]
        direction TB
        StartWI(["Upload Document"]) --> ExtractWI["Extract & Normalize Text"]
        ExtractWI --> LengthCheckWI{"Text length > 100K chars?"}
        
        LengthCheckWI -->|No: Short| IngestSingleWI["Single-Call Compilation<br/>• INGEST_PROMPT_TEMPLATE"]
        
        LengthCheckWI -->|Yes: Long| Phase1WI["Phase 1: Extract Overview<br/>• Read first 6K + last 3K chars<br/>• Generate global topics & overview"]
        Phase1WI --> Phase2WI["Phase 2: Segment Iteration<br/>• Chunk text into 40K segments<br/>• ThreadPoolExecutor 5 workers parallel"]
        
        IngestSingleWI --> ParseWI["Parse & Repair JSON Output"]
        Phase2WI --> ParseWI
        
        ParseWI --> MergeLockWI["Acquire Session Lock<br/>Load index.json"]
        MergeLockWI --> CombineWI["Merge content with --- separator<br/>Deduplicate relationships"]
        CombineWI --> CrossRefWI["O(N²) Cross-Reference Pass<br/>• Scan content for matches to link pages"]
        CrossRefWI --> SaveWI["Write index.json & Release Lock"]
    end

    subgraph Querying ["Querying & Retrieval"]
        direction TB
        StartWQ(["User Question Input"]) --> LoadWI["Load index.json"]
        LoadWI --> SizeCheckWQ{"Wiki pages > 20?"}
        
        SizeCheckWQ -->|Yes: Large| SelectWQ["LLM Page Selection<br/>• Feed list of page titles + summaries<br/>• LLM returns 10-15 relevant page names"]
        SizeCheckWQ -->|No: Small| SelectAllWQ["Use all pages in wiki"]
        
        SelectWQ --> AssembleWQ["Join selected pages content"]
        SelectAllWQ --> AssembleWQ
        
        AssembleWQ --> PromptWQ["Compile Wiki Answer Prompt"]
        PromptWQ --> LLMWQ["Call LLM Engine"]
        LLMWQ --> ParseAnsWQ["Parse response:<br/>• Strip CoT reasoning trace<br/>• Calculate Confidence Score"]
        
        ParseAnsWQ --> CitationsWQ["Identify & link page citations e.g. [Page Title]"]
    end
    
    SaveWI -.->|Reads index.json| LoadWI

    class StartWI ingest
    class StartWQ query
    class ExtractWI,IngestSingleWI,Phase1WI,Phase2WI,ParseWI,MergeLockWI,CombineWI,CrossRefWI,SaveWI,LoadWI,SelectWQ,SelectAllWQ,AssembleWQ,PromptWQ,LLMWQ,ParseAnsWQ,CitationsWQ process
    class LengthCheckWI,SizeCheckWQ decision
```

---

## 4. Upload, Progress, and Session System

**Uploads**:
The upload route is **non-blocking** and supports nested folders:
1. Files are saved to disk (preserving relative paths via frontend `relative_paths` array) and ingestion tasks are submitted to the thread pool.
2. The server returns immediately with `{"status": "accepted", "files_queued": N}`.
3. The frontend polls `/progress` every 1.5 seconds, receiving document-level counters.
4. When all documents complete processing, `phase` transitions to `"complete"` and the UI enables querying.

**Sessions & Isolation**:
- The application supports multiple persistent sessions (`data/sessions.json`).
- Sessions are automatically renamed based on the first question asked (first 30 characters).
- **Strict Isolation**: Every chat session maintains its own completely independent `index.json` and `UPLOAD_PATH`. Knowledge is strictly sandboxed per session to prevent cross-contamination.
- **Context-Aware UI**: Advanced UI modes (`[Ask]`, `[Compare]`, `[Review]`) dynamically appear only within an active session (after upload completion or when resuming an older session), ensuring a clean, context-focused workspace.
- Full session CRUD is supported, including clearing all Wiki index data per session.

---

## 5. UI Architecture (Light-Mode SPA)

The frontend is a custom-built, lightweight Single-Page Application (SPA) designed to feel like a premium analytical tool.
- **Technology Stack**: HTML5, Vanilla JavaScript, and minimal Bootstrap 5 (only used for structural grids).
- **Aesthetics**: Complete departure from dark-mode; utilizes a clean Light Theme defined by custom CSS variables (`--bg-surface`, `--accent`, etc.) and Google Fonts (`Lora` and `DM Sans`).
- **Layout**: Three fixed zones (Topbar, Sidebar, Main Workspace) with no page reloads. Modals (Doc Reader, Knowledge Graph) are implemented as absolute-positioned overlays.
---

## 6. Technical Specifications

- **LLM & Embeddings**: Supports both Azure OpenAI and OpenRouter configuration (via `config.py`).
- **Dual-Model Routing**: Uses a large model for synthesis-heavy tasks (ingest compilation, answer generation, drafting, compare narrative) and a fast/cheap model for lightweight structured tasks (page selection, contradiction pre-flight, JSON repair, cell extraction). Routing is handled in `llm.py` via the `fast=True` flag — no code changes needed to swap models, only `.env` updates.
- **Token Budgets**: All LLM calls have explicit `max_tokens` caps defined as constants in `config.py` (`MAX_TOKENS_*`). This prevents silently burning token budget on calls whose outputs are small JSON objects.
- **Graphing**: D3.js Force Simulation for navigable knowledge graph.
- **Isolation**: `session_id` used for per-user database and wiki isolation.
- **Concurrency**: `ThreadPoolExecutor` for parallel document ingestion + per-session `threading.Lock` for wiki index safety. Uses nested `ThreadPoolExecutor` (concurrency limited by `WIKI_MAX_WORKERS` from config, default: 3) for concurrent segment compilation inside `wiki.py`.
- **Configurables** (via `.env`):
  - `LLM_PROVIDER` — Provider for LLM calls (`azure` or `openrouter`, default: `azure`)
  - `EMBEDDING_PROVIDER` — Provider for embedding generation (`azure` or `openrouter`, default: `azure`)
  - `AZURE_OPENAI_API_KEY` — Azure API key
  - `AZURE_OPENAI_ENDPOINT` — Azure Endpoint URL
  - `AZURE_OPENAI_API_VERSION` — Azure API version
  - `AZURE_OPENAI_DEPLOYMENT` — **Big model** deployment for synthesis tasks (e.g. `gpt-5.4`)
  - `AZURE_FAST_DEPLOYMENT` — **Fast/cheap model** deployment for selection, contradiction checks, repair (e.g. `gpt-5.4-mini`)
  - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` — Deployment name for Azure embeddings (e.g. `text-embedding-3-large`)
  - `EMBEDDING_DIMENSIONS` — Embedding dimensions (default: `1536`)
  - `OPENROUTER_API_KEY` — OpenRouter API key
  - `OPENROUTER_MODEL` — **Big model** for synthesis tasks (e.g. `openai/gpt-oss-120b:free`)
  - `OPENROUTER_FAST_MODEL` — **Fast/cheap model** for selection, contradiction checks, repair (e.g. `openai/gpt-oss-20b:free`)
  - `OPENROUTER_EMBEDDING_MODEL` — OpenRouter embedding model name (e.g. `nvidia/llama-nemotron-embed-vl-1b-v2:free`)
  - `WIKI_MAX_WORKERS` — Thread workers for concurrent chunk processing (default: `3`)
  - `TESSERACT_CMD` — Path to Tesseract executable (for OCR fallback on scanned PDFs)
  - `PORT` — server port (default: 5001)

