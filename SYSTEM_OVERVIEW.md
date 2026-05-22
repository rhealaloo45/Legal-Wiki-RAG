# System Overview: RAG vs LLM Wiki

This application is a research tool designed to compare two distinct paradigms of working with large-scale document knowledge: **Retrieval-Augmented Generation (RAG)** and **LLM Wiki Synthesis**. It is optimized for batch ingestion of up to 50+ documents at once.

## 1. High-Level Architecture

The system is built as a **Single-Page Flask Application**:
- **Backend**: Python/Flask utilizing a `ThreadPoolExecutor` (up to 10 workers) for parallel file ingestion and query processing. Uploads are **non-blocking** — the server accepts files and returns immediately while ingestion runs in background threads.
- **Frontend**: Bootstrap 5 (Dark Mode), Vanilla JS, and D3.js for knowledge graph visualization. Features document-level progress bars with ETA, a responsive side-by-side layout, and a wiki page browser.
- **LLM Engine**: Dual-provider support (Ollama for local embeddings, OpenRouter for cloud LLM reasoning).

### Concurrency & Routing Flow

```mermaid
graph TD
    Client[Client / Web Browser] -- "1. POST /upload (Files + session_id)" --> UploadRoute["Flask: upload() [app.py]"]
    Client -- "4. Polls /progress every 1.5s" --> ProgressRoute["Flask: progress() [app.py]"]
    Client -- "5. POST /query (Question + session_id)" --> QueryRoute["Flask: query_route() [app.py]"]

    subgraph BackendPool ["Backend Concurrency and Execution Pool"]
        UploadRoute -- "2a. Save files (Sync)" --> SavedDisk["data/uploads/{session_id}_{name}"]
        UploadRoute -- "2b. Initialize PROGRESS_STORE" --> StoreInit["PROGRESS_STORE[session_id]"]
        UploadRoute -- "3. Submit tasks to ThreadPoolExecutor" --> Pool["ThreadPoolExecutor (max_workers=10)"]

        Pool -- "Async Worker Thread" --> IngestRAG["_ingest_single_doc_rag()"]
        Pool -- "Async Worker Thread" --> IngestWiki["_ingest_single_doc_wiki()"]

        QueryRoute -- "Submit parallel query tasks" --> QueryPool["ThreadPoolExecutor"]
        QueryPool -- "Parallel Thread" --> RunRAGQuery["rag.query()"]
        QueryPool -- "Parallel Thread" --> RunWikiQuery["wiki.query()"]
    end

    style BackendPool fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff
    style IngestRAG fill:#2c5282,stroke:#3182ce,color:#fff
    style IngestWiki fill:#2c5282,stroke:#3182ce,color:#fff
```

---

## 2. Pipeline 1: RAG (Retrieval-Augmented Generation)

RAG treats documents as a **searchable database** of raw fragments.

### Ingest Phase
1. **Extraction**: Reads `.pdf` (via `pypdf`) or `.txt` files.
2. **Chunking**: Splits text into 2000-character windows with a 200-character overlap.
3. **Batch Embedding**: Generates vector representations for all chunks in a single API call using local **Ollama** (`nomic-embed-text`) batch embedding.
4. **Storage**: Persists vectors and metadata in **ChromaDB** collections (scoped per session). All writes are synchronized using a global re-entrant lock (`threading.RLock`) to ensure SQLite thread safety when multiple documents ingest concurrently.

### Query Phase
1. **Retrieval**: Embeds the user's question and performs a vector similarity search (top-8 results by default, configurable via `TOP_K` in `.env`). Bypasses the LLM and immediately returns a standard fallback answer (`"I’m sorry, but I can’t provide an answer based on the excerpts you requested."`) if no chunks exist.
2. **Augmentation**: Injects the raw text fragments into a specialized prompt.
3. **Generation**: The LLM answers based *only* on the retrieved context, citing specific chunks. If the retrieved context does not contain the answer, the LLM is strictly instructed to return the fallback statement exactly: `"I’m sorry, but I can’t provide an answer based on the excerpts you requested."`

### RAG Pipeline Flowchart

```mermaid
flowchart TD
    classDef ingest fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef query fill:#065f46,stroke:#10b981,color:#fff
    classDef process fill:#1f2937,stroke:#4b5563,color:#fff
    classDef decision fill:#78350f,stroke:#d97706,color:#fff

    subgraph Ingestion
        StartI(["Upload Document"]) --> ExtractI["Extract & Normalize Text"]
        ExtractI --> ChunkI["Sliding Window Chunker<br/>• Size: 2000 chars, Overlap: 200 chars"]
        ChunkI --> EmbedI["Batch Embed via Ollama<br/>• nomic-embed-text"]
        EmbedI --> DBWriteI{"ChromaDB Store<br/>Acquire sqlite lock"}
        DBWriteI -->|Thread-Safe Upsert| DBStoreI[("Persist Vectors in collection: rag_session_id")]
    end

    subgraph Querying
        StartQ(["User Question Input"]) --> EmbedQ["Embed Query via nomic-embed-text"]
        EmbedQ --> DBSearchQ["ChromaDB Vector Query<br/>• Fetch Top-8 nearest chunks"]
        DBSearchQ --> CheckQ{"Chunks Retained?"}
        
        CheckQ -->|No Chunks Found| FallbackQ["Return standard fallback message:<br/>'I'm sorry, but I can't provide an answer...'"]
        CheckQ -->|Yes| PromptQ["Compile RAG Prompt with chunks"]
        
        PromptQ --> LLMQ["Call LLM Engine"]
        LLMQ --> AnsQ["Answer Generation & Citation parsing"]
    end
    
    DBStoreI -.->|Vector Lookup| DBSearchQ

    class StartI ingest
    class StartQ query
    class ExtractI,ChunkI,EmbedI,DBStoreI,EmbedQ,DBSearchQ,FallbackQ,PromptQ,LLMQ,AnsQ process
    class DBWriteI,CheckQ decision
```


---


## 3. Pipeline 2: LLM Wiki Synthesis

The Wiki pipeline treats documents as a **source for narrative synthesis**, not entity extraction. The goal is to build wiki pages that explain what provisions *mean* and how they *interact*, not just list facts.

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

**Merge & Persistence:**
- **New Pages**: Added to the index with content and a one-line summary.
- **Existing Pages**: Content is appended with a separator (`---`). Contradictions are flagged.
- **Cross-Referencing**: Automatic pass to detect mentions of page titles within other pages.
- **Thread Safety**: Per-session `threading.Lock` protects the load→merge→save cycle, preventing data loss during parallel multi-document ingestion.
- **Storage**: The wiki is stored as a structured `index.json` per session, where each page contains both full `content` and a `summary` field.

### Query Phase — Index-Based Retrieval

To prevent hallucination at scale (50 documents can produce 300-1500 wiki pages), the query uses a **two-step retrieval** approach:

1. **Page Selection**: The LLM receives only the **list of page titles + one-line summaries** (compact — even 500 pages fits easily). It selects the 10-15 most relevant pages for the question.
2. **Deep Answer**: Only the selected pages' full content is sent in the answer prompt, keeping the context focused and grounded.
3. **Dynamic Learning**: If the answer introduces new concepts or insights, the LLM extracts them into new pages and relations, compounding the wiki's knowledge.
4. **Citation**: Answers use `[Page Title]` notation which links directly to the visual graph.
5. **Graph Rendering**: D3.js renders the `index.json` as a force-directed graph where nodes are entities and edges are relationships.

> For small wikis (≤ 20 pages), the selection step is skipped and all pages are sent directly.

### LLM Wiki Pipeline Flowchart

```mermaid
flowchart TD
    classDef ingest fill:#1e3a8a,stroke:#3b82f6,color:#fff
    classDef query fill:#065f46,stroke:#10b981,color:#fff
    classDef process fill:#1f2937,stroke:#4b5563,color:#fff
    classDef decision fill:#78350f,stroke:#d97706,color:#fff

    subgraph Ingestion
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

    subgraph Querying
        StartWQ(["User Question Input"]) --> LoadWI["Load index.json"]
        LoadWI --> SizeCheckWQ{"Wiki pages > 20?"}
        
        SizeCheckWQ -->|Yes: Large| SelectWQ["LLM Page Selection<br/>• Feed list of page titles + summaries<br/>• LLM returns 10-15 relevant page names"]
        SizeCheckWQ -->|No: Small| SelectAllWQ["Use all pages in wiki"]
        
        SelectWQ --> AssembleWQ["Join selected pages content"]
        SelectAllWQ --> AssembleWQ
        
        AssembleWQ --> PromptWQ["Compile Wiki Answer Prompt"]
        PromptWQ --> LLMWQ["Call LLM Engine"]
        LLMWQ --> ParseAnsWQ["Parse response JSON:<br/>• answer text<br/>• new_pages / new_relations"]
        
        ParseAnsWQ --> LearnCheckWQ{"Any new knowledge generated?"}
        LearnCheckWQ -->|Yes| DynamicMergeWQ["Acquire Lock & Merge new facts into index.json"]
        LearnCheckWQ -->|No| CitationsWQ["Identify & link page citations e.g. Page Title"]
        DynamicMergeWQ --> CitationsWQ
    end
    
    SaveWI -.->|Reads index.json| LoadWI

    class StartWI ingest
    class StartWQ query
    class ExtractWI,IngestSingleWI,Phase1WI,Phase2WI,ParseWI,MergeLockWI,CombineWI,CrossRefWI,SaveWI,LoadWI,SelectWQ,SelectAllWQ,AssembleWQ,PromptWQ,LLMWQ,ParseAnsWQ,DynamicMergeWQ,CitationsWQ process
    class LengthCheckWI,SizeCheckWQ,LearnCheckWQ decision
```

---


## 4. Upload & Progress System

The upload route is **non-blocking**:
1. Files are saved to disk and ingestion tasks are submitted to the thread pool.
2. The server returns immediately with `{"status": "accepted", "files_queued": N}`.
3. The frontend polls `/progress` every 1.5 seconds, receiving document-level counters:
   ```json
   {"phase": "processing", "docs": {"total": 50, "rag_done": 12, "wiki_done": 8}}
   ```
4. Progress bars show `RAG: 12/50 docs | Wiki: 8/50 docs` with an ETA based on average per-document processing time.
5. When all documents complete both pipelines, `phase` transitions to `"complete"` and the UI enables querying.


## 5. Key Comparisons

| Feature | RAG | LLM Wiki |
| :--- | :--- | :--- |
| **Data Unit** | Raw Chunks (Fixed Size) | Semantic Pages (Narrative Synthesis) |
| **Logic** | Find similar text at query time | Build structured model at ingest time |
| **Ingest Strategy** | Chunk → batch embed → thread-safe store | Adaptive: single-call (≤100K) or two-phase (overview → parallel detail 40K segments) |
| **Query Strategy** | Vector similarity (top-8 chunks) | Index-based: select relevant pages by summary, then deep read |
| **Visuals** | List of retrieved snippets with similarity scores | Interactive D3.js Knowledge Graph & Page Browser |
| **Conflict Handling** | Standalone snippets; fallback if not found | Merges with `---` separators, flags contradictions |
| **Strengths** | Faster ingest, exact retrieval, zero-hallucination fallback | Deep legal synthesis, relationship mapping, dynamic knowledge growth |
| **Scale Behavior** | Handles 50+ docs via batch embedding and locked SQLite writes | Handles 50+ docs via index-based retrieval & parallel segment processing |

---

## 6. Technical Specifications

- **Embeddings**: `nomic-embed-text` (Local via Ollama, batch API)
- **LLM**: `openai/gpt-oss-120b:free` (OpenRouter) or local Ollama fallback
- **Vector DB**: ChromaDB (Persistent at `app/data/chroma`, protected by a global `threading.RLock`)
- **Graphing**: D3.js Force Simulation
- **Isolation**: `session_id` used for per-user database and wiki isolation
- **Concurrency**: `ThreadPoolExecutor(10)` for parallel document ingestion + per-session `threading.Lock` for wiki index safety. In addition, uses nested `ThreadPoolExecutor(5)` for concurrent segment compilation inside `wiki.py` and a global re-entrant lock for thread-safe SQLite writes in `rag.py`.
- **Security**: Split OpenRouter API keys for RAG and Wiki pipelines to monitor quota/usage independently
- **Configurables** (via `.env`):
  - `TOP_K` — number of chunks to retrieve for RAG queries (default: 8)
  - `LLM_PROVIDER` — `openrouter` or `ollama`
  - `OPENROUTER_MODEL` — model to use for LLM calls
  - `PORT` — server port (default: 5001)
