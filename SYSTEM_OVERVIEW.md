# System Overview: RAG vs LLM Wiki

This application is a research tool designed to compare two distinct paradigms of working with large-scale document knowledge: **Retrieval-Augmented Generation (RAG)** and **LLM Wiki Synthesis**.

## 1. High-Level Architecture

The system is built as a **Single-Page Flask Application**:
- **Backend**: Python/Flask utilizing a `ThreadPoolExecutor` (up to 10 workers) for parallel file ingestion and query processing.
- **Frontend**: Bootstrap 5 (Dark Mode), Vanilla JS, and D3.js for knowledge graph visualization. Features a responsive side-by-side layout displaying an upload card alongside a wiki page browser.
- **LLM Engine**: Dual-provider support (Ollama for local embeddings, OpenRouter for cloud LLM reasoning).

---

## 2. Pipeline 1: RAG (Retrieval-Augmented Generation)

RAG treats documents as a **searchable database** of raw fragments.

### Ingest Phase
1. **Extraction**: Reads `.pdf` (via `pypdf`) or `.txt` files.
2. **Chunking**: Splits text into 2000-character windows with a 200-character overlap.
3. **Embedding**: Generates vector representations using local **Ollama** (`nomic-embed-text`).
4. **Storage**: Persists vectors and metadata in **ChromaDB** collections (scoped per session).

### Query Phase
1. **Retrieval**: Embeds the user's question and performs a vector similarity search (top-3 results).
2. **Augmentation**: Injects the raw text fragments into a specialized prompt.
3. **Generation**: The LLM answers based *only* on the retrieved context, citing specific chunks (e.g., `[Source, chunk 4]`).

---

## 3. Pipeline 2: LLM Wiki Synthesis

The Wiki pipeline treats documents as a **source for structured synthesis**.

### Ingest Phase (The "Compiler")
1. **Extraction**: Reads the full source text and splits it into large overlapping segments (15,000 characters) to ensure the LLM can process the whole document.
2. **Analysis**: Sends each segment to the LLM with a structural extraction prompt.
3. **Entity Discovery**: The LLM extracts specific entities, definitions, and relations into a JSON structure.
4. **Incremental Merge**: 
    - **New Pages**: Added to the index.
    - **Existing Pages**: Information is appended with a separator (`---`). If contradictions are detected, they are explicitly flagged.
    - **Cross-Referencing**: Automatic pass to detect mentions of page titles within other pages.
5. **Persistence**: The wiki is stored as a structured `index.json` per session.

### Query Phase
1. **Synthesis**: The LLM reads the *pre-compiled* wiki pages as context, instead of the raw source.
2. **Dynamic Learning**: If the answer introduces new concepts or insights, the LLM extracts them into new pages and relations, compounding the wiki's knowledge.
3. **Citation**: Answers use `[Page Title]` notation which links directly to the visual graph.
4. **Graph Rendering**: D3.js renders the `index.json` as a force-directed graph where nodes are entities and edges are relationships.

---

## 4. Key Comparisons

| Feature | RAG | LLM Wiki |
| :--- | :--- | :--- |
| **Data Unit** | Raw Chunks (Fixed Size) | Semantic Pages (Entities/Concepts) |
| **Logic** | Find similar text at query time | Build structured model at ingest time |
| **Visuals** | List of retrieved snippets | Interactive D3.js Knowledge Graph & Page Browser |
| **Conflict Handling**| Shows multiple snippets | Merges and aggregates context |
| **Strengths** | Faster ingest, exact retrieval | Deep synthesis, entity relationship mapping, dynamic knowledge growth |

---

## 5. Technical Specifications

- **Embeddings**: `nomic-embed-text` (Local via Ollama)
- **LLM**: `openai/gpt-oss-120b:free` (OpenRouter) or local fallback.
- **Vector DB**: ChromaDB (Persistent at `app/data/chroma`)
- **Graphing**: D3.js Force Simulation
- **Isolation**: `session_id` used for per-user database and wiki isolation.
- **Concurrency**: Parallel processing for multi-file upload and simultaneous querying.
- **Security**: Split OpenRouter API keys for RAG and Wiki pipelines to monitor quota/usage independently.
