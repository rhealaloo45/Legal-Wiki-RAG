# Legal Wiki RAG

Legal Wiki RAG is a research tool built as a Single-Page Flask Application to compare two paradigms of querying large-scale document knowledge: **Retrieval-Augmented Generation (RAG)** vs **LLM Wiki Synthesis**.

## Features

- **Side-by-Side Comparison**: Upload `.txt` or `.pdf` files to ingest data into both pipelines and run parallel queries.
- **RAG Pipeline**: Vector-based semantic search using local embeddings via Ollama and ChromaDB.
- **Wiki Pipeline**: LLM-driven structured knowledge extraction that compounds over time into a persistent index, complete with automatic cross-referencing and interactive D3.js graphs.
- **Deep Understanding**: Granular progress tracking, deep insight views into retrieved chunks (RAG), and interactive knowledge graphs + wiki page browsing.
- **Local & Cloud LLMs**: Configurable to use OpenRouter (cloud LLM) or local models via Ollama.

## Project Architecture

For a detailed technical and architectural breakdown of how both pipelines operate, please refer to the [System Overview](SYSTEM_OVERVIEW.md).

## Setup & Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally
- (Optional) [OpenRouter](https://openrouter.ai/) account for cloud LLM usage

### Step 1: Environment Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd Legal-wiki-RAG/app
pip install -r requirements.txt
```

### Step 2: Configuration

Create a `.env` file in the `app/` directory by copying the `.env.example`:

```bash
cp app/.env.example app/.env
```

Edit `app/.env` to include your OpenRouter API keys if you plan to use `LLM_PROVIDER=openrouter`. Ensure the following models are pulled via Ollama if running locally:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Step 3: Run the Application

Start the Flask server from the `app/` directory:

```bash
cd app
python app.py
```

The application will be accessible at `http://localhost:5000/`.

## Usage

1. **Upload Documents**: Drag and drop `.pdf` or `.txt` files into the upload card. 
2. **Monitor Ingestion**: The system processes documents in parallel and tracks chunking, RAG embeddings, and Wiki page generation separately.
3. **Explore Knowledge**: Browse generated Wiki pages and click them to view detailed structured text.
4. **Query**: Ask a question. The application queries both the RAG pipeline and Wiki pipeline concurrently and displays side-by-side answers, citations, and metrics (latency and tokens).

## License

MIT
