# ⚖️ Legal Wiki

Legal Wiki is a research tool built as a Single-Page Flask Application to process large-scale document knowledge using **LLM Wiki Synthesis**. 🚀

## ✨ Features

- **🧠 Wiki Pipeline**: LLM-driven structured knowledge extraction that compounds over time into a persistent index, complete with automatic cross-referencing and interactive D3.js graphs.
- **✍️ Draft Mode**: Context-aware legal drafting with automatic stance detection (e.g. Tata-friendly, Neutral) and DOCX export.
- **📊 Review Mode**: Concurrently extract structured cells across multiple documents into a confidence-coded Excel export.
- **🔄 Compare Mode**: Automatically identify comparison aspects across existing wiki docs + new uploaded docs, flag outliers, and generate narratives.
- **📈 Deep Understanding**: Granular progress tracking, interactive knowledge graphs, and wiki page browsing.
- **🔒 Session Isolation**: Every chat session maintains its own completely independent wiki, isolating knowledge context and document uploads securely.
- **🎨 Premium UI**: A custom-built, lightweight Light Mode Single-Page Application (SPA) designed without heavy frontend frameworks, offering a seamless and context-aware experience.
- **☁️ Azure OpenAI & OpenRouter Powered**: Configurable to use either Azure OpenAI or OpenRouter for all LLM extraction, embedding, synthesis, and querying tasks.

## 🏗️ Project Architecture

For a detailed technical and architectural breakdown of how the pipeline operates, please refer to the [System Overview](SYSTEM_OVERVIEW.md) 📘 and [System Flow](SYSTEM_FLOW.md) 📄.

## 🛠️ Setup & Installation

## Prerequisites

- Python 3.9+ 🐍
- Azure OpenAI credentials OR an OpenRouter API key 🌐
- (Optional) Tesseract OCR executable installed on your system if you need to process scanned/image-based PDFs.

### Step 1: Environment Setup ⚙️

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd Legal-wiki-RAG/app
pip install -r requirements.txt
```

### Step 2: Configuration 🔑

Create a `.env` file in the `app/` directory by copying the `.env.example`:

```bash
cp .env.example .env
```

Edit `app/.env` to configure your selected provider:

#### Option A: OpenRouter Config
```env
LLM_PROVIDER=openrouter
EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
```

#### Option B: Azure OpenAI Config
```env
LLM_PROVIDER=azure
EMBEDDING_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
EMBEDDING_DIMENSIONS=1536
```

#### OCR Config (Optional)
If Tesseract OCR is not on your system PATH, define its path:
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Step 3: Run the Application 🚀

Start the Flask server from the `app/` directory:

```bash
python app.py
```

The application will be accessible at `http://localhost:5001/`. 🌐

## 📖 Usage

1. **📄 Upload Documents**: Drag and drop `.pdf` or `.txt` files into the upload card. 
2. **⏳ Monitor Ingestion**: The system processes documents in parallel and generates interconnected Wiki pages.
3. **🌐 Explore Knowledge**: Browse generated Wiki pages and click them to view detailed structured text and source citations.
4. **❓ Query**: Ask a question in the Ask tab to generate a synthesized answer with inline citations.
5. **✍️ Draft**: Use the Draft tab to generate, refine, and export legal clauses and agreements grounded in your wiki's knowledge.

## 📜 License

MIT 📝
