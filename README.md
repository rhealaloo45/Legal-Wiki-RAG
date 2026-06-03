# ⚖️ Legal Wiki

Legal Wiki is a research tool built as a Single-Page Flask Application to process large-scale document knowledge using **LLM Wiki Synthesis**. 🚀

## ✨ Features

- **🧠 Wiki Pipeline**: LLM-driven structured knowledge extraction that compounds over time into a persistent index, complete with automatic cross-referencing and interactive D3.js graphs.
- **📊 Review Mode**: Concurrently extract structured cells across multiple documents into a confidence-coded Excel export.
- **🔄 Compare Mode**: Automatically identify comparison aspects across existing wiki docs + new uploaded docs, flag outliers, and generate narratives.
- **📈 Deep Understanding**: Granular progress tracking, interactive knowledge graphs, and wiki page browsing.
- **🔒 Session Isolation**: Every chat session maintains its own completely independent wiki, isolating knowledge context and document uploads securely.
- **🎨 Premium UI**: A custom-built, lightweight Light Mode Single-Page Application (SPA) designed without heavy frontend frameworks, offering a seamless and context-aware experience.
- **☁️ Azure OpenAI Powered**: Uses Azure OpenAI for all LLM extraction, synthesis, and querying tasks.

## 🏗️ Project Architecture

For a detailed technical and architectural breakdown of how the pipeline operates, please refer to the [System Overview](SYSTEM_OVERVIEW.md) 📘 and [System Flow](SYSTEM_FLOW.md) 📄.

## 🛠️ Setup & Installation

### 📋 Prerequisites

- Python 3.9+ 🐍
- Azure OpenAI account and credentials 🌐

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
cp app/.env.example app/.env
```

Edit `app/.env` to include your Azure OpenAI API keys, endpoint, and deployment name:

```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Step 3: Run the Application 🚀

Start the Flask server from the `app/` directory:

```bash
cd app
python app.py
```

The application will be accessible at `http://localhost:5001/`. 🌐

## 📖 Usage

1. **📄 Upload Documents**: Drag and drop `.pdf` or `.txt` files into the upload card. 
2. **⏳ Monitor Ingestion**: The system processes documents in parallel and generates interconnected Wiki pages.
3. **🌐 Explore Knowledge**: Browse generated Wiki pages and click them to view detailed structured text and source citations.
4. **❓ Query**: Ask a question in the Ask tab to generate a synthesized answer with inline citations.

## 📜 License

MIT 📝
