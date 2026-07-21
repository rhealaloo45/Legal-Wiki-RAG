# Opik Tracing & Evaluation Guide

This guide explains how to run Comet ML's Opik tracing platform locally, integrate it into agentic AI workflows, and prepare the RAG application for production using Docker.

## 1. Running Opik Locally

We use a custom `docker-compose.opik.yml` stack to run Opik entirely locally. It includes ClickHouse, Redis, Zookeeper, MySQL, MinIO, and the Opik Backend/Frontend.

### Prerequisites
- Docker & Docker Compose installed.
- Ensure ports `5173` (Frontend) and `8080` (Backend) are free.

### Starting the Stack
1. Open a terminal in the root directory.
2. Run the following command:
   ```bash
   docker compose -f docker-compose.opik.yml up -d
   ```
3. The stack will start. Note that **ClickHouse** takes some time to initialize and run database migrations. Wait for the `opik-backend-1` container to report as healthy.
4. Access the Opik dashboard at: **http://localhost:5173**

## 2. Integrating Opik with Agents

To trace agents and LLMs, Opik provides a lightweight SDK. In this project, we integrated it via the `app/services/opik_tracing.py` wrapper.

### Step 1: Environment Variables
Ensure your application knows where the local Opik instance is hosted. Set this in your `.env` file:
```env
OPIK_URL_OVERRIDE=http://localhost:5173
OPIK_PROJECT_NAME=legal-wiki-rag
```

### Step 2: Code Integration
You can use the `@track` decorator on any function to log its inputs, outputs, and execution time to Opik.

```python
from opik import track

@track(name="llm_agent_query", type="llm")
def query_agent(prompt: str) -> str:
    # Your agent logic here (e.g. LiteLLM, LangChain, or direct OpenAI calls)
    response = llm.generate(prompt)
    return response
```

### Step 3: LLM-as-a-Judge Metrics
Opik supports automated evaluations (like Hallucination and Answer Relevance). We use these in the `/query` endpoint:
```python
from opik.evaluation.metrics import Hallucination, AnswerRelevance

# Evaluate an answer based on the provided context
hallucination_metric = Hallucination()
score = hallucination_metric.score(
    input=user_question, 
    output=agent_answer, 
    context=[retrieved_docs]
)
```

## 3. Creating Docker Images for Production

To deploy the entire RAG application alongside Opik in a production environment, you should containerize the Flask application.

### Dockerfile (App)
Create a `Dockerfile` in the `app/` directory:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (e.g. for OCR)
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
```

### Production `docker-compose.yml`
In production, you merge your app with the Opik stack. Add an `app` service to your compose file:
```yaml
services:
  app:
    build: 
      context: ./app
    ports:
      - "5001:5001"
    environment:
      - OPIK_URL_OVERRIDE=http://opik-frontend:5173
      - OPIK_PROJECT_NAME=legal-wiki-rag
    depends_on:
      opik-frontend:
        condition: service_started
```

### Build & Deploy
1. Build the production image:
   ```bash
   docker compose -f docker-compose.prod.yml build
   ```
2. Run the full production stack:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```
