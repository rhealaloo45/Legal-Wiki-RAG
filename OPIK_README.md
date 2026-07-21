# Opik Integration Guide

This guide explains how to run Comet ML's Opik platform locally, how it integrates with the AI agents in this codebase, and how to build Docker images for production.

## 1. Running Opik Locally

We have configured a lightweight, self-hosted Opik stack using Docker Compose. This stack includes the Opik backend, frontend (UI), ClickHouse (for traces), MySQL, Redis, and MinIO.

### Starting the Stack

To start the local Opik platform, run:

```bash
docker compose -f docker-compose.opik.yml up -d
```

*(Note: The first time you run this, it will take a few minutes to pull the necessary Docker images.)*

### Accessing the UI

Once the containers are healthy, the Opik dashboard will be available at:
**http://localhost:5173**

### Stopping the Stack

To stop the Opik services:
```bash
docker compose -f docker-compose.opik.yml down
```

To stop the services and wipe all traced data (volumes):
```bash
docker compose -f docker-compose.opik.yml down -v
```

---

## 2. Integrating with the Application (Agents)

The main application integrates with Opik via the Python SDK (`opik`). 

### Enabling Tracing

To tell the Legal-Wiki-RAG application to send traces to your local Opik instance, you need to set the `OPIK_URL_OVERRIDE` environment variable.

If you are running the application via `docker compose up`, open `docker-compose.yml` and uncomment the Opik section under the `web` service environment variables:

```yaml
    environment:
      # ... other vars ...
      # ── Opik tracing (optional) ──────────────────────────────────────────────
      OPIK_URL_OVERRIDE: http://host.docker.internal:8080
      OPIK_PROJECT_NAME: legal-wiki-rag
```

If you are running the application natively (without Docker), set the variable in your `app/.env` file:
```env
OPIK_URL_OVERRIDE=http://localhost:8080
OPIK_PROJECT_NAME=legal-wiki-rag
```

### How Tracing Works in the Codebase

1. **`app/services/opik_tracing.py`**: This helper module initializes the Opik SDK if the `OPIK_URL_OVERRIDE` is present. It provides a conditional `@track` decorator that safely becomes a no-op if Opik is disabled.
2. **LLM Calls (`app/services/llm.py`)**: The `ask()` and `fast_ask()` functions are decorated with `@opik_tracing.track(type="llm")`. Every prompt, completion, and token usage count is automatically logged.
3. **Agent Pipeline (`app/services/intent_agent.py`)**: 
    - The `validate_response_node` is traced as a general span.
    - After the application generates an answer, we call `opik_tracing.run_evals()`.
    - This triggers Opik's built-in **LLM-as-a-judge** metrics:
        - **Hallucination:** Checks if the answer contains claims not present in the retrieved context.
        - **Answer Relevance:** Checks if the answer directly addresses the user's question.
        - **Context Precision:** Checks if the retrieved documents were actually useful.
    - The scores are attached to the active Opik trace as "Feedback Scores" so you can filter and visualize them in the UI.

---

## 3. Creating Docker Images for Production

To deploy the application to a production environment, you need to build the Docker image. The `Dockerfile` has been optimized to exclude unnecessary build dependencies (like `gcc` and `tesseract-ocr` if not required by your system).

### Building the Image

To build the production image manually:

```bash
docker build -t legal-wiki-rag-web:latest .
```

Or, using Docker Compose:

```bash
docker compose build --no-cache web
```

### Production Considerations for Opik

1. **Managed vs. Self-Hosted**: 
   - **Local/Dev**: Use the lightweight `docker-compose.opik.yml` provided in this repository.
   - **Production**: It is highly recommended to use **Comet ML's managed Opik Cloud** to avoid managing databases (ClickHouse, MySQL) at scale.
2. **Switching to Managed Opik**: 
   To use Opik Cloud in production, simply **do not set** `OPIK_URL_OVERRIDE`. Instead, set the standard Opik authentication tokens in your production environment variables:
   ```env
   OPIK_API_KEY=your_comet_api_key
   OPIK_WORKSPACE=your_workspace_name
   OPIK_PROJECT_NAME=legal-wiki-rag
   ```
   The SDK will automatically route traces to the cloud.
3. **Performance**: LLM-as-a-judge evaluations (`Hallucination`, etc.) require additional LLM calls. If latency or cost is a strict concern in production, you may want to sample requests or run evaluations asynchronously outside the main request lifecycle.
