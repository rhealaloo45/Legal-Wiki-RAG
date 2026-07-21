# Docker Operations Guide: Legal-Wiki-RAG & Opik

This document provides complete instructions on how to start, stop, monitor, and troubleshoot Docker containers for both **Opik Tracing** and **Legal-Wiki-RAG**.

---

## 1. Stack Architecture Overview

The system consists of two separate Docker Compose stacks:

1. **Opik Stack (`docker-compose.opik.yml`)**
   - **Opik Frontend (Nginx)**: Port `5173` (Dashboard UI & API proxy)
   - **Opik Backend**: Port `8080` (Java REST API)
   - **Opik Storage**: ClickHouse, Redis, MySQL, Zookeeper, MinIO

2. **Legal-Wiki-RAG Stack (`docker-compose.yml`)**
   - **Legal-Wiki Web App**: Port `5001` (Flask application)
   - **Legal-Wiki Database**: Port `5433` (PostgreSQL 17 + `pgvector`)

---

## 2. Starting the Services

### Option A: Complete Docker Setup (Recommended for Full Containerization)

#### Step 1: Start Opik Tracing Stack
Run Opik in detached mode (`-d`):
```bash
docker compose -f docker-compose.opik.yml up -d
```
> ⏳ **Note**: ClickHouse database migrations take ~1-2 minutes on first startup.
> Verify Opik is ready at: **http://localhost:5173**

#### Step 2: Start Legal-Wiki-RAG Stack
Build and start Legal-Wiki-RAG:
```bash
docker compose up -d --build
```
> Verify Legal-Wiki-RAG is running at: **http://localhost:5001**

---

### Option B: Mixed Setup (Local Python App + Docker Services)

If you prefer running `py app.py` locally for fast development/code hot-reloading:

1. **Start Opik & Postgres DB in Docker**:
   ```bash
   docker compose -f docker-compose.opik.yml up -d
   docker compose up -d db
   ```

2. **Run Python App locally**:
   ```bash
   cd app
   python app.py
   ```
   > Access Legal-Wiki-RAG at: **http://127.0.0.1:5001**

---

## 3. Stopping the Services

### Stop Legal-Wiki-RAG
To stop the web application and PostgreSQL container (data volumes preserved):
```bash
docker compose down
```

### Stop Opik Tracing Stack
To stop all Opik backend/frontend services:
```bash
docker compose -f docker-compose.opik.yml down
```

### Stop a Specific Container
If you want to stop a single container (e.g. freeing port `5001` to run `py app.py` locally):
```bash
docker stop legal-wiki-rag-web-1
```

---

## 4. Monitoring & Diagnostics

### View Running Containers & Ports
```bash
docker ps
```

### View Real-Time Logs

- **Legal-Wiki App Logs**:
  ```bash
  docker logs legal-wiki-rag-web-1 -f
  ```

- **Opik Backend Logs**:
  ```bash
  docker logs opik-backend-1 -f
  ```

- **Opik Frontend Logs**:
  ```bash
  docker logs opik-frontend-1 -f
  ```

---

## 5. Troubleshooting & Useful Commands

- **Freeing Port Conflicts**:
  If port `5001` or `5173` is reported as occupied:
  ```bash
  docker stop legal-wiki-rag-web-1 opik-frontend-1
  ```

- **Clean Reset (Wipe Database Volumes)**:
  To completely reset databases and volumes:
  ```bash
  docker compose down -v
  docker compose -f docker-compose.opik.yml down -v
  ```
