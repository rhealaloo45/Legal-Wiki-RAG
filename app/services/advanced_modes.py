import os
import io
import json
import uuid
import logging
import concurrent.futures
from threading import Lock
from typing import Optional

import config
from services import reader, llm, wiki

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def get_raw_doc_text(session_id: str, doc_name: str) -> str:
    """Retrieve raw text from an uploaded document."""
    file_path = os.path.join(config.UPLOAD_PATH, f"{session_id}_{doc_name}")
    if not os.path.exists(file_path):
        logger.error(f"Failed to find raw doc text at path: {file_path}")
        return ""
    try:
        return reader.read_file(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def extract_cell(doc_text: str, column_name: str) -> dict:
    """Extract a specific piece of information from text."""
    prompt = f"""\
Extract one specific piece of information from legal text.
Return JSON only, no preamble:
{{"value": str|null, "confidence": float 0-1, "quote": str|null}}
Rules: Never infer. Never hallucinate. 
If not found: value=null, confidence=0.0, quote=null.
quote must be verbatim from text, max 100 chars.

Text:
{doc_text[:8000]}

Extract: {column_name}"""

    try:
        raw, _ = llm.ask(prompt, max_tokens=150)
        import re
        # remove potential reasoning block or markdown
        raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'```json', '', raw)
        raw = re.sub(r'```', '', raw)
        parsed = json.loads(raw.strip())
        return {
            "value": parsed.get("value"),
            "confidence": float(parsed.get("confidence", 0.0)),
            "quote": parsed.get("quote")
        }
    except Exception as e:
        logger.error(f"Cell extraction failed for '{column_name}': {e}")
        return {"value": None, "confidence": 0.0, "quote": None}

def _apply_cell_styling(ws, cell, confidence):
    """Apply color styling based on confidence."""
    from openpyxl.styles import PatternFill
    if confidence is None:
        confidence = 0.0
        
    if confidence >= 0.8:
        fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid") # green
    elif confidence >= 0.5:
        fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid") # yellow
    else:
        fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid") # red
    
    cell.fill = fill

def _build_review_sheet(ws, store_data: dict):
    """docs=rows, columns=cols"""
    from openpyxl.styles import Font
    columns = store_data.get("columns", [])
    rows = store_data.get("rows", {})
    
    # Header
    ws.append(["Document Name"] + columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    
    # Rows
    for r_idx, (doc_name, col_data) in enumerate(rows.items(), start=2):
        row_values = [doc_name]
        for col_name in columns:
            cell_data = col_data.get(col_name, {})
            row_values.append(cell_data.get("value"))
            
        ws.append(row_values)
        
        # Style
        for c_idx, col_name in enumerate(columns, start=2):
            cell_data = col_data.get(col_name, {})
            confidence = cell_data.get("confidence", 0.0)
            _apply_cell_styling(ws, ws.cell(row=r_idx, column=c_idx), confidence)

def _build_compare_sheet(ws, store_data: dict):
    """aspects=rows, docs=cols"""
    from openpyxl.styles import Font
    sources = store_data.get("sources", [])
    aspects = store_data.get("aspects", [])
    table = store_data.get("table", {})
    
    doc_headers = [s.get("label", s.get("name")) for s in sources]
    
    # Header
    ws.append(["Aspect"] + doc_headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    
    # Rows
    for r_idx, aspect in enumerate(aspects, start=2):
        row_values = [aspect]
        aspect_data = table.get(aspect, {})
        for s in sources:
            doc_key = s.get("label", s.get("name"))
            cell_data = aspect_data.get(doc_key, {})
            row_values.append(cell_data.get("value"))
            
        ws.append(row_values)
        
        # Style
        for c_idx, s in enumerate(sources, start=2):
            doc_key = s.get("label", s.get("name"))
            cell_data = aspect_data.get(doc_key, {})
            confidence = cell_data.get("confidence", 0.0)
            _apply_cell_styling(ws, ws.cell(row=r_idx, column=c_idx), confidence)

def export_matrix_to_xlsx(store_data: dict, mode: str) -> bytes:
    """Generate Excel bytes for either review or compare store data."""
    try:
        import openpyxl
    except ImportError:
        logger.error("openpyxl not installed.")
        return b""
        
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Export"
    
    if mode == "review":
        _build_review_sheet(ws, store_data)
    elif mode == "compare":
        _build_compare_sheet(ws, store_data)
        
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()

# ---------------------------------------------------------------------------
# Background Jobs
# ---------------------------------------------------------------------------

def _run_review_job(job_id: str, session_id: str, doc_names: list, columns: list, store_ref: dict, locks_ref: dict):
    """Background job for review mode."""
    lock = locks_ref[job_id]
    
    def _extract_worker(doc_name, col_name, doc_text):
        res = extract_cell(doc_text, col_name)
        with lock:
            store_ref[job_id]["rows"][doc_name][col_name] = res
            store_ref[job_id]["completed"] += 1
            if res.get("confidence", 0.0) < 0.5 or res.get("value") is None:
                store_ref[job_id]["flagged"].append([doc_name, col_name])

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for doc_name in doc_names:
                doc_text = get_raw_doc_text(session_id, doc_name)
                for col_name in columns:
                    futures.append(executor.submit(_extract_worker, doc_name, col_name, doc_text))
            
            concurrent.futures.wait(futures)
            
        with lock:
            store_ref[job_id]["status"] = "complete"
            
        wiki._log_event(session_id, "REVIEW", f"Job: {job_id} | Docs: {len(doc_names)} | Cols: {len(columns)} | Flagged: {len(store_ref[job_id]['flagged'])}")
            
    except Exception as e:
        logger.error(f"Review job {job_id} failed: {e}")
        with lock:
            store_ref[job_id]["status"] = "error"
            store_ref[job_id]["error"] = str(e)


def _get_scoped_wiki_pages(session_id: str, doc_name: str) -> dict:
    """Filter wiki index to pages where source_doc == doc_name."""
    index = wiki._load_index(session_id)
    pages = index.get("pages", {})
    scoped = {k: v for k, v in pages.items() if isinstance(v, dict) and v.get("source_doc") == doc_name}
    return scoped

def _run_compare_job(job_id: str, session_id: str, doc_names: list, question: str, 
                     uploaded_text: Optional[str], uploaded_name: Optional[str], temp_path: Optional[str],
                     store_ref: dict, locks_ref: dict):
    """Background job for compare mode."""
    lock = locks_ref[job_id]
    
    try:
        # STEP 1 - NORMALIZE SOURCES
        with lock:
            store_ref[job_id]["stage"] = "retrieving"
            
        sources = []
        for doc_name in doc_names:
            scoped_pages = _get_scoped_wiki_pages(session_id, doc_name)
            if not scoped_pages:
                structured_text = get_raw_doc_text(session_id, doc_name)
            else:
                # BM25 Fallback/Logic
                try:
                    from rank_bm25 import BM25Okapi
                    corpus = []
                    keys = list(scoped_pages.keys())
                    for k in keys:
                        summary = scoped_pages[k].get("summary", "") if isinstance(scoped_pages[k], dict) else ""
                        corpus.append(f"{k} {summary}".lower().split())
                    bm25 = BM25Okapi(corpus)
                    tokenized_query = question.lower().split()
                    top_keys = bm25.get_top_n(tokenized_query, keys, n=5)
                    
                    parts = []
                    for k in top_keys:
                        parts.append(scoped_pages[k].get("content", ""))
                    structured_text = "\n\n".join(parts)
                except Exception as e:
                    logger.error(f"BM25 failed during compare: {e}")
                    structured_text = get_raw_doc_text(session_id, doc_name)
            
            sources.append({"name": doc_name, "type": "wiki", "text": structured_text})
            
        if uploaded_text and uploaded_name:
            sources.append({"name": uploaded_name, "type": "upload", "text": uploaded_text[:12000], "label": f"{uploaded_name} ⚡"})
            
        with lock:
            store_ref[job_id]["sources"] = [{"name": s["name"], "type": s["type"], "label": s.get("label", s["name"])} for s in sources]
            
        # STEP 2 - ASPECT IDENTIFICATION
        with lock:
            store_ref[job_id]["stage"] = "extracting"
            
        aspect_prompt = f"""\
Question: {question}
Documents being compared: {[s['name'] for s in sources]}
List 4-6 specific aspects to extract for this comparison.
Return JSON only: {{"aspects": [str]}}"""

        raw, _ = llm.ask(aspect_prompt, max_tokens=150)
        import re
        raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'```json', '', raw)
        raw = re.sub(r'```', '', raw)
        parsed = json.loads(raw.strip())
        aspects = parsed.get("aspects", [])
        
        with lock:
            store_ref[job_id]["aspects"] = aspects
            for aspect in aspects:
                store_ref[job_id]["table"][aspect] = {}
        
        # STEP 3 - EXTRACT PER SOURCE PER ASPECT
        def _extract_compare_worker(source_dict, aspect):
            res = extract_cell(source_dict["text"], aspect)
            doc_key = source_dict.get("label", source_dict["name"])
            with lock:
                store_ref[job_id]["table"][aspect][doc_key] = res
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for s in sources:
                for a in aspects:
                    futures.append(executor.submit(_extract_compare_worker, s, a))
            concurrent.futures.wait(futures)
            
        # STEP 4 - OUTLIER DETECTION
        with lock:
            store_ref[job_id]["stage"] = "analyzing"
            
        outliers = []
        for aspect in aspects:
            vals = {}
            for s in sources:
                doc_key = s.get("label", s["name"])
                val = store_ref[job_id]["table"][aspect][doc_key].get("value")
                if val:
                    vals[doc_key] = val
                    
            outlier_prompt = f"""\
Values for {aspect}: {json.dumps(vals)}
Identify any outliers or contradictions. 
Return JSON only: 
[{{"doc": str, "reason": str}}] or []"""

            raw, _ = llm.ask(outlier_prompt, max_tokens=100)
            raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
            raw = re.sub(r'```json', '', raw)
            raw = re.sub(r'```', '', raw)
            try:
                parsed_outliers = json.loads(raw.strip())
                if isinstance(parsed_outliers, list):
                    for po in parsed_outliers:
                        po["aspect"] = aspect
                        outliers.append(po)
            except:
                pass
                
        with lock:
            store_ref[job_id]["outliers"] = outliers
            
        # STEP 5 - NARRATIVE GENERATION
        narrative_prompt = f"""\
Question: {question}
Comparison table: {json.dumps(store_ref[job_id]["table"])}
Outliers: {json.dumps(outliers)}
Write a 3-5 sentence legal synthesis. 
Cite doc names inline. Flag contradictions explicitly."""

        narrative, _ = llm.ask(narrative_prompt, max_tokens=400)
        
        # STEP 6 - STORE + COMPLETE
        with lock:
            store_ref[job_id]["narrative"] = narrative
            store_ref[job_id]["status"] = "complete"
            
        wiki._log_event(session_id, "COMPARE", f"Job: {job_id} | Docs: {len(doc_names) + (1 if uploaded_text else 0)} | Aspects: {len(aspects)} | Outliers: {len(outliers)}")
        
    except Exception as e:
        logger.error(f"Compare job {job_id} failed: {e}")
        with lock:
            store_ref[job_id]["status"] = "error"
            store_ref[job_id]["error"] = str(e)
            
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Failed to clean up temp file {temp_path}: {e}")
