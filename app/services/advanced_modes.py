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
    flat_name = doc_name.replace("/", "_").replace("\\", "_")
    file_path = os.path.join(config.UPLOAD_PATH, f"{session_id}_{flat_name}")
    
    if not os.path.exists(file_path):
        # Resilient fallback: search across all uploads for any session matching flat_name
        found = False
        for fname in os.listdir(config.UPLOAD_PATH):
            if fname.endswith(flat_name):
                file_path = os.path.join(config.UPLOAD_PATH, fname)
                found = True
                break
        if not found:
            logger.error(f"Failed to find raw doc text at path: {file_path} or any fallback")
            return ""
            
    try:
        return reader.read_file(file_path)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""

def _get_wiki_text_for_doc(session_id: str, doc_name: str) -> str:
    """Retrieve synthesized wiki content for a document.
    
    Looks up wiki pages tagged with this source_doc and concatenates their
    content. This is much smaller and more focused than raw document text,
    leading to faster and more accurate extraction.
    Falls back to raw text if no wiki pages are found.
    
    The output preserves source attribution and supporting quotes embedded
    during wiki ingest, so downstream LLM calls can cite them.
    """
    scoped = _get_scoped_wiki_pages(session_id, doc_name)
    if scoped:
        parts = [f"[Source Document: {doc_name}]"]
        for title, page_data in scoped.items():
            content = page_data.get("content", "") if isinstance(page_data, dict) else str(page_data)
            summary = page_data.get("summary", "") if isinstance(page_data, dict) else ""
            parts.append(f"## {title}\n{summary}\n{content}")
        wiki_text = "\n\n".join(parts)
        if len(wiki_text) > 200:  # Only use wiki if substantial content exists
            return wiki_text
    
    # Fallback to raw text — prefix with source doc name for traceability
    raw = get_raw_doc_text(session_id, doc_name)
    if raw:
        return f"[Source Document: {doc_name}]\n\n{raw}"
    return ""

# Max chars of context to send per cell extraction call.
# Set to 6000 to ensure wiki quotes and multi-page content aren't truncated.
_CELL_CONTEXT_LIMIT = 6000

def extract_cell(doc_text: str, column_name: str) -> dict:
    """Extract a specific piece of information from text using fast LLM path."""
    prompt = f"""\
Extract the specific piece of information requested from the legal text below.
Return JSON only, no preamble:
{{"value": str|null, "confidence": float 0-1, "quote": str|null}}

STRICT RULES:
- The "value" field must be a concise, informative summary (1-3 sentences) explaining the extracted information clearly, not just a single keyword.
- ONLY extract information that is EXPLICITLY stated in the text below.
- If the requested information is NOT found in the text, you MUST return null for the "value" field. Do NOT write sentences explaining that it is missing.
- DO NOT infer, assume, or hallucinate any values.
- The "quote" field MUST be a verbatim excerpt copied exactly from the text (max 120 chars).
  It serves as evidence for your extracted value.
- If the text contains "Supporting Quotes" sections, prefer those as your quote source.
- If the information is not found anywhere in the text: value=null, confidence=0.0, quote=null.
- confidence should reflect how directly the text states the information:
  1.0 = exact verbatim match, 0.8 = clearly stated, 0.5 = implied, 0.0 = not found.

Text:
{doc_text[:_CELL_CONTEXT_LIMIT]}

Extract: {column_name}"""

    try:
        raw, _ = llm.fast_ask(prompt, max_tokens=300)
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
        
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter
    
    # Auto-adjust column widths and enable text wrapping for all cells
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 35 if col_idx == 1 else 55
        
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()

# ---------------------------------------------------------------------------
# Background Jobs
# ---------------------------------------------------------------------------

# Timeout for individual future results (seconds)
_FUTURE_TIMEOUT = 90

def _run_review_job(job_id: str, session_id: str, doc_names: list, columns: list, store_ref: dict, locks_ref: dict):
    """Background job for review mode.
    
    Optimisation: uses wiki-synthesized content instead of raw document text
    where available, dramatically reducing prompt size and latency.
    """
    lock = locks_ref[job_id]
    
    def _extract_worker(doc_name, col_name, doc_text):
        try:
            res = extract_cell(doc_text, col_name)
        except Exception as e:
            logger.error(f"Worker failed for {doc_name}/{col_name}: {e}")
            res = {"value": None, "confidence": 0.0, "quote": None}
        with lock:
            store_ref[job_id]["rows"][doc_name][col_name] = res
            store_ref[job_id]["completed"] += 1
            if res.get("confidence", 0.0) < 0.5 or res.get("value") is None:
                store_ref[job_id]["flagged"].append([doc_name, col_name])

    try:
        # Pre-fetch all document texts (wiki-first, raw fallback)
        doc_texts = {}
        for doc_name in doc_names:
            doc_texts[doc_name] = _get_wiki_text_for_doc(session_id, doc_name)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for doc_name in doc_names:
                doc_text = doc_texts[doc_name]
                for col_name in columns:
                    futures.append(executor.submit(_extract_worker, doc_name, col_name, doc_text))
            
            # Wait with timeout to prevent indefinite hanging
            done, not_done = concurrent.futures.wait(futures, timeout=_FUTURE_TIMEOUT * len(futures) / 5 + 30)
            
            # Cancel any stragglers
            for f in not_done:
                f.cancel()
                with lock:
                    store_ref[job_id]["completed"] += 1
            
        with lock:
            store_ref[job_id]["status"] = "complete"
            
        wiki._log_event(session_id, "REVIEW", f"Job: {job_id} | Docs: {len(doc_names)} | Cols: {len(columns)} | Flagged: {len(store_ref[job_id]['flagged'])}")
            
    except Exception as e:
        logger.error(f"Review job {job_id} failed: {e}")
        with lock:
            store_ref[job_id]["status"] = "error"
            store_ref[job_id]["error"] = str(e)


def _get_scoped_wiki_pages(session_id: str, doc_name: str) -> dict:
    """Filter wiki index to pages where source_doc matches doc_name."""
    index = wiki._load_index(session_id)
    pages = index.get("pages", {})
    
    # Normalize doc_name to match how wiki.py saves source_doc
    flat_name = doc_name.replace("/", "_").replace("\\", "_")
    expected_source_doc = f"{session_id}_{flat_name}"
    
    scoped = {k: v for k, v in pages.items() if isinstance(v, dict) and (v.get("source_doc") == expected_source_doc or v.get("source_doc") == doc_name or v.get("source_doc") == flat_name)}
    return scoped

def _run_compare_job(job_id: str, session_id: str, doc_names: list, question: str, 
                     uploaded_text: Optional[str], uploaded_name: Optional[str], temp_path: Optional[str],
                     store_ref: dict, locks_ref: dict):
    """Background job for compare mode.
    
    Optimisations over the original implementation:
    - Uses fast_ask for cell extraction calls
    - Batches outlier detection into a single LLM call instead of per-aspect
    - Uses wiki content with BM25 pre-filtering
    """
    lock = locks_ref[job_id]
    import re
    
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
                # BM25 retrieval from wiki pages
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
            
        # STEP 2 - ASPECT IDENTIFICATION (single fast call)
        with lock:
            store_ref[job_id]["stage"] = "extracting"
            
        aspect_prompt = f"""\
User Query: {question}
Documents being compared: {[s['name'] for s in sources]}

Based on the User Query, list the specific factual aspects that need to be extracted for comparison.
If the query explicitly asks for certain points (e.g., "limits of liability, liability caps"), use those exact points as the aspects.
If the query is open-ended (e.g., "Compare these documents"), list 4-6 key legal and commercial aspects to compare.
Aspects must be concrete, extractable data points (e.g., "Liability Cap", "Termination Period"), NOT abstract concepts.
Return JSON only: {{"aspects": [str]}}"""

        raw, _ = llm.fast_ask(aspect_prompt, max_tokens=150)
        raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
        raw = re.sub(r'```json', '', raw)
        raw = re.sub(r'```', '', raw)
        parsed = json.loads(raw.strip())
        aspects = parsed.get("aspects", [])
        
        with lock:
            store_ref[job_id]["aspects"] = aspects
            for aspect in aspects:
                store_ref[job_id]["table"][aspect] = {}
        
        # STEP 3 - EXTRACT PER SOURCE PER ASPECT (parallel with fast_ask)
        def _extract_compare_worker(source_dict, aspect):
            try:
                res = extract_cell(source_dict["text"], aspect)
            except Exception as e:
                logger.error(f"Compare extract failed {source_dict['name']}/{aspect}: {e}")
                res = {"value": None, "confidence": 0.0, "quote": None}
            doc_key = source_dict.get("label", source_dict["name"])
            with lock:
                store_ref[job_id]["table"][aspect][doc_key] = res
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for s in sources:
                for a in aspects:
                    futures.append(executor.submit(_extract_compare_worker, s, a))
            
            # Wait with timeout
            total_cells = len(sources) * len(aspects)
            timeout = max(60, total_cells * 15)  # ~15s per cell, minimum 60s
            done, not_done = concurrent.futures.wait(futures, timeout=timeout)
            
            # Fill in failures for timed-out cells
            for f in not_done:
                f.cancel()
            
        # STEP 4 - OUTLIER DETECTION (batched into a single LLM call)
        with lock:
            store_ref[job_id]["stage"] = "analyzing"
            
        # Build a compact summary of all values for batch outlier detection
        all_values = {}
        for aspect in aspects:
            aspect_vals = {}
            for s in sources:
                doc_key = s.get("label", s["name"])
                cell_data = store_ref[job_id]["table"].get(aspect, {}).get(doc_key, {})
                val = cell_data.get("value")
                if val:
                    aspect_vals[doc_key] = val
            if aspect_vals:
                all_values[aspect] = aspect_vals

        outliers = []
        if all_values:
            outlier_prompt = f"""\
Compare the extracted values below across documents. Identify ONLY contradictions
or significant differences that are DIRECTLY visible in the data provided.
DO NOT infer, speculate, or flag differences that are not clearly present.
Return JSON only — an array of objects:
[{{"aspect": str, "doc": str, "reason": str}}] or []
Return [] if no clear contradictions exist.

Extracted values:
{json.dumps(all_values, indent=1)}"""

            try:
                raw, _ = llm.fast_ask(outlier_prompt, max_tokens=1000)
                raw = re.sub(r'<reasoning>.*?</reasoning>', '', raw, flags=re.DOTALL)
                raw = re.sub(r'```json', '', raw)
                raw = re.sub(r'```', '', raw)
                parsed_outliers = json.loads(raw.strip())
                if isinstance(parsed_outliers, list):
                    outliers = parsed_outliers
            except Exception as e:
                logger.error(f"Outlier detection failed: {e}")
                
        with lock:
            store_ref[job_id]["outliers"] = outliers
            
        # STEP 5 - NARRATIVE GENERATION (single call, using ask for quality)
        narrative_prompt = f"""\
Question: {question}
Comparison table: {json.dumps(store_ref[job_id]["table"])}
Outliers: {json.dumps(outliers)}

Write a well-structured legal synthesis based ONLY on the comparison table data above.

STRICT RULES:
- Format your response using markdown with clear headings, bullet points, and bold text for readability.
- Write fluid, cohesive summary paragraphs describing trends across the documents. DO NOT output repetitive nested lists detailing every single document's individual value. 
- Group similar findings together into single sentences (e.g. "Most documents do not specify a notice period, except [1] which requires 30 days").
- Do NOT redundantly enumerate "not specified" for every document. Synthesize it.
- Cite your sources using standard IEEE format inline (e.g., [1], [2], [3]). Do NOT mention the exact document names anywhere in the paragraph text.
- ONLY state facts that appear in the comparison table. DO NOT add information not present in the data.
- Flag contradictions explicitly, citing both documents and their conflicting values verbatim.
- DO NOT invent legal conclusions, implications, or recommendations beyond what the data shows.
- PROPER CITATIONS (CRITICAL): You MUST create a "References" list at the very end of your answer starting with a "References" heading. Each entry must strictly follow this pattern: "[X] File_Name.pdf, Clause/Page" (e.g. "[1] Service Agreement 1_redacted.pdf, Clause 14.1"). If the exact clause/page is not in the table, just map it to the document name: "[1] Service Agreement 1_redacted.pdf". Do not wrap file names in formatting."""

        narrative, _ = llm.ask(narrative_prompt, max_tokens=1500)
        
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
