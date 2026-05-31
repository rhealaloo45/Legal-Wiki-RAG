import re
import os

HTML_PATH = "/Users/rhea/Desktop/Rhea Code/Legal-Wiki-RAG/app/templates/index.html"

def patch_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Add Tabs right inside the container
    tabs_html = """
  <!-- Mode Tabs -->
  <ul class="nav nav-pills mb-4" id="modeTabs" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="ask-tab" data-bs-toggle="pill" data-bs-target="#ask-view" type="button" role="tab">Ask</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="compare-tab" data-bs-toggle="pill" data-bs-target="#compare-view" type="button" role="tab">Compare</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="review-tab" data-bs-toggle="pill" data-bs-target="#review-view" type="button" role="tab">Review</button>
    </li>
  </ul>

  <div class="tab-content" id="modeTabsContent">
    <!-- ASK VIEW -->
    <div class="tab-pane fade show active" id="ask-view" role="tabpanel">
"""

    if "<!-- Mode Tabs -->" not in html:
        html = html.replace('<div class="container-fluid px-4 py-4" style="max-width:1320px">', 
                            '<div class="container-fluid px-4 py-4" style="max-width:1320px">' + tabs_html)

    # 2. Close Ask View and add Compare / Review Views just before modals
    closing_tabs = """
    </div> <!-- /ask-view -->
    
    <!-- COMPARE VIEW -->
    <div class="tab-pane fade" id="compare-view" role="tabpanel">
      <div class="row g-4">
        <div class="col-12 col-md-4">
          <div class="glass-card p-4 h-100">
            <h5 class="mb-3">1. Select Existing Documents</h5>
            <div id="compare-doc-list" class="mb-4" style="max-height: 200px; overflow-y: auto;">
              <p class="text-secondary small">No documents ingested yet.</p>
            </div>
            
            <h5 class="mb-3">2. Upload Document (Optional)</h5>
            <div class="upload-zone p-3 mb-4" id="compare-upload-zone">
              <input type="file" id="compare-file" accept=".txt,.pdf" class="d-none">
              <p class="mb-0 text-light small">Click to upload document</p>
              <p class="text-secondary small mt-1 mb-0" id="compare-staged-label"></p>
            </div>
            
            <h5 class="mb-3">3. Compare Topic</h5>
            <input type="text" class="form-control mb-3" id="compare-q" placeholder="e.g. Limitation of Liability">
            <button class="btn btn-accent w-100" id="run-compare-btn">Run Compare</button>
          </div>
        </div>
        <div class="col-12 col-md-8">
          <div class="glass-card p-4 h-100">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <h5 class="mb-0">Compare Results</h5>
              <button class="btn btn-outline-light btn-sm d-none" id="compare-export-btn">Export to Excel</button>
            </div>
            <div id="compare-status" class="mb-3 text-secondary small"></div>
            <div class="table-responsive d-none" id="compare-table-wrap">
              <table class="table table-dark table-bordered table-sm" id="compare-table">
                <thead id="compare-thead"></thead>
                <tbody id="compare-tbody"></tbody>
              </table>
            </div>
            <div id="compare-outliers-wrap" class="mt-4 d-none">
              <h6>Outliers & Contradictions</h6>
              <ul id="compare-outliers-list" class="text-danger small"></ul>
            </div>
            <div id="compare-narrative-wrap" class="mt-4 d-none">
              <h6>Narrative Synthesis</h6>
              <p id="compare-narrative" class="text-light small"></p>
            </div>
          </div>
        </div>
      </div>
    </div> <!-- /compare-view -->
    
    <!-- REVIEW VIEW -->
    <div class="tab-pane fade" id="review-view" role="tabpanel">
      <div class="row g-4">
        <div class="col-12 col-md-4">
          <div class="glass-card p-4 h-100">
            <h5 class="mb-3">1. Select Documents</h5>
            <div id="review-doc-list" class="mb-4" style="max-height: 200px; overflow-y: auto;">
              <p class="text-secondary small">No documents ingested yet.</p>
            </div>
            
            <h5 class="mb-3">2. Define Columns</h5>
            <div id="review-cols-list">
              <input type="text" class="form-control mb-2 review-col-input" placeholder="e.g. Governing Law" value="Governing Law">
              <input type="text" class="form-control mb-2 review-col-input" placeholder="e.g. Notice Period" value="Notice Period">
            </div>
            <button class="btn btn-outline-secondary btn-sm mb-4" id="add-review-col-btn">+ Add Column</button>
            
            <button class="btn btn-accent w-100" id="run-review-btn">Run Review</button>
          </div>
        </div>
        <div class="col-12 col-md-8">
          <div class="glass-card p-4 h-100">
            <div class="d-flex justify-content-between align-items-center mb-3">
              <h5 class="mb-0">Review Results</h5>
              <button class="btn btn-outline-light btn-sm d-none" id="review-export-btn">Export to Excel</button>
            </div>
            <div id="review-status" class="mb-3">
              <div class="progress d-none" id="review-progress-wrap"><div class="progress-bar progress-bar-wiki" id="review-progress" style="width:0%"></div></div>
              <div class="text-secondary small mt-1" id="review-status-text"></div>
            </div>
            <div class="table-responsive d-none" id="review-table-wrap">
              <table class="table table-dark table-bordered table-sm" id="review-table">
                <thead id="review-thead"></thead>
                <tbody id="review-tbody"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div> <!-- /review-view -->
  </div> <!-- /tab-content -->
"""
    if "<!-- /ask-view -->" not in html:
        html = html.replace('<!-- RAG Modal -->', closing_tabs + '\n<!-- RAG Modal -->')

    # 3. Inject javascript for Review and Compare at the very end
    js_code = """
// ============================================================
// Advanced Modes (Review & Compare)
// ============================================================

let currentDocs = []; // populated after upload/resume

function updateDocLists() {
  const reviewDiv = document.getElementById('review-doc-list');
  const compareDiv = document.getElementById('compare-doc-list');
  
  if (!currentDocs.length) {
    reviewDiv.innerHTML = '<p class="text-secondary small">No documents available.</p>';
    compareDiv.innerHTML = '<p class="text-secondary small">No documents available.</p>';
    return;
  }
  
  const htmlStr = currentDocs.map(d => `
    <div class="form-check">
      <input class="form-check-input doc-checkbox" type="checkbox" value="${d}" id="cb-${d}" checked>
      <label class="form-check-label text-light small" for="cb-${d}" style="word-break: break-all;">${d}</label>
    </div>
  `).join('');
  
  reviewDiv.innerHTML = htmlStr;
  compareDiv.innerHTML = htmlStr;
}

// Hook into existing ingest logic:
const origLoadFileTree = window.loadFileTree || function(){};
window.loadFileTree = async function() {
  await origLoadFileTree();
  try {
    const res = await fetch(`/session/${SID}`);
    const data = await res.json();
    currentDocs = data.files_list || [];
    updateDocLists();
  } catch(e) {}
};

// REVIEW LOGIC
document.getElementById('add-review-col-btn')?.addEventListener('click', () => {
  const list = document.getElementById('review-cols-list');
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'form-control mb-2 review-col-input';
  input.placeholder = 'New Column';
  list.appendChild(input);
});

document.getElementById('run-review-btn')?.addEventListener('click', async () => {
  const docs = Array.from(document.getElementById('review-doc-list').querySelectorAll('input:checked')).map(cb => cb.value);
  const cols = Array.from(document.querySelectorAll('.review-col-input')).map(i => i.value.trim()).filter(Boolean);
  
  if (!docs.length || !cols.length) {
    showToast('Select at least one doc and define one column', 'error');
    return;
  }
  
  document.getElementById('run-review-btn').disabled = true;
  document.getElementById('review-table-wrap').classList.add('d-none');
  document.getElementById('review-export-btn').classList.add('d-none');
  const pwrap = document.getElementById('review-progress-wrap');
  const pbar = document.getElementById('review-progress');
  const stext = document.getElementById('review-status-text');
  
  pwrap.classList.remove('d-none');
  pbar.style.width = '0%';
  stext.textContent = 'Starting...';
  
  try {
    const res = await fetch('/review/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: SID, doc_names: docs, columns: cols})
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    const jobId = data.job_id;
    
    const timer = setInterval(async () => {
      const pRes = await fetch(`/review/progress?job_id=${jobId}`);
      const pData = await pRes.json();
      
      pbar.style.width = `${pData.percent}%`;
      stext.textContent = `Completed ${pData.completed} / ${pData.total} cells. Flagged: ${pData.flagged_count}`;
      
      if (pData.status === 'complete' || pData.status === 'error') {
        clearInterval(timer);
        document.getElementById('run-review-btn').disabled = false;
        
        if (pData.status === 'complete') {
          stext.textContent = 'Review Complete.';
          renderReviewResult(jobId);
        } else {
          stext.textContent = 'Error occurred during review.';
        }
      }
    }, 2000);
  } catch(e) {
    showToast(e.message, 'error');
    document.getElementById('run-review-btn').disabled = false;
  }
});

function getStyleForConfidence(conf) {
  if (conf === null || conf === undefined) return '';
  if (conf >= 0.8) return 'background-color: #14532d; color: #fff;'; // dark green
  if (conf >= 0.5) return 'background-color: #713f12; color: #fff;'; // dark yellow
  return 'background-color: #7f1d1d; color: #fff;'; // dark red
}

async function renderReviewResult(jobId) {
  const res = await fetch(`/review/result?job_id=${jobId}`);
  const data = await res.json();
  
  const thead = document.getElementById('review-thead');
  const tbody = document.getElementById('review-tbody');
  
  thead.innerHTML = `<tr><th>Document</th>${data.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
  
  let rowsHtml = '';
  for (const doc in data.rows) {
    rowsHtml += `<tr><td>${doc}</td>`;
    for (const col of data.columns) {
      const cell = data.rows[doc][col] || {};
      const val = cell.value || '-';
      const style = getStyleForConfidence(cell.confidence);
      const quote = cell.quote ? `title="Quote: ${cell.quote.replace(/"/g, '&quot;')}"` : '';
      rowsHtml += `<td style="${style} cursor:help;" ${quote}>${val}</td>`;
    }
    rowsHtml += `</tr>`;
  }
  tbody.innerHTML = rowsHtml;
  
  document.getElementById('review-table-wrap').classList.remove('d-none');
  const expBtn = document.getElementById('review-export-btn');
  expBtn.classList.remove('d-none');
  expBtn.onclick = () => window.open(`/review/export?job_id=${jobId}`);
}


// COMPARE LOGIC
let compareFile = null;
const $compZone = document.getElementById('compare-upload-zone');
const $compInput = document.getElementById('compare-file');
$compZone?.addEventListener('click', () => $compInput.click());
$compInput?.addEventListener('change', () => {
  if ($compInput.files.length) {
    compareFile = $compInput.files[0];
    document.getElementById('compare-staged-label').textContent = compareFile.name;
    $compZone.style.borderColor = 'var(--accent)';
  }
});

document.getElementById('run-compare-btn')?.addEventListener('click', async () => {
  const docs = Array.from(document.getElementById('compare-doc-list').querySelectorAll('input:checked')).map(cb => cb.value);
  const q = document.getElementById('compare-q').value.trim();
  
  if (!q) {
    showToast('Enter a compare topic/question', 'error');
    return;
  }
  if (!docs.length && !compareFile) {
    showToast('Select at least one doc or upload one', 'error');
    return;
  }
  
  document.getElementById('run-compare-btn').disabled = true;
  document.getElementById('compare-table-wrap').classList.add('d-none');
  document.getElementById('compare-outliers-wrap').classList.add('d-none');
  document.getElementById('compare-narrative-wrap').classList.add('d-none');
  document.getElementById('compare-export-btn').classList.add('d-none');
  
  const stext = document.getElementById('compare-status');
  stext.textContent = 'Starting job...';
  
  const fd = new FormData();
  fd.append('session_id', SID);
  fd.append('doc_names', JSON.stringify(docs));
  fd.append('question', q);
  if (compareFile) fd.append('uploaded_file', compareFile);
  
  try {
    const res = await fetch('/compare/start', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    
    const jobId = data.job_id;
    
    const timer = setInterval(async () => {
      const pRes = await fetch(`/compare/progress?job_id=${jobId}`);
      const pData = await pRes.json();
      
      stext.textContent = `Stage: ${pData.stage || pData.status}...`;
      
      if (pData.status === 'complete' || pData.status === 'error') {
        clearInterval(timer);
        document.getElementById('run-compare-btn').disabled = false;
        
        if (pData.status === 'complete') {
          stext.textContent = 'Compare Complete.';
          renderCompareResult(jobId);
        } else {
          stext.textContent = 'Error occurred during compare.';
        }
      }
    }, 2000);
  } catch(e) {
    showToast(e.message, 'error');
    document.getElementById('run-compare-btn').disabled = false;
  }
});

async function renderCompareResult(jobId) {
  const res = await fetch(`/compare/result?job_id=${jobId}`);
  const data = await res.json();
  
  const thead = document.getElementById('compare-thead');
  const tbody = document.getElementById('compare-tbody');
  
  // docs as columns
  const sourceNames = data.sources.map(s => s.label || s.name);
  thead.innerHTML = `<tr><th>Aspect</th>${sourceNames.map(n => `<th>${n}</th>`).join('')}</tr>`;
  
  let rowsHtml = '';
  for (const aspect of data.aspects) {
    rowsHtml += `<tr><td>${aspect}</td>`;
    for (const s of sourceNames) {
      const cell = data.table[aspect][s] || {};
      const val = cell.value || '-';
      const style = getStyleForConfidence(cell.confidence);
      const quote = cell.quote ? `title="Quote: ${cell.quote.replace(/"/g, '&quot;')}"` : '';
      rowsHtml += `<td style="${style} cursor:help;" ${quote}>${val}</td>`;
    }
    rowsHtml += `</tr>`;
  }
  tbody.innerHTML = rowsHtml;
  
  document.getElementById('compare-table-wrap').classList.remove('d-none');
  
  const outliersList = document.getElementById('compare-outliers-list');
  if (data.outliers && data.outliers.length > 0) {
    outliersList.innerHTML = data.outliers.map(o => `<li><strong>${o.doc} (${o.aspect}):</strong> ${o.reason}</li>`).join('');
    document.getElementById('compare-outliers-wrap').classList.remove('d-none');
  }
  
  if (data.narrative) {
    document.getElementById('compare-narrative').textContent = data.narrative;
    document.getElementById('compare-narrative-wrap').classList.remove('d-none');
  }
  
  const expBtn = document.getElementById('compare-export-btn');
  expBtn.classList.remove('d-none');
  expBtn.onclick = () => window.open(`/compare/export?job_id=${jobId}`);
}
"""

    if "// REVIEW LOGIC" not in html:
        html = html.replace('</script>\n</body>', js_code + '\n</script>\n</body>')

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    patch_html()
    print("Done")
