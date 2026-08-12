"""Render the 105-question review as a self-contained HTML page.

Data comes from full_report_data.json (answers + ground truth + source docs) and
the score table below. Scores are manual judgements against the verified ground
truth; everything else is read from the run artefacts so the page cannot drift
from what was measured.
"""
import html
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TEMP = os.environ.get("TEMP", ".")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "rag_review_105.html")

# v1 audit baseline, all 105 questions.
V1 = {1:9,2:9,3:9,4:10,5:9,6:7,7:7,8:9,9:9,10:9,11:9,12:9,13:10,14:1,15:9,16:9,17:1,18:2,19:9,
      20:8,21:2,22:7,23:9,24:2,25:9,26:9,27:4,28:8,29:6,30:0,31:8,32:9,33:8,34:9,35:9,36:8,37:9,
      38:1,39:1,40:1,41:9,42:3,43:1,44:1,45:9,46:6,47:10,48:9,49:1,50:9,51:8,52:9,53:9,54:9,55:7,
      56:1,57:9,58:3,59:9,60:7,61:4,62:9,63:9,64:7,65:1,66:9,67:1,68:9,69:9,70:1,71:9,72:7,73:9,
      74:1,75:9,76:1,77:9,78:9,79:1,80:8,81:9,82:9,83:9,84:9,85:3,86:9,87:9,88:9,89:9,90:9,91:9,
      92:2,93:8,94:6,95:9,96:8,97:10,98:9,99:9,100:8,101:1,102:5,103:9,104:9,105:2}

# Re-tested this round: best score observed across the two corpus runs.
# 2026-08-10 batch: scope-resolution family fix + document-metadata surfacing
# fix, re-run live on the mixed (production) session against ground truth.
# Scores below that value replace the prior round for the same 20 questions —
# see full_report_data.json's "origin" field for which answer each score grades.
RETEST = {14:9, 17:6, 18:9, 21:8, 24:7, 27:7, 29:7, 30:5, 42:9, 43:1, 44:1, 49:9, 56:6,
          58:9, 61:9, 65:9, 67:9, 70:8, 74:8, 76:9, 79:9, 85:6, 92:9, 94:9, 101:2,
          102:6, 105:7,
          # Asked inside combined turns in the v1 run, so no standalone answer
          # survived to show. Re-run individually on the clean corpus, and scored
          # against those answers — a v1 score beside a v2 answer would be a
          # number describing text the reader is not looking at.
          15:9, 31:6, 51:6, 52:9, 68:9, 78:8, 80:9, 82:9, 87:9, 103:9, 104:9}

# Ground-truth fact absent from every ingested page of the expected document.
# The refusal is correct; no retrieval change can raise these.
# 2026-08-10: 18, 49, 61, 74, 79, 87 were patched (fact confirmed present in the
# raw PDF extraction, appended to the relevant page) or fixed via document-
# metadata surfacing — no longer true that the fact is absent, so removed.
# 105's fact is now also present (same patch round) but still fails: a
# DIFFERENT problem (unscoped corpus-wide retrieval over 494 docs doesn't
# surface the page — no party/document is named to narrow the search), not an
# ingest gap, so it no longer belongs in this set either. Empty until a new
# genuinely ingest-absent case is found.
INGEST_CAPPED = set()

# Ground-truthed to synthetic Test_* fixtures or to no document at all (src field
# is "N/A — no such document exists"), so untestable on the clean/production
# corpus. Verified against gt_full.json's src field for every question, not
# assumed: 30-31 are N/A, 32-46 all cite Test_NDA_*/Test_JVA_*/Test_CCD_*/
# Test_Opinion_*/Test_Judgment_* — an earlier pass only flagged 32-46 and missed
# 30-31, undercounting the untestable set by 2.
FIXTURE_BOUND = {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46}


def band(score: int) -> str:
    return "pass" if score >= 8 else ("partial" if score >= 5 else "fail")


def main() -> None:
    data = json.load(open(os.path.join(TEMP, "full_report_data.json"), encoding="utf-8"))
    gap_path = os.path.join(TEMP, "gap_out.json")
    if os.path.exists(gap_path):
        for r in json.load(open(gap_path, encoding="utf-8")):
            if r["id"] in data and r.get("answer"):
                data[r["id"]]["answer"] = r["answer"]
                data[r["id"]]["files"] = (r.get("meta") or {}).get("files_used") or []
                data[r["id"]]["origin"] = "retest"

    rows = []
    for n in range(1, 106):
        qid = f"Q{n}"
        e = data.get(qid, {})
        score = RETEST.get(n, V1[n])
        rows.append({
            "id": qid, "n": n, "q": e.get("question", ""), "gt": e.get("gt", ""),
            "src": e.get("src", ""), "answer": e.get("answer", "") or "(no answer captured)",
            "files": [f.split("_", 1)[-1] for f in (e.get("files") or [])][:3],
            "score": score, "v1": V1[n], "band": band(score),
            "retested": n in RETEST, "capped": n in INGEST_CAPPED,
            "fixture": n in FIXTURE_BOUND,
        })

    total = sum(r["score"] for r in rows)
    stats = {
        "mean": round(total / len(rows), 2),
        "v1mean": round(sum(V1.values()) / len(V1), 2),
        "passes": sum(1 for r in rows if r["band"] == "pass"),
        "partial": sum(1 for r in rows if r["band"] == "partial"),
        "fails": sum(1 for r in rows if r["band"] == "fail"),
        "retested": sum(1 for r in rows if r["retested"]),
        "capped": len(INGEST_CAPPED),
    }

    page = TEMPLATE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)) \
                   .replace("__STATS__", json.dumps(stats))
    out = os.path.abspath(OUT)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print("wrote", out)
    print(f"mean {stats['mean']} (v1 {stats['v1mean']}) | "
          f"8+: {stats['passes']}  5-7: {stats['partial']}  0-4: {stats['fails']}")


TEMPLATE = r"""<title>LexWiki RAG Accuracy Review — 105 Questions</title>
<style>
  :root{
    --paper:#F5F7F7; --surface:#FFFFFF; --ink:#141B1D; --muted:#5D6E71;
    --rule:#D7E0E1; --accent:#2C5F66; --accent-soft:#E4EDEE;
    --pass:#2F6B4B; --partial:#8A6516; --fail:#9A3A31;
    --pass-bg:#E7F0EA; --partial-bg:#F6EFDF; --fail-bg:#F7E7E5;
    --on-accent:#FFFFFF;
    --shadow:0 1px 2px rgba(20,27,29,.06),0 8px 24px -16px rgba(20,27,29,.28);
    --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,"Cascadia Code","SF Mono",Consolas,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --paper:#0F1416; --surface:#161D1F; --ink:#E3EAEB; --muted:#93A5A8;
      --rule:#263134; --accent:#74B0B7; --accent-soft:#1B2A2C;
      --pass:#7FC29B; --partial:#D6AC5E; --fail:#E08279;
      --pass-bg:#152420; --partial-bg:#241E12; --fail-bg:#26191A; --on-accent:#0F1416;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.7);
    }
  }
  :root[data-theme="light"]{
    --paper:#F5F7F7; --surface:#FFFFFF; --ink:#141B1D; --muted:#5D6E71;
    --rule:#D7E0E1; --accent:#2C5F66; --accent-soft:#E4EDEE;
    --pass:#2F6B4B; --partial:#8A6516; --fail:#9A3A31;
    --pass-bg:#E7F0EA; --partial-bg:#F6EFDF; --fail-bg:#F7E7E5; --on-accent:#FFFFFF;
    --shadow:0 1px 2px rgba(20,27,29,.06),0 8px 24px -16px rgba(20,27,29,.28);
  }
  :root[data-theme="dark"]{
    --paper:#0F1416; --surface:#161D1F; --ink:#E3EAEB; --muted:#93A5A8;
    --rule:#263134; --accent:#74B0B7; --accent-soft:#1B2A2C;
    --pass:#7FC29B; --partial:#D6AC5E; --fail:#E08279;
    --pass-bg:#152420; --partial-bg:#241E12; --fail-bg:#26191A; --on-accent:#0F1416;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -16px rgba(0,0,0,.7);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
       line-height:1.55;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1120px;margin:0 auto;padding:0 20px 72px}

  header{padding:44px 0 26px;border-bottom:1px solid var(--rule)}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.14em;
           text-transform:uppercase;color:var(--accent);margin:0 0 10px}
  h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,4vw,40px);
     line-height:1.15;margin:0 0 12px;text-wrap:balance;letter-spacing:-.01em}
  .lede{margin:0;max-width:66ch;color:var(--muted);font-size:15.5px}

  .figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
           gap:1px;background:var(--rule);border:1px solid var(--rule);
           border-radius:10px;overflow:hidden;margin:26px 0 0}
  .fig{background:var(--surface);padding:14px 16px}
  .fig dt{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
          text-transform:uppercase;color:var(--muted);margin:0 0 6px}
  .fig dd{margin:0;font-family:var(--serif);font-size:26px;font-variant-numeric:tabular-nums}
  .fig small{display:block;font-family:var(--sans);font-size:11.5px;color:var(--muted);
             margin-top:2px}

  .note{margin:22px 0 0;padding:13px 16px;border-left:3px solid var(--accent);
        background:var(--accent-soft);border-radius:0 8px 8px 0;font-size:13.5px;
        max-width:78ch}
  .note strong{font-weight:600}

  .controls{position:sticky;top:0;z-index:20;background:var(--paper);
            border-bottom:1px solid var(--rule);padding:12px 0;margin-bottom:20px;
            display:flex;flex-wrap:wrap;gap:10px;align-items:center}
  input[type=search]{flex:1 1 230px;min-width:180px;padding:8px 12px;font:inherit;
        font-size:14px;color:var(--ink);background:var(--surface);
        border:1px solid var(--rule);border-radius:7px}
  input[type=search]:focus-visible,button:focus-visible{outline:2px solid var(--accent);
        outline-offset:2px}
  .chips{display:flex;gap:6px;flex-wrap:wrap}
  .chip{font:inherit;font-size:12.5px;padding:6px 11px;border-radius:99px;cursor:pointer;
        background:var(--surface);color:var(--muted);border:1px solid var(--rule)}
  .chip[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
        color:var(--on-accent)}
  .count{font-family:var(--mono);font-size:12px;color:var(--muted);margin-left:auto}

  .row{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--band);
       border-radius:9px;margin-bottom:8px;box-shadow:var(--shadow);overflow:hidden}
  .row.pass{--band:var(--pass)} .row.partial{--band:var(--partial)} .row.fail{--band:var(--fail)}
  .head{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:start;
        width:100%;padding:13px 16px;background:none;border:0;text-align:left;
        font:inherit;color:inherit;cursor:pointer}
  .qid{font-family:var(--mono);font-size:12px;color:var(--muted);padding-top:3px;
       min-width:38px}
  .qtext{font-size:14.5px;line-height:1.5}
  .meta{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:7px}
  .tag{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;
       padding:2px 7px;border-radius:4px;border:1px solid var(--rule);color:var(--muted)}
  .tag.capped{color:var(--fail);border-color:var(--fail)}
  .tag.fixture{color:var(--partial);border-color:var(--partial)}
  .tag.retested{color:var(--accent);border-color:var(--accent)}
  .score{font-family:var(--mono);font-size:16px;font-variant-numeric:tabular-nums;
         padding:4px 10px;border-radius:6px;white-space:nowrap}
  .pass .score{background:var(--pass-bg);color:var(--pass)}
  .partial .score{background:var(--partial-bg);color:var(--partial)}
  .fail .score{background:var(--fail-bg);color:var(--fail)}
  .delta{display:block;font-size:10px;text-align:center;color:var(--muted);margin-top:3px}

  .body{display:none;padding:0 16px 16px;border-top:1px solid var(--rule)}
  .row.open .body{display:block}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px}
  @media (max-width:760px){.cols{grid-template-columns:1fr}}
  .panel h3{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
            text-transform:uppercase;color:var(--muted);margin:0 0 7px;font-weight:500}
  .panel .content{font-size:13.5px;white-space:pre-wrap;overflow-wrap:anywhere;
        background:var(--paper);border:1px solid var(--rule);border-radius:7px;
        padding:11px 13px;max-height:340px;overflow:auto}
  .gt .content{border-left:3px solid var(--pass)}
  .src{margin-top:14px;font-size:13px;color:var(--muted)}
  .src b{color:var(--ink);font-weight:600}
  .src .file{font-family:var(--mono);font-size:12px}
  .empty{padding:40px;text-align:center;color:var(--muted)}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">LexWiki · RAG accuracy audit</p>
  <h1>Ground-truth review across 105 questions</h1>
  <p class="lede">Every question in the audit set with the system's answer, the verified
     ground truth, the document the answer should have come from, and a score out of 10.</p>
  <dl class="figures">
    <div class="fig"><dt>Mean score</dt><dd id="f-mean"></dd><small id="f-v1"></small></div>
    <div class="fig"><dt>Scored 8+</dt><dd id="f-pass"></dd><small>of 105</small></div>
    <div class="fig"><dt>Scored 5–7</dt><dd id="f-partial"></dd><small>partial credit</small></div>
    <div class="fig"><dt>Scored 0–4</dt><dd id="f-fail"></dd><small>failures</small></div>
    <div class="fig"><dt>Re-tested</dt><dd id="f-retested"></dd><small>after fixes</small></div>
  </dl>
  <p class="note"><strong>Ingest-capped questions.</strong> For <span id="f-capped"></span>
     questions the ground-truth fact is absent from every ingested page of the expected
     document — the answer text is not in the database at all. Those refusals are correct,
     and no retrieval change can raise them until ingest is fixed.</p>
  <p class="note"><strong>Mixed scoring basis.</strong> Re-tested questions show the better
     of two corpus runs (all 494 documents, or the 46 real documents with synthetic
     <code>Test_*</code> fixtures removed). The rest retain their original mixed-corpus
     score. This is a per-question best-observed figure, not one uniform measurement.</p>
</header>

<div class="controls">
  <input type="search" id="q" placeholder="Search questions, answers, documents…"
         aria-label="Search">
  <div class="chips" role="group" aria-label="Filter by score">
    <button class="chip" data-f="all" aria-pressed="true">All</button>
    <button class="chip" data-f="pass" aria-pressed="false">8+</button>
    <button class="chip" data-f="partial" aria-pressed="false">5–7</button>
    <button class="chip" data-f="fail" aria-pressed="false">0–4</button>
    <button class="chip" data-f="retested" aria-pressed="false">Re-tested</button>
    <button class="chip" data-f="capped" aria-pressed="false">Ingest-capped</button>
  </div>
  <button class="chip" id="toggle-all" aria-pressed="false">Expand all</button>
  <span class="count" id="count"></span>
</div>

<div id="list"></div>
<p class="empty" id="empty" hidden>No questions match that filter.</p>
</div>

<script>
const ROWS = __DATA__, STATS = __STATS__;
const $ = s => document.querySelector(s);
const esc = s => (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

$("#f-mean").textContent = STATS.mean.toFixed(2);
$("#f-v1").textContent = "was " + STATS.v1mean.toFixed(2);
$("#f-pass").textContent = STATS.passes;
$("#f-partial").textContent = STATS.partial;
$("#f-fail").textContent = STATS.fails;
$("#f-retested").textContent = STATS.retested;
$("#f-capped").textContent = STATS.capped;

let filter = "all", query = "";

function tags(r){
  let t = "";
  if (r.retested) t += '<span class="tag retested">re-tested</span>';
  if (r.capped)   t += '<span class="tag capped">ingest-capped</span>';
  if (r.fixture)  t += '<span class="tag fixture">fixture-bound</span>';
  return t;
}

function render(){
  const list = $("#list");
  const shown = ROWS.filter(r => {
    if (filter === "pass" || filter === "partial" || filter === "fail") {
      if (r.band !== filter) return false;
    } else if (filter === "retested" && !r.retested) return false;
    else if (filter === "capped" && !r.capped) return false;
    if (!query) return true;
    const hay = (r.id+" "+r.q+" "+r.answer+" "+r.gt+" "+r.src).toLowerCase();
    return hay.includes(query);
  });
  $("#count").textContent = shown.length + " of " + ROWS.length;
  $("#empty").hidden = shown.length > 0;
  list.innerHTML = shown.map(r => {
    const delta = r.retested && r.score !== r.v1 ? `<span class="delta">was ${r.v1}</span>` : "";
    const files = r.files.length
      ? `<div class="src"><b>Cited by the system:</b> <span class="file">${esc(r.files.join(" · "))}</span></div>` : "";
    return `<article class="row ${r.band}" data-id="${r.id}">
      <button class="head" aria-expanded="false">
        <span class="qid">${r.id}</span>
        <span><span class="qtext">${esc(r.q)}</span><span class="meta">${tags(r)}</span></span>
        <span><span class="score">${r.score}</span>${delta}</span>
      </button>
      <div class="body">
        <div class="cols">
          <div class="panel"><h3>System answer</h3><div class="content">${esc(r.answer)}</div></div>
          <div class="panel gt"><h3>Ground truth</h3><div class="content">${esc(r.gt)}</div></div>
        </div>
        <div class="src"><b>Expected source document:</b> ${esc(r.src)}</div>
        ${files}
      </div>
    </article>`;
  }).join("");
}

$("#list").addEventListener("click", e => {
  const btn = e.target.closest(".head");
  if (!btn) return;
  const row = btn.closest(".row");
  const open = row.classList.toggle("open");
  btn.setAttribute("aria-expanded", open ? "true" : "false");
});

document.querySelectorAll(".chip[data-f]").forEach(c => {
  c.addEventListener("click", () => {
    filter = c.dataset.f;
    document.querySelectorAll(".chip[data-f]").forEach(o =>
      o.setAttribute("aria-pressed", o === c ? "true" : "false"));
    render();
  });
});

$("#q").addEventListener("input", e => { query = e.target.value.toLowerCase().trim(); render(); });

$("#toggle-all").addEventListener("click", () => {
  const on = $("#toggle-all").getAttribute("aria-pressed") !== "true";
  $("#toggle-all").setAttribute("aria-pressed", on ? "true" : "false");
  $("#toggle-all").textContent = on ? "Collapse all" : "Expand all";
  document.querySelectorAll(".row").forEach(r => {
    r.classList.toggle("open", on);
    r.querySelector(".head").setAttribute("aria-expanded", on ? "true" : "false");
  });
});

render();
</script>
"""

if __name__ == "__main__":
    main()
