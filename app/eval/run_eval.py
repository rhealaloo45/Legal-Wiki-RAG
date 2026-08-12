"""All-in-one RAG evaluation script.

Two passes, both sourced from data/logs/rag_query_log.json:

1. General pass (no ground truth needed) — Groundedness, Relevance, Retrieval,
   Coherence, Fluency — over the most recent logged queries.
2. Ground-truth pass — Similarity, F1 — over the hand-verified Q&A pairs in
   GROUND_TRUTH below (each query must match one already in the log so its
   logged response/context can be pulled automatically).

To add more ground-truth questions: ask the question in the app once (so it's
logged), then append a {"query": ..., "ground_truth": ...} pair to GROUND_TRUTH.

Usage:
    python eval/run_eval.py                # all logged queries + ground-truth pass
    python eval/run_eval.py --limit 50     # most recent 50 logged queries + ground-truth pass
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# The SDK's own batch runner logs a "Finished X / Y lines" line per row and a
# "not a dictionary" warning per evaluator — quiet those so the terminal ends
# with just our own summary tables instead of a wall of progress lines.
logging.getLogger("execution.bulk").setLevel(logging.ERROR)
logging.getLogger("azure.ai.evaluation").setLevel(logging.ERROR)

from azure.ai.evaluation import (
    evaluate,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    F1ScoreEvaluator,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(EVAL_DIR), "data", "eval_results")
MAIN_DATASET_PATH = os.path.join(EVAL_DIR, "eval_dataset.jsonl")
GROUND_TRUTH_DATASET_PATH = os.path.join(EVAL_DIR, "ground_truth_dataset.jsonl")

# Hand-verified answers for a fixed set of Joint Venture Agreement questions.
# Each query must exact-match an entry already logged in rag_query_log.json.
GROUND_TRUTH = [
    {
        "query": "In the Joint Venture Agreement between Tata Consumer Products Limited and BrewSphere Ready Beverages Private Limited, what are the precise scope-of-use and time-bound exclusivity restrictions imposed on the parties?",
        "ground_truth": "The agreement does not set out a precise exclusivity scope or duration. It states only that exclusivity, where agreed, must be narrow, sector-specific, and time-bound, and that each party may continue its pre-existing businesses unless expressly restricted. It also limits brand and external use to approved use and notes that foreground IP is subject to approved field-of-use limitations. However, it does not identify the exclusive sector, products, territory, field of use, start/end date, or duration. Therefore, the requested precise scope and time limit are not stated in the document.",
    },
    {
        "query": "How does the ReVolt Circular Mobility Materials Private Limited JV Agreement allocate initial and subsequent capital contribution obligations among Tata Motors Limited, Tata Power Company Limited, and ReVolt?",
        "ground_truth": "The agreement does not allocate precise initial or subsequent capital-contribution amounts among Tata Motors Limited, Tata Power Company Limited, and the recycling-technology partner. It states that each party will contribute the assets, licences, personnel support, commercial relationships, and other inputs described in the schedules, including capital funding in agreed tranches. However, the document does not provide the contribution amounts, shareholding percentages, funding dates, or tranche allocation. It further states that participation in workshops, design reviews, or preliminary coordination does not create an additional contribution or enlarged commitment unless it is expressly included in the approved business plan or a written amendment. Therefore, the precise capital allocation is not stated in the document.",
    },
    {
        "query": "Review the Reserved Matters list under the JV Agreement between Tata Steel Limited and SteelLoop Resource Recovery Private Limited. Which corporate decisions require unanimous board approval or special majority consent?",
        "ground_truth": "The agreement does not contain the Reserved Matters list, nor does it identify any decisions requiring unanimous board approval or special-majority consent. It states only that material deviations from the annual business plan require board approval in accordance with a reserved-matter matrix. It also refers generally to budget-approval discipline. However, the reserved-matter matrix and any voting thresholds are not included in the document. Therefore, the specific corporate decisions and approval thresholds are not stated in the document.",
    },
    {
        "query": "Does the Joint venture agreement between Tata Power Renewable Energy Limited and Cold Chain Energy Services Private Limited contain a \"no implied additional contributions\" clause, and what are the legal consequences if a party participates in joint planning workshops?",
        "ground_truth": "The agreement contains an express provision preventing implied additional contributions. It states that a party will not be deemed to have made an additional contribution or incurred an enlarged commitment merely because its personnel participate in workshops, design reviews, or preliminary coordination meetings. Such participation creates an additional commitment only if it is expressly captured in the approved business plan or a written amendment. However, the agreement does not state further consequences, such as waiver, indemnity, or liability arising from participation in joint planning workshops.",
    },
    {
        "query": "Analyze the governance and deadlock-resolution mechanisms under the DriveConnect Experience Labs Private Limited JV Agreement. What exit triggers and call/put option rights are established?",
        "ground_truth": "The agreement provides for feature-release approvals, data partitioning and privacy safeguards, foreground-IP allocation rules, customer-communication approval rights, and escalation of deadlock through a senior steering committee. Deadlock resolution is described as escalation, cooling-off, mediation, and, where required, buy-sell or tailored-separation mechanisms. The stated exit triggers are change of control, insolvency, sanctions exposure, persistent KPI failure, uncured material breach, extended force majeure, and illegality. However, the agreement does not establish express call-option or put-option rights, or specify option holders, exercise periods, valuation mechanics, or transfer procedures. Therefore, detailed call/put rights are not stated in the document.",
    },
    {
        "query": "Under the JV Agreement between Tata Electronics Private Limited and MicroFab Precision Manufacturing Private Limited, how is the risk of chip fabrication delays or waver run failures contractually allocated?",
        "ground_truth": "The agreement does not address chip-fabrication delays, wafer-run failures, yield failures, or the allocation of those specific risks. It provides for quality-gate approvals, restricted use of manufacturing know-how, a tooling-ownership matrix, integrity and export-control compliance, and a buy-sell option for unresolved strategic deadlock. It also distinguishes ordinary business underperformance from fraud, wilful misconduct, confidentiality breaches, IP infringement, and regulatory misconduct. However, it does not state which party bears the loss from chip-fabrication delays or wafer-run failures, nor does it provide service levels, liquidated damages, or delay-specific remedies. Therefore, the requested risk allocation is not stated in the document.",
    },
    {
        "query": "In the joint vemture agreement between Tata Consumer Products Limited and VitalSpring Wellness Platforms Private Limited, is there an express ring-fencing mechanism to isolate operational liabilities and proprietary intellectual property?",
        "ground_truth": "The agreement describes the JV as a “ring-fenced operating vehicle” and provides that each party retains its background IP. It further states that foreground IP created specifically for the JV will vest in, or be licensed to, the JV under an agreed ownership matrix, subject to pre-existing rights and approved field-of-use limitations. However, it does not contain an express operational-liability ring-fencing mechanism, such as limited recourse, exclusion of parent liability, liability caps, or creditor-separation provisions. Therefore, while the document contains a high-level ring-fenced-JV reference and an IP-ownership framework, it does not expressly establish a mechanism that isolates both operational liabilities and proprietary IP.",
    },
]


def print_summary_table(rows: list[dict], evaluator_names: list[str]) -> None:
    """One line per evaluator: avg score / pass rate / min / max, aligned in columns."""
    header = f"{'Evaluator':<14}{'Avg':>7}{'Pass %':>9}{'Min':>7}{'Max':>7}{'N':>6}"
    print(header)
    print("-" * len(header))
    for name in evaluator_names:
        score_key = f"outputs.{name}.{name}"
        passed_key = f"outputs.{name}.{name}_passed"
        scores = [r[score_key] for r in rows if isinstance(r.get(score_key), (int, float))]
        passed = [r[passed_key] for r in rows if passed_key in r]
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        pass_rate = (sum(1 for p in passed if p) / len(passed) * 100) if passed else float("nan")
        print(f"{name:<14}{avg:>7.2f}{pass_rate:>8.0f}%{min(scores):>7.2f}{max(scores):>7.2f}{len(scores):>6}")
    print("-" * len(header))


def write_rows_csv(rows: list[dict], evaluator_names: list[str], csv_path: str) -> None:
    """Compact per-query CSV — query, response, and each evaluator's score/pass/reason.

    Kept separate from the full JSON (which includes the full retrieved context per
    row and can run to tens of MB) so day-to-day review can happen in a spreadsheet.
    """
    fieldnames = ["query", "response"]
    for name in evaluator_names:
        fieldnames += [f"{name}_score", f"{name}_pass", f"{name}_reason"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {
                "query": r.get("inputs.query", ""),
                "response": r.get("inputs.response", ""),
            }
            for name in evaluator_names:
                out[f"{name}_score"] = r.get(f"outputs.{name}.{name}")
                out[f"{name}_pass"] = r.get(f"outputs.{name}.{name}_passed")
                out[f"{name}_reason"] = r.get(f"outputs.{name}.{name}_reason", "")
            writer.writerow(out)


def build_main_dataset(limit: int = None) -> str:
    log_path = os.path.join(config.LOGS_PATH, "rag_query_log.json")
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        records = records[-limit:]

    with open(MAIN_DATASET_PATH, "w", encoding="utf-8") as f:
        count = 0
        for rec in records:
            if not rec.get("query") or not rec.get("response"):
                continue
            row = {
                "query": rec["query"],
                "context": "\n\n".join(rec.get("contexts", [])),
                "response": rec["response"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Main dataset: {count} rows -> {MAIN_DATASET_PATH}")
    return MAIN_DATASET_PATH


def build_ground_truth_dataset() -> str:
    log_path = os.path.join(config.LOGS_PATH, "rag_query_log.json")
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for entry in GROUND_TRUTH:
        matches = [r for r in records if r.get("query", "").strip() == entry["query"].strip()]
        if not matches:
            print(f"WARNING: no logged response found for ground-truth query: {entry['query'][:80]}...")
            continue
        latest = matches[-1]  # most recent logged run of this question
        rows.append({
            "query": entry["query"],
            "context": "\n\n".join(latest.get("contexts", [])),
            "response": latest["response"],
            "ground_truth": entry["ground_truth"],
        })

    with open(GROUND_TRUTH_DATASET_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Ground-truth dataset: {len(rows)}/{len(GROUND_TRUTH)} rows -> {GROUND_TRUTH_DATASET_PATH}")
    return GROUND_TRUTH_DATASET_PATH if rows else None


def run_main_eval(dataset_path: str, model_config: dict, timestamp: str) -> None:
    evaluators = {
        "groundedness": GroundednessEvaluator(model_config, is_reasoning_model=True),
        "relevance": RelevanceEvaluator(model_config, is_reasoning_model=True),
        "retrieval": RetrievalEvaluator(model_config, is_reasoning_model=True),
        "coherence": CoherenceEvaluator(model_config, is_reasoning_model=True),
        "fluency": FluencyEvaluator(model_config, is_reasoning_model=True),
    }

    output_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    result = evaluate(data=dataset_path, evaluators=evaluators, output_path=output_path)

    rows = result.get("rows", [])
    csv_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.csv")
    write_rows_csv(rows, list(evaluators.keys()), csv_path)

    print("\n" + "=" * 50)
    print(f"GENERAL EVALUATION SUMMARY ({len(rows)} queries)")
    print("=" * 50)
    print_summary_table(rows, list(evaluators.keys()))
    print(f"\nPer-query table (open in Excel/Sheets): {csv_path}")
    print(f"Full raw results (incl. retrieved context): {output_path}")


def run_ground_truth_eval(dataset_path: str, model_config: dict, timestamp: str) -> None:
    evaluators = {
        "similarity": SimilarityEvaluator(model_config, is_reasoning_model=True),
        "f1_score": F1ScoreEvaluator(),  # pure string-overlap metric, no judge model needed
    }

    output_path = os.path.join(RESULTS_DIR, f"ground_truth_eval_{timestamp}.json")
    result = evaluate(data=dataset_path, evaluators=evaluators, output_path=output_path)

    rows = result.get("rows", [])
    csv_path = os.path.join(RESULTS_DIR, f"ground_truth_eval_{timestamp}.csv")
    write_rows_csv(rows, list(evaluators.keys()), csv_path)

    print("\n" + "=" * 50)
    print(f"GROUND TRUTH EVALUATION SUMMARY ({len(rows)} queries)")
    print("=" * 50)
    print_summary_table(rows, list(evaluators.keys()))
    print(f"\nPer-query table (open in Excel/Sheets): {csv_path}")
    print(f"Full raw results: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max recent queries for the general pass")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    model_config = {
        "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "api_key": config.AZURE_OPENAI_API_KEY,
        "azure_deployment": config.AZURE_OPENAI_DEPLOYMENT,
        "api_version": config.AZURE_OPENAI_API_VERSION,
    }

    main_dataset_path = build_main_dataset(args.limit)
    run_main_eval(main_dataset_path, model_config, timestamp)

    gt_dataset_path = build_ground_truth_dataset()
    if gt_dataset_path:
        run_ground_truth_eval(gt_dataset_path, model_config, timestamp)


if __name__ == "__main__":
    main()
