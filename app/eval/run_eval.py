"""Run offline RAG evaluation (Groundedness/Relevance/Retrieval) over eval_dataset.jsonl.

No ground truth required — these evaluators only need {query, context, response}.
Judge model is the same Azure OpenAI deployment the app already uses (config.AZURE_OPENAI_DEPLOYMENT).
Results are always written locally; if AZURE_AI_PROJECT_ENDPOINT is set (see .env),
the run is also logged to that Azure AI Foundry project's dashboard — requires
`az login` locally so DefaultAzureCredential can authenticate the upload.

Usage: python eval/run_eval.py
"""
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
# with just our own summary table instead of a wall of progress lines.
logging.getLogger("execution.bulk").setLevel(logging.ERROR)
logging.getLogger("azure.ai.evaluation").setLevel(logging.ERROR)

from azure.ai.evaluation import (
    evaluate,
    GroundednessEvaluator,
    RelevanceEvaluator,
    RetrievalEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
)
from azure.identity import DeviceCodeCredential

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_results")


def _print_summary_table(rows: list[dict], evaluator_names: list[str]) -> None:
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


def _write_rows_csv(rows: list[dict], evaluator_names: list[str], csv_path: str) -> None:
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


def main():
    if not os.path.exists(DATASET_PATH):
        raise SystemExit(f"{DATASET_PATH} not found — run build_dataset.py first")

    model_config = {
        "azure_endpoint": config.AZURE_OPENAI_ENDPOINT,
        "api_key": config.AZURE_OPENAI_API_KEY,
        "azure_deployment": config.AZURE_OPENAI_DEPLOYMENT,
        "api_version": config.AZURE_OPENAI_API_VERSION,
    }

    # gpt-5-nano (config.AZURE_OPENAI_DEPLOYMENT) is a reasoning model — it rejects
    # the legacy max_tokens param the SDK's prompty templates send by default;
    # is_reasoning_model switches them to max_completion_tokens (same fix already
    # applied for the app's own LLM calls, see config.py AZURE_REASONING_EFFORT).
    groundedness = GroundednessEvaluator(model_config, is_reasoning_model=True)
    relevance = RelevanceEvaluator(model_config, is_reasoning_model=True)
    retrieval = RetrievalEvaluator(model_config, is_reasoning_model=True)
    coherence = CoherenceEvaluator(model_config, is_reasoning_model=True)
    fluency = FluencyEvaluator(model_config, is_reasoning_model=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")

    evaluators = {
        "groundedness": groundedness,
        "relevance": relevance,
        "retrieval": retrieval,
        "coherence": coherence,
        "fluency": fluency,
    }

    # Local run first, unconditionally — the SDK writes output_path only *after*
    # any Foundry upload succeeds, so passing azure_ai_project into this same
    # call risks losing the whole (expensive, judge-model) local run if the
    # cloud upload throws (e.g. tenant mismatch). Guarantee the local save,
    # then attempt the cloud upload as a separate, non-destructive step.
    result = evaluate(data=DATASET_PATH, evaluators=evaluators, output_path=output_path)

    rows = result.get("rows", [])
    csv_path = os.path.join(RESULTS_DIR, f"eval_{timestamp}.csv")
    _write_rows_csv(rows, list(evaluators.keys()), csv_path)

    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY ({len(rows)} queries)")
    print("=" * 50)
    _print_summary_table(rows, list(evaluators.keys()))
    print(f"\nPer-query table (open in Excel/Sheets): {csv_path}")
    print(f"Full raw results (incl. retrieved context): {output_path}")

    azure_ai_project = os.getenv("AZURE_AI_PROJECT_ENDPOINT") or None
    if azure_ai_project:
        try:
            # AzureCliCredential shells out to `az`, which fails to resolve from
            # this process even when `az` works fine interactively in the same
            # terminal. InteractiveBrowserCredential auto-launches a browser tab
            # that crashed in this VS Code environment. DeviceCodeCredential prints
            # a code + URL instead and doesn't launch anything itself — open the
            # printed URL in any working browser and enter the code there.
            credential = DeviceCodeCredential(tenant_id=os.getenv("AZURE_TENANT_ID"))
            cloud_result = evaluate(
                data=DATASET_PATH,
                evaluators=evaluators,
                azure_ai_project=azure_ai_project,
                credential=credential,
            )
            print(f"Logged to Azure AI Foundry: {cloud_result.get('studio_url') or azure_ai_project}")
        except Exception as exc:
            print(f"\nFoundry upload failed (local results above are unaffected): {exc}")
    else:
        print("AZURE_AI_PROJECT_ENDPOINT not set — skipped Foundry dashboard logging.")


if __name__ == "__main__":
    main()
