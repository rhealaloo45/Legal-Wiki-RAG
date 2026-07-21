"""Convert data/logs/rag_query_log.json into a JSONL eval dataset.

Usage: python eval/build_dataset.py [--limit N]
Writes eval/eval_dataset.jsonl with one {query, context, response} row per logged turn.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dataset.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max rows to include (most recent first)")
    args = parser.parse_args()

    log_path = os.path.join(config.LOGS_PATH, "rag_query_log.json")
    with open(log_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit:
        records = records[-args.limit:]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            if not rec.get("query") or not rec.get("response"):
                continue
            row = {
                "query": rec["query"],
                "context": "\n\n".join(rec.get("contexts", [])),
                "response": rec["response"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
