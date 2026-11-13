from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            metrics = run["metrics"]
            delta = run["minus_global"]
            changes = run["top1_ranking_changes"]
            rows.append(
                {
                    "model": path.stem,
                    "beta": run["beta"],
                    "rerank_top_k": run["rerank_top_k"],
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "group_recall_at_1": metrics["group_recall_at_1"],
                    "delta_recall_at_1": delta["recall_at_1"],
                    "delta_recall_at_5": delta["recall_at_5"],
                    "delta_recall_at_10": delta["recall_at_10"],
                    "delta_group_recall_at_1": delta["group_recall_at_1"],
                    "corrected_queries": changes["corrected"],
                    "regressed_queries": changes["regressed"],
                    "net_corrections": changes["net_corrections"],
                    "top1_changed": changes["top1_changed"],
                }
            )
    if not rows:
        raise SystemExit(f"no reranker JSON files found in {args.root}")
    output = args.root / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
