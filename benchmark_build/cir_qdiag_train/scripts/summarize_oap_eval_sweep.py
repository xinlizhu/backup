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
    for path in sorted(args.root.glob("*/beta0_1/eval_sweep/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = path.parents[2].name
        final = payload["oap_final"]
        delta = payload["oap_minus_global"]
        changes = payload["top1_ranking_changes"]
        rows.append(
            {
                "model": model,
                "result": path.stem,
                "beta": payload["beta"],
                "recall_at_1": final["recall_at_1"],
                "recall_at_5": final["recall_at_5"],
                "recall_at_10": final["recall_at_10"],
                "group_recall_at_1": final["group_recall_at_1"],
                "delta_recall_at_1": delta["recall_at_1"],
                "delta_group_recall_at_1": delta["group_recall_at_1"],
                "corrected_queries": changes["corrected_queries"],
                "regressed_queries": changes["regressed_queries"],
                "net_corrections": changes["net_corrections"],
            }
        )
    if not rows:
        raise SystemExit(f"no sweep JSON files found below {args.root}")
    output = args.root / "eval_sweep_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
