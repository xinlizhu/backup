#!/usr/bin/env python3
"""Combine the original-model CIRR-train recall JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        ("Model", "model"),
        ("R@1", "recall_at_1"),
        ("R@5", "recall_at_5"),
        ("R@10", "recall_at_10"),
        ("R@32", "recall_at_32"),
        ("R@50", "recall_at_50"),
        ("Mean rank", "mean_rank"),
        ("Median rank", "median_rank"),
    ]
    lines = [
        "# Original-model CIRR train full-gallery retrieval",
        "",
        "Protocol: rank the ground-truth target over all 16,939 train images after excluding the reference image.",
        "",
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
