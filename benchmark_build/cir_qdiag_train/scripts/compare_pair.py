from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected metric object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a controlled baseline and hard-negative run."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = _read(args.baseline)
    hard = _read(args.hard)
    rows = []
    for key in sorted(set(baseline) & set(hard)):
        if not key.startswith(("recall_at_", "group_recall_at_")):
            continue
        if not isinstance(baseline[key], (int, float)) or not isinstance(
            hard[key], (int, float)
        ):
            continue
        rows.append(
            {
                "metric": key,
                "baseline": baseline[key],
                "hard_weight_0_5": hard[key],
                "hard_minus_baseline": round(hard[key] - baseline[key], 4),
            }
        )
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
