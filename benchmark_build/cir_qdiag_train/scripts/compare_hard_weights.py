from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare batch-InfoNCE baseline with hard-negative weights 0.5 and 1.0."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--weight-0-5", type=Path, required=True)
    parser.add_argument("--weight-1-0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = {
        "baseline": _read(args.baseline),
        "hard_weight_0_5": _read(args.weight_0_5),
        "hard_weight_1_0": _read(args.weight_1_0),
    }
    common = set.intersection(*(set(value) for value in metrics.values()))
    rows = []
    for key in sorted(common):
        values = {name: payload[key] for name, payload in metrics.items()}
        if not key.startswith(("recall_at_", "group_recall_at_")) or not all(
            isinstance(value, (int, float)) for value in values.values()
        ):
            continue
        rows.append(
            {
                "metric": key,
                **values,
                "weight_0_5_minus_baseline": round(
                    values["hard_weight_0_5"] - values["baseline"], 4
                ),
                "weight_1_0_minus_baseline": round(
                    values["hard_weight_1_0"] - values["baseline"], 4
                ),
                "weight_0_5_minus_1_0": round(
                    values["hard_weight_0_5"] - values["hard_weight_1_0"], 4
                ),
            }
        )
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.output)
    print(text, end="")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected metric object: {path}")
    return value


if __name__ == "__main__":
    main()
