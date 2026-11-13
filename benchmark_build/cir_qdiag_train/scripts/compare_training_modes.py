from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare standard InfoNCE, Add/Substitute, and +Delete training."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--add-substitute", type=Path, required=True)
    parser.add_argument("--add-substitute-delete", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = {
        "baseline": _read(args.baseline),
        "add_substitute": _read(args.add_substitute),
        "add_substitute_delete": _read(args.add_substitute_delete),
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
                "add_substitute_minus_baseline": round(
                    values["add_substitute"] - values["baseline"], 4
                ),
                "add_substitute_delete_minus_baseline": round(
                    values["add_substitute_delete"] - values["baseline"], 4
                ),
                "delete_increment": round(
                    values["add_substitute_delete"] - values["add_substitute"], 4
                ),
            }
        )
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
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
