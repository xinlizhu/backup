from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-delete", type=Path, required=True)
    parser.add_argument("--without-delete", type=Path, required=True)
    args = parser.parse_args()
    with_delete = json.loads(args.with_delete.read_text(encoding="utf-8"))
    without_delete = json.loads(args.without_delete.read_text(encoding="utf-8"))
    rows = []
    for key in sorted(set(with_delete) & set(without_delete)):
        first = with_delete[key]
        second = without_delete[key]
        if (
            key.startswith(("recall_at_", "group_recall_at_"))
            and isinstance(first, (int, float))
            and isinstance(second, (int, float))
        ):
            rows.append(
                {
                    "metric": key,
                    "with_delete": first,
                    "without_delete": second,
                    "delta_with_minus_without": round(first - second, 4),
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
