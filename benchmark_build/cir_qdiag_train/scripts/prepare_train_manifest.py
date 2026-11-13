from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir_qdiag_train.manifest import build_train_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-split", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--negative-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple CIR-QDiag train output shards.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all-samples-have-negatives", action="store_true")
    parser.add_argument(
        "--only-samples-with-negatives",
        action="store_true",
        help="Write only CIRR annotations represented by accepted generated negatives.",
    )
    args = parser.parse_args()
    summary = build_train_manifest(
        annotation_path=args.annotations,
        image_split_path=args.image_split,
        image_root=args.image_root,
        negative_jsonl_paths=args.negative_jsonl,
        output_path=args.output,
        require_all_samples_have_negatives=args.require_all_samples_have_negatives,
        only_samples_with_negatives=args.only_samples_with_negatives,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
