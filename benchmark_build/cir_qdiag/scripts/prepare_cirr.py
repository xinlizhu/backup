from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag.config import load_config
from cir_qdiag.pipeline import prepare_source
from cir_qdiag.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_20.yaml"))
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--annotation-path", type=Path)
    parser.add_argument("--image-split-path", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.annotation_root:
        root = args.annotation_root
        nested = root / "cirr" if (root / "cirr" / "captions").is_dir() else root
        config.data.annotation_path = nested / "captions" / f"cap.rc2.{args.split}.json"
        config.data.image_split_path = nested / "image_splits" / f"split.rc2.{args.split}.json"
    if args.annotation_path:
        config.data.annotation_path = args.annotation_path
    if args.image_split_path:
        config.data.image_split_path = args.image_split_path
    if args.image_root:
        config.data.image_root = args.image_root
    if args.limit:
        config.sampling.limit = args.limit
    if args.seed is not None:
        config.sampling.seed = args.seed
    configure_logging()
    prepare_source(config, args.output, start_index=args.start_index, end_index=args.end_index)


if __name__ == "__main__":
    main()
