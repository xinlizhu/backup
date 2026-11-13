from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from cir_qdiag_openai_cirr.config import load_config
from cir_qdiag_openai_cirr.pipeline import run_pipeline
from cir_qdiag_openai_cirr.report import export_audit_html, write_quality_reports
from cir_qdiag_openai_cirr.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", choices=["prepare", "generate", "validate"])
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(verbose=args.verbose)
    config = load_config(args.config)
    config.runtime.resume = args.resume or config.runtime.resume
    config.runtime.force_stage = args.force_stage
    if args.seed is not None:
        config.sampling.seed = args.seed
    paths = run_pipeline(config, start_index=args.start_index, end_index=args.end_index)
    export_audit_html(paths["final"], config.runtime.output_dir / "audit.html")
    write_quality_reports(paths["final"], config.runtime.output_dir)


if __name__ == "__main__":
    main()
