from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag.config import load_config
from cir_qdiag.paraphrasing import positive_paraphrase_stage, write_paraphrase_reports
from cir_qdiag.pipeline import (
    create_client,
    generate_stage,
    prepare_source,
    run_pipeline,
    screen_source_stage,
    validate_stage,
)
from cir_qdiag.report import export_audit_html, write_quality_reports
from cir_qdiag.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cir-qdiag")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    _config_arg(prepare)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--start-index", type=int)
    prepare.add_argument("--end-index", type=int)

    screen = sub.add_parser("screen")
    _config_arg(screen)
    screen.add_argument("--input", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--rejected-output", type=Path, required=True)
    screen.add_argument("--failed-output", type=Path)

    generate = sub.add_parser("generate")
    _stage_args(generate)
    validate = sub.add_parser("validate")
    _stage_args(validate)

    run = sub.add_parser("run")
    _config_arg(run)
    run.add_argument("--force-stage", choices=["prepare", "generate", "validate"])
    run.add_argument("--start-index", type=int)
    run.add_argument("--end-index", type=int)

    audit = sub.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    report = sub.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)

    paraphrase = sub.add_parser("paraphrase")
    _stage_args(paraphrase)
    paraphrase.add_argument("--num-variants", type=int)
    paraphrase.add_argument("--report-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    if args.command == "audit":
        export_audit_html(args.input, args.output)
        return
    if args.command == "report":
        write_quality_reports(args.input, args.output_dir)
        return
    config = load_config(args.config)
    if args.command == "prepare":
        prepare_source(config, args.output, start_index=args.start_index, end_index=args.end_index)
    elif args.command == "screen":
        screen_source_stage(
            config,
            args.input,
            args.output,
            args.rejected_output,
            failed_path=args.failed_output,
        )
    elif args.command == "generate":
        generate_stage(config, args.input, args.output, force=args.force)
    elif args.command == "validate":
        validate_stage(config, args.input, args.output, force=args.force)
    elif args.command == "run":
        config.runtime.force_stage = args.force_stage
        paths = run_pipeline(config, start_index=args.start_index, end_index=args.end_index)
        export_audit_html(paths["final"], config.runtime.output_dir / "audit.html")
        write_quality_reports(paths["final"], config.runtime.output_dir)
    elif args.command == "paraphrase":
        output = positive_paraphrase_stage(
            config,
            args.input,
            args.output,
            create_client(config),
            force=args.force,
            num_variants=args.num_variants,
        )
        write_paraphrase_reports(output, args.report_dir or args.output.parent)


def _config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def _stage_args(parser: argparse.ArgumentParser) -> None:
    _config_arg(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")


if __name__ == "__main__":
    main()
