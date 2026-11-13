#!/usr/bin/env python3
"""Run trained CLIP-like checkpoints or original BGE on official PinPoint.

This runner deliberately stays separate from the PinPoint V7 diagnostics.  It
reads all 7,635 official queries from ``pinpoint_licensed.parquet``, performs
full-corpus retrieval, writes the standard PinPoint results JSON, and invokes
the repository's unmodified ``evaluate.py``.  The cross-model summary contains
only the two headline metrics requested for model comparison: mAP@10 and
NegRecall@10.
"""

from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import write_json_atomic
from pinpoint_full_diagnostics import (
    QuerySpec,
    load_or_build_query_embeddings,
    load_or_search,
    read_available_signatures,
)
from trained_clip_adapter import (
    FROZEN_CLIP_MODELS,
    TRAINED_CLIP_MODELS,
    OAPResidualAdapter,
    OriginalCLIPAdapter,
    TrainedCLIPAdapter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
CIR_PLAY_ROOT = WORKSPACE_ROOT / "CIR_PLAY"
DEFAULT_PINPOINT_ROOT = WORKSPACE_ROOT / "pinpoint-dataset"
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR.parent / "cir_qdiag_openai" / "outputs" / "pinpoint_official_full"
)
DEFAULT_MODELS = "CLIP-L-baseline,CLIP-L-hard-w0.5"
ORIGINAL_MODELS = {
    "BGE-VL-MLLM-S2": (
        WORKSPACE_ROOT / "CIR_PLAY" / "models" / "checkpoints" / "BGE-VL-MLLM-S2"
    ),
    "BGE-VL-large-original": (
        WORKSPACE_ROOT / "CIR_PLAY" / "models" / "checkpoints" / "BGE-VL-large"
    ),
    "Qwen3-VL-Embedding-2B": (
        WORKSPACE_ROOT / "CIR_PLAY" / "models" / "Qwen3-VL-Embedding-2B"
    ),
    "Qwen3-VL-Embedding-8B": (
        WORKSPACE_ROOT / "CIR_PLAY" / "models" / "Qwen3-VL-Embedding-8B"
    ),
}
PRETRAINED_CLIP_MODELS = {
    "CLIP-L-original": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "models"
        / "clip-vit-large-patch14"
    ),
}
OAP_MODELS = {
    "CLIP-L-OAP-epoch2-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_residual"
        / "clip_l"
        / "beta0_1"
        / "checkpoint_epoch_002.pt",
        0.1,
    ),
    "CLIP-L-OAP-epoch5-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_residual"
        / "clip_l"
        / "beta0_1"
        / "checkpoint_epoch_005.pt",
        0.1,
    ),
    "BGE-VL-large-OAP-epoch5-beta0.025": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_residual"
        / "bge_vl_large"
        / "beta0_1"
        / "checkpoint_epoch_005.pt",
        0.025,
    ),
    "CLIP-L-OAP-Attention-epoch2-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_attention"
        / "clip_l"
        / "beta0_1"
        / "checkpoint_epoch_002.pt",
        0.1,
    ),
    "CLIP-L-OAP-Attention-epoch5-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_attention"
        / "clip_l"
        / "beta0_1"
        / "checkpoint_epoch_005.pt",
        0.1,
    ),
    "BGE-VL-large-OAP-Attention-epoch2-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_attention"
        / "bge_vl_large"
        / "beta0_1"
        / "checkpoint_epoch_002.pt",
        0.1,
    ),
    "BGE-VL-large-OAP-Attention-epoch5-beta0.1": (
        SCRIPT_DIR.parent
        / "cir_qdiag_train"
        / "outputs"
        / "oap_attention"
        / "bge_vl_large"
        / "beta0_1"
        / "checkpoint_epoch_005.pt",
        0.1,
    ),
}
SUPPORTED_MODELS = {
    **ORIGINAL_MODELS,
    **PRETRAINED_CLIP_MODELS,
    **TRAINED_CLIP_MODELS,
    **{name: path for name, (path, _) in OAP_MODELS.items()},
}
FROZEN_CLIP_MODELS.add("CLIP-L-original")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate supported CLIP-like/BGE models on all official PinPoint queries, "
            "reporting mAP@10 and NegRecall@10."
        )
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register an additional trained checkpoint; repeat for multiple models.",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Register a trained checkpoint loaded through the CIR_PLAY adapter "
            "registry (for example a Qwen query-LoRA checkpoint)."
        ),
    )
    parser.add_argument(
        "--pretrained-clip",
        action="append",
        default=[],
        metavar="NAME=BACKBONE_DIR",
        help="Register an ordinary pretrained CLIP backbone.",
    )
    parser.add_argument("--pinpoint-root", type=Path, default=DEFAULT_PINPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--frozen-corpus-cache",
        type=Path,
        default=None,
        help=(
            "Optional existing corpus cache shared by frozen CLIP-L models. "
            "For example, reuse the cache created by the pilot diagnostics."
        ),
    )
    parser.add_argument(
        "--shared-corpus-models",
        default="",
        help=(
            "Comma-separated model names known to have the exact same image encoder. "
            "Their corpus is encoded once and shared."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--search-device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--processor-workers", type=int, default=2)
    parser.add_argument("--processor-prefetch-batches", type=int, default=2)
    parser.add_argument("--flush-interval", type=int, default=16)
    parser.add_argument("--query-search-batch-size", type=int, default=256)
    parser.add_argument("--corpus-search-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Keep 50 for the standard PinPoint result format; must be at least 10.",
    )
    parser.add_argument("--force-corpus", action="store_true")
    parser.add_argument("--force-queries", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    return parser.parse_args()


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "none", "nan", "null"}


def normalize_query_id(value: object) -> str:
    text = str(value).removeprefix("query_")
    return text.zfill(5)


def load_official_queries(path: Path) -> tuple[pd.DataFrame, list[QuerySpec]]:
    frame = pd.read_parquet(path)
    required = {
        "query_id",
        "query_image_signature",
        "query_image_signature2",
        "instruction",
        "positive_candidates",
        "negative_candidates",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Official PinPoint parquet is missing columns: {missing_columns}")

    queries: list[QuerySpec] = []
    seen_ids: set[str] = set()
    for query_index, (_, row) in enumerate(frame.iterrows()):
        query_id = normalize_query_id(row["query_id"])
        if query_id in seen_ids:
            raise ValueError(f"Duplicate normalized PinPoint query ID: {query_id}")
        seen_ids.add(query_id)
        references = [str(row["query_image_signature"])]
        second_reference = row.get("query_image_signature2")
        if not _is_missing(second_reference):
            references.append(str(second_reference))
        queries.append(
            QuerySpec(
                query_index=query_index,
                query_key=f"official::{query_id}",
                query_kind="positive",
                group_id=query_id,
                instruction_id=query_id,
                source_instruction_id=query_id,
                negative_id="",
                text=str(row["instruction"]),
                reference_ids=tuple(references),
                semantic_target="",
                edit_operation="",
                query_category=str(row.get("query_category") or ""),
                length_category=str(row.get("length_category") or ""),
                is_question=bool(row.get("is_question", False)),
                is_imperative=bool(row.get("is_imperative", False)),
            )
        )
    if not queries:
        raise ValueError(f"Official PinPoint parquet contains no queries: {path}")
    return frame, queries


def write_standard_results(
    path: Path,
    frame: pd.DataFrame,
    top_indices: np.ndarray,
    corpus_signatures: list[str],
) -> None:
    if top_indices.shape[0] != len(frame):
        raise ValueError(
            f"Search rows {top_indices.shape[0]} do not match queries {len(frame)}"
        )
    signatures = np.asarray(corpus_signatures)
    results: dict[str, dict[str, list[str]]] = {}
    for row_index, query_id in enumerate(frame["query_id"]):
        normalized = normalize_query_id(query_id)
        retrieved = signatures[top_indices[row_index]].astype(str).tolist()
        results[normalized] = {"retrieved_items": retrieved}
    write_json_atomic(path, results)


def run_official_evaluator(
    pinpoint_root: Path,
    results_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(pinpoint_root / "evaluate.py"),
        "--results",
        str(results_path),
        "--ground_truth",
        str(pinpoint_root / "pinpoint_licensed.parquet"),
        "--output",
        str(metrics_path),
    ]
    print("[official-eval] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=pinpoint_root, check=True)
    metrics = pd.read_csv(metrics_path)
    if len(metrics) != 1:
        raise ValueError(f"Expected one official metric row in {metrics_path}")
    return metrics.iloc[0].to_dict()


def preflight(
    models: list[str],
    supported_models: dict[str, Path],
    pinpoint_root: Path,
    top_k: int,
) -> None:
    if not models:
        raise ValueError("--models selected no models")
    unknown = sorted(set(models) - set(supported_models))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    missing_checkpoints = [
        f"{name}: {supported_models[name]}"
        for name in models
        if not supported_models[name].exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "Selected checkpoints do not exist:\n" + "\n".join(missing_checkpoints)
        )
    for required in (
        pinpoint_root / "pinpoint_licensed.parquet",
        pinpoint_root / "index_signatures.txt",
        pinpoint_root / "image_cache",
        pinpoint_root / "evaluate.py",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if top_k < 10:
        raise ValueError("--top-k must be at least 10 for mAP@10/NegRecall@10")


def main() -> None:
    args = parse_args()
    custom_models: dict[str, Path] = {}
    for specification in args.checkpoint:
        if "=" not in specification:
            raise ValueError(
                f"invalid --checkpoint {specification!r}; expected NAME=PATH"
            )
        name, raw_path = specification.split("=", 1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise ValueError(
                f"invalid --checkpoint {specification!r}; expected NAME=PATH"
            )
        if name in SUPPORTED_MODELS or name in custom_models:
            raise ValueError(f"duplicate model registration: {name}")
        custom_models[name] = Path(raw_path.strip()).resolve()
    custom_adapters: dict[str, Path] = {}
    for specification in args.adapter:
        if "=" not in specification:
            raise ValueError(
                f"invalid --adapter {specification!r}; expected NAME=PATH"
            )
        name, raw_path = specification.split("=", 1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise ValueError(
                f"invalid --adapter {specification!r}; expected NAME=PATH"
            )
        if (
            name in SUPPORTED_MODELS
            or name in custom_models
            or name in custom_adapters
        ):
            raise ValueError(f"duplicate model registration: {name}")
        custom_adapters[name] = Path(raw_path.strip()).resolve()
    custom_pretrained_clips: dict[str, Path] = {}
    for specification in args.pretrained_clip:
        if "=" not in specification:
            raise ValueError(
                f"invalid --pretrained-clip {specification!r}; expected NAME=PATH"
            )
        name, raw_path = specification.split("=", 1)
        name = name.strip()
        if not name or not raw_path.strip():
            raise ValueError(
                f"invalid --pretrained-clip {specification!r}; expected NAME=PATH"
            )
        if (
            name in SUPPORTED_MODELS
            or name in custom_models
            or name in custom_adapters
            or name in custom_pretrained_clips
        ):
            raise ValueError(f"duplicate model registration: {name}")
        custom_pretrained_clips[name] = Path(raw_path.strip()).resolve()
    supported_models = {
        **SUPPORTED_MODELS,
        **custom_models,
        **custom_adapters,
        **custom_pretrained_clips,
    }
    models = list(dict.fromkeys(item.strip() for item in args.models.split(",") if item.strip()))
    pinpoint_root = args.pinpoint_root.resolve()
    output_root = args.output_root.resolve()
    preflight(models, supported_models, pinpoint_root, args.top_k)
    output_root.mkdir(parents=True, exist_ok=True)

    parquet_path = pinpoint_root / "pinpoint_licensed.parquet"
    frame, queries = load_official_queries(parquet_path)
    image_cache = pinpoint_root / "image_cache"
    signatures = read_available_signatures(
        pinpoint_root / "index_signatures.txt",
        image_cache,
        limit=None,
        report_path=output_root / "missing_corpus_signatures.txt",
    )
    reference_ids = {value for query in queries for value in query.reference_ids}
    missing_references = sorted(reference_ids - set(signatures))
    if missing_references:
        raise ValueError(
            f"{len(missing_references)} query reference images are unavailable; "
            f"first={missing_references[:5]}"
        )

    if str(CIR_PLAY_ROOT) not in sys.path:
        sys.path.insert(0, str(CIR_PLAY_ROOT))
    from run_pinpoint_bge_lincir import (
        choose_device,
        load_or_build_corpus_embeddings,
        make_adapter,
        prepare_hf_runtime_cache,
    )

    prepare_hf_runtime_cache(output_root / "_hf_runtime")
    device = choose_device(args.device)
    search_device = str(choose_device(args.search_device or args.device))
    summary_rows: list[dict[str, object]] = []
    forced_frozen_corpus = False
    explicitly_shared_corpus_models = {
        value.strip()
        for value in args.shared_corpus_models.split(",")
        if value.strip()
    }
    unknown_shared = sorted(explicitly_shared_corpus_models - set(models))
    if unknown_shared:
        raise ValueError(
            f"--shared-corpus-models contains unselected models: {unknown_shared}"
        )

    print(
        f"[dataset] official_queries={len(frame)} "
        f"available_corpus={len(signatures)} top_k={args.top_k}",
        flush=True,
    )
    for model_name in models:
        model_path = supported_models[model_name].resolve()
        model_dir = output_root / model_name
        cache_root = model_dir / "cache"
        shares_frozen_corpus = (
            model_name in FROZEN_CLIP_MODELS
            or model_name in explicitly_shared_corpus_models
        )
        if shares_frozen_corpus:
            corpus_cache = (
                args.frozen_corpus_cache.resolve()
                if args.frozen_corpus_cache is not None
                else output_root / "_shared_frozen_clip_l_corpus"
            )
        else:
            corpus_cache = cache_root / "corpus"

        print(f"[model] {model_name}: {model_path}", flush=True)
        if model_name in ORIGINAL_MODELS or model_name in custom_adapters:
            adapter = make_adapter(
                model_name,
                model_path,
                device,
                args.dtype,
                hf_cache_dir=(
                    WORKSPACE_ROOT
                    / "CIR_PLAY"
                    / "models"
                    / "checkpoints"
                    / "lincir"
                    / ".cache"
                    / "huggingface"
                ),
                lincir_clip_model_name="large",
            )
        elif (
            model_name in PRETRAINED_CLIP_MODELS
            or model_name in custom_pretrained_clips
        ):
            adapter = OriginalCLIPAdapter(model_path, device, args.dtype)
        elif model_name in OAP_MODELS:
            adapter = OAPResidualAdapter(
                model_path,
                device,
                args.dtype,
                beta=OAP_MODELS[model_name][1],
            )
        else:
            adapter = TrainedCLIPAdapter(model_path, device, args.dtype)
        force_this_corpus = args.force_corpus and (
            not shares_frozen_corpus or not forced_frozen_corpus
        )
        corpus_embeddings, cached_signatures = load_or_build_corpus_embeddings(
            adapter,
            signatures,
            image_cache=image_cache,
            cache_dir=corpus_cache,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=force_this_corpus,
        )
        if shares_frozen_corpus and force_this_corpus:
            forced_frozen_corpus = True
        query_embeddings, query_fingerprint = load_or_build_query_embeddings(
            adapter,
            queries,
            model_name=model_name,
            model_path=model_path,
            dtype=args.dtype,
            image_cache=image_cache,
            cache_root=cache_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_batches=args.prefetch_batches,
            processor_workers=args.processor_workers,
            processor_prefetch_batches=args.processor_prefetch_batches,
            flush_interval=args.flush_interval,
            force=args.force_queries,
        )
        del adapter
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        _, top_indices, _ = load_or_search(
            query_embeddings,
            corpus_embeddings,
            query_fingerprint_value=query_fingerprint,
            corpus_signatures=signatures,
            cache_root=cache_root,
            top_k=min(args.top_k, len(signatures)),
            device=search_device,
            query_batch_size=args.query_search_batch_size,
            corpus_chunk_size=args.corpus_search_chunk_size,
            force=args.force_search,
        )
        if cached_signatures.astype(str).tolist() != signatures:
            raise ValueError(f"Corpus cache identifiers do not match for {model_name}")

        # evaluate.py uses the results filename stem as its reported model name.
        results_path = model_dir / f"{model_name}.json"
        metrics_path = model_dir / "official_metrics.csv"
        model_dir.mkdir(parents=True, exist_ok=True)
        write_standard_results(results_path, frame, top_indices, signatures)
        metrics = run_official_evaluator(pinpoint_root, results_path, metrics_path)
        summary_rows.append(
            {
                "model": model_name,
                "mAP@10": metrics["mAP@10"],
                "NegRecall@10": metrics["NegRecall@10"],
                "total_queries": metrics["total_queries"],
            }
        )
        print(
            f"[metric] {model_name} mAP@10={float(metrics['mAP@10']):.6f} "
            f"NegRecall@10={float(metrics['NegRecall@10']):.6f}",
            flush=True,
        )

    summary_path = output_root / "official_map10_negrecall10.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format="%.6f")
    write_json_atomic(
        output_root / "run_metadata.json",
        {
            "official_query_count": len(frame),
            "index_signature_count": sum(
                1
                for line in (pinpoint_root / "index_signatures.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ),
            "available_corpus_count": len(signatures),
            "missing_corpus_count": sum(
                1
                for line in (output_root / "missing_corpus_signatures.txt").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ),
            "models": models,
            "headline_metrics": ["mAP@10", "NegRecall@10"],
            "official_evaluator": str(pinpoint_root / "evaluate.py"),
        },
    )
    print(f"[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
