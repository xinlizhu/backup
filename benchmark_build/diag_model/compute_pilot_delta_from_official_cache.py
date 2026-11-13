#!/usr/bin/env python3
"""Compute PinPoint V7 negative-text Delta from official-eval caches.

This avoids re-encoding the image corpus and original positive queries.  Only
the generated negative texts are encoded; reference/candidate embeddings and
positive composed-query embeddings come from ``run_clip_pinpoint_official``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pinpoint_full_diagnostics import load_v7_dataset, summarize_negative_pairs
from run_clip_pinpoint_official import OAP_MODELS
from trained_clip_adapter import (
    TRAINED_CLIP_MODELS,
    OAPResidualAdapter,
    TrainedCLIPAdapter,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "final.jsonl"
)
DEFAULT_CACHE_ROOT = (
    SCRIPT_DIR.parent
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_official_full_lora"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent
    / "cir_qdiag_openai"
    / "outputs"
    / "pinpoint_pilot_500_v7"
    / "full_delta_from_official_cache"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--official-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        default="CLIP-L-full-baseline,CLIP-L-full-hard-w0.5",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def only_child(path: Path) -> Path:
    children = sorted(item for item in path.iterdir() if item.is_dir())
    if len(children) != 1:
        raise ValueError(f"Expected one cache fingerprint directory in {path}, got {children}")
    return children[0]


def load_cache(model_dir: Path) -> tuple[np.ndarray, dict[str, int], np.ndarray, dict[str, int]]:
    corpus_dir = model_dir / "cache" / "corpus"
    query_dir = only_child(model_dir / "cache" / "queries")
    corpus = np.load(corpus_dir / "corpus_embeddings.npy", mmap_mode="r")
    signatures = np.load(corpus_dir / "corpus_signatures.npy", allow_pickle=False).astype(str)
    positive_queries = np.load(query_dir / "query_embeddings.npy", mmap_mode="r")
    query_keys = np.load(query_dir / "query_keys.npy", allow_pickle=False).astype(str)
    return (
        corpus,
        {value: index for index, value in enumerate(signatures)},
        positive_queries,
        {value: index for index, value in enumerate(query_keys)},
    )


def official_key(instruction_id: str) -> str:
    value = instruction_id.removeprefix("query_")
    return f"official::{int(value):05d}"


def pooled_references(
    reference_ids: tuple[str, ...],
    corpus: np.ndarray,
    signature_to_index: dict[str, int],
    embedding_dim: int | None = None,
) -> np.ndarray:
    rows = np.asarray(
        corpus[[signature_to_index[value] for value in reference_ids]],
        dtype=np.float32,
    )
    if embedding_dim is not None:
        rows = rows[:, :embedding_dim]
    pooled = rows.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm == 0:
        raise ValueError(f"Zero-norm reference pool for {reference_ids}")
    return pooled / norm


def evaluate_model(
    model_name: str,
    *,
    groups,
    queries,
    cache_root: Path,
    output_root: Path,
    device: torch.device,
    dtype: str,
    batch_size: int,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    model_dir = cache_root / model_name
    corpus, signature_to_index, positive_queries, key_to_index = load_cache(model_dir)
    if model_name in OAP_MODELS:
        checkpoint, beta = OAP_MODELS[model_name]
        adapter = OAPResidualAdapter(checkpoint, device, dtype, beta=beta)
        reference_dim = adapter.model.global_dim
    else:
        adapter = TrainedCLIPAdapter(TRAINED_CLIP_MODELS[model_name], device, dtype)
        reference_dim = None

    group_by_id = {group.group_id: group for group in groups}
    negatives = [query for query in queries if query.query_kind == "negative"]
    reference_rows = np.stack(
        [
            pooled_references(
                query.reference_ids,
                corpus,
                signature_to_index,
                reference_dim,
            )
            for query in negatives
        ]
    ).astype(np.float32, copy=False)

    negative_embeddings: list[np.ndarray] = []
    for start in range(0, len(negatives), batch_size):
        batch = negatives[start : start + batch_size]
        text_inputs = adapter.processor(
            text=[query.text for query in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        moved = adapter._to_device(dict(text_inputs))
        references = torch.from_numpy(reference_rows[start : start + len(batch)]).to(
            device=device,
            dtype=adapter.target_dtype,
        )
        with torch.inference_mode():
            if isinstance(adapter, OAPResidualAdapter):
                text_features = adapter.model.base.encode_text(
                    moved["input_ids"],
                    moved.get("attention_mask"),
                )
                if adapter.model.base.normalize_components_before_fusion:
                    references = F.normalize(references, dim=-1)
                global_query = adapter.model.base.compose(references, text_features)
                semantic_query = adapter.model.project_query_semantics(global_query)
                composed = adapter.model.pack_query(global_query, semantic_query)
            else:
                text_features = adapter.model.encode_text(
                    moved["input_ids"],
                    moved.get("attention_mask"),
                )
                composed = adapter.model.compose(
                    F.normalize(references, dim=-1), text_features
                )
        negative_embeddings.append(composed.float().cpu().numpy())
        print(f"[{model_name}] negative queries {start + len(batch)}/{len(negatives)}", flush=True)
    negative_embeddings_array = np.concatenate(negative_embeddings, axis=0)

    rows: list[dict] = []
    for query, negative_embedding in zip(negatives, negative_embeddings_array, strict=True):
        group = group_by_id[query.group_id]
        positive_indices = [signature_to_index[value] for value in group.positive_ids]
        targets = np.asarray(corpus[positive_indices], dtype=np.float32)
        positive_embedding = np.asarray(
            positive_queries[key_to_index[official_key(query.instruction_id)]],
            dtype=np.float32,
        )
        positive_scores = positive_embedding @ targets.T
        negative_scores = np.asarray(negative_embedding, dtype=np.float32) @ targets.T
        deltas = positive_scores - negative_scores
        rows.append(
            {
                "group_id": query.group_id,
                "instruction_id": query.instruction_id,
                "negative_id": query.negative_id,
                "semantic_target": query.semantic_target,
                "edit_operation": query.edit_operation,
                "positive_target_count": len(positive_indices),
                "mean_positive_query_target_score": float(positive_scores.mean()),
                "mean_negative_query_target_score": float(negative_scores.mean()),
                "mean_target_delta": float(deltas.mean()),
                "target_raw_win_rate": float(np.mean(deltas > 0.0)),
                # Robust calibration needs native paraphrases; it is intentionally
                # omitted in this cache-only Delta calculation.
                "target_robust_win_rate": float("nan"),
            }
        )

    frame = pd.DataFrame(rows)
    summary = summarize_negative_pairs(
        frame,
        ks=(),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(model_output / "negative_pair_metrics.csv", index=False)
    summary.to_csv(model_output / "negative_summary.csv", index=False)
    del adapter
    return summary


def main() -> None:
    args = parse_args()
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    supported_models = set(TRAINED_CLIP_MODELS) | set(OAP_MODELS)
    unknown = sorted(set(models) - supported_models)
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    groups, queries, coverage = load_v7_dataset(args.input.resolve(), limit_groups=None)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cir_play_root = SCRIPT_DIR.parents[1] / "CIR_PLAY"
    if str(cir_play_root) not in sys.path:
        sys.path.insert(0, str(cir_play_root))
    from run_pinpoint_bge_lincir import prepare_hf_runtime_cache

    prepare_hf_runtime_cache(args.output_root / "_hf_runtime")
    summaries = []
    for model_name in models:
        summary = evaluate_model(
            model_name,
            groups=groups,
            queries=queries,
            cache_root=args.official_cache_root.resolve(),
            output_root=args.output_root.resolve(),
            device=torch.device(args.device),
            dtype=args.dtype,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        summary.insert(0, "model", model_name)
        summaries.append(summary)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output_root / "negative_summary.csv", index=False)
    compact = combined.loc[
        combined["subset"].isin(["all", "strict"]),
        [
            "model",
            "subset",
            "group_count",
            "instruction_count",
            "negative_count",
            "pair_balanced_mean_target_delta",
            "pair_balanced_target_raw_win_rate",
            "target_raw_win_rate_ci_low",
            "target_raw_win_rate_ci_high",
        ],
    ]
    compact.to_csv(args.output_root / "comparison.csv", index=False)
    (args.output_root / "metadata.json").write_text(
        json.dumps({"coverage": coverage, "models": models}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(compact.to_string(index=False))
    print(f"[done] {args.output_root / 'comparison.csv'}")


if __name__ == "__main__":
    main()
