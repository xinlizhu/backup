from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from cir_qdiag_train.oap_config import OAPConfig
from cir_qdiag_train.oap_model import build_oap_model, load_oap_checkpoint


@torch.no_grad()
def evaluate_oap_cirr(
    config: OAPConfig,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = build_oap_model(config)
    load_oap_checkpoint(checkpoint_path, model=model)
    model.to(device).eval()

    annotations = _load_json(config.data.val_annotation_path)
    image_map = _load_json(config.data.val_image_split_path)
    if not isinstance(annotations, list) or not isinstance(image_map, dict):
        raise TypeError("invalid CIRR validation annotations or image split")
    image_ids = sorted(image_map)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    target_global, target_semantic = _encode_gallery(
        config, model, processor, image_ids, image_map, device
    )

    modes = ("global", "final")
    recall_hits = {
        mode: {cutoff: 0 for cutoff in config.evaluation.recall_k}
        for mode in modes
    }
    group_hits = {
        mode: {cutoff: 0 for cutoff in config.evaluation.group_recall_k}
        for mode in modes
    }
    rank_sums = {mode: 0.0 for mode in modes}
    reciprocal_rank_sums = {mode: 0.0 for mode in modes}
    top1_corrections = 0
    top1_regressions = 0
    processed = 0
    maximum = max(config.evaluation.recall_k)

    for batch in tqdm(
        list(_chunks(annotations, config.evaluation.batch_size)),
        desc="OAP CIRR validation",
    ):
        references = [
            _load_image(
                _resolve_split_image(row["reference"], image_map, config.data.image_root)
            )
            for row in batch
        ]
        image_inputs = processor(images=references, return_tensors="pt")
        text_inputs = processor(
            text=[row["caption"] for row in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with _autocast(device, config.optimizer.mixed_precision):
            query_global, query_semantic = model.encode_queries(
                image_inputs["pixel_values"].to(device),
                text_inputs["input_ids"].to(device),
                _optional_to(text_inputs.get("attention_mask"), device),
            )
            global_scores = query_global @ target_global.t()
            final_scores = model.pairwise_score(
                query_global,
                query_semantic,
                target_global,
                target_semantic,
            )

        for row_index, row in enumerate(batch):
            reference_id = row["reference"]
            target_id = row.get("target_hard", row.get("target"))
            if reference_id not in image_index or target_id not in image_index:
                raise ValueError("validation query references an image absent from split")
            target_index = image_index[target_id]
            per_mode_scores = {
                "global": global_scores[row_index].clone(),
                "final": final_scores[row_index].clone(),
            }
            for scores in per_mode_scores.values():
                scores[image_index[reference_id]] = -torch.inf

            top1 = {}
            for mode, scores in per_mode_scores.items():
                top = torch.topk(
                    scores, k=min(maximum, len(image_ids) - 1)
                ).indices.tolist()
                top1[mode] = bool(top and top[0] == target_index)
                for cutoff in recall_hits[mode]:
                    recall_hits[mode][cutoff] += int(
                        target_index in top[:cutoff]
                    )
                rank = int((scores > scores[target_index]).sum().item()) + 1
                rank_sums[mode] += rank
                reciprocal_rank_sums[mode] += 1.0 / rank

                members = row.get("img_set", {}).get("members", [])
                group_ids = [
                    image_id
                    for image_id in members
                    if image_id != reference_id and image_id in image_index
                ]
                if target_id not in group_ids:
                    raise ValueError(
                        f"pair {row.get('pairid')} target is absent from its group"
                    )
                ranked_group = sorted(
                    group_ids,
                    key=lambda image_id: float(scores[image_index[image_id]]),
                    reverse=True,
                )
                for cutoff in group_hits[mode]:
                    group_hits[mode][cutoff] += int(
                        target_id in ranked_group[:cutoff]
                    )

            top1_corrections += int(not top1["global"] and top1["final"])
            top1_regressions += int(top1["global"] and not top1["final"])
            processed += 1

    metrics_by_mode: dict[str, dict[str, float | int]] = {}
    for mode in modes:
        metrics: dict[str, float | int] = {
            "num_queries": processed,
            "gallery_size": len(image_ids),
        }
        metrics.update(
            {
                f"recall_at_{cutoff}": round(100.0 * hits / processed, 4)
                for cutoff, hits in recall_hits[mode].items()
            }
        )
        metrics.update(
            {
                f"group_recall_at_{cutoff}": round(100.0 * hits / processed, 4)
                for cutoff, hits in group_hits[mode].items()
            }
        )
        metrics["mean_rank"] = round(rank_sums[mode] / processed, 4)
        metrics["mrr"] = round(reciprocal_rank_sums[mode] / processed, 6)
        metrics_by_mode[mode] = metrics

    delta = {
        key: round(
            float(metrics_by_mode["final"][key])
            - float(metrics_by_mode["global"][key]),
            4,
        )
        for key in metrics_by_mode["global"]
        if key.startswith(("recall_at_", "group_recall_at_"))
    }
    result: dict[str, Any] = {
        "architecture": model.architecture,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "base_checkpoint": str(config.model.base_checkpoint),
        "beta": config.model.beta,
        "global_baseline": metrics_by_mode["global"],
        "oap_final": metrics_by_mode["final"],
        "oap_minus_global": delta,
        "top1_ranking_changes": {
            "corrected_queries": top1_corrections,
            "regressed_queries": top1_regressions,
            "net_corrections": top1_corrections - top1_regressions,
            "correction_rate_over_all_queries": round(
                top1_corrections / processed, 6
            ),
            "regression_rate_over_all_queries": round(
                top1_regressions / processed, 6
            ),
        },
    }
    output = Path(
        output_path or config.experiment.output_dir / "oap_cirr_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


@torch.no_grad()
def evaluate_oap_reranker(
    config: OAPConfig,
    *,
    checkpoint_path: str | Path,
    betas: list[float],
    rerank_top_ks: list[int],
    output_path: str | Path,
) -> dict[str, Any]:
    """Retrieve globally, then reorder only the original global Top-K with OAP."""

    if not betas or min(betas) <= 0:
        raise ValueError("reranker betas must be positive")
    if not rerank_top_ks or min(rerank_top_ks) <= 0:
        raise ValueError("rerank Top-K values must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = build_oap_model(config)
    load_oap_checkpoint(checkpoint_path, model=model)
    model.to(device).eval()

    annotations = _load_json(config.data.val_annotation_path)
    image_map = _load_json(config.data.val_image_split_path)
    if not isinstance(annotations, list) or not isinstance(image_map, dict):
        raise TypeError("invalid CIRR validation annotations or image split")
    image_ids = sorted(image_map)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    target_global, target_semantic = _encode_gallery(
        config, model, processor, image_ids, image_map, device
    )

    cutoffs = config.evaluation.recall_k
    group_cutoffs = config.evaluation.group_recall_k
    baseline = _new_rank_totals(cutoffs, group_cutoffs)
    run_keys = [(float(beta), int(top_k)) for beta in betas for top_k in rerank_top_ks]
    totals = {
        key: _new_rank_totals(cutoffs, group_cutoffs) for key in run_keys
    }
    changes = {
        key: {"corrected": 0, "regressed": 0, "top1_changed": 0}
        for key in run_keys
    }
    processed = 0

    for batch in tqdm(
        list(_chunks(annotations, config.evaluation.batch_size)),
        desc="OAP CIRR reranker",
    ):
        references = [
            _load_image(
                _resolve_split_image(row["reference"], image_map, config.data.image_root)
            )
            for row in batch
        ]
        image_inputs = processor(images=references, return_tensors="pt")
        text_inputs = processor(
            text=[row["caption"] for row in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with _autocast(device, config.optimizer.mixed_precision):
            query_global, query_semantic = model.encode_queries(
                image_inputs["pixel_values"].to(device),
                text_inputs["input_ids"].to(device),
                _optional_to(text_inputs.get("attention_mask"), device),
            )
            global_scores = (query_global @ target_global.t()).float()
            semantic_scores = torch.einsum(
                "nkd,mkd->nmk", query_semantic, target_semantic
            ).mean(dim=-1).float()

        for row_index, row in enumerate(batch):
            reference_id = row["reference"]
            target_id = row.get("target_hard", row.get("target"))
            target_index = image_index[target_id]
            scores = global_scores[row_index].clone()
            scores[image_index[reference_id]] = -torch.inf
            global_order = torch.argsort(scores, descending=True)
            group_indices = _group_indices(row, reference_id, target_id, image_index)
            _update_rank_totals(
                baseline,
                order=global_order,
                target_index=target_index,
                group_indices=group_indices,
                cutoffs=cutoffs,
                group_cutoffs=group_cutoffs,
            )
            global_top1_correct = int(global_order[0]) == target_index
            global_positions = _rank_positions(global_order)
            group_tensor = torch.tensor(
                group_indices, dtype=torch.long, device=global_order.device
            )
            group_global_positions = global_positions[group_tensor]
            target_group_position = group_indices.index(target_index)

            semantic = semantic_scores[row_index]
            for beta, top_k in run_keys:
                effective_k = min(top_k, len(image_ids) - 1)
                selected = global_order[:effective_k]
                selected_final_scores = scores[selected] + beta * semantic[selected]
                local_order = torch.argsort(selected_final_scores, descending=True)
                selected_new_positions = _rank_positions(local_order)
                target_global_position = global_positions[target_index]
                target_position = torch.where(
                    target_global_position < effective_k,
                    selected_new_positions[
                        torch.clamp(target_global_position, max=effective_k - 1)
                    ],
                    target_global_position,
                )
                group_final_positions = group_global_positions.clone()
                group_inside = group_global_positions < effective_k
                group_final_positions[group_inside] = selected_new_positions[
                    group_global_positions[group_inside]
                ]
                group_rank = int(
                    (
                        group_final_positions
                        < group_final_positions[target_group_position]
                    )
                    .sum()
                    .item()
                ) + 1
                _update_rank_values(
                    totals[(beta, top_k)],
                    rank=int(target_position.item()) + 1,
                    group_rank=group_rank,
                    cutoffs=cutoffs,
                    group_cutoffs=group_cutoffs,
                )
                reranked_top1 = int(selected[local_order[0]])
                reranked_top1_correct = reranked_top1 == target_index
                changes[(beta, top_k)]["corrected"] += int(
                    not global_top1_correct and reranked_top1_correct
                )
                changes[(beta, top_k)]["regressed"] += int(
                    global_top1_correct and not reranked_top1_correct
                )
                changes[(beta, top_k)]["top1_changed"] += int(
                    reranked_top1 != int(global_order[0])
                )
            processed += 1

    baseline_metrics = _finalize_rank_totals(
        baseline, processed, len(image_ids), cutoffs, group_cutoffs
    )
    runs: list[dict[str, Any]] = []
    for beta, top_k in run_keys:
        metrics = _finalize_rank_totals(
            totals[(beta, top_k)],
            processed,
            len(image_ids),
            cutoffs,
            group_cutoffs,
        )
        delta = {
            key: round(float(metrics[key]) - float(baseline_metrics[key]), 4)
            for key in baseline_metrics
            if key.startswith(("recall_at_", "group_recall_at_"))
        }
        row_changes = changes[(beta, top_k)]
        runs.append(
            {
                "beta": beta,
                "rerank_top_k": top_k,
                "metrics": metrics,
                "minus_global": delta,
                "top1_ranking_changes": {
                    **row_changes,
                    "net_corrections": (
                        row_changes["corrected"] - row_changes["regressed"]
                    ),
                },
            }
        )
    result = {
        "architecture": model.architecture,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "protocol": "global retrieval followed by OAP-only reordering inside Top-K",
        "global_baseline": baseline_metrics,
        "runs": runs,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _new_rank_totals(
    cutoffs: list[int], group_cutoffs: list[int]
) -> dict[str, Any]:
    return {
        "recall": {cutoff: 0 for cutoff in cutoffs},
        "group_recall": {cutoff: 0 for cutoff in group_cutoffs},
        "rank_sum": 0.0,
        "reciprocal_rank_sum": 0.0,
    }


def _group_indices(
    row: dict[str, Any],
    reference_id: str,
    target_id: str,
    image_index: dict[str, int],
) -> list[int]:
    members = row.get("img_set", {}).get("members", [])
    group_ids = [
        image_id
        for image_id in members
        if image_id != reference_id and image_id in image_index
    ]
    if target_id not in group_ids:
        raise ValueError(f"pair {row.get('pairid')} target is absent from its group")
    return [image_index[image_id] for image_id in group_ids]


def _update_rank_totals(
    totals: dict[str, Any],
    *,
    order: torch.Tensor,
    target_index: int,
    group_indices: list[int],
    cutoffs: list[int],
    group_cutoffs: list[int],
) -> None:
    positions = _rank_positions(order)
    rank = int(positions[target_index].item()) + 1
    group_positions = positions[
        torch.tensor(group_indices, dtype=torch.long, device=order.device)
    ]
    target_group_position = group_indices.index(target_index)
    group_rank = int(
        (group_positions < group_positions[target_group_position]).sum().item()
    ) + 1
    _update_rank_values(
        totals,
        rank=rank,
        group_rank=group_rank,
        cutoffs=cutoffs,
        group_cutoffs=group_cutoffs,
    )


def _rank_positions(order: torch.Tensor) -> torch.Tensor:
    positions = torch.empty_like(order)
    positions[order] = torch.arange(order.numel(), device=order.device)
    return positions


def _update_rank_values(
    totals: dict[str, Any],
    *,
    rank: int,
    group_rank: int,
    cutoffs: list[int],
    group_cutoffs: list[int],
) -> None:
    totals["rank_sum"] += rank
    totals["reciprocal_rank_sum"] += 1.0 / rank
    for cutoff in cutoffs:
        totals["recall"][cutoff] += int(rank <= cutoff)
    for cutoff in group_cutoffs:
        totals["group_recall"][cutoff] += int(group_rank <= cutoff)


def _finalize_rank_totals(
    totals: dict[str, Any],
    processed: int,
    gallery_size: int,
    cutoffs: list[int],
    group_cutoffs: list[int],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "num_queries": processed,
        "gallery_size": gallery_size,
    }
    metrics.update(
        {
            f"recall_at_{cutoff}": round(
                100.0 * totals["recall"][cutoff] / processed, 4
            )
            for cutoff in cutoffs
        }
    )
    metrics.update(
        {
            f"group_recall_at_{cutoff}": round(
                100.0 * totals["group_recall"][cutoff] / processed, 4
            )
            for cutoff in group_cutoffs
        }
    )
    metrics["mean_rank"] = round(totals["rank_sum"] / processed, 4)
    metrics["mrr"] = round(totals["reciprocal_rank_sum"] / processed, 6)
    return metrics


@torch.no_grad()
def _encode_gallery(
    config: OAPConfig,
    model,
    processor,
    image_ids: list[str],
    image_map: dict[str, str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_rows: list[torch.Tensor] = []
    semantic_rows: list[torch.Tensor] = []
    for ids in tqdm(
        list(_chunks(image_ids, config.evaluation.batch_size)),
        desc="OAP CIRR gallery",
    ):
        images = [
            _load_image(
                _resolve_split_image(image_id, image_map, config.data.image_root)
            )
            for image_id in ids
        ]
        inputs = processor(images=images, return_tensors="pt")
        with _autocast(device, config.optimizer.mixed_precision):
            global_target, semantic_target = model.encode_targets(
                inputs["pixel_values"].to(device)
            )
        global_rows.append(global_target)
        semantic_rows.append(semantic_target)
    return torch.cat(global_rows), torch.cat(semantic_rows)


def _resolve_split_image(
    image_id: str, image_map: dict[str, str], image_root: Path
) -> Path:
    mapped = image_map.get(image_id)
    if not isinstance(mapped, str) or not mapped:
        raise ValueError(f"image {image_id} is absent from split map")
    path = (image_root / mapped.removeprefix("./")).resolve()
    if not path.is_relative_to(image_root.resolve()):
        raise ValueError(f"unsafe image path for {image_id}: {mapped}")
    return path


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _optional_to(value: torch.Tensor | None, device: torch.device):
    return None if value is None else value.to(device)
