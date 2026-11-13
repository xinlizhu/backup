from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from cir_qdiag_train.config import TrainConfig
from cir_qdiag_train.evaluation import (
    _autocast,
    _chunks,
    _encode_gallery,
    _load_image,
    _load_json,
    _resolve_image,
)
from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint


@torch.no_grad()
def evaluate_cirr_reverse(
    config: TrainConfig,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Retrieve composed CIRR queries with each unique target image.

    A target image may be shared by several CIRR validation rows. All rows whose
    ``target_hard`` (or legacy ``target``) is that image are therefore treated as
    relevant, instead of forcing a false one-to-one diagonal correspondence.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = ComposedRetrievalModel.from_pretrained(
        config.model.pretrained_model_name_or_path,
        variant=config.model.variant,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        composition_head=config.model.composition_head,
        fusion_init_path=config.model.fusion_init_path,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        freeze_backbone=config.model.freeze_backbone,
        freeze_vision_encoder=config.model.freeze_vision_encoder,
        gradient_checkpointing=False,
        temperature=config.optimizer.temperature,
        learnable_temperature=config.optimizer.learnable_temperature,
        unfreeze_last_n_layers=config.model.unfreeze_last_n_layers,
        lora_rank=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
    )
    load_checkpoint(checkpoint_path, model=model)
    model.to(device).eval()

    annotations = _load_json(config.data.val_annotation_path)
    image_map = _load_json(config.data.val_image_split_path)
    if not isinstance(annotations, list) or not isinstance(image_map, dict):
        raise TypeError("invalid CIRR validation files")
    if not annotations:
        raise ValueError("CIRR validation annotations are empty")

    query_target_ids = [_target_id(row) for row in annotations]
    target_ids = sorted(set(query_target_ids))
    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    missing_targets = sorted(set(target_ids) - set(image_map))
    if missing_targets:
        raise ValueError(f"reverse targets are absent from split map: {missing_targets[:5]}")

    target_features = _encode_gallery(
        model,
        processor,
        target_ids,
        image_map,
        config.data.image_root,
        config.evaluation.batch_size,
        device,
        config.optimizer.mixed_precision,
    )
    query_features = _encode_composed_queries(
        model,
        processor,
        annotations,
        image_map,
        config.data.image_root,
        config.evaluation.batch_size,
        device,
        config.optimizer.mixed_precision,
    )
    scores = (target_features @ query_features.t()).float()
    query_target_indices = torch.tensor(
        [target_index[target_id] for target_id in query_target_ids],
        dtype=torch.long,
        device=scores.device,
    )
    metrics = compute_reverse_metrics(
        scores,
        query_target_indices,
        cutoffs=config.evaluation.recall_k,
    )
    metrics.update(
        {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "direction": "target_image_to_composed_query",
            "score": "cosine",
            "positive_definition": (
                "all validation query rows sharing the reverse target image"
            ),
            "metric_units": {
                "target_hit_rate_at_k": "percent of unique targets with any positive in top-k",
                "macro_query_recall_at_k": "percent, averaged equally over unique targets",
                "pair_recall_at_k": "percent of all target-query positive pairs retrieved",
                "map_at_k": "fraction in [0, 1], macro over unique targets",
                "ndcg_at_k": "fraction in [0, 1], macro over unique targets",
                "mrr": "fraction in [0, 1], reciprocal rank of first positive",
            },
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def compute_reverse_metrics(
    scores: torch.Tensor,
    query_target_indices: torch.Tensor,
    *,
    cutoffs: list[int],
) -> dict[str, object]:
    """Compute multi-positive target-to-query retrieval metrics from a score matrix."""

    if scores.ndim != 2:
        raise ValueError(f"scores must be a matrix, got shape={tuple(scores.shape)}")
    num_targets, num_queries = scores.shape
    if query_target_indices.shape != (num_queries,):
        raise ValueError("query target indices do not match the score matrix")
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError("reverse retrieval cutoffs must be positive")
    if query_target_indices.numel() and (
        query_target_indices.min() < 0 or query_target_indices.max() >= num_targets
    ):
        raise ValueError("query target index is out of range")

    positive_counts = torch.bincount(
        query_target_indices.detach().cpu(), minlength=num_targets
    )
    if num_targets == 0 or num_queries == 0 or (positive_counts == 0).any():
        raise ValueError("every reverse target must have at least one positive query")

    order = torch.argsort(scores, dim=1, descending=True).detach().cpu()
    query_targets_cpu = query_target_indices.detach().cpu()
    cutoffs = sorted({int(cutoff) for cutoff in cutoffs})
    target_hits = {cutoff: 0 for cutoff in cutoffs}
    macro_recall = {cutoff: 0.0 for cutoff in cutoffs}
    pair_hits = {cutoff: 0 for cutoff in cutoffs}
    average_precision = {cutoff: 0.0 for cutoff in cutoffs}
    ndcg = {cutoff: 0.0 for cutoff in cutoffs}
    best_ranks: list[int] = []
    positive_rank_sum = 0

    for target_id in range(num_targets):
        ranks = torch.nonzero(
            query_targets_cpu[order[target_id]] == target_id,
            as_tuple=False,
        ).flatten() + 1
        positive_count = int(positive_counts[target_id])
        best_ranks.append(int(ranks[0]))
        positive_rank_sum += int(ranks.sum())
        for cutoff in cutoffs:
            retrieved = ranks[ranks <= cutoff]
            retrieved_count = int(retrieved.numel())
            target_hits[cutoff] += int(retrieved_count > 0)
            macro_recall[cutoff] += retrieved_count / positive_count
            pair_hits[cutoff] += retrieved_count
            if retrieved_count:
                precision = torch.arange(1, retrieved_count + 1) / retrieved.float()
                average_precision[cutoff] += float(
                    precision.sum() / min(positive_count, cutoff)
                )
                gains = 1.0 / torch.log2(retrieved.float() + 1.0)
                ideal_ranks = torch.arange(1, min(positive_count, cutoff) + 1).float()
                ideal_gain = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()
                ndcg[cutoff] += float(gains.sum() / ideal_gain)

    sorted_best_ranks = sorted(best_ranks)
    midpoint = num_targets // 2
    if num_targets % 2:
        median_best_rank = float(sorted_best_ranks[midpoint])
    else:
        median_best_rank = 0.5 * (
            sorted_best_ranks[midpoint - 1] + sorted_best_ranks[midpoint]
        )

    metrics: dict[str, object] = {
        "num_reverse_targets": num_targets,
        "num_query_candidates": num_queries,
        "num_positive_pairs": int(positive_counts.sum()),
        "targets_with_multiple_queries": int((positive_counts > 1).sum()),
        "max_queries_per_target": int(positive_counts.max()),
        "mean_queries_per_target": round(float(positive_counts.float().mean()), 6),
        "mean_best_positive_rank": round(sum(best_ranks) / num_targets, 4),
        "median_best_positive_rank": round(median_best_rank, 4),
        "mean_positive_rank": round(positive_rank_sum / num_queries, 4),
        "mean_positive_percentile": round(
            (positive_rank_sum / num_queries - 1) / max(num_queries - 1, 1), 8
        ),
        "mrr": round(sum(1.0 / rank for rank in best_ranks) / num_targets, 8),
    }
    for cutoff in cutoffs:
        metrics[f"target_hit_rate_at_{cutoff}"] = round(
            100.0 * target_hits[cutoff] / num_targets, 4
        )
        metrics[f"macro_query_recall_at_{cutoff}"] = round(
            100.0 * macro_recall[cutoff] / num_targets, 4
        )
        metrics[f"pair_recall_at_{cutoff}"] = round(
            100.0 * pair_hits[cutoff] / num_queries, 4
        )
        metrics[f"map_at_{cutoff}"] = round(
            average_precision[cutoff] / num_targets, 8
        )
        metrics[f"ndcg_at_{cutoff}"] = round(ndcg[cutoff] / num_targets, 8)
    return metrics


@torch.no_grad()
def _encode_composed_queries(
    model: ComposedRetrievalModel,
    processor,
    annotations: list[dict],
    image_map: dict,
    image_root: Path,
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for batch in tqdm(
        list(_chunks(annotations, batch_size)),
        desc="CIRR composed-query gallery",
    ):
        reference_images = [
            _load_image(_resolve_image(row["reference"], image_map, image_root))
            for row in batch
        ]
        image_inputs = processor(images=reference_images, return_tensors="pt")
        text_inputs = processor(
            text=[row["caption"] for row in batch],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with _autocast(device, mixed_precision):
            reference_features = model.encode_reference(
                image_inputs["pixel_values"].to(device)
            )
            text_features = model.encode_text(
                text_inputs["input_ids"].to(device),
                (
                    None
                    if text_inputs.get("attention_mask") is None
                    else text_inputs["attention_mask"].to(device)
                ),
            )
            features.append(model.compose(reference_features, text_features))
    return torch.cat(features, dim=0)


def _target_id(row: dict) -> str:
    target = row.get("target_hard", row.get("target"))
    if not isinstance(target, str) or not target:
        raise ValueError(f"CIRR validation row has no target image: pairid={row.get('pairid')}")
    return target
