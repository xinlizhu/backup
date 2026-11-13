from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm import tqdm

from cir_qdiag_train.config import TrainConfig
from cir_qdiag_train.inference_data import (
    build_inference_loader,
    inference_loader_settings,
    inference_tensor_to_device,
)
from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint


@torch.no_grad()
def evaluate_cirr(
    config: TrainConfig,
    *,
    checkpoint_path: str | Path | None,
    output_path: str | Path | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
) -> dict[str, float | int | str]:
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
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model=model)
    model.to(device).eval()

    annotations = _load_json(config.data.val_annotation_path)
    image_map = _load_json(config.data.val_image_split_path)
    if not isinstance(annotations, list) or not isinstance(image_map, dict):
        raise ValueError("invalid CIRR validation files")
    image_ids = sorted(image_map)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    workers, prefetch = inference_loader_settings(
        default_workers=config.evaluation.num_workers,
        default_prefetch=config.evaluation.prefetch_factor,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    pin_memory = config.evaluation.pin_memory and device.type == "cuda"
    image_features = _encode_gallery(
        model,
        processor,
        image_ids,
        image_map,
        config.data.image_root,
        config.evaluation.batch_size,
        device,
        config.optimizer.mixed_precision,
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )

    recall_hits = {cutoff: 0 for cutoff in config.evaluation.recall_k}
    group_hits = {cutoff: 0 for cutoff in config.evaluation.group_recall_k}
    processed = 0
    max_recall = max(config.evaluation.recall_k)
    query_loader = build_inference_loader(
        image_paths=[
            _resolve_image(row["reference"], image_map, config.data.image_root)
            for row in annotations
        ],
        texts=[row["caption"] for row in annotations],
        processor=processor,
        batch_size=config.evaluation.batch_size,
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )
    for inputs in tqdm(query_loader, desc="CIRR validation queries"):
        batch = [annotations[index] for index in inputs["indices"].tolist()]
        with _autocast(device, config.optimizer.mixed_precision):
            reference_features = model.encode_reference(
                inference_tensor_to_device(inputs["pixel_values"], device)
            )
            text_features = model.encode_text(
                inference_tensor_to_device(inputs["input_ids"], device),
                (
                    None
                    if inputs.get("attention_mask") is None
                    else inference_tensor_to_device(inputs["attention_mask"], device)
                ),
            )
            queries = model.compose(reference_features, text_features)
            scores = queries @ image_features.t()

        for row_index, row in enumerate(batch):
            reference_id = row["reference"]
            target_id = row.get("target_hard", row.get("target"))
            if reference_id not in image_index or target_id not in image_index:
                raise ValueError("validation query references an image absent from split map")
            row_scores = scores[row_index].clone()
            row_scores[image_index[reference_id]] = -torch.inf
            top_indices = torch.topk(
                row_scores,
                k=min(max_recall, len(image_ids) - 1),
            ).indices.tolist()
            for cutoff in recall_hits:
                recall_hits[cutoff] += image_index[target_id] in top_indices[:cutoff]

            image_set = row.get("img_set", {}).get("members", [])
            group_ids = [
                image_id
                for image_id in image_set
                if image_id != reference_id and image_id in image_index
            ]
            if target_id not in group_ids:
                raise ValueError(
                    f"pair {row.get('pairid')} target is absent from its CIRR image set"
                )
            group_scores = torch.tensor(
                [row_scores[image_index[image_id]].item() for image_id in group_ids]
            )
            group_order = torch.argsort(group_scores, descending=True).tolist()
            ranked_group_ids = [group_ids[index] for index in group_order]
            for cutoff in group_hits:
                group_hits[cutoff] += target_id in ranked_group_ids[:cutoff]
            processed += 1

    metrics: dict[str, float | int | str] = {
        "checkpoint": (
            "pretrained"
            if checkpoint_path is None
            else str(Path(checkpoint_path).resolve())
        ),
        "num_queries": processed,
    }
    metrics.update(
        {
            f"recall_at_{cutoff}": round(100.0 * hits / processed, 4)
            for cutoff, hits in recall_hits.items()
        }
    )
    metrics.update(
        {
            f"group_recall_at_{cutoff}": round(100.0 * hits / processed, 4)
            for cutoff, hits in group_hits.items()
        }
    )
    output = Path(output_path or config.experiment.output_dir / "val_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


@torch.no_grad()
def _encode_gallery(
    model: ComposedRetrievalModel,
    processor,
    image_ids: list[str],
    image_map: dict,
    image_root: Path,
    batch_size: int,
    device: torch.device,
    mixed_precision: str,
    *,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    loader = build_inference_loader(
        image_paths=[_resolve_image(image_id, image_map, image_root) for image_id in image_ids],
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    for inputs in tqdm(loader, desc="CIRR validation gallery"):
        with _autocast(device, mixed_precision):
            batch_features = model.encode_image(
                inference_tensor_to_device(inputs["pixel_values"], device)
            )
        features.append(batch_features)
    return torch.cat(features, dim=0)


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        from contextlib import nullcontext

        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _resolve_image(image_id: str, image_map: dict, image_root: Path) -> Path:
    mapped = image_map.get(image_id)
    if not isinstance(mapped, str):
        raise ValueError(f"image is absent from split map: {image_id}")
    root = Path(image_root).resolve()
    relative = Path(mapped.removeprefix("./"))
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError(f"unsafe image path: {mapped}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _chunks(values: list, size: int) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
