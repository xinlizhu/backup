"""Adapter for CLIP-like composed-retrieval checkpoints from cir_qdiag_train."""

from __future__ import annotations

import sys
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
TRAIN_PROJECT_ROOT = BENCHMARK_ROOT / "cir_qdiag_train"
TRAIN_SOURCE_ROOT = TRAIN_PROJECT_ROOT / "src"

TRAINED_CLIP_MODELS = {
    "BGE-VL-large-full-hard-w0.1": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "bge_vl_large"
        / "full"
        / "hard_w0_1"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-baseline": (
        TRAIN_PROJECT_ROOT / "outputs" / "clip_l" / "batch64" / "baseline" / "last.pt"
    ),
    "CLIP-L-hard-w0.5": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "batch64"
        / "hard_w0_5"
        / "last.pt"
    ),
    "CLIP-L-last4-baseline": (
        TRAIN_PROJECT_ROOT / "outputs" / "clip_l" / "last4" / "baseline" / "last.pt"
    ),
    "CLIP-L-last4-hard-w0.5": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "last4"
        / "hard_w0_5"
        / "last.pt"
    ),
    "CLIP-L-full-baseline": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "baseline"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-full-hard-w0.1": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "hard_w0_1"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-full-hard-w0.25": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "hard_w0_25"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-full-hard-w0.5": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "hard_w0_5"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-full-hard-w0.75": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "hard_w0_75"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-full-hard-w1.0": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "full"
        / "hard_w1_0"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-lora-r16-baseline": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "lora_r16"
        / "baseline"
        / "checkpoint_epoch_010.pt"
    ),
    "CLIP-L-lora-r16-hard-w0.5": (
        TRAIN_PROJECT_ROOT
        / "outputs"
        / "clip_l"
        / "lora_r16"
        / "hard_w0_5"
        / "checkpoint_epoch_010.pt"
    ),
}

FROZEN_CLIP_MODELS = {"CLIP-L-baseline", "CLIP-L-hard-w0.5"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in {"", "none", "nan", "null"}


def _reference_count(row: pd.Series) -> int:
    return sum(
        not _is_missing(row.get(column))
        for column in ("query_image_signature", "query_image_signature2")
    )


def _target_dtype(dtype: str, device: torch.device) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype not in aliases:
        raise ValueError(f"Unsupported dtype: {dtype}")
    selected = aliases[dtype]
    if device.type == "cpu" and selected == torch.float16:
        raise ValueError("float16 CLIP inference is not supported on CPU; use float32")
    return selected


class TrainedCLIPAdapter:
    """Expose a trained residual CLIP composer through the PinPoint interface.

    Candidate images use the frozen CLIP image encoder. For a two-reference
    PinPoint query, the normalized reference embeddings are averaged and
    normalized before applying the same learned composition head used in
    training. One-reference queries therefore exactly match the training path.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        dtype: str,
    ) -> None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if str(TRAIN_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(TRAIN_SOURCE_ROOT))
        from cir_qdiag_train.model import ComposedRetrievalModel

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_config = checkpoint.get("config", {}).get("model")
        if not isinstance(model_config, dict):
            raise TypeError(f"Checkpoint has no model config: {checkpoint_path}")
        if model_config.get("variant") not in {
            "clip_vit_b16",
            "clip_vit_l14",
            "bge_vl_large",
        }:
            raise ValueError(
                "PinPoint comparison expects a supported CLIP-like checkpoint; "
                f"found {model_config.get('variant')!r}"
            )
        freeze_backbone = bool(model_config.get("freeze_backbone"))
        unfreeze_last_n_layers = model_config.get("unfreeze_last_n_layers")
        lora_rank = model_config.get("lora_rank")

        backbone_path = Path(str(model_config["pretrained_model_name_or_path"]))
        if not backbone_path.is_absolute():
            backbone_path = (TRAIN_PROJECT_ROOT / backbone_path).resolve()
        model, processor = ComposedRetrievalModel.from_pretrained(
            str(backbone_path),
            variant=str(model_config["variant"]),
            local_files_only=True,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            freeze_backbone=freeze_backbone,
            freeze_vision_encoder=bool(
                model_config.get("freeze_vision_encoder", False)
            ),
            gradient_checkpointing=False,
            temperature=float(
                checkpoint.get("config", {}).get("optimizer", {}).get(
                    "temperature", 0.07
                )
            ),
            learnable_temperature=bool(
                checkpoint.get("config", {}).get("optimizer", {}).get(
                    "learnable_temperature", True
                )
            ),
            unfreeze_last_n_layers=(
                None
                if unfreeze_last_n_layers is None
                else int(unfreeze_last_n_layers)
            ),
            lora_rank=None if lora_rank is None else int(lora_rank),
            lora_alpha=float(model_config.get("lora_alpha", 32.0)),
            lora_dropout=float(model_config.get("lora_dropout", 0.05)),
        )
        from cir_qdiag_train.model import load_checkpoint

        load_checkpoint(checkpoint_path, model=model)

        self.model_path = checkpoint_path
        self.backbone_path = backbone_path
        self.device = device
        self.target_dtype = _target_dtype(dtype, device)
        self.model = model.to(device=self.device, dtype=self.target_dtype).eval()
        self.processor = processor
        self.freeze_backbone = freeze_backbone
        self.freeze_vision_encoder = bool(
            model_config.get("freeze_vision_encoder", False)
        )
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.lora_rank = lora_rank
        self.variant = str(model_config["variant"])

    def load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def prepare_image_batch(self, images: Sequence[Image.Image]) -> dict[str, Any]:
        return dict(self.processor(images=list(images), return_tensors="pt"))

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device, non_blocking=self.device.type == "cuda")
                if value.is_floating_point():
                    value = value.to(self.target_dtype)
            moved[key] = value
        return moved

    def encode_prepared_image_batch(
        self,
        prepared: dict[str, Any],
        identifiers: Sequence[str] | None = None,
    ) -> np.ndarray:
        del identifiers
        inputs = self._to_device(prepared)
        with torch.inference_mode():
            embeddings = self.model.encode_image(inputs["pixel_values"])
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def prepare_query_batch(
        self,
        rows: pd.DataFrame,
        image_cache: Path,
        preloaded_images: Sequence[Image.Image],
    ) -> dict[str, Any]:
        del image_cache
        reference_counts = [_reference_count(row) for _, row in rows.iterrows()]
        if any(count == 0 for count in reference_counts):
            raise ValueError("Every composed query must have at least one reference image")
        if sum(reference_counts) != len(preloaded_images):
            raise ValueError(
                "Prepared reference-image count does not match the query frame: "
                f"{len(preloaded_images)} != {sum(reference_counts)}"
            )
        texts = [str(row.get("instruction") or "") for _, row in rows.iterrows()]
        return {
            "images": dict(
                self.processor(images=list(preloaded_images), return_tensors="pt")
            ),
            "texts": dict(
                self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            ),
            "reference_counts": reference_counts,
        }

    def encode_prepared_query_batch(
        self,
        prepared: dict[str, Any],
        image_cache: Path,
    ) -> np.ndarray:
        del image_cache
        image_inputs = self._to_device(prepared["images"])
        text_inputs = self._to_device(prepared["texts"])
        reference_counts = prepared["reference_counts"]
        with torch.inference_mode():
            flat_references = self.model.encode_reference(image_inputs["pixel_values"])
            pooled_references = []
            start = 0
            for count in reference_counts:
                end = start + int(count)
                pooled_references.append(flat_references[start:end].mean(dim=0))
                start = end
            reference_features = torch.stack(pooled_references, dim=0)
            if self.model.normalize_components_before_fusion:
                reference_features = F.normalize(reference_features, dim=-1)
            text_features = self.model.encode_text(
                text_inputs["input_ids"],
                text_inputs.get("attention_mask"),
            )
            embeddings = self.model.compose(reference_features, text_features)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.model_path),
            "backbone": str(self.backbone_path),
            "candidate": (
                f"normalized {self.variant} image embedding"
            ),
            "query": "normalize(mean(reference embeddings)) + text + learned residual",
            "multiple_references": "normalized mean before learned composition",
            "dtype": str(self.target_dtype).removeprefix("torch."),
            "backbone_finetune_mode": (
                f"lora_r{self.lora_rank}"
                if self.lora_rank is not None
                else (
                    "frozen"
                    if self.freeze_backbone
                    else (
                        f"last_{self.unfreeze_last_n_layers}"
                        if self.unfreeze_last_n_layers is not None
                        else (
                            "text_and_fusion"
                            if self.freeze_vision_encoder
                            else "full"
                        )
                    )
                )
            ),
            "vision_encoder_frozen": self.freeze_backbone
            or self.freeze_vision_encoder,
        }


class OAPResidualAdapter:
    """PinPoint adapter for frozen-global O/A/P residual or attention checkpoints."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        dtype: str,
        *,
        beta: float,
    ) -> None:
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if beta <= 0:
            raise ValueError("OAP inference beta must be positive")
        if str(TRAIN_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(TRAIN_SOURCE_ROOT))
        from cir_qdiag_train.oap_config import OAPConfig
        from cir_qdiag_train.oap_model import build_oap_model, load_oap_checkpoint

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config_payload = checkpoint.get("config")
        if not isinstance(config_payload, dict):
            raise TypeError(f"OAP checkpoint has no config: {checkpoint_path}")
        config = OAPConfig.model_validate(config_payload)
        config.model.beta = float(beta)
        model, processor = build_oap_model(config)
        load_oap_checkpoint(checkpoint_path, model=model)

        self.model_path = checkpoint_path
        self.backbone_path = Path(config.model.pretrained_model_name_or_path)
        self.base_checkpoint = Path(config.model.base_checkpoint)
        self.device = device
        self.target_dtype = _target_dtype(dtype, device)
        self.model = model.to(device=self.device, dtype=self.target_dtype).eval()
        self.processor = processor
        self.beta = float(beta)
        self.architecture = config.model.architecture

    def load_image(self, path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def prepare_image_batch(self, images: Sequence[Image.Image]) -> dict[str, Any]:
        return dict(self.processor(images=list(images), return_tensors="pt"))

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.device, non_blocking=self.device.type == "cuda")
                if value.is_floating_point():
                    value = value.to(self.target_dtype)
            moved[key] = value
        return moved

    def encode_prepared_image_batch(
        self,
        prepared: dict[str, Any],
        identifiers: Sequence[str] | None = None,
    ) -> np.ndarray:
        del identifiers
        inputs = self._to_device(prepared)
        with torch.inference_mode():
            global_target, semantic_target = self.model.encode_targets(
                inputs["pixel_values"]
            )
            embeddings = self.model.pack_target(global_target, semantic_target)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def prepare_query_batch(
        self,
        rows: pd.DataFrame,
        image_cache: Path,
        preloaded_images: Sequence[Image.Image],
    ) -> dict[str, Any]:
        del image_cache
        reference_counts = [_reference_count(row) for _, row in rows.iterrows()]
        if any(count == 0 for count in reference_counts):
            raise ValueError("Every composed query must have at least one reference image")
        if sum(reference_counts) != len(preloaded_images):
            raise ValueError(
                "Prepared reference-image count does not match the query frame: "
                f"{len(preloaded_images)} != {sum(reference_counts)}"
            )
        texts = [str(row.get("instruction") or "") for _, row in rows.iterrows()]
        return {
            "images": dict(
                self.processor(images=list(preloaded_images), return_tensors="pt")
            ),
            "texts": dict(
                self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
            ),
            "reference_counts": reference_counts,
        }

    def encode_prepared_query_batch(
        self,
        prepared: dict[str, Any],
        image_cache: Path,
    ) -> np.ndarray:
        del image_cache
        image_inputs = self._to_device(prepared["images"])
        text_inputs = self._to_device(prepared["texts"])
        reference_counts = prepared["reference_counts"]
        with torch.inference_mode():
            if self.architecture == "attention_tokens":
                flat_references, flat_reference_tokens = (
                    self.model.encode_reference_with_tokens(
                        image_inputs["pixel_values"]
                    )
                )
            else:
                flat_references = self.model.base.encode_reference(
                    image_inputs["pixel_values"]
                )
                flat_reference_tokens = None
            pooled_references = []
            pooled_reference_tokens = []
            start = 0
            for count in reference_counts:
                end = start + int(count)
                pooled_references.append(flat_references[start:end].mean(dim=0))
                if flat_reference_tokens is not None:
                    # ViT patch positions have a common layout. Averaging the token
                    # sequence mirrors the existing mean-pooled global handling for
                    # PinPoint queries with two reference images.
                    pooled_reference_tokens.append(
                        flat_reference_tokens[start:end].mean(dim=0)
                    )
                start = end
            reference_features = torch.stack(pooled_references, dim=0)
            if self.model.base.normalize_components_before_fusion:
                reference_features = F.normalize(reference_features, dim=-1)
            if self.architecture == "attention_tokens":
                text_features, text_tokens = self.model.encode_text_with_tokens(
                    text_inputs["input_ids"],
                    text_inputs.get("attention_mask"),
                )
                global_query, semantic_query = self.model.compose_queries_from_tokens(
                    reference_features,
                    torch.stack(pooled_reference_tokens, dim=0),
                    text_features,
                    text_tokens,
                    text_inputs.get("attention_mask"),
                )
            else:
                text_features = self.model.base.encode_text(
                    text_inputs["input_ids"],
                    text_inputs.get("attention_mask"),
                )
                global_query = self.model.base.compose(
                    reference_features, text_features
                )
                semantic_query = self.model.project_query_semantics(global_query)
            embeddings = self.model.pack_query(global_query, semantic_query)
        return embeddings.float().cpu().numpy().astype(np.float32, copy=False)

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.model_path),
            "base_checkpoint": str(self.base_checkpoint),
            "backbone": str(self.backbone_path),
            "architecture": self.architecture,
            "candidate": f"packed frozen global + O/A/P {self.architecture} target",
            "query": f"packed frozen global + O/A/P {self.architecture} query",
            "score": "global dot + beta/3 * sum(O/A/P dot)",
            "beta": self.beta,
            "embedding_dimension": self.model.packed_dim,
            "multiple_references": "mean before frozen learned composition",
            "dtype": str(self.target_dtype).removeprefix("torch."),
            "backbone_finetune_mode": "frozen trained global checkpoint",
        }


class OriginalCLIPAdapter(TrainedCLIPAdapter):
    """Standard CLIP-B/L image/text sum composition without CIRR training."""

    def __init__(
        self,
        backbone_path: Path,
        device: torch.device,
        dtype: str,
    ) -> None:
        backbone_path = Path(backbone_path).resolve()
        if not backbone_path.is_dir():
            raise FileNotFoundError(backbone_path)
        if str(TRAIN_SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(TRAIN_SOURCE_ROOT))
        from cir_qdiag_train.model import ComposedRetrievalModel

        backbone_config = json.loads(
            (backbone_path / "config.json").read_text(encoding="utf-8")
        )
        vision_hidden = int(backbone_config.get("vision_config", {}).get("hidden_size", 0))
        if vision_hidden == 768:
            variant = "clip_vit_b16"
        elif vision_hidden == 1024:
            variant = "clip_vit_l14"
        else:
            raise ValueError(
                f"unsupported original CLIP vision hidden size: {vision_hidden}"
            )
        model, processor = ComposedRetrievalModel.from_pretrained(
            str(backbone_path),
            variant=variant,
            local_files_only=True,
            hidden_dim=1024,
            dropout=0.1,
            freeze_backbone=True,
            gradient_checkpointing=False,
            temperature=0.07,
            learnable_temperature=True,
        )
        self.model_path = backbone_path
        self.backbone_path = backbone_path
        self.device = device
        self.target_dtype = _target_dtype(dtype, device)
        self.model = model.to(device=self.device, dtype=self.target_dtype).eval()
        self.processor = processor
        self.freeze_backbone = True
        self.freeze_vision_encoder = True
        self.unfreeze_last_n_layers = None
        self.lora_rank = None
        self.variant = variant

    def protocol_metadata(self) -> dict[str, Any]:
        metadata = super().protocol_metadata()
        metadata.update(
            {
                "checkpoint": None,
                "query": "normalize(CLIP image embedding + CLIP text embedding)",
                "backbone_finetune_mode": "pretrained",
            }
        )
        return metadata
