from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class ComposedRetrievalModel(nn.Module):
    """Compose a reference-image embedding and modification-text embedding."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
        temperature: float,
        learnable_temperature: bool,
        freeze_vision_encoder: bool = False,
        unfreeze_last_n_layers: int | None = None,
        lora_rank: int | None = None,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        normalize_components_before_fusion: bool = True,
        composition_head: str = 'residual_mlp',
        fusion_init_path: str | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "get_image_features") or not hasattr(
            backbone, "get_text_features"
        ):
            raise TypeError(
                "backbone must implement get_image_features() and get_text_features(); "
                "use a CLIP-compatible Transformers checkpoint"
            )
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        self.freeze_vision_encoder = freeze_vision_encoder
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.normalize_components_before_fusion = normalize_components_before_fusion
        self.composition_head = composition_head
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Standard CLIP has no pretrained composed-query head. We use the common
        # normalize(image + text) zero-shot baseline, then learn a residual on top.
        # Zero initialization makes every experiment start from exactly the same
        # ordinary CLIP composition instead of a random fusion function.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        if composition_head != 'residual_mlp':
            from cir_qdiag_train.composition_heads import make_head
            self.fusion = make_head(composition_head, embedding_dim, fusion_init_path)
            self.normalize_components_before_fusion = False
        self.register_buffer('empty_input_ids', None, persistent=False)
        self.register_buffer('empty_attention_mask', None, persistent=False)
        logit_scale = torch.tensor(math.log(1.0 / temperature), dtype=torch.float32)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)
        _configure_backbone_trainability(
            self.backbone,
            freeze_backbone=freeze_backbone,
            freeze_vision_encoder=freeze_vision_encoder,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.backbone_trainable = any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        self.image_encoder_trainable = any(
            parameter.requires_grad
            for module in _image_encoder_modules(self.backbone)
            for parameter in module.parameters()
        )
        self.text_encoder_trainable = any(
            parameter.requires_grad
            for module in _text_encoder_modules(self.backbone)
            for parameter in module.parameters()
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        variant: str,
        local_files_only: bool,
        trust_remote_code: bool = False,
        hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
        gradient_checkpointing: bool,
        temperature: float,
        learnable_temperature: bool,
        freeze_vision_encoder: bool = False,
        unfreeze_last_n_layers: int | None = None,
        lora_rank: int | None = None,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        composition_head: str = 'residual_mlp',
        fusion_init_path: str | None = None,
    ) -> tuple["ComposedRetrievalModel", Any]:
        if trust_remote_code:
            _prepare_hf_runtime_cache()
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        _validate_clip_backbone(backbone, variant)
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            _enable_gradient_checkpointing(backbone)
        embedding_dim = getattr(backbone.config, "projection_dim", None)
        if not isinstance(embedding_dim, int):
            raise ValueError(
                "could not infer projection_dim from the CLIP-compatible backbone config"
            )
        model = cls(
            backbone,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            freeze_vision_encoder=freeze_vision_encoder,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            normalize_components_before_fusion=variant != "bge_vl_large",
            composition_head=composition_head,
            fusion_init_path=fusion_init_path,
        )
        if composition_head == 'magiclens':
            empty = processor(text=[''], padding='max_length', max_length=77,
                              truncation=True, return_tensors='pt')
            model.empty_input_ids = empty['input_ids']
            model.empty_attention_mask = empty.get('attention_mask')
        return model, processor

    def train(self, mode: bool = True) -> "ComposedRetrievalModel":
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()
        elif not self.image_encoder_trainable:
            for module in _image_encoder_modules(self.backbone):
                module.eval()
        return self

    def _encode_image_raw(self, pixel_values: torch.Tensor) -> torch.Tensor:
        context = (
            torch.no_grad() if not self.image_encoder_trainable else _null_context()
        )
        with context:
            features = self.backbone.get_image_features(pixel_values=pixel_values)
        return _as_tensor(features)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.target_from_raw(self._encode_image_raw(pixel_values))

    def target_from_raw(self, features: torch.Tensor) -> torch.Tensor:
        if self.composition_head == 'magiclens':
            if self.empty_input_ids is None:
                raise RuntimeError('MagicLens target requires tokenized empty text')
            empty = self.encode_text(self.empty_input_ids, self.empty_attention_mask)
            return self.compose(features, empty.expand(features.shape[0], -1))
        return F.normalize(features, dim=-1)

    def reference_from_raw(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, dim=-1) if self.normalize_components_before_fusion else features

    def encode_reference(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self._encode_image_raw(pixel_values)
        if self.normalize_components_before_fusion:
            features = F.normalize(features, dim=-1)
        return features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        context = (
            torch.no_grad() if not self.text_encoder_trainable else _null_context()
        )
        with context:
            features = self.backbone.get_text_features(**kwargs)
        features = _as_tensor(features)
        if self.normalize_components_before_fusion:
            features = F.normalize(features, dim=-1)
        return features

    def compose(
        self,
        reference_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.composition_head == 'clip4cir_combiner':
            return self.fusion.combine_features(reference_features, text_features)
        if self.composition_head != 'residual_mlp':
            return self.fusion(reference_features, text_features)
        interaction = reference_features * text_features
        residual = self.fusion(
            torch.cat([reference_features, text_features, interaction], dim=-1)
        )
        composed = reference_features + text_features + residual
        return F.normalize(composed, dim=-1)

    def forward(
        self,
        *,
        reference_pixel_values: torch.Tensor,
        target_pixel_values: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor | None,
        negative_input_ids: torch.Tensor | None,
        negative_attention_mask: torch.Tensor | None,
        negative_owner: torch.Tensor,
        candidate_pixel_values: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        reference_features = self.encode_reference(reference_pixel_values)
        target_features = self.encode_image(target_pixel_values)
        positive_text_features = self.encode_text(
            positive_input_ids,
            positive_attention_mask,
        )
        positive_queries = self.compose(reference_features, positive_text_features)
        negative_queries = None
        if negative_input_ids is not None and negative_input_ids.shape[0] > 0:
            negative_text_features = self.encode_text(
                negative_input_ids,
                negative_attention_mask,
            )
            negative_queries = self.compose(
                reference_features[negative_owner],
                negative_text_features,
            )
        candidate_features = None
        if candidate_pixel_values is not None and candidate_pixel_values.shape[0] > 0:
            candidate_features = self.encode_image(candidate_pixel_values)
        return {
            "positive_queries": positive_queries,
            "target_features": target_features,
            "negative_queries": negative_queries,
            "candidate_features": candidate_features,
            "scale": self.scale(),
        }

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("backbone.") or key in trainable_names
        }

    def load_trainable_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected = list(unexpected)
        expected = set(self.trainable_state_dict())
        missing = [key for key in missing if key in expected]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )

    def parameter_summary(self) -> dict[str, int | str | None]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        backbone_total = sum(
            parameter.numel() for parameter in self.backbone.parameters()
        )
        backbone_trainable = sum(
            parameter.numel()
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )
        mode = (
            f"lora_r{self.lora_rank}"
            if self.lora_rank is not None
            else (
                "frozen"
                if not self.backbone_trainable
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
        )
        return {
            "backbone_finetune_mode": mode,
            "composition_head": self.composition_head,
            "fusion_parameter_count": sum(p.numel() for p in self.fusion.parameters()),
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha if self.lora_rank is not None else None,
            "lora_dropout": self.lora_dropout if self.lora_rank is not None else None,
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "backbone_parameter_count": backbone_total,
            "trainable_backbone_parameter_count": backbone_trainable,
        }


def _prepare_hf_runtime_cache() -> None:
    """Keep local custom-model modules in the writable project tree."""

    cache_root = Path(__file__).resolve().parents[2] / ".hf_runtime"
    paths = {
        "HF_HOME": cache_root / "hf_home",
        "HF_MODULES_CACHE": cache_root / "modules",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "HF_HUB_CACHE": cache_root / "hub",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def _configure_backbone_trainability(
    backbone: nn.Module,
    *,
    freeze_backbone: bool,
    freeze_vision_encoder: bool,
    unfreeze_last_n_layers: int | None,
    lora_rank: int | None,
    lora_alpha: float,
    lora_dropout: float,
) -> None:
    if lora_rank is not None:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        _inject_attention_lora(
            backbone,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        return
    if freeze_backbone:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        return
    if unfreeze_last_n_layers is None:
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        # The composed-retrieval model uses its own temperature and never calls
        # CLIPModel.forward(), so the pretrained CLIP logit parameters are unused.
        for name, parameter in backbone.named_parameters():
            if name in {"logit_scale", "logit_bias"}:
                parameter.requires_grad = False
        if freeze_vision_encoder:
            for module in _image_encoder_modules(backbone):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        return

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    vision_model = getattr(backbone, "vision_model", None)
    text_model = getattr(backbone, "text_model", None)
    vision_encoder = getattr(vision_model, "encoder", None)
    text_encoder = getattr(text_model, "encoder", None)
    vision_layers = getattr(vision_encoder, "layers", None)
    text_layers = getattr(text_encoder, "layers", None)
    if vision_layers is None or text_layers is None:
        raise TypeError(
            "last-N finetuning requires a standard CLIP backbone with vision/text "
            "encoder.layers"
        )
    if unfreeze_last_n_layers > min(len(vision_layers), len(text_layers)):
        raise ValueError(
            f"cannot unfreeze last {unfreeze_last_n_layers} layers; "
            f"vision_layers={len(vision_layers)}, text_layers={len(text_layers)}"
        )

    modules: list[nn.Module] = [
        *vision_layers[-unfreeze_last_n_layers:],
        *text_layers[-unfreeze_last_n_layers:],
    ]
    for optional in (
        getattr(vision_model, "post_layernorm", None),
        getattr(text_model, "final_layer_norm", None),
        getattr(backbone, "visual_projection", None),
        getattr(backbone, "text_projection", None),
    ):
        if isinstance(optional, nn.Module):
            modules.append(optional)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _image_encoder_modules(backbone: nn.Module) -> list[nn.Module]:
    modules = []
    for name in ("vision_model", "visual_projection"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    return modules


def _text_encoder_modules(backbone: nn.Module) -> list[nn.Module]:
    modules = []
    for name in ("text_model", "text_projection"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    return modules


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper that keeps the pretrained linear layer frozen."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base_output + update * self.scaling


def _inject_attention_lora(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    targets = []
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(("vision_model.", "text_model.")):
            continue
        if name.rsplit(".", 1)[-1] not in {"q_proj", "v_proj"}:
            continue
        targets.append((name, module))
    if not targets:
        raise TypeError("no CLIP vision/text q_proj or v_proj modules found for LoRA")
    for name, module in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = backbone.get_submodule(parent_name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )


def _enable_gradient_checkpointing(backbone: nn.Module) -> None:
    """Use non-reentrant checkpointing so frozen inputs do not drop tail gradients."""

    try:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        backbone.gradient_checkpointing_enable()
        for tower_name in ("vision_model", "text_model"):
            tower = getattr(backbone, tower_name, None)
            embeddings = getattr(tower, "embeddings", None)
            if embeddings is not None:
                embeddings.register_forward_hook(_require_output_grad)


def _require_output_grad(
    _module: nn.Module,
    _inputs: tuple[object, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    return output.requires_grad_(True)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *_: object) -> None:
        return None


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    pooler = getattr(value, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        return pooler
    raise TypeError(f"backbone feature method returned unsupported type: {type(value)}")


def _validate_clip_backbone(backbone: nn.Module, variant: str) -> None:
    expected = {
        "clip_vit_b16": {
            "projection_dim": 512,
            "vision_hidden_size": 768,
            "vision_layers": 12,
            "patch_size": 16,
            "requires_custom_code": False,
        },
        "clip_vit_l14": {
            "projection_dim": 768,
            "vision_hidden_size": 1024,
            "vision_layers": 24,
            "patch_size": 14,
            "requires_custom_code": False,
        },
        "bge_vl_large": {
            "projection_dim": 768,
            "vision_hidden_size": 1024,
            "vision_layers": 24,
            "patch_size": 14,
            "requires_custom_code": True,
        },
    }
    if variant not in expected:
        raise ValueError(f"unsupported CLIP-like variant: {variant}")
    config = getattr(backbone, "config", None)
    if config is None or getattr(config, "model_type", None) != "clip":
        raise ValueError("checkpoint is not a standard Transformers CLIP model")
    specification = expected[variant]
    has_custom_code = bool(getattr(config, "auto_map", None))
    requires_custom_code = bool(specification.get("requires_custom_code", False))
    if has_custom_code and not requires_custom_code:
        raise ValueError(
            "checkpoint declares custom model code; expected ordinary OpenAI CLIP, not BGE"
        )
    if requires_custom_code and not has_custom_code:
        raise ValueError("BGE-VL-large must declare its local custom AutoModel code")
    vision = getattr(config, "vision_config", None)
    actual = {
        "projection_dim": getattr(config, "projection_dim", None),
        "vision_hidden_size": getattr(vision, "hidden_size", None),
        "vision_layers": getattr(vision, "num_hidden_layers", None),
        "patch_size": getattr(vision, "patch_size", None),
    }
    expected_architecture = {
        key: value
        for key, value in specification.items()
        if key != "requires_custom_code"
    }
    if actual != expected_architecture:
        raise ValueError(
            f"checkpoint architecture does not match {variant}; "
            f"expected={expected_architecture}, actual={actual}"
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: ComposedRetrievalModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    base_checkpoint: str | Path | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model": model.trainable_state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "metrics": metrics or {},
    }
    if base_checkpoint is not None:
        payload["base_checkpoint"] = str(Path(base_checkpoint).resolve())
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    *,
    model: ComposedRetrievalModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base_checkpoint = checkpoint.get("base_checkpoint")
    if base_checkpoint is not None:
        base_path = Path(str(base_checkpoint))
        if not base_path.is_absolute():
            base_path = (Path(path).resolve().parent / base_path).resolve()
        if base_path.resolve() == Path(path).resolve():
            raise ValueError(f"checkpoint cannot inherit from itself: {path}")
        load_checkpoint(base_path, model=model)
    model.load_trainable_state_dict(checkpoint["model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError(
                f"{path} is model-only and cannot resume training; use last.pt"
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise ValueError(
                f"{path} is model-only and cannot resume training; use last.pt"
            )
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
