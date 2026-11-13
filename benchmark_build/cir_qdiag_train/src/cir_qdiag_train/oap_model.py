from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cir_qdiag_train.model import ComposedRetrievalModel, load_checkpoint
from cir_qdiag_train.oap_config import OAPConfig

NUM_SEMANTIC_COMPONENTS = 3


class SemanticAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(inputs), dim=-1)


class OAPSemanticModel(nn.Module):
    """Common packed-score interface for both O/A/P architectures."""

    architecture = "semantic_base"

    def __init__(
        self,
        base: ComposedRetrievalModel,
        *,
        semantic_dim: int,
        beta: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.beta = beta
        self.semantic_dim = semantic_dim
        self.global_dim = base.embedding_dim
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.base.eval()

    @property
    def packed_dim(self) -> int:
        return self.global_dim + NUM_SEMANTIC_COMPONENTS * self.semantic_dim

    def train(self, mode: bool = True) -> OAPSemanticModel:
        super().train(mode)
        # The immutable global checkpoint contains a dropout-bearing fusion head.
        # Keeping it in eval mode makes global geometry deterministic.
        self.base.eval()
        return self

    def aligned_score(
        self,
        query_global: torch.Tensor,
        query_semantic: torch.Tensor,
        target_global: torch.Tensor,
        target_semantic: torch.Tensor,
    ) -> torch.Tensor:
        global_score = (query_global * target_global).sum(dim=-1)
        semantic_score = (query_semantic * target_semantic).sum(dim=-1).mean(dim=-1)
        return global_score + self.beta * semantic_score

    def pairwise_score(
        self,
        query_global: torch.Tensor,
        query_semantic: torch.Tensor,
        target_global: torch.Tensor,
        target_semantic: torch.Tensor,
    ) -> torch.Tensor:
        global_score = query_global @ target_global.t()
        semantic_score = torch.einsum(
            "nkd,mkd->nmk", query_semantic, target_semantic
        ).mean(dim=-1)
        return global_score + self.beta * semantic_score

    def pack_query(
        self, global_query: torch.Tensor, semantic_query: torch.Tensor
    ) -> torch.Tensor:
        scale = math.sqrt(self.beta / NUM_SEMANTIC_COMPONENTS)
        return torch.cat([global_query, (scale * semantic_query).flatten(1)], dim=-1)

    def pack_target(
        self, global_target: torch.Tensor, semantic_target: torch.Tensor
    ) -> torch.Tensor:
        scale = math.sqrt(self.beta / NUM_SEMANTIC_COMPONENTS)
        return torch.cat([global_target, (scale * semantic_target).flatten(1)], dim=-1)

    def semantic_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("base.")
        }

    def load_semantic_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.startswith("base.")]
        if missing or unexpected:
            raise RuntimeError(
                f"OAP checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )

    def parameter_summary(self) -> dict[str, int | float | str]:
        return {
            "architecture": self.architecture,
            "global_dim": self.global_dim,
            "semantic_dim": self.semantic_dim,
            "packed_dim": self.packed_dim,
            "beta": self.beta,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


class OAPResidualModel(OAPSemanticModel):
    """Frozen global CIR model plus Object/Attribute/Predicate residual matchers."""

    architecture = "residual_adapter"

    def __init__(
        self,
        base: ComposedRetrievalModel,
        *,
        semantic_hidden_dim: int,
        semantic_dim: int,
        beta: float,
    ) -> None:
        super().__init__(base, semantic_dim=semantic_dim, beta=beta)

        query_adapters = [
            SemanticAdapter(self.global_dim, semantic_hidden_dim, semantic_dim)
            for _ in range(NUM_SEMANTIC_COMPONENTS)
        ]
        # Start each target adapter from its query counterpart. Parameters are
        # independent after initialization, but the initial residual is aligned
        # instead of adding unrelated random query/target geometry.
        target_adapters = [copy.deepcopy(adapter) for adapter in query_adapters]
        self.query_adapters = nn.ModuleList(query_adapters)
        self.target_adapters = nn.ModuleList(target_adapters)

    def project_query_semantics(self, global_query: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [adapter(global_query) for adapter in self.query_adapters], dim=1
        )

    def project_target_semantics(self, global_target: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [adapter(global_target) for adapter in self.target_adapters], dim=1
        )

    def encode_queries(
        self,
        reference_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            reference = self.base.encode_reference(reference_pixel_values)
            text = self.base.encode_text(input_ids, attention_mask)
            global_query = self.base.compose(reference, text)
        return global_query, self.project_query_semantics(global_query)

    def encode_targets(
        self, target_pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            global_target = self.base.encode_image(target_pixel_values)
        return global_target, self.project_target_semantics(global_target)

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
    ) -> dict[str, torch.Tensor | None]:
        with torch.no_grad():
            reference = self.base.encode_reference(reference_pixel_values)
            positive_text = self.base.encode_text(
                positive_input_ids, positive_attention_mask
            )
            positive_global = self.base.compose(reference, positive_text)
            target_global = self.base.encode_image(target_pixel_values)
        positive_semantic = self.project_query_semantics(positive_global)
        target_semantic = self.project_target_semantics(target_global)
        negative_global = None
        negative_semantic = None
        if negative_input_ids is not None and negative_input_ids.shape[0] > 0:
            with torch.no_grad():
                negative_text = self.base.encode_text(
                    negative_input_ids, negative_attention_mask
                )
                negative_global = self.base.compose(
                    reference[negative_owner], negative_text
                )
            negative_semantic = self.project_query_semantics(negative_global)
        return {
            "positive_global": positive_global,
            "positive_semantic": positive_semantic,
            "target_global": target_global,
            "target_semantic": target_semantic,
            "negative_global": negative_global,
            "negative_semantic": negative_semantic,
        }


class SemanticCrossAttentionPooler(nn.Module):
    """Three learned O/A/P queries over a pre-projected token context."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        semantic_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.semantic_queries = nn.Parameter(
            torch.empty(NUM_SEMANTIC_COMPONENTS, hidden_dim)
        )
        nn.init.normal_(self.semantic_queries, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.semantic_projection = nn.Sequential(
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
        )

    def forward(
        self,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        queries = self.semantic_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=self.context_norm(context),
            value=self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        states = self.attention_norm(queries + self.attention_dropout(attended))
        states = self.ffn_norm(states + self.ffn_dropout(self.ffn(states)))
        return F.normalize(self.semantic_projection(states), dim=-1)


class OAPAttentionModel(OAPSemanticModel):
    """Frozen global CIR model with token-level O/A/P semantic attention."""

    architecture = "attention_tokens"

    def __init__(
        self,
        base: ComposedRetrievalModel,
        *,
        attention_hidden_dim: int,
        attention_num_heads: int,
        attention_ffn_dim: int,
        attention_dropout: float,
        semantic_dim: int,
        beta: float,
    ) -> None:
        super().__init__(base, semantic_dim=semantic_dim, beta=beta)
        backbone_config = getattr(base.backbone, "config", None)
        vision_config = getattr(backbone_config, "vision_config", None)
        text_config = getattr(backbone_config, "text_config", None)
        vision_hidden_dim = getattr(vision_config, "hidden_size", None)
        text_hidden_dim = getattr(text_config, "hidden_size", None)
        if not isinstance(vision_hidden_dim, int) or not isinstance(
            text_hidden_dim, int
        ):
            raise TypeError(
                "attention-token OAP requires CLIP vision/text hidden-size configs"
            )
        if not all(
            hasattr(base.backbone, name)
            for name in (
                "vision_model",
                "text_model",
                "visual_projection",
                "text_projection",
            )
        ):
            raise TypeError(
                "attention-token OAP requires CLIP vision/text towers and projections"
            )

        self.attention_hidden_dim = attention_hidden_dim
        self.attention_num_heads = attention_num_heads
        self.attention_ffn_dim = attention_ffn_dim
        self.query_vision_projection = nn.Sequential(
            nn.Linear(vision_hidden_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
        )
        self.query_text_projection = nn.Sequential(
            nn.Linear(text_hidden_dim, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
        )
        self.target_vision_projection = copy.deepcopy(
            self.query_vision_projection
        )
        self.query_attention = SemanticCrossAttentionPooler(
            hidden_dim=attention_hidden_dim,
            num_heads=attention_num_heads,
            ffn_dim=attention_ffn_dim,
            semantic_dim=semantic_dim,
            dropout=attention_dropout,
        )
        # Query and target parameters are independent, but aligned at
        # initialization so their residual score does not start from two unrelated
        # random geometries.
        self.target_attention = copy.deepcopy(self.query_attention)

    def encode_reference_with_tokens(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_features, tokens = self._encode_vision_tower(pixel_values)
        reference_features = raw_features
        if self.base.normalize_components_before_fusion:
            reference_features = F.normalize(reference_features, dim=-1)
        return reference_features, tokens

    def encode_text_with_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.base.backbone.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden, pooled = _hidden_and_pooled(outputs)
            text_features = self.base.backbone.text_projection(pooled)
            if self.base.normalize_components_before_fusion:
                text_features = F.normalize(text_features, dim=-1)
        return text_features, hidden

    def compose_queries_from_tokens(
        self,
        reference_features: torch.Tensor,
        reference_tokens: torch.Tensor,
        text_features: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            global_query = self.base.compose(reference_features, text_features)
        semantic_query = self._encode_query_semantics(
            reference_tokens,
            text_tokens,
            attention_mask,
        )
        return global_query, semantic_query

    def encode_queries(
        self,
        reference_pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference, reference_tokens = self.encode_reference_with_tokens(
            reference_pixel_values
        )
        text, text_tokens = self.encode_text_with_tokens(input_ids, attention_mask)
        return self.compose_queries_from_tokens(
            reference,
            reference_tokens,
            text,
            text_tokens,
            attention_mask,
        )

    def encode_targets(
        self, target_pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_target, target_tokens = self._encode_vision_tower(target_pixel_values)
        global_target = F.normalize(raw_target, dim=-1)
        semantic_target = self.target_attention(
            self.target_vision_projection(_patch_tokens(target_tokens))
        )
        return global_target, semantic_target

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
    ) -> dict[str, torch.Tensor | None]:
        reference, reference_tokens = self.encode_reference_with_tokens(
            reference_pixel_values
        )
        positive_text, positive_text_tokens = self.encode_text_with_tokens(
            positive_input_ids, positive_attention_mask
        )
        with torch.no_grad():
            positive_global = self.base.compose(reference, positive_text)
        reference_context = self.query_vision_projection(
            _patch_tokens(reference_tokens)
        )
        positive_semantic = self._encode_query_semantics_from_vision_context(
            reference_context,
            positive_text_tokens,
            positive_attention_mask,
        )
        target_global, target_semantic = self.encode_targets(target_pixel_values)

        negative_global = None
        negative_semantic = None
        if negative_input_ids is not None and negative_input_ids.shape[0] > 0:
            negative_text, negative_text_tokens = self.encode_text_with_tokens(
                negative_input_ids, negative_attention_mask
            )
            with torch.no_grad():
                negative_global = self.base.compose(
                    reference[negative_owner], negative_text
                )
            negative_semantic = self._encode_query_semantics_from_vision_context(
                reference_context[negative_owner],
                negative_text_tokens,
                negative_attention_mask,
            )
        return {
            "positive_global": positive_global,
            "positive_semantic": positive_semantic,
            "target_global": target_global,
            "target_semantic": target_semantic,
            "negative_global": negative_global,
            "negative_semantic": negative_semantic,
        }

    def parameter_summary(self) -> dict[str, int | float | str]:
        summary = super().parameter_summary()
        summary.update(
            {
                "attention_hidden_dim": self.attention_hidden_dim,
                "attention_num_heads": self.attention_num_heads,
                "attention_ffn_dim": self.attention_ffn_dim,
            }
        )
        return summary

    def _encode_vision_tower(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.base.backbone.vision_model(
                pixel_values=pixel_values,
            )
            hidden, pooled = _hidden_and_pooled(outputs)
            image_features = self.base.backbone.visual_projection(pooled)
        return image_features, hidden

    def _encode_query_semantics(
        self,
        reference_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        vision_context = self.query_vision_projection(
            _patch_tokens(reference_tokens)
        )
        return self._encode_query_semantics_from_vision_context(
            vision_context, text_tokens, attention_mask
        )

    def _encode_query_semantics_from_vision_context(
        self,
        vision_context: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        text_context = self.query_text_projection(text_tokens)
        context = torch.cat([vision_context, text_context], dim=1)
        key_padding_mask = None
        if attention_mask is not None:
            vision_mask = torch.zeros(
                vision_context.shape[:2],
                dtype=torch.bool,
                device=vision_context.device,
            )
            text_mask = ~attention_mask.to(dtype=torch.bool)
            key_padding_mask = torch.cat([vision_mask, text_mask], dim=1)
        return self.query_attention(context, key_padding_mask)


def _hidden_and_pooled(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = getattr(outputs, "last_hidden_state", None)
    pooled = getattr(outputs, "pooler_output", None)
    if not isinstance(hidden, torch.Tensor) or not isinstance(pooled, torch.Tensor):
        try:
            hidden, pooled = outputs[0], outputs[1]
        except (IndexError, TypeError) as error:
            raise TypeError("CLIP tower returned no hidden/pooler tensors") from error
    if not isinstance(hidden, torch.Tensor) or not isinstance(pooled, torch.Tensor):
        raise TypeError("CLIP tower returned unsupported hidden/pooler values")
    return hidden, pooled


def _patch_tokens(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError(
            f"vision hidden states must contain CLS + patch tokens, got {hidden.shape}"
        )
    return hidden[:, 1:, :]


def build_oap_model(config: OAPConfig) -> tuple[OAPSemanticModel, Any]:
    base, processor = ComposedRetrievalModel.from_pretrained(
        config.model.pretrained_model_name_or_path,
        variant=config.model.variant,
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        hidden_dim=config.model.base_hidden_dim,
        dropout=config.model.base_dropout,
        freeze_backbone=True,
        gradient_checkpointing=False,
        temperature=config.model.base_temperature,
        learnable_temperature=config.model.base_learnable_temperature,
    )
    load_checkpoint(config.model.base_checkpoint, model=base)
    if config.model.architecture == "residual_adapter":
        model: OAPSemanticModel = OAPResidualModel(
            base,
            semantic_hidden_dim=config.model.semantic_hidden_dim,
            semantic_dim=config.model.semantic_dim,
            beta=config.model.beta,
        )
    elif config.model.architecture == "attention_tokens":
        model = OAPAttentionModel(
            base,
            attention_hidden_dim=config.model.attention_hidden_dim,
            attention_num_heads=config.model.attention_num_heads,
            attention_ffn_dim=config.model.attention_ffn_dim,
            attention_dropout=config.model.attention_dropout,
            semantic_dim=config.model.semantic_dim,
            beta=config.model.beta,
        )
    else:  # pragma: no cover - rejected by the strict config model
        raise ValueError(f"unsupported OAP architecture: {config.model.architecture}")
    return model, processor


def save_oap_checkpoint(
    path: str | Path,
    *,
    model: OAPSemanticModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "semantic_model": model.semantic_state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_oap_checkpoint(
    path: str | Path,
    *,
    model: OAPSemanticModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_semantic_state_dict(checkpoint["semantic_model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError(f"{path} is model-only and cannot resume training")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise ValueError(f"{path} is model-only and cannot resume training")
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
