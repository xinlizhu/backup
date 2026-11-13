from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from cir_qdiag_train.config import EvalConfig, ModelConfig, StrictConfig


class OnPolicyExperimentConfig(StrictConfig):
    name: str = Field(min_length=1)
    seed: int = 42
    output_dir: Path
    initial_checkpoint: Path | None = None


class OnPolicyDataConfig(StrictConfig):
    train_annotation_path: Path
    train_image_split_path: Path
    val_annotation_path: Path
    val_image_split_path: Path
    image_root: Path
    teacher_cache_dir: Path | None = None
    negative_query_manifest: Path | None = None
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)


class OnPolicyTeacherConfig(StrictConfig):
    backend: Literal["embedding_cache", "http_reranker"] = "embedding_cache"
    endpoint_url: str | None = None
    model_name: str | None = None
    api_format: Literal["rerank", "score"] = "rerank"
    instruction: str | None = None
    image_transport: Literal["data_url", "file_url"] = "data_url"
    max_concurrent_requests: int = Field(default=8, ge=1)
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0)

    @model_validator(mode="after")
    def validate_backend(self) -> "OnPolicyTeacherConfig":
        if self.backend == "http_reranker":
            if not self.endpoint_url or not self.endpoint_url.startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    "http_reranker requires an http(s) teacher.endpoint_url"
                )
            if not self.model_name:
                raise ValueError("http_reranker requires teacher.model_name")
        return self


class OnPolicyObjectiveConfig(StrictConfig):
    loss_type: Literal[
        "rank_kd",
        "candidate_infonce",
        "centered_mse",
        "listwise_group_relative",
        "causal_reinforce",
        "causal_grpo",
        "causal_gspo",
        "reward_tilted_reverse_opd",
        "reward_tilted_reverse_gspo",
    ] = "rank_kd"
    rank_weight: float = Field(default=0.0, ge=0.0)
    centered_response_weight: float = Field(default=0.0, ge=0.0)
    rank_temperature: float = Field(default=0.05, gt=0.0)
    rank_kl_direction: Literal["forward", "reverse"] = "forward"
    centered_response_beta: float = Field(default=0.01, gt=0.0)
    rollout_count: int = Field(default=16, ge=2)
    rollout_top_k: int = Field(default=10, ge=1)
    policy_update_epochs: int = Field(default=1, ge=1)
    policy_clip_epsilon: float | None = Field(default=None, gt=0.0)
    deterministic_policy_dropout: bool = True
    reward_tilt_beta: float = Field(default=0.1, gt=0.0)
    anchor_kl_enabled: bool = False
    anchor_kl_target: float = Field(default=0.02, gt=0.0)
    anchor_kl_beta_initial: float = Field(default=0.1, gt=0.0)
    anchor_kl_dual_lr: float = Field(default=0.05, gt=0.0)
    anchor_kl_ema_decay: float = Field(default=0.95, ge=0.0, lt=1.0)
    anchor_kl_update_interval: int = Field(default=20, ge=1)
    anchor_kl_beta_min: float = Field(default=0.0001, gt=0.0)
    anchor_kl_beta_max: float = Field(default=10.0, gt=0.0)
    require_teacher_target_top1: bool = True
    teacher_target_max_rank: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_objective(self) -> "OnPolicyObjectiveConfig":
        if (
            self.loss_type == "rank_kd"
            and self.rank_weight == 0
            and self.centered_response_weight == 0
        ):
            raise ValueError("at least one on-policy KD loss weight must be positive")
        if self.loss_type in {
            "candidate_infonce",
            "centered_mse",
            "listwise_group_relative",
            "causal_reinforce",
            "causal_grpo",
            "causal_gspo",
            "reward_tilted_reverse_opd",
            "reward_tilted_reverse_gspo",
        }:
            if self.rank_weight != 0 or self.centered_response_weight != 0:
                raise ValueError(
                    f"{self.loss_type} must disable Teacher KD loss weights"
                )
        if self.loss_type in {
            "listwise_group_relative",
            "causal_reinforce",
            "causal_grpo",
            "causal_gspo",
            "reward_tilted_reverse_gspo",
        } and self.uses_teacher_filter:
            raise ValueError(
                "group-relative policy loss uses gold-target reward, "
                "not Teacher filtering"
            )
        if self.loss_type == "causal_reinforce":
            if self.policy_update_epochs != 1:
                raise ValueError("causal REINFORCE requires one policy update")
            if self.policy_clip_epsilon is not None:
                raise ValueError("causal REINFORCE does not use policy clipping")
        if self.loss_type == "causal_grpo":
            if self.policy_update_epochs < 2:
                raise ValueError("causal GRPO requires at least two policy updates")
            if self.policy_clip_epsilon is None:
                raise ValueError("causal GRPO requires policy_clip_epsilon")
        if self.loss_type == "causal_gspo" and self.policy_update_epochs < 2:
            raise ValueError("causal GSPO requires at least two policy updates")
        if self.loss_type == "reward_tilted_reverse_gspo":
            if self.policy_update_epochs < 2:
                raise ValueError(
                    "reward-tilted reverse-KL GSPO requires at least two policy updates"
                )
            if self.policy_clip_epsilon is None:
                raise ValueError(
                    "reward-tilted reverse-KL GSPO requires policy_clip_epsilon"
                )
        if self.loss_type == "reward_tilted_reverse_opd":
            if self.policy_update_epochs != 1:
                raise ValueError(
                    "reward-tilted Reverse-OPD requires one direct KL update"
                )
            if self.policy_clip_epsilon is not None:
                raise ValueError(
                    "reward-tilted Reverse-OPD does not use policy clipping"
                )
        if self.anchor_kl_enabled and not self.is_causal_policy_loss:
            raise ValueError("anchor-policy KL requires a causal policy objective")
        if self.anchor_kl_beta_min > self.anchor_kl_beta_initial:
            raise ValueError("anchor KL beta minimum exceeds its initial value")
        if self.anchor_kl_beta_initial > self.anchor_kl_beta_max:
            raise ValueError("anchor KL beta initial value exceeds its maximum")
        if (
            self.require_teacher_target_top1
            and self.teacher_target_max_rank not in (None, 1)
        ):
            raise ValueError(
                "require_teacher_target_top1=true conflicts with "
                "teacher_target_max_rank > 1"
            )
        return self

    @property
    def uses_teacher_filter(self) -> bool:
        return self.require_teacher_target_top1 or self.teacher_target_max_rank is not None

    @property
    def is_policy_loss(self) -> bool:
        return self.loss_type in {
            "listwise_group_relative",
            "causal_reinforce",
            "causal_grpo",
            "causal_gspo",
            "reward_tilted_reverse_opd",
            "reward_tilted_reverse_gspo",
        }

    @property
    def is_causal_policy_loss(self) -> bool:
        return self.loss_type in {
            "causal_reinforce",
            "causal_grpo",
            "causal_gspo",
            "reward_tilted_reverse_opd",
            "reward_tilted_reverse_gspo",
        }

    @property
    def needs_teacher_scores(self) -> bool:
        return (
            self.loss_type in {
                "rank_kd",
                "centered_mse",
                "reward_tilted_reverse_opd",
                "reward_tilted_reverse_gspo",
            }
            or self.uses_teacher_filter
        )


class OnPolicyMiningConfig(StrictConfig):
    force_target_insertion: bool = True
    top_k: int = Field(default=32, ge=2)
    reservoir_top_k: int | None = Field(default=None, ge=2)
    live_reservoir_selection: bool = False
    live_selection_image_batch_size: int = Field(default=1024, ge=1)
    pixel_cache_enabled: bool = False
    pixel_cache_dtype: Literal["float16", "float32"] = "float16"
    pixel_cache_build_batch_size: int = Field(default=128, ge=1)
    pixel_cache_workers: int = Field(default=8, ge=0)
    pixel_cache_build_prefetch_factor: int = Field(default=2, ge=1)
    pixel_cache_transfer_prefetch_batches: int = Field(default=2, ge=1)
    pixel_cache_pin_memory: bool = True
    persist_candidate_cache: bool = True
    candidate_source: Literal["student", "teacher", "random"] = "student"
    refresh_mode: Literal["every_epoch", "once"] = "every_epoch"
    refresh_every_epochs: int = Field(default=1, ge=1)
    image_batch_size: int = Field(default=512, ge=1)
    query_batch_size: int = Field(default=1024, ge=1)
    teacher_score_batch_size: int = Field(default=512, ge=1)
    cache_dtype: Literal["float16", "float32"] = "float16"
    include_negative_queries_as_candidates: bool = False
    max_negative_query_candidates_per_sample: int | None = Field(
        default=None,
        ge=1,
    )


class OnPolicyOptimizerConfig(StrictConfig):
    epochs: int = Field(default=20, ge=1)
    batch_size_per_gpu: int = Field(default=32, ge=1)
    live_image_batch_size: int = Field(
        default=256,
        ge=1,
        description="Vision-forward chunk size for deduplicated live candidates.",
    )
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    backbone_learning_rate: float = Field(default=5e-7, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    temperature: float = Field(default=0.07, gt=0.0)
    learnable_temperature: bool = False
    mixed_precision: Literal["no", "bf16", "fp16"] = "bf16"
    log_every_steps: int = Field(default=20, ge=1)
    save_every_epochs: int = Field(default=4, ge=1)


class OnPolicyDistillationConfig(StrictConfig):
    experiment: OnPolicyExperimentConfig
    data: OnPolicyDataConfig
    model: ModelConfig
    objective: OnPolicyObjectiveConfig
    teacher: OnPolicyTeacherConfig = Field(default_factory=OnPolicyTeacherConfig)
    mining: OnPolicyMiningConfig = Field(default_factory=OnPolicyMiningConfig)
    optimizer: OnPolicyOptimizerConfig = Field(default_factory=OnPolicyOptimizerConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def validate_experiment(self) -> "OnPolicyDistillationConfig":
        if self.model.variant not in {"clip_vit_b16", "clip_vit_l14"}:
            raise ValueError(
                "CLIP on-policy distillation supports ordinary CLIP-B/16 or CLIP-L/14"
            )
        if self.model.freeze_backbone or self.model.freeze_vision_encoder:
            raise ValueError("CLIP on-policy KD must full-finetune the Student")
        if self.model.unfreeze_last_n_layers is not None or self.model.lora_rank is not None:
            raise ValueError("CLIP on-policy KD requires ordinary full finetuning")
        if self.optimizer.learnable_temperature:
            raise ValueError("KD uses explicit temperatures; learnable_temperature must be false")
        if self.mining.refresh_every_epochs != 1:
            raise ValueError("candidate mining supports a refresh interval of one epoch")
        if self.optimizer.save_every_epochs > self.optimizer.epochs:
            raise ValueError("save_every_epochs cannot exceed total epochs")
        if (
            self.objective.is_policy_loss
            and self.objective.rollout_top_k > self.mining.top_k
        ):
            raise ValueError("rollout_top_k cannot exceed the candidate Top-K")
        if (
            self.objective.is_causal_policy_loss
            and self.optimizer.gradient_accumulation_steps != 1
        ):
            raise ValueError(
                "causal policy rollout reuse currently requires "
                "gradient_accumulation_steps=1"
            )
        if self.mining.live_reservoir_selection:
            if not (
                self.objective.is_causal_policy_loss
                or self.objective.loss_type == "rank_kd"
            ):
                raise ValueError(
                    "live reservoir selection requires rank_kd or a causal policy objective"
                )
            if self.teacher.backend != "embedding_cache":
                raise ValueError("live reservoir selection requires an embedding-cache Teacher")
            if self.optimizer.gradient_accumulation_steps != 1:
                raise ValueError("live reservoir selection requires one optimizer update per batch")
            if self.mining.candidate_source != "student":
                raise ValueError("live reservoir selection requires Student mining")
            if self.mining.refresh_mode != "every_epoch":
                raise ValueError(
                    "live reservoir selection requires epoch-level refresh"
                )
            if self.mining.persist_candidate_cache:
                raise ValueError(
                    "live reservoir selection keeps ephemeral candidates in memory"
                )
            if self.mining.include_negative_queries_as_candidates:
                raise ValueError(
                    "live reservoir selection supports image candidates only"
                )
            reservoir_top_k = self.mining.reservoir_top_k
            if reservoir_top_k is None or reservoir_top_k <= self.mining.top_k:
                raise ValueError(
                    "live reservoir Top-K must exceed the rollout candidate Top-K"
                )
            if not self.mining.pixel_cache_enabled:
                raise ValueError(
                    "live reservoir selection requires the in-memory pixel cache"
                )
        elif self.mining.reservoir_top_k is not None:
            raise ValueError(
                "mining.reservoir_top_k requires live_reservoir_selection=true"
            )
        elif self.mining.pixel_cache_enabled:
            raise ValueError(
                "mining.pixel_cache_enabled currently requires "
                "live_reservoir_selection=true"
            )
        if not self.mining.force_target_insertion:
            if (
                self.objective.loss_type != "rank_kd"
                or self.objective.centered_response_weight != 0
                or self.mining.candidate_source != "student"
                or self.mining.refresh_mode != "every_epoch"
                or self.mining.live_reservoir_selection
                or self.mining.include_negative_queries_as_candidates
                or self.teacher.backend != "embedding_cache"
            ):
                raise ValueError("no target insertion requires epoch Student image-only Rank KD with embedding Teacher")
        if self.objective.anchor_kl_enabled and not self.mining.live_reservoir_selection:
            raise ValueError(
                "anchor-policy KL currently requires live reservoir selection"
            )
        if self.teacher.backend == "embedding_cache":
            if (
                self.objective.needs_teacher_scores
                or self.mining.candidate_source == "teacher"
            ) and self.data.teacher_cache_dir is None:
                raise ValueError(
                    "embedding_cache Teacher requires data.teacher_cache_dir"
                )
        else:
            if self.mining.candidate_source == "teacher":
                raise ValueError(
                    "an HTTP reranker cannot retrieve a full-gallery Teacher Top-K; "
                    "use candidate_source=student"
                )
            if self.objective.needs_teacher_scores and self.mining.cache_dtype != "float32":
                raise ValueError(
                    "HTTP reranker scores must use mining.cache_dtype=float32"
                )
        mixed_queries = self.mining.include_negative_queries_as_candidates
        if mixed_queries:
            if self.data.negative_query_manifest is None:
                raise ValueError(
                    "mixed q- candidates require data.negative_query_manifest"
                )
            if self.teacher.backend != "embedding_cache":
                raise ValueError(
                    "mixed q- candidates currently require an embedding-cache Teacher"
                )
            if self.mining.candidate_source != "student":
                raise ValueError(
                    "mixed q- candidates require candidate_source=student"
                )
            if self.objective.loss_type != "rank_kd":
                raise ValueError("mixed q- candidates require rank_kd")
            if self.mining.top_k < 3:
                raise ValueError("mixed q- candidate Top-K must be at least three")
            if (
                self.mining.max_negative_query_candidates_per_sample is not None
                and self.mining.max_negative_query_candidates_per_sample
                > self.mining.top_k - 2
            ):
                raise ValueError(
                    "the q- candidate cap must retain target plus at least one hard image"
                )
        elif self.data.negative_query_manifest is not None:
            raise ValueError(
                "data.negative_query_manifest is set but mixed q- candidates are disabled"
            )
        elif self.mining.max_negative_query_candidates_per_sample is not None:
            raise ValueError(
                "a q- candidate cap is set but mixed q- candidates are disabled"
            )
        return self

    def finalize_paths(self, config_path: Path) -> "OnPolicyDistillationConfig":
        project_root = config_path.resolve().parent.parent
        for owner, field_name in (
            (self.experiment, "output_dir"),
            (self.experiment, "initial_checkpoint"),
            (self.data, "train_annotation_path"),
            (self.data, "train_image_split_path"),
            (self.data, "val_annotation_path"),
            (self.data, "val_image_split_path"),
            (self.data, "image_root"),
            (self.data, "teacher_cache_dir"),
            (self.data, "negative_query_manifest"),
        ):
            value = getattr(owner, field_name)
            if value is None:
                continue
            if not value.is_absolute():
                setattr(owner, field_name, (project_root / value).resolve())
        local_model = Path(self.model.pretrained_model_name_or_path)
        local_candidate = (project_root / local_model).resolve()
        if not local_model.is_absolute() and local_candidate.exists():
            self.model.pretrained_model_name_or_path = str(local_candidate)
        return self


def load_on_policy_config(path: str | Path) -> OnPolicyDistillationConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return OnPolicyDistillationConfig.model_validate(payload).finalize_paths(config_path)
