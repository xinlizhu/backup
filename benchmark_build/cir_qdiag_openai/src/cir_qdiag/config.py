from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(ConfigModel):
    parquet_path: Path
    image_root: Path
    min_group_size: int = Field(default=2, ge=2)
    max_group_size: int = Field(default=6, le=6)

    @model_validator(mode="after")
    def validate_group_sizes(self) -> "DataConfig":
        if self.max_group_size < self.min_group_size:
            raise ValueError("max_group_size must be >= min_group_size")
        return self


class SamplingConfig(ConfigModel):
    limit: int | None = Field(default=50, gt=0)
    seed: int = 42
    strategy: Literal["random", "first"] = "random"


class ModelConfig(ConfigModel):
    backend: Literal["openai"] = "openai"
    api_base: str = "${OPENAI_BASE_URL}"
    api_key: str = "${OPENAI_API_KEY}"
    organization: str | None = None
    project: str | None = None
    timeout_seconds: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    image_detail: Literal["low", "high", "original", "auto"] = "high"
    structured_output_mode: Literal["json_schema", "prompt"] = "prompt"
    generation_model: str = "gpt5.6-terra"
    selection_model: str = "gpt5.6-terra"
    generation_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None = None
    selection_reasoning_effort: Literal[
        "none", "low", "medium", "high", "xhigh", "max"
    ] | None = None


class GenerationConfig(ConfigModel):
    max_output_tokens: int = Field(default=4096, gt=0)


class SelectionConfig(ConfigModel):
    max_output_tokens: int = Field(default=16384, gt=0)
    max_positive_images: int = Field(default=8, ge=1, le=8)


class RuntimeConfig(ConfigModel):
    output_dir: Path
    cache_dir: Path | None = None
    prompt_dir: Path | None = None
    prompt_version: Literal["v7"] = "v7"
    resume: bool = True
    force_stage: Literal["prepare", "generate", "validate"] | None = None
    max_workers: int = Field(default=2, ge=1)


class PipelineConfig(ConfigModel):
    data: DataConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    runtime: RuntimeConfig

    def finalize_paths(self, config_path: Path) -> "PipelineConfig":
        project_root = config_path.resolve().parent.parent
        if not self.data.parquet_path.is_absolute():
            self.data.parquet_path = (project_root / self.data.parquet_path).resolve()
        if not self.data.image_root.is_absolute():
            self.data.image_root = (project_root / self.data.image_root).resolve()
        if self.runtime.prompt_dir is None:
            self.runtime.prompt_dir = project_root / "prompts"
        elif not self.runtime.prompt_dir.is_absolute():
            self.runtime.prompt_dir = (project_root / self.runtime.prompt_dir).resolve()
        if not self.runtime.output_dir.is_absolute():
            self.runtime.output_dir = (project_root / self.runtime.output_dir).resolve()
        if self.runtime.cache_dir is None:
            self.runtime.cache_dir = self.runtime.output_dir / "cache"
        elif not self.runtime.cache_dir.is_absolute():
            self.runtime.cache_dir = (project_root / self.runtime.cache_dir).resolve()
        return self


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if overrides:
        _deep_update(raw, overrides)
    return PipelineConfig.model_validate(raw).finalize_paths(config_path)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
