"""Strict, separate configuration for the TG-CIR baseline (not OPRD)."""

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs/tg_cir_clip_l_e4.yaml"


def clip_backbone_name(clip_config):
    """Validate the two supported 224px CLIP backbones using actual dimensions."""
    vision, text = clip_config["vision_config"], clip_config["text_config"]
    shape = (
        clip_config["projection_dim"],
        vision["hidden_size"],
        vision["patch_size"],
        vision["image_size"],
        text["hidden_size"],
    )
    names = {(512, 768, 16, 224, 512): "CLIP-B", (768, 1024, 14, 224, 768): "CLIP-L"}
    if shape not in names:
        raise ValueError(f"TG-CIR expects CLIP ViT-B/16 or ViT-L/14 at 224px; got {shape}")
    return names[shape]


def model_label(config):
    path = Path(config["model"]["pretrained_model_name_or_path"]) / "config.json"
    return "TG-CIR-" + clip_backbone_name(json.loads(path.read_text()))


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TGModel(StrictConfig):
    pretrained_model_name_or_path: Path
    global_tokens: int = Field(default=4, ge=1, le=512)
    local_tokens: int = Field(default=8, ge=1)
    temperature: float = Field(default=0.05, gt=0)
    gradient_checkpointing: bool = True


class TGData(StrictConfig):
    image_root: Path
    train_annotation_path: Path
    train_image_split_path: Path
    val_annotation_path: Path
    val_image_split_path: Path


class TGTraining(StrictConfig):
    epochs: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=2)
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    clip_learning_rate: float = Field(default=1e-5, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    mixed_precision: Literal["fp16", "bf16", "no"] = "fp16"
    save_every_steps: int = Field(default=100, ge=1)
    log_every_steps: int = Field(default=20, ge=1)


class TGLosses(StrictConfig):
    stu_rank: float = Field(default=1, ge=0)
    tea_rank: float = Field(default=1, ge=0)
    tea_mask: float = Field(default=1, ge=0)
    ortho: float = Field(default=0.1, ge=0)
    ckd: float = Field(default=1, ge=0)
    kl: float = Field(default=0.1, ge=0)


class TGEvaluation(StrictConfig):
    batch_size: int = Field(default=128, ge=1)
    num_workers: int = Field(default=8, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    fashioniq_root: Path


class TGConfig(StrictConfig):
    method: Literal["tg_cir_v1"]
    seed: int = 42
    output_dir: Path
    model: TGModel
    data: TGData
    training: TGTraining
    loss_weights: TGLosses
    evaluation: TGEvaluation


def load_config(path):
    config = TGConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    config.output_dir = resolve(config.output_dir)
    config.model.pretrained_model_name_or_path = resolve(config.model.pretrained_model_name_or_path)
    for name in TGData.model_fields:
        setattr(config.data, name, resolve(getattr(config.data, name)))
    config.evaluation.fashioniq_root = resolve(config.evaluation.fashioniq_root)
    return config.model_dump(mode="json")


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def training_identity(config):
    # Loader/runtime changes do not change the deterministic sample order.
    return {
        "method": config["method"],
        "seed": config["seed"],
        "model": config["model"],
        "data": config["data"],
        "loss_weights": config["loss_weights"],
        "training": {
            key: value
            for key, value in config["training"].items()
            if key not in {"num_workers", "prefetch_factor", "save_every_steps", "log_every_steps"}
        },
    }
