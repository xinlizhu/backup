from __future__ import annotations

import sys
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
import numpy as np
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image

from cir_qdiag_train.qwen_on_policy_config import QwenQueryLoRAConfig


class QwenQueryLoRAStudent:
    """Asymmetric Qwen3-VL-Embedding Student for efficient retrieval OPD.

    The original image/index encoder stays immutable, so the gallery can be
    cached exactly once.  A LoRA policy is active only for composed queries.
    This makes Student Top-K refresh exact for the deployed asymmetric model
    without re-encoding every candidate image after every optimizer step.
    """

    QUERY_INSTRUCTION = "Represent the composed image retrieval query."
    TARGET_INSTRUCTION = "Represent the image for retrieval."
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        config: QwenQueryLoRAConfig,
        *,
        device: torch.device,
        mixed_precision: str,
    ) -> None:
        self.config = config
        self.device = device
        self.mixed_precision = mixed_precision
        model_root = Path(config.pretrained_model_name_or_path).resolve()
        if not model_root.is_dir():
            raise FileNotFoundError(model_root)
        torch_dtype = _model_dtype(device, mixed_precision)
        if config.variant == "qwen3_vl_instruct_query_lora":
            from cir_qdiag_train.qwen_instruct_retrieval import InstructQueryRetrieval

            self.embedder = InstructQueryRetrieval(config, torch_dtype)
        elif config.shared_lora:
            from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval

            self.embedder = NativeQwenRetrieval(config, torch_dtype)
        else:
            model_root_text = str(model_root)
            if model_root_text not in sys.path:
                sys.path.insert(0, model_root_text)
            from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

            self.embedder = Qwen3VLEmbedder(
                model_name_or_path=model_root_text,
                max_length=config.max_length,
                min_pixels=config.min_pixels,
                max_pixels=config.max_pixels,
                default_instruction="Represent the user's input.",
                local_files_only=config.local_files_only,
                torch_dtype=torch_dtype,
            )
        base_model = self.embedder.model
        base_model.to(device)
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False
        if hasattr(base_model.config, "text_config"):
            base_model.config.text_config.use_cache = False
        if config.gradient_checkpointing:
            try:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                base_model.gradient_checkpointing_enable()
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=("all-linear" if config.lora_target_modules == "all-linear"
                            else self.LORA_TARGET_MODULES),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.model = get_peft_model(base_model, lora).to(device)
        self.embedder.model = self.model

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.trainable_parameters())
        return {
            "total_parameter_count": total,
            "trainable_parameter_count": trainable,
            "frozen_parameter_count": total - trainable,
        }

    def prepare_query_batch(
        self,
        images: Sequence[Path | Image.Image],
        texts: Sequence[str],
    ) -> dict[str, Any]:
        if len(images) != len(texts):
            raise ValueError("Qwen query images/texts have different batch sizes")
        return self.prepare_composed_query_batch(
            [[image] for image in images], texts
        )

    def prepare_composed_query_batch(
        self,
        image_groups: Sequence[Sequence[Path | Image.Image]],
        texts: Sequence[str],
    ) -> dict[str, Any]:
        if len(image_groups) != len(texts):
            raise ValueError("Qwen query image groups/texts have different batch sizes")
        if any(not group for group in image_groups):
            raise ValueError("every Qwen composed query needs at least one image")
        conversations = [
            _composed_conversation(
                images,
                str(text),
                instruction=self.QUERY_INSTRUCTION,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )
            for images, text in zip(image_groups, texts, strict=True)
        ]
        return dict(self.embedder._preprocess_inputs(conversations))

    def prepare_image_batch(
        self, images: Sequence[Path | Image.Image]
    ) -> dict[str, Any]:
        if self.config.native_retrieval:
            return dict(self.embedder._preprocess_inputs([
                _composed_conversation(
                    [image], "", instruction=self.TARGET_INSTRUCTION,
                    min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels,
                ) for image in images
            ]))
        conversations = [
            self.embedder.format_model_input(
                image=_image_input(image),
                instruction=self.TARGET_INSTRUCTION,
            )
            for image in images
        ]
        return dict(self.embedder._preprocess_inputs(conversations))

    def encode_prepared_queries(
        self,
        prepared: dict[str, Any],
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        context = nullcontext() if requires_grad else torch.inference_mode()
        with context:
            inputs = _move_batch(prepared, self.device)
            with _autocast(self.device, self.mixed_precision):
                outputs = self.model(**inputs)
                embeddings = self.embedder._pooling_last(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
            return F.normalize(embeddings.float(), p=2, dim=-1)

    def encode_prepared_images(
        self, prepared: dict[str, Any], *, requires_grad: bool = False
    ) -> torch.Tensor:
        if self.config.shared_lora:
            # Shared adapter stays enabled on BOTH branches, in training and evaluation.
            return self.encode_prepared_queries(prepared, requires_grad=requires_grad)
        if requires_grad:
            raise ValueError("query-only LoRA cannot backpropagate through gallery")
        self.eval()
        with torch.inference_mode(), self.model.disable_adapter():
            inputs = _move_batch(prepared, self.device)
            with _autocast(self.device, self.mixed_precision):
                outputs = self.model(**inputs)
                embeddings = self.embedder._pooling_last(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
            return F.normalize(embeddings.float(), p=2, dim=-1)

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(self.model).items()
        }

    def load_lora_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = get_peft_model_state_dict(self.model)
        if expected.keys() != state.keys() or any(expected[k].shape != state[k].shape for k in expected):
            raise ValueError("Incomplete or incompatible Qwen adapter state dict")
        incompatible = set_peft_model_state_dict(self.model, state)
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        if unexpected:
            raise ValueError(f"unexpected Qwen LoRA checkpoint keys: {unexpected}")


def save_qwen_checkpoint(
    path: str | Path,
    *,
    student: QwenQueryLoRAStudent,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    scaler: Any | None = None,
) -> None:
    destination = Path(path)
    if student.config.qwen35:
        from cir_qdiag_train.qwen35_student import save_qwen35_checkpoint
        return save_qwen35_checkpoint(
            destination, student=student, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, global_step=global_step, config=config, scaler=scaler,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "student_type": student.config.variant,
        "model": student.lora_state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if student.config.shared_lora:
        local_cuda_rng = config.get("optimizer", {}).get("world_size", 1) > 1
        payload["rng_state"] = {
            "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and not local_cuda_rng else None,
        }
        if local_cuda_rng and student.device.type == "cuda":
            # Do not initialize contexts on another rank's GPU just to save RNG.
            # Distributed epoch resume reseeds every rank after candidate mining.
            payload["rng_state"]["cuda_local"] = torch.cuda.get_rng_state(student.device)
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_qwen_checkpoint(
    path: str | Path,
    *,
    student: QwenQueryLoRAStudent,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if student.config.qwen35:
        from cir_qdiag_train.qwen35_student import load_qwen35_checkpoint
        return load_qwen35_checkpoint(
            path, student=student, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, expected_config=expected_config,
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if expected_config is not None:
        from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig
        from cir_qdiag_train.resume_config import without_loader_settings
        old = QwenOnPolicyDistillationConfig.model_validate(checkpoint["config"]).model_dump(mode="json")
        if without_loader_settings(old) != without_loader_settings(expected_config):
            raise ValueError("Qwen resume checkpoint experiment configuration mismatch")
    if checkpoint.get("student_type") != student.config.variant:
        raise ValueError(f"Qwen checkpoint mode does not match Student: {path}")
    previous = QwenQueryLoRAConfig.model_validate(checkpoint["config"]["model"])
    if previous != student.config:
        raise ValueError("Qwen checkpoint model/LoRA configuration mismatch")
    student.load_lora_state_dict(checkpoint["model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError("Qwen checkpoint has no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        if "scheduler" not in checkpoint:
            raise ValueError("Qwen checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if student.config.shared_lora and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("cuda_local") is not None and student.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda_local"], student.device)
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def qwen_student_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    mixed_precision: str,
) -> tuple[QwenQueryLoRAStudent, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = checkpoint.get("config", {}).get("model")
    if not isinstance(payload, dict):
        raise TypeError("Qwen checkpoint has no model config")
    model_config = QwenQueryLoRAConfig.model_validate(payload)
    if checkpoint.get("student_type") != model_config.variant:
        raise ValueError("Qwen checkpoint type/config mismatch")
    student_class = QwenQueryLoRAStudent
    if model_config.llava:
        from cir_qdiag_train.llava_student import LlavaLoRAStudent
        student_class = LlavaLoRAStudent
    elif model_config.qwen35:
        from cir_qdiag_train.qwen35_student import Qwen35Student
        student_class = Qwen35Student
    student = student_class(
        model_config,
        device=device,
        mixed_precision=mixed_precision,
    )
    if model_config.qwen35:
        student.load_checkpoint_weights(checkpoint)
    else:
        student.load_lora_state_dict(checkpoint["model"])
    student.eval()
    return student, checkpoint


def _image_input(value: Path | Image.Image) -> str | Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return str(Path(value).resolve())


def _composed_conversation(
    images: Sequence[Path | Image.Image],
    text: str,
    *,
    instruction: str,
    min_pixels: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "image": _image_input(image),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        for image in images
    ]
    if text:
        content.append({"type": "text", "text": text})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {"role": "user", "content": content},
    ]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _model_dtype(device: torch.device, mixed_precision: str) -> torch.dtype:
    if device.type != "cuda" or mixed_precision == "no":
        return torch.float32
    return torch.bfloat16 if mixed_precision == "bf16" else torch.float16


def _autocast(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
