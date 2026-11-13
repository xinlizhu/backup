"""Qwen3.5 retrieval: shared all-linear LoRA or full backbone fine-tuning."""

import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model

from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig
from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval
from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent, _autocast, _model_dtype, _move_batch
from cir_qdiag_train.resume_config import without_loader_settings


def preprocess_qwen35(processor, conversations, max_length):
    from qwen_vl_utils import process_vision_info

    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    images, videos = process_vision_info(conversations, image_patch_size=16)
    if videos is not None:
        raise ValueError("Qwen3.5 retrieval currently supports images, not videos")
    prepared = dict(processor(
        text=texts, images=images, padding=True, truncation=False, do_resize=False,
        return_mm_token_type_ids=True, return_tensors="pt",
    ))
    if prepared["input_ids"].shape[1] > max_length:
        raise ValueError("Qwen3.5 input exceeds max_length; increase it or reduce max_pixels")
    if "mm_token_type_ids" not in prepared:
        raise ValueError("Qwen3.5 requires mm_token_type_ids for correct multimodal positions")
    return prepared


def load_qwen35_backbone(config, dtype):
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Qwen3.5 requires the separate .venv_qwen35 environment; see QWEN35_TRAINING.md") from exc
    generation, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path, local_files_only=config.local_files_only,
        dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
    )
    # MTP is a generation-only auxiliary module; the tied LM head is unused by
    # retrieval. Every retrieval-backbone weight must come from the checkpoint.
    unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("mtp.")]
    if info.get("missing_keys") or info.get("mismatched_keys") or unexpected:
        raise RuntimeError(f"Incomplete Qwen3.5 loading: {info}")
    return generation.model


def component(name):
    if "merger" in name:
        return "merger"
    if "visual" in name:
        return "vision"
    return "language"


class Qwen35Student(QwenQueryLoRAStudent):
    def __init__(self, config, *, device, mixed_precision):
        from transformers import AutoProcessor

        if not config.qwen35:
            raise ValueError("Qwen35Student requires a Qwen3.5 configuration")
        self.config, self.device, self.mixed_precision = config, device, mixed_precision
        self.full_finetuning = config.variant == "qwen3_5_full"
        # Full FT keeps FP32 parameters / Adam states and uses BF16 autocast.
        # PEFT similarly keeps trainable adapters in FP32 over a BF16 base.
        dtype = torch.float32 if self.full_finetuning else _model_dtype(device, mixed_precision)
        base = load_qwen35_backbone(config, dtype)
        self.linear_module_names = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)]
        base.config.use_cache = False
        base.config.text_config.use_cache = False
        if config.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if not self.full_finetuning:
                base.enable_input_require_grads()
        if self.full_finetuning:
            base.requires_grad_(True)
            self.model = base.to(device)
        else:
            lora = LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout, target_modules="all-linear",
                bias="none", task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model = get_peft_model(base, lora).to(device)
        processor = AutoProcessor.from_pretrained(
            config.pretrained_model_name_or_path, local_files_only=config.local_files_only,
            padding_side="right",
        )
        self.embedder = SimpleNamespace(
            processor=processor, model=self.model, _pooling_last=NativeQwenRetrieval._pooling_last,
            _preprocess_inputs=lambda conversations: preprocess_qwen35(processor, conversations, config.max_length),
        )
        self.trainability_audit()

    def encode_prepared_queries(self, prepared, *, requires_grad):
        with nullcontext() if requires_grad else torch.inference_mode():
            inputs = _move_batch(prepared, self.device)
            with _autocast(self.device, self.mixed_precision):
                outputs = self.model(**inputs, use_cache=False)
                pooled = self.embedder._pooling_last(outputs.last_hidden_state, inputs["attention_mask"])
            return F.normalize(pooled.float(), p=2, dim=-1)

    def encode_prepared_images(self, prepared, *, requires_grad=False):
        # Both modes always use the CURRENT shared model, including during eval.
        return self.encode_prepared_queries(prepared, requires_grad=requires_grad)

    def trainability_audit(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as runtime
        modules = {"language": [], "vision": [], "merger": []}
        for name in self.linear_module_names:
            modules[component(name)].append(name)
        if not all(modules.values()):
            raise ValueError("Qwen3.5 must cover language, vision and merger")
        if self.full_finetuning:
            if any(not p.requires_grad for p in self.model.parameters()):
                raise ValueError("Full fine-tuning cannot contain frozen backbone parameters")
            if any("lora_" in n for n, _ in self.model.named_parameters()):
                raise ValueError("Full fine-tuning cannot contain LoRA modules")
        else:
            base = self.model.get_base_model()
            for name in self.linear_module_names:
                if not hasattr(base.get_submodule(name), "lora_A"):
                    raise ValueError(f"Unadapted linear module: {name}")
            if any(p.requires_grad and "lora_" not in n for n, p in self.model.named_parameters()):
                raise ValueError("LoRA mode cannot train original backbone parameters")
        return {"mode": "full" if self.full_finetuning else "all_linear_lora",
                "both_sides_trainable": True, "modules": modules,
                "fla_delta_kernel_available": runtime.chunk_gated_delta_rule is not None,
                "causal_conv1d_kernel_available": runtime.causal_conv1d_fn is not None,
                **self.parameter_summary()}

    def checkpoint_weights(self):
        if self.full_finetuning:
            return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return self.lora_state_dict()

    def load_checkpoint_weights(self, checkpoint):
        expected = "full_backbone" if self.full_finetuning else "lora_adapter"
        if checkpoint.get("weight_format") != expected or checkpoint.get("student_type") != self.config.variant:
            raise ValueError("Qwen3.5 full/LoRA checkpoint formats are not interchangeable")
        if self.full_finetuning:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.load_lora_state_dict(checkpoint["model"])


def save_qwen35_checkpoint(path, *, student, optimizer, scheduler, epoch, global_step, config, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2, "student_type": student.config.variant,
        "weight_format": "full_backbone" if student.full_finetuning else "lora_adapter",
        "model": student.checkpoint_weights(), "epoch": epoch, "global_step": global_step,
        "config": config, "rng_state": {
            "torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_qwen35_checkpoint(path, *, student, optimizer=None, scheduler=None, scaler=None, expected_config=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    previous = QwenOnPolicyDistillationConfig.model_validate(payload["config"])
    if previous.model != student.config:
        raise ValueError("Qwen3.5 checkpoint model/LoRA configuration mismatch")
    if expected_config is not None:
        if without_loader_settings(previous.model_dump(mode="json")) != without_loader_settings(expected_config):
            raise ValueError("Qwen3.5 resume configuration mismatch")
    student.load_checkpoint_weights(payload)
    for name, owner in [("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)]:
        if owner is not None:
            if name not in payload:
                raise ValueError(f"Qwen3.5 checkpoint has no {name} state")
            owner.load_state_dict(payload[name])
    rng = payload["rng_state"]
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload
