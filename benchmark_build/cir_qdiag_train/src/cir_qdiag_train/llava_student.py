"""LLaVA-NeXT retrieval with shared, all-linear LoRA (no generation logits)."""
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent, _model_dtype

QUERY_INSTRUCTION = (
    "Retrieve the target image that best meets the combined criteria by using both "
    "the provided image and the image retrieval instructions: "
)


def load_llava_backbone(config, dtype):
    generation, info = LlavaNextForConditionalGeneration.from_pretrained(
        config.pretrained_model_name_or_path, local_files_only=config.local_files_only,
        torch_dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("unexpected_keys"):
        raise RuntimeError(f"Incomplete LLaVA checkpoint loading: {info}")
    # Native loading also handles older HF checkpoint key names. Discard only
    # the unneeded vocabulary head; never randomly initialize a retrieval layer.
    return generation.model


def load_llava_processor(config):
    from transformers import LlavaNextConfig

    native = LlavaNextConfig.from_pretrained(config.pretrained_model_name_or_path,
                                            local_files_only=config.local_files_only)
    processor = LlavaNextProcessor.from_pretrained(
        config.pretrained_model_name_or_path, local_files_only=config.local_files_only,
        use_fast=False,
    )
    processor.patch_size = native.vision_config.patch_size
    processor.vision_feature_select_strategy = native.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1  # CLIP's CLS is removed by `default`.
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_eos_token = False  # We append exactly one explicit EOS.
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if processor.tokenizer.convert_tokens_to_ids(processor.image_token) != native.image_token_index:
        raise ValueError("LLaVA processor image token does not match model")
    return processor


def _resize_image(value, size):
    if isinstance(value, Image.Image):
        return value.resize((size, size)).convert("RGB")
    with Image.open(Path(value)) as source:
        return source.resize((size, size)).convert("RGB")


def preprocess_llava(processor, config, image_groups, texts=None, *, query):
    """One CPU-only implementation for training, mining and all evaluations.

    Use BGE-style role prompts with explicit EOS pooling. The input is resized
    to 512 square; the native AnyRes processor still produces multiple crops.
    Flat images follow prompt order, including multi-reference diagnostic rows.
    """
    if texts is None:
        texts = [None] * len(image_groups)
    if not image_groups or len(texts) != len(image_groups) or any(not g for g in image_groups):
        raise ValueError("LLaVA needs matching, nonempty image groups and texts")
    eos = processor.tokenizer.eos_token
    if eos is None:
        raise ValueError("LLaVA tokenizer has no EOS token")
    prompts, images = [], []
    for group, text in zip(image_groups, texts, strict=True):
        if text is not None and processor.image_token in str(text):
            raise ValueError("Query text must not introduce extra image placeholders")
        prompt = f"[INST] \n <instruct> {QUERY_INSTRUCTION} <query> " if query else "[INST] "
        if text is not None:
            prompt += f" {text} \n"
        prompt += " " + " ".join([processor.image_token] * len(group))
        prompts.append(prompt + " [/INST]" + eos)
        images.extend(_resize_image(im, config.llava_image_size) for im in group)
    prepared = dict(processor(text=prompts, images=images, padding=True,
                              truncation=False, return_tensors="pt"))
    ids, mask = prepared["input_ids"], prepared["attention_mask"]
    if ids.shape[1] > config.max_length:
        raise ValueError(f"LLaVA input has {ids.shape[1]} tokens > max_length={config.max_length}; "
                         "do not truncate image tokens")
    last = mask.shape[1] - 1 - mask.flip(1).argmax(1)
    if not torch.all(ids[torch.arange(len(ids)), last] == processor.tokenizer.eos_token_id):
        raise ValueError("Every LLaVA input must end in an attended EOS token")
    # Some tokenizer versions supply token_type_ids; Mistral does not use them.
    return {k: v for k, v in prepared.items()
            if k in {"input_ids", "attention_mask", "pixel_values", "image_sizes"}}


class LlavaLoRAStudent(QwenQueryLoRAStudent):
    """Reuse the tested score/checkpoint interface, not Qwen's model or prompts."""
    QUERY_INSTRUCTION = QUERY_INSTRUCTION

    def __init__(self, config, *, device, mixed_precision):
        if not config.llava:
            raise ValueError("LLaVA requires its own shared-LoRA variant")
        self.config, self.device, self.mixed_precision = config, device, mixed_precision
        processor = load_llava_processor(config)
        base = load_llava_backbone(config, _model_dtype(device, mixed_precision)).to(device)
        base.config.use_cache = base.config.text_config.use_cache = False
        if config.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        targets = ("all-linear" if config.lora_target_modules == "all-linear" else
                   [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)
                    and n.startswith("language_model.") and ".self_attn." in n
                    and n.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}])
        if not targets:
            raise ValueError("No language attention LoRA targets found")
        self.model = get_peft_model(base, LoraConfig(
            target_modules=targets, r=config.lora_rank, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
        )).to(device)
        self.embedder = SimpleNamespace(processor=processor, model=self.model, _pooling_last=self.pool_last)

    @staticmethod
    def pool_last(hidden_state, mask):
        last = mask.shape[1] - 1 - mask.flip(1).argmax(1)
        return hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), last]

    def prepare_query_batch(self, images, texts):
        return self.prepare_composed_query_batch([[im] for im in images], texts)

    def prepare_composed_query_batch(self, image_groups, texts):
        return preprocess_llava(self.embedder.processor, self.config, image_groups, texts, query=True)

    def prepare_image_batch(self, images):
        return preprocess_llava(self.embedder.processor, self.config, [[im] for im in images], query=False)

    def trainability_audit(self):
        adapters = [n for n, m in self.model.named_modules() if hasattr(m, "lora_A")]
        uncovered = [n for n, m in self.model.named_modules() if isinstance(m, torch.nn.Linear)
                     and not any(n.startswith(a + ".") for a in adapters)]
        groups = {name: [n for n in adapters if fragment in n] for name, fragment in
                  (("language", "language_model"), ("vision", "vision_tower"),
                   ("projector", "multi_modal_projector"))}
        if self.config.lora_target_modules == "all-linear" and (uncovered or any(not g for g in groups.values())):
            raise RuntimeError(f"Incomplete all-linear coverage: {uncovered}, {groups}")
        if self.config.lora_target_modules == "attention":
            expected = self.model.get_base_model().config.text_config.num_hidden_layers * 4
            if (len(adapters) != expected or len(groups["language"]) != expected
                    or groups["vision"] or groups["projector"]
                    or any(".self_attn." not in n or n.rsplit(".", 1)[-1]
                           not in {"q_proj", "k_proj", "v_proj", "o_proj"} for n in adapters)):
                raise RuntimeError(f"Invalid language-only attention coverage: {groups}")
        if any(p.requires_grad and "lora_" not in n for n, p in self.model.named_parameters()):
            raise RuntimeError("Non-LoRA parameter unexpectedly trainable")
        # Native LLaVA selects the penultimate vision hidden state. Adapters in
        # later blocks are instantiated but disconnected from retrieval loss.
        native = self.model.get_base_model().config
        layers = native.vision_config.num_hidden_layers
        selected = native.vision_feature_layer
        selected = [selected] if isinstance(selected, int) else selected
        depth = max(i if i >= 0 else layers + 1 + i for i in selected)
        inactive = [n for n in adapters if any(f"vision_model.encoder.layers.{i}." in n
                                               for i in range(depth, layers))]
        return {"target_modules": self.config.lora_target_modules, "shared_query_gallery": True, "modules": groups,
                "native_vision_feature_layer": selected,
                "inactive_vision_adapter_modules": inactive, **self.parameter_summary()}
