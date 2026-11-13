"""Native Qwen3-VL retrieval wrapper: no dependency on files in an embedding repo."""

import torch


def preprocess(processor, conversations, max_length):
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    # Fail on invalid images; never silently substitute a text-only NULL sample.
    images, videos = process_vision_info(conversations, image_patch_size=16)
    if videos is not None:
        raise ValueError("This retrieval trainer supports images, not video")
    prepared = processor(
        text=text, images=images, padding=True, truncation=False,
        do_resize=False, return_tensors="pt",
    )
    if prepared["input_ids"].shape[1] > max_length:
        raise ValueError("Qwen input exceeds max_length; increase it or reduce max_pixels")
    return dict(prepared)


class NativeQwenRetrieval:
    def __init__(self, config, dtype):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        self.config = config
        self.processor = Qwen3VLProcessor.from_pretrained(
            config.pretrained_model_name_or_path,
            local_files_only=config.local_files_only, padding_side="right",
        )
        # Load using the checkpoint's native class first. Loading Qwen3VLModel
        # directly can mis-map the `model.` prefix in Transformers 4.57.
        loader = Qwen3VLForConditionalGeneration
        if getattr(config, "variant", None) == "qwen3_vl_embedding_shared_lora":
            # Official embedding weights have a model.* backbone and no lm_head.
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel
            class EmbeddingBackbone(Qwen3VLPreTrainedModel):
                _checkpoint_conversion_mapping = {}

                def __init__(self, native_config):
                    super().__init__(native_config)
                    self.model = Qwen3VLModel(native_config)
                    self.post_init()
            loader = EmbeddingBackbone
        generation_model, info = loader.from_pretrained(
            config.pretrained_model_name_or_path,
            local_files_only=config.local_files_only,
            torch_dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
        )
        # Instruct's generation head is not needed. No backbone may be random-init.
        unexpected = [k for k in info.get("unexpected_keys", []) if k != "lm_head.weight"]
        if info.get("missing_keys") or info.get("mismatched_keys") or unexpected:
            raise RuntimeError(f"Incomplete Qwen backbone loading: {info}")
        self.model = generation_model.model

    def _preprocess_inputs(self, conversations):
        return preprocess(self.processor, conversations, self.config.max_length)

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        positions = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)
        return hidden_state[torch.arange(hidden_state.shape[0], device=hidden_state.device), positions]
