"""Instruct initialization with the old Embedding query-only input protocol.

Use the native checkpoint loader (including its prefix mapping), then discard
the generation head. The training branch remains the original frozen-gallery
query-only attention LoRA path, not shared all-linear LoRA.
"""

from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval


def preprocess_query_only(processor, conversations, max_length):
    from qwen_vl_utils.vision_process import process_vision_info

    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    # Same valid-image preprocessing arguments as Qwen3VLEmbedder. Invalid images
    # raise instead of silently substituting the upstream text-only NULL fallback.
    images, videos, video_kwargs = process_vision_info(
        conversations, image_patch_size=16,
        return_video_metadata=True, return_video_kwargs=True,
    )
    if videos is not None:
        raise ValueError("This retrieval trainer supports images, not video")
    return dict(processor(
        text=text, images=images, videos=None, video_metadata=None,
        truncation=True, max_length=max_length, padding=True, do_resize=False,
        return_tensors="pt", **video_kwargs,
    ))


class InstructQueryRetrieval(NativeQwenRetrieval):
    def _preprocess_inputs(self, conversations):
        return preprocess_query_only(self.processor, conversations, self.config.max_length)
