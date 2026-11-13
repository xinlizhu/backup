"""Inference-only TG-CIR interface for the existing PinPoint/QDiag evaluators."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cir_qdiag_train.tg_cir_data import TGCIRProcessor
from cir_qdiag_train.tg_cir_runtime import build_model, load_checkpoint


class TGCIRAdapter:
    def __init__(self, model_path, device, dtype="auto"):
        self.model_path = Path(model_path).resolve()
        self.device = torch.device(device)
        self.dtype_arg = dtype
        self.native_qkv_enabled = False
        types = {
            "auto": torch.float16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype not in types:
            raise ValueError(f"Unsupported TG-CIR inference dtype: {dtype}")
        self.amp_dtype = types[dtype] if self.device.type == "cuda" else torch.float32
        checkpoint = load_checkpoint(self.model_path)
        config = checkpoint["config"]
        self.model = build_model(config, checkpoint["clip_config"])
        self.model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        self.model.to(self.device).eval()
        self.processor = TGCIRProcessor(config["model"]["pretrained_model_name_or_path"])

    def load_image(self, path):
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda" and self.amp_dtype != torch.float32,
        )

    def prepare_image_batch(self, images):
        return self.processor(images=images)

    @torch.inference_mode()
    def encode_prepared_image_batch(self, prepared, identifiers=None):
        del identifiers
        with self._autocast():
            values = self.model.extract_retrieval_target(
                prepared["pixel_values"].to(self.device, non_blocking=True)
            )
        return values.float().cpu().numpy()

    def encode_image_batch(self, image_paths, identifiers=None):
        images = [self.load_image(path) for path in image_paths]
        try:
            return self.encode_prepared_image_batch(self.prepare_image_batch(images), identifiers)
        finally:
            for image in images:
                image.close()

    def prepare_query_batch(self, rows, image_cache, preloaded_images):
        from run_pinpoint_bge_lincir import query_image_signatures

        del image_cache
        counts = [len(query_image_signatures(row)) for _, row in rows.iterrows()]
        if not counts or any(count < 1 for count in counts):
            raise ValueError("TG-CIR queries require at least one reference image")
        if sum(counts) != len(preloaded_images):
            raise ValueError("TG-CIR reference counts do not match preloaded images")
        return {
            **self.processor(
                images=preloaded_images, text=rows["instruction"].astype(str).tolist()
            ),
            "reference_counts": counts,
        }

    @torch.inference_mode()
    def encode_prepared_query_batch(self, prepared, image_cache):
        del image_cache
        with self._autocast():
            references = self.model.backbone.extract_img_fea(
                prepared["pixel_values"].to(self.device, non_blocking=True)
            )
            counts = prepared["reference_counts"]
            # A documented extension for two-reference PinPoint, not an original
            # TG-CIR training operation. Single-reference is exactly unchanged.
            pooled = torch.stack([group.mean(dim=0) for group in references.split(counts)])
            texts = self.model.backbone.extract_text_fea(
                prepared["input_ids"].to(self.device, non_blocking=True)
            )
            values = self.model.pool(self.model.student_attributes(pooled, texts)[0])
        return values.float().cpu().numpy().astype(np.float32, copy=False)

    def encode_query_batch(self, rows, image_cache, preloaded_images=None):
        from run_pinpoint_bge_lincir import flat_query_image_paths

        own_images = preloaded_images is None
        images = (
            preloaded_images
            if preloaded_images is not None
            else [self.load_image(path) for path in flat_query_image_paths(rows, image_cache)]
        )
        try:
            prepared = self.prepare_query_batch(rows, image_cache, images)
            return self.encode_prepared_query_batch(prepared, image_cache)
        finally:
            if own_images:
                for image in images:
                    image.close()

    def protocol_metadata(self):
        return {
            "checkpoint": str(self.model_path),
            "method": "TG-CIR",
            "teacher_at_inference": False,
            "query": "TG-CIR student attribute composition; mean then L2 normalize",
            "candidate": "TG-CIR image attributes; mean then L2 normalize",
            "multiple_references": "mean aligned reference attributes before student composition",
            "text": "OpenAI CLIP-compatible fixed 77 tokens, zero padding, truncate preserving EOS",
            "dtype": str(self.amp_dtype),
        }
