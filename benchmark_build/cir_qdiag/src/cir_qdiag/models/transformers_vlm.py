from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cir_qdiag.models.vllm_openai import _extract_json
from cir_qdiag.schemas import BackendResult
from cir_qdiag.utils.hashing import local_model_fingerprint


class TransformersVLMBackend:
    """Direct local fallback. The vLLM API backend is preferred for batch construction."""

    def __init__(
        self,
        name_or_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        model_revision: str | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.name_or_path = str(name_or_path)
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.model_revision = model_revision
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.model_artifact_fingerprint = local_model_fingerprint(self.name_or_path)
        self._processor = None
        self._model = None

    def cache_identity(self) -> dict[str, Any]:
        return {
            "backend_contract": "transformers-v2",
            "name_or_path": self.name_or_path,
            "model_revision": self.model_revision,
            "model_artifact_fingerprint": self.model_artifact_fingerprint,
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "mm_processor_kwargs": self.mm_processor_kwargs,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install the transformers optional dependencies") from exc
        torch_dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(
            self.name_or_path,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code,
            **self.mm_processor_kwargs,
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else self.device,
            revision=self.model_revision,
            trust_remote_code=self.trust_remote_code,
        )

    def generate_json(
        self,
        images: list[str],
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        *,
        stage: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_new_tokens: int = 1024,
    ) -> BackendResult:
        self._load()
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        content: list[dict[str, Any]] = []
        for path in images:
            content.append({"type": "image", "url": str(Path(path).resolve())})
        schema_instruction = "\nRequired JSON Schema:\n" + json.dumps(
            json_schema, ensure_ascii=False
        )
        content.append({"type": "text", "text": user_prompt + schema_instruction})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            chat_template_kwargs={"enable_thinking": True},
        ).to(self._model.device)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 0.8
        output = self._model.generate(**inputs, **kwargs)
        input_length = inputs["input_ids"].shape[-1]
        text = self._processor.decode(output[0][input_length:], skip_special_tokens=True)
        return BackendResult(
            data=_extract_json(text),
            raw_response={"text": text},
            model_name=self.name_or_path,
            model_revision=self.model_revision,
        )
