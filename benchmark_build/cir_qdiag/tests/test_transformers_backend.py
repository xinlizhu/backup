from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from cir_qdiag.models.transformers_vlm import TransformersVLMBackend


def test_transformers_loader_honors_revision_and_processor_kwargs(monkeypatch) -> None:
    processor_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    class ProcessorFactory:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            processor_calls.append((path, kwargs))
            return SimpleNamespace()

    class ModelFactory:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs):
            model_calls.append((path, kwargs))
            return SimpleNamespace(device="cpu")

    torch = ModuleType("torch")
    torch.bfloat16 = object()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = ProcessorFactory
    transformers.AutoModelForMultimodalLM = ModelFactory
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    backend = TransformersVLMBackend(
        "/models/qwen",
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=True,
        model_revision="revision-7",
        mm_processor_kwargs={"max_pixels": 1234},
    )
    backend._load()

    assert processor_calls == [
        (
            "/models/qwen",
            {
                "revision": "revision-7",
                "trust_remote_code": True,
                "max_pixels": 1234,
            },
        )
    ]
    assert model_calls[0][0] == "/models/qwen"
    assert model_calls[0][1]["revision"] == "revision-7"
    assert model_calls[0][1]["trust_remote_code"] is True
    assert model_calls[0][1]["device_map"] == "cpu"
    assert backend.cache_identity()["mm_processor_kwargs"] == {"max_pixels": 1234}
