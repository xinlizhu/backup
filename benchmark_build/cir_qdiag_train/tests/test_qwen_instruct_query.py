import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from test_qwen_shared_lora import tiny_config, tiny_inputs

from cir_qdiag_train.qwen_on_policy_config import QwenQueryLoRAConfig, load_qwen_on_policy_config
from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent, _composed_conversation

PROJECT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
spec = importlib.util.spec_from_file_location("instruct_query_runner_test", PROJECT / "scripts/run_qwen_instruct_query_e4.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("suffix", list(runner.SOURCE_CONFIGS))
def test_exact_old_config_except_initialization_and_output(suffix):
    name = "qwen2_instruct_query_" + suffix
    c = load_qwen_on_policy_config(PROJECT / f"configs/{name}.yaml")
    runner.assert_only_initialization_changed(name, c)
    assert not c.model.shared_lora and c.model.native_retrieval
    assert c.model.lora_target_modules == "attention"
    assert c.optimizer.batch_size_per_gpu == 8
    assert c.optimizer.gradient_accumulation_steps == 4
    assert c.mining.query_batch_size == c.mining.image_batch_size == 16
    c.optimizer.batch_size_per_gpu = 16
    with pytest.raises(ValueError, match="differs"):
        runner.assert_only_initialization_changed(name, c)


def test_instruct_query_rejects_all_linear():
    with pytest.raises(ValueError, match="attention LoRA"):
        QwenQueryLoRAConfig(variant="qwen3_vl_instruct_query_lora",
                           pretrained_model_name_or_path="unused", lora_target_modules="all-linear")


def test_attention_lora_gradients_frozen_images_and_checkpoint(monkeypatch, tmp_path):
    from transformers import Qwen3VLModel

    from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval
    from cir_qdiag_train.qwen_student import qwen_student_from_checkpoint, save_qwen_checkpoint

    template = Qwen3VLModel(tiny_config())
    def init(self, config, dtype):
        self.config, self.model = config, deepcopy(template)
    monkeypatch.setattr(NativeQwenRetrieval, "__init__", init)
    c = QwenQueryLoRAConfig(variant="qwen3_vl_instruct_query_lora",
                           pretrained_model_name_or_path=str(PROJECT), lora_rank=2, lora_alpha=4)
    student = QwenQueryLoRAStudent(c, device=torch.device("cpu"), mixed_precision="no")
    names = [n for n, p in student.model.named_parameters() if p.requires_grad]
    assert names and all("language_model" in n and "lora_" in n for n in names)
    assert all(any(f".{target}." in n for target in student.LORA_TARGET_MODULES) for n in names)
    inputs = tiny_inputs()
    frozen = student.encode_prepared_images(inputs)
    student.train()
    query = student.encode_prepared_queries(inputs, requires_grad=True)
    (query * torch.arange(32)).sum().backward()
    assert all(p.grad is not None for p in student.trainable_parameters())
    assert all(p.grad is None for p in student.parameters() if not p.requires_grad)
    with torch.no_grad():
        for n, param in student.model.named_parameters():
            if "lora_B" in n:
                param.add_(torch.randn_like(param) * .1)
    torch.testing.assert_close(frozen, student.encode_prepared_images(inputs))
    changed_query = student.encode_prepared_queries(inputs, requires_grad=False)
    assert not torch.allclose(frozen, changed_query)
    with pytest.raises(ValueError, match="query-only"):
        student.encode_prepared_images(inputs, requires_grad=True)
    path = tmp_path / "native_query.pt"
    save_qwen_checkpoint(path, student=student, optimizer=None, scheduler=None, epoch=0,
                         global_step=1, config={"model": c.model_dump(mode="json")})
    restored, payload = qwen_student_from_checkpoint(path, device=torch.device("cpu"), mixed_precision="no")
    assert payload["student_type"] == "qwen3_vl_instruct_query_lora"
    torch.testing.assert_close(changed_query, restored.encode_prepared_queries(inputs, requires_grad=False))
    torch.testing.assert_close(frozen, restored.encode_prepared_images(inputs))


def test_real_2b_attention_parameter_count(monkeypatch):
    from transformers import Qwen3VLConfig, Qwen3VLModel

    from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval

    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_query_infonce.yaml").model
    with torch.device("meta"):
        template = Qwen3VLModel(Qwen3VLConfig.from_pretrained(c.pretrained_model_name_or_path, local_files_only=True))
    def init(self, config, dtype):
        self.config, self.model = config, template
    monkeypatch.setattr(NativeQwenRetrieval, "__init__", init)
    with torch.device("meta"):
        student = QwenQueryLoRAStudent(c, device=torch.device("meta"), mixed_precision="no")
    assert student.parameter_summary()["trainable_parameter_count"] == 6422528
    assert all("language_model" in n for n, p in student.model.named_parameters() if p.requires_grad)


def test_preprocessing_matches_old_embedding_for_valid_images(tmp_path):
    from transformers import Qwen3VLProcessor

    from cir_qdiag_train.qwen_instruct_retrieval import preprocess_query_only
    from cir_qdiag_train.qwen_training_data import QueryCollator

    old_path = PROJECT.parents[1] / "CIR_PLAY/models/Qwen3-VL-Embedding-2B/scripts/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("old_embedding_preprocessor_test", old_path)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_query_infonce.yaml").model
    processor = Qwen3VLProcessor.from_pretrained(c.pretrained_model_name_or_path, local_files_only=True, padding_side="right")
    path = tmp_path / "reference.png"
    Image.new("RGB", (64, 96), (20, 90, 130)).save(path)
    texts = ["make it blue", "long " * 1600]
    for text in texts:
        conversations = [_composed_conversation([path], text, instruction=QwenQueryLoRAStudent.QUERY_INSTRUCTION,
                                                min_pixels=c.min_pixels, max_pixels=c.max_pixels)]
        state = SimpleNamespace(processor=processor, max_length=c.max_length)
        expected = dict(old.Qwen3VLEmbedder._preprocess_inputs(state, conversations))
        actual = preprocess_query_only(processor, conversations, c.max_length)
        worker = QueryCollator(processor, c)([(0, path, text)])[1]
        assert actual.keys() == expected.keys() == worker.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            torch.testing.assert_close(worker[key], value, rtol=0, atol=0)
    state = SimpleNamespace(default_instruction="Represent the user's input.", min_pixels=c.min_pixels, max_pixels=c.max_pixels)
    old_conversation = old.Qwen3VLEmbedder.format_model_input(state, image=str(path), instruction=QwenQueryLoRAStudent.TARGET_INSTRUCTION)
    new_conversation = _composed_conversation([path], "", instruction=QwenQueryLoRAStudent.TARGET_INSTRUCTION,
                                             min_pixels=c.min_pixels, max_pixels=c.max_pixels)
    old_image = preprocess_query_only(processor, [old_conversation], c.max_length)
    new_image = preprocess_query_only(processor, [new_conversation], c.max_length)
    for key in old_image:
        torch.testing.assert_close(old_image[key], new_image[key], rtol=0, atol=0)


def test_fixed_gallery_rejects_embedding_cache(tmp_path):
    from cir_qdiag_train.qwen_on_policy_trainer import _load_student_gallery
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_query_infonce.yaml")
    c.data.student_gallery_cache_dir = tmp_path
    metadata = {"complete": True, "student_type": "qwen3_vl_embedding_query_lora"}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="do not reuse Embedding"):
        _load_student_gallery(c, ["0"])
    metadata.update(student_type=c.model.variant, base_model_path=c.model.pretrained_model_name_or_path,
                    preprocessing={"min_pixels": c.model.min_pixels, "max_pixels": c.model.max_pixels,
                                   "max_length": c.model.max_length, "instruction": QwenQueryLoRAStudent.TARGET_INSTRUCTION})
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "image_ids.json").write_text('["0"]')
    np.save(tmp_path / "gallery_features.npy", np.ones((1, 4), dtype=np.float16))
    assert _load_student_gallery(c, ["0"]).shape == (1, 4)


def test_runner_prepares_cache_then_trains_and_evaluates(tmp_path, monkeypatch):
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_query_infonce.yaml")
    name = "qwen2_instruct_query_infonce"
    c.experiment.output_dir = tmp_path / name
    c.data.student_gallery_cache_dir = tmp_path / "cache/qwen3_vl_instruct_2b_fixed_gallery"
    calls = []
    suite = SimpleNamespace(runner=SimpleNamespace(), load=lambda _: c,
                            preflight=lambda config: calls.append("preflight"),
                            run_qwen=lambda n, **kwargs: calls.append((n, kwargs)),
                            summarize=lambda: calls.append("summary"))
    monkeypatch.setattr(runner, "suite", suite)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prepare_gallery", lambda path, cfg: calls.append("cache"))
    monkeypatch.setattr(sys, "argv", ["run", "--run-indices", "2"])
    monkeypatch.setenv("QWEN2_EVAL_BATCH_SIZE", "256")
    monkeypatch.setenv("QWEN2_DIAG_BATCH_SIZE", "128")
    runner.main()
    assert calls == ["cache", "preflight", (name, {"extended": True}), "summary"]
    import os
    assert os.environ["QWEN2_EVAL_BATCH_SIZE"] == os.environ["QWEN2_DIAG_BATCH_SIZE"] == "16"


def test_gallery_builder_reuses_only_new_model_cache(tmp_path, monkeypatch):
    import prepare_qwen_student_gallery_cache as builder

    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_query_infonce.yaml")
    c.data.student_gallery_cache_dir = tmp_path
    calls = []

    class Student:
        TARGET_INSTRUCTION = QwenQueryLoRAStudent.TARGET_INSTRUCTION

        def __init__(self, config, **kwargs):
            calls.append("load")
            assert config.variant == "qwen3_vl_instruct_query_lora"

        def eval(self):
            pass

        def prepare_image_batch(self, paths):
            return paths

        def encode_prepared_images(self, paths):
            return torch.ones(len(paths), 4)

    def encode(**kwargs):
        pending = json.loads((tmp_path / "metadata.json").read_text())
        assert pending["student_type"] == c.model.variant and not pending["complete"]
        features = kwargs["encode"](0, 2)
        np.save(kwargs["path"], features)
        return features

    monkeypatch.setattr(builder, "load_qwen_on_policy_config", lambda _: c)
    monkeypatch.setattr(builder, "load_image_map", lambda *args: (["0", "1"], ["a", "b"]))
    monkeypatch.setattr(builder, "QwenQueryLoRAStudent", Student)
    monkeypatch.setattr(builder, "_open_resumable_features", encode)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["cache", "--config", "unused", "--image-batch-size", "16"])
    builder.main()
    builder.main()
    assert calls == ["load"]  # complete matching cache skips even model initialization
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    metadata["base_model_path"] = "old-embedding-checkpoint"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata mismatch"):
        builder.main()
