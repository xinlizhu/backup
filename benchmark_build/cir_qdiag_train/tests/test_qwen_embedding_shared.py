from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config

PROJECT = Path(__file__).parents[1]


@pytest.mark.parametrize("suffix", ["oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8"])
def test_embedding_two_gpu_config(suffix):
    c = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_embedding_shared_{suffix}.yaml")
    assert c.model.shared_lora and c.model.train_gallery and c.model.native_retrieval
    assert c.model.variant == "qwen3_vl_embedding_shared_lora"
    assert c.model.pretrained_model_name_or_path.endswith("Qwen3-VL-Embedding-2B")
    assert c.model.lora_target_modules == "all-linear" and c.model.lora_rank == 16
    assert c.data.student_gallery_cache_dir is None
    assert c.data.num_workers == 8 and c.data.prefetch_factor == 2
    assert c.optimizer.world_size == 2 and c.optimizer.batch_size_per_gpu == 16
    assert c.optimizer.gradient_accumulation_steps == 1 and c.optimizer.epochs == 1
    assert c.mining.force_target_insertion and c.mining.top_k == 32
    assert c.mining.candidate_source == ("student" if suffix.startswith("oprd") else "random")
    assert not c.objective.uses_teacher_filter
    assert "qwen_embedding_shared_2gpu_e1" in str(c.experiment.output_dir)


def test_embedding_runner_dry_run():
    result = subprocess.run([sys.executable, "scripts/run_qwen_embedding_shared_e4.py", "--dry-run"],
                            cwd=PROJECT, capture_output=True, text=True, check=True)
    assert result.stdout.count("GPUs=2; global_batch=32") == 5
    names = [line.split()[1].rstrip(":") for line in result.stdout.splitlines() if line.startswith("[")]
    assert names == ["qwen2_embedding_shared_" + suffix for suffix in
                     ("oprd_s2", "infonce", "cmse_s2", "oprd_qwen8", "cmse_qwen8")]
    assert "evaluation-only" not in result.stdout


def test_embedding_headless_load_and_both_sides(monkeypatch, tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from test_qwen_shared_lora import tiny_config, tiny_inputs
    from cir_qdiag_train.qwen_retrieval import NativeQwenRetrieval
    from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent
    from cir_qdiag_train.qwen_on_policy_trainer import audit_shared_adapters
    import accelerate.utils.other
    monkeypatch.setattr(accelerate.utils.other, "is_deepspeed_available", lambda: False)
    model = Qwen3VLForConditionalGeneration(tiny_config())
    model.config.save_pretrained(tmp_path)
    save_file({k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"},
              str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    monkeypatch.setattr(Qwen3VLProcessor, "from_pretrained", lambda *a, **k: None)
    c = load_qwen_on_policy_config(PROJECT / "configs/qwen2_embedding_shared_oprd_s2.yaml").model
    c.pretrained_model_name_or_path = str(tmp_path)
    c.lora_rank, c.lora_alpha = 2, 4
    loaded = NativeQwenRetrieval(c, torch.float32)
    for n, p in model.model.named_parameters():
        torch.testing.assert_close(p, dict(loaded.model.named_parameters())[n])
    student = QwenQueryLoRAStudent(c, device=torch.device("cpu"), mixed_precision="no")
    assert all(audit_shared_adapters(student)["modules"].values())
    for encode in (student.encode_prepared_queries, student.encode_prepared_images):
        student.model.zero_grad(set_to_none=True)
        (encode(tiny_inputs(), requires_grad=True) * torch.arange(32)).sum().backward()
        for component in ("language_model", "visual.blocks", "merger"):
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for n, p in student.model.named_parameters() if component in n and p.requires_grad)
