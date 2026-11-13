import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

PROJECT = Path(__file__).parents[1]


def test_original_dry_run_has_four_evaluations_no_writes(tmp_path):
    output = tmp_path / "org"
    result = subprocess.run([sys.executable, "scripts/evaluate_qwen_instruct_original.py",
                             "--dry-run", "--output-dir", str(output)],
                            cwd=PROJECT, text=True, capture_output=True, check=True)
    assert result.stdout.count("[evaluate]") == 4
    assert "Qwen3-VL-2B-Instruct-original" in result.stdout
    assert not output.exists()
    for script in ("evaluate_cirr_original_adapter.py", "evaluate_fashioniq_original_adapter.py",
                   "run_clip_pinpoint_official.py", "evaluate_winrate_random10.py"):
        assert script in result.stdout


def test_zero_adapter_matches_original_and_runner_resumes(monkeypatch, tmp_path):
    from test_qwen_shared_lora import build_tiny_student, tiny_inputs
    from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location("qwen_org_test", PROJECT / "scripts/evaluate_qwen_instruct_original.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    student = build_tiny_student(monkeypatch)
    student.eval()
    inputs = tiny_inputs()
    for encode in (student.encode_prepared_queries, student.encode_prepared_images):
        with torch.no_grad():
            original = encode(inputs, requires_grad=False)
            with student.model.disable_adapter():
                base = encode(inputs, requires_grad=False)
        torch.testing.assert_close(original, base, rtol=0, atol=0)
    config = load_qwen_on_policy_config(PROJECT / "configs/qwen2_instruct_shared_oprd_s2.yaml")
    config.model = student.config
    monkeypatch.setattr(module, "QwenQueryLoRAStudent", lambda *a, **kw: student)
    calls = []
    monkeypatch.setattr(module.runner, "execute", lambda cmd, log: calls.append(cmd))
    args = SimpleNamespace(device="cpu", output_dir=tmp_path)
    checkpoint = tmp_path / "initialization.pt"
    completed = tmp_path / "done"
    completed.touch()
    module.evaluate(args, config, checkpoint, [(completed, ["skip"]), (tmp_path / "todo", ["evaluate"])])
    assert calls == [["evaluate"]]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["epoch"] == -1 and payload["global_step"] == 0
    assert "optimizer" not in payload
    assert all(not v.count_nonzero() for k, v in payload["model"].items() if "lora_B" in k)
    module.evaluate(args, config, checkpoint, [])
    payload["global_step"] = 10
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="initialization"):
        module.evaluate(args, config, checkpoint, [])
