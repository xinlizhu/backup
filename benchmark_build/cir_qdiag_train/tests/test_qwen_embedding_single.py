import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config

PROJECT = Path(__file__).parents[1]


@pytest.mark.parametrize("suffix", ["oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8"])
def test_exact_reference_match(suffix):
    current = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_embedding_single_{suffix}.yaml")
    old = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_instruct_shared_{suffix}.yaml")
    a, b = current.model_dump(mode="json"), old.model_dump(mode="json")
    for value in (a, b):
        value.pop("experiment")
        value["model"].pop("variant")
        value["model"].pop("pretrained_model_name_or_path")
    assert a == b
    assert current.optimizer.world_size == 1
    assert current.optimizer.batch_size_per_gpu == 16 and current.optimizer.gradient_accumulation_steps == 2
    assert current.model.train_gallery and current.model.shared_lora
    assert current.data.student_gallery_cache_dir is None
    assert current.model.pretrained_model_name_or_path.endswith("Qwen3-VL-Embedding-2B")
    assert "qwen_embedding_single" in str(current.experiment.output_dir)


@pytest.mark.parametrize("indices", [["0", "3"], ["1", "2", "4"]])
def test_independent_queue_dry_run(indices):
    result = subprocess.run([sys.executable, "scripts/run_qwen_embedding_single.py", "--dry-run",
                             "--run-indices", *indices], cwd=PROJECT, capture_output=True, text=True, check=True)
    assert result.stdout.count("single GPU; batch=16 x 2") == len(indices)
    assert "evaluation-only" not in result.stdout


def test_runner_dispatches_selected_jobs_without_suite_lock(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location("embedding_single_test_runner", PROJECT / "scripts/run_qwen_embedding_single.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.suite, "preflight", lambda c: None)
    monkeypatch.setattr(module.suite, "run_qwen", lambda name, **kwargs: calls.append(name))
    monkeypatch.setattr(module.suite, "summarize", lambda: None)
    monkeypatch.setattr(sys, "argv", ["runner", "--run-indices", "1", "2", "4"])
    # Keep shared suite module state local to this test.
    for obj, key in [(module.suite, "RUNS"), (module.suite, "EVALUATION_EPOCHS"),
                     (module.suite, "ROOT"), (module.suite.runner, "ROOT"),
                     (module.suite.runner, "LABEL_PREFIX"), (module.suite.runner, "CONFIG_PATHS")]:
        monkeypatch.setattr(obj, key, getattr(obj, key))
    module.main()
    assert calls == [module.RUNS[i] for i in (1, 2, 4)]
    assert not (tmp_path / ".suite.lock").exists()
    assert module.suite.EVALUATION_EPOCHS == {}
