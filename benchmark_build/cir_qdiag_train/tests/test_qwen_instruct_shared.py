from pathlib import Path
import subprocess
import sys

import pytest

from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config

PROJECT = Path(__file__).parents[1]
SUFFIXES = ("oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_shared_comparison_configs(suffix):
    c = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_instruct_shared_{suffix}.yaml")
    old = load_qwen_on_policy_config(PROJECT / f"configs/qwen2_instruct_query_{suffix}.yaml")
    assert c.model.train_gallery and c.model.shared_lora
    assert c.model.lora_target_modules == "all-linear"
    assert c.data.student_gallery_cache_dir is None
    assert old.data.student_gallery_cache_dir is not None
    assert c.experiment.output_dir != old.experiment.output_dir
    assert not c.objective.uses_teacher_filter
    assert c.mining.force_target_insertion and c.mining.top_k == 32
    assert c.mining.refresh_mode == "every_epoch"
    assert c.mining.candidate_source == ("student" if suffix.startswith("oprd") else "random")
    assert not c.optimizer.candidate_gradient_cache
    assert c.optimizer.candidate_encode_batch_size == 32
    assert c.optimizer.epochs == (4 if suffix == "oprd_s2" else 2)
    assert c.optimizer.batch_size_per_gpu == 16
    assert c.optimizer.gradient_accumulation_steps == 2
    assert c.optimizer.batch_size_per_gpu * c.optimizer.gradient_accumulation_steps == 32
    if suffix != "oprd_s2":
        assert c.data.num_workers == 8
        assert c.data.prefetch_factor == 4
    a, b = c.model_dump(mode="json"), old.model_dump(mode="json")
    # Preserve the prior protocol except the explicitly requested trainability changes,
    # disabled teacher filter and isolated output/cache identities.
    for d in (a, b):
        d.pop("experiment")
        d["data"].pop("student_gallery_cache_dir")
        if suffix != "oprd_s2":
            d["data"].pop("num_workers")
            d["data"].pop("prefetch_factor")
        for key in ("variant", "lora_target_modules", "trust_remote_code"):
            d["model"].pop(key)
        d["objective"].pop("teacher_target_max_rank")
        d["optimizer"].pop("candidate_encode_batch_size")
        d["optimizer"].pop("candidate_gradient_cache")
        d["optimizer"].pop("batch_size_per_gpu")
        d["optimizer"].pop("gradient_accumulation_steps")
        d["optimizer"].pop("epochs")
    assert a == b


def test_runner_dry_run():
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/run_qwen_instruct_shared_e4.py"),
                             "--dry-run"], cwd=PROJECT, capture_output=True, text=True, check=True, timeout=30)
    for suffix in SUFFIXES:
        assert f"qwen2_instruct_shared_{suffix}" in result.stdout
    assert result.stdout.count("no fixed gallery") == 5


def test_runner_rejects_invalid_index():
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/run_qwen_instruct_shared_e4.py"),
                             "--dry-run", "--run-indices", "5"], cwd=PROJECT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
