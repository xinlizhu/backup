import os
import subprocess
import sys
from pathlib import Path

import pytest

from cir_qdiag_train.llava_parallel import shard_order
from cir_qdiag_train.qwen_on_policy_config import QwenOnPolicyDistillationConfig, load_qwen_on_policy_config

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 35, 28245])
def test_disjoint_batches_without_padding(length):
    order = list(reversed(range(length)))
    ranks = [shard_order(order, 16, rank, 2) for rank in range(2)]
    assert set(ranks[0]).isdisjoint(ranks[1])
    assert sorted(ranks[0] + ranks[1]) == list(range(length))
    for step, start in enumerate(range(0, length, 32)):
        assert ranks[0][step * 16:(step + 1) * 16] + ranks[1][step * 16:(step + 1) * 16] == order[start:start + 32]


def test_parallel_config_rejects_unsupported_semantics():
    c = load_qwen_on_policy_config(PROJECT / "configs/llava_oprd_s2.yaml")
    c.optimizer.gradient_accumulation_steps = 2
    with pytest.raises(ValueError, match="accumulation=1"):
        QwenOnPolicyDistillationConfig.model_validate(c.model_dump())
    c.optimizer.gradient_accumulation_steps = 1
    c.objective.teacher_target_max_rank = 10
    with pytest.raises(ValueError, match="unfiltered"):
        QwenOnPolicyDistillationConfig.model_validate(c.model_dump())


def test_smoke_launch_is_two_rank_and_no_run_writes():
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/smoke_llava_lora.py"), "--dry-run"],
                            cwd=PROJECT, text=True, capture_output=True, timeout=60, check=True)
    assert "2 GPUs x 16" in result.stdout and "global_batch=32" in result.stdout
    assert "candidate_chunk=8" in result.stdout and "gradient_cache=True" in result.stdout
    assert "--smoke-only" in result.stdout and "--nproc_per_node=2" in result.stdout


@pytest.mark.parametrize("mode", ["gradients", "training"])
def test_two_process_cpu_integration(mode, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", GLOO_SOCKET_IFNAME="lo")
    variants = [("oprd_s2", "full")] if mode == "gradients" else [
        (suffix, phase) for suffix in ("oprd_s2", "infonce", "cmse_s2") for phase in ("full", "resume")]
    for suffix, phase in variants:
        result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                                 "--max_restarts=0", str(PROJECT / "tests/llava_parallel_check.py"),
                                 "--mode", mode, "--work-dir", str(tmp_path), "--suffix", suffix, "--phase", phase],
                                cwd=PROJECT, env=env, text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, (result.stdout + result.stderr)[-14000:]
        assert ("GRADIENT_EQUIVALENCE_OK" if mode == "gradients" else "TINY_TRAINING_AND_RESUME_OK") in result.stdout
