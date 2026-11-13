import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import yaml
from cir_qdiag_train.qwen_on_policy_config import QwenQueryLoRAConfig

PROJECT = Path(__file__).parents[1]


@pytest.fixture
def evaluator():
    spec = importlib.util.spec_from_file_location("cirr_eval_config_test",
        PROJECT / "scripts/evaluate_cirr_original_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("suffix", ["oprd_s2", "oprd_qwen8", "infonce", "cmse_s2", "cmse_qwen8"])
def test_real_embedding_configs(evaluator, suffix):
    c = evaluator.load_evaluation_config(PROJECT / f"configs/qwen2_embedding_shared_{suffix}.yaml")
    assert c.model.variant == "qwen3_vl_embedding_shared_lora"
    assert c.optimizer.epochs == 1 and c.optimizer.world_size == 2


@pytest.mark.parametrize("variant", get_args(QwenQueryLoRAConfig.model_fields["variant"].annotation))
def test_all_schema_variants_route_to_qwen(evaluator, monkeypatch, tmp_path, variant):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"variant": variant}}))
    sentinel = object()
    monkeypatch.setattr(evaluator, "load_qwen_on_policy_config", lambda p: sentinel)
    monkeypatch.setattr(evaluator, "load_on_policy_config",
                        lambda p: pytest.fail("Qwen variant routed to CLIP"))
    assert evaluator.load_evaluation_config(path) is sentinel


def test_clip_unchanged(evaluator):
    c = evaluator.load_evaluation_config(PROJECT / "configs/on_policy_clip_l_rank.yaml")
    assert c.model.variant == "clip_vit_l14"


def test_main_passes_embedding_config_validation(evaluator, monkeypatch):
    args = SimpleNamespace(config=PROJECT / "configs/qwen2_embedding_shared_oprd_s2.yaml",
                           image_batch_size=16, query_batch_size=16, score_batch_size=512)
    monkeypatch.setattr(evaluator, "parse_args", lambda: args)
    class ReachedDataset(Exception):
        pass
    def stop_before_data(*args, **kwargs):
        raise ReachedDataset
    monkeypatch.setattr(evaluator, "load_image_map", stop_before_data)
    with pytest.raises(ReachedDataset):
        evaluator.main()
