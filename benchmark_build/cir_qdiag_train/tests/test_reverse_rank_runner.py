"""Verify launch/resume/evaluation orchestration without loading a real model."""

import importlib.util
import json
from pathlib import Path

import pytest

from cir_qdiag_train.on_policy_config import load_on_policy_config


def load_runner():
    path = Path(__file__).parents[1] / "scripts/run_reverse_rank_r128.py"
    spec = importlib.util.spec_from_file_location("reverse_rank_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_resumes_then_skips_completed_stages(tmp_path, monkeypatch):
    runner = load_runner()
    config = load_on_policy_config(
        runner.PROJECT / "configs/clip_l_reverse_rank_opd_r128_e4.yaml"
    )
    config.experiment.output_dir = tmp_path / "opd"
    run = config.experiment.output_dir
    run.mkdir()
    (run / "last.pt").touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "load_on_policy_config", lambda path: config)
    commands = []

    def execute(argv, log, cwd=None):
        argv = [str(x) for x in argv]
        commands.append(argv)
        script = Path(argv[1]).name

        def argument(name):
            return argv[argv.index(name) + 1]

        if script == "train_on_policy_distillation.py":
            assert argument("--resume-from") == str(run / "last.pt")
            for epoch in range(1, 5):
                (run / f"checkpoint_epoch_{epoch:03d}.pt").touch()
            (run / "effective_config.json").write_text(config.model_dump_json())
        elif script == "evaluate_on_policy_distillation.py":
            Path(argument("--output")).write_text(json.dumps({"recall_at_1": 40.0}))
        elif script == "evaluate_fashioniq_clip.py":
            output = Path(argument("--output-dir"))
            output.mkdir(parents=True)
            (output / "metrics.json").write_text(
                json.dumps({"macro_average": {"recall_at_10": 30.0}})
            )
        else:
            output = Path(argument("--output-root")) / "CLIP-L-REV-opd"
            output.mkdir(parents=True)
            assert argument("--checkpoint").endswith("checkpoint_epoch_004.pt")
            if script == "run_clip_pinpoint_official.py":
                (output / "official_metrics.csv").write_text("mAP@10,NegRecall@10\n0.1,0.2\n")
            elif script == "evaluate_winrate_random10.py":
                (output / "summary.csv").write_text(
                    "subset,target_stable_win_rate\nstrict,0.15\n"
                )
            else:
                raise AssertionError(script)

    monkeypatch.setattr(runner, "execute", execute)
    runner.run_one("opd", extended=True)
    assert len(commands) == 11  # train, four CIRR, four FashionIQ, two final diagnostics
    runner.run_one("opd", extended=True)
    assert len(commands) == 11
    runner.summarize()
    report = (tmp_path / "RESULTS.md").read_text()
    assert "best E1=40.0" in report
    assert "0.15" in report
    assert "static: pending" in report
    config.objective.rank_temperature = 0.1
    with pytest.raises(RuntimeError, match="objective configuration changed"):
        runner.run_one("opd", extended=True)
