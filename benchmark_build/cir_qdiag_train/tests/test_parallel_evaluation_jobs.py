import importlib.util
import io
import threading
from pathlib import Path

import pytest


@pytest.fixture
def suite():
    path = Path(__file__).parents[1] / "scripts/run_main_comparison_e4.py"
    spec = importlib.util.spec_from_file_location("parallel_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jobs():
    return [(["python", script, "--device", "cuda:0"], Path("/tmp")) for script in (
        "evaluate_cirr_original_adapter.py", "evaluate_fashioniq_original_adapter.py",
        "run_clip_pinpoint_official.py", "evaluate_winrate_random10.py")]


def test_parallel_lanes_and_order(suite, monkeypatch, tmp_path):
    barrier = threading.Barrier(2, timeout=5)
    seen = {"cuda:0": [], "cuda:1": []}

    def execute(command, log, cwd):
        device = command[-1]
        if not seen[device]:
            barrier.wait()
        seen[device].append(command[1])
        assert Path(log.name).name == f"evaluation_gpu{device[-1]}.log"

    monkeypatch.setattr(suite.runner, "execute", execute)
    original = jobs()
    suite.execute_evaluation_jobs(original, io.StringIO(), tmp_path, parallel=True)
    assert seen["cuda:0"] == [j[0][1] for j in original[:2]]
    assert seen["cuda:1"] == [j[0][1] for j in original[2:]]
    assert all(j[0][-1] == "cuda:0" for j in original)


def test_serial_unchanged(suite, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(suite.runner, "execute", lambda command, log, cwd: seen.append((command, cwd)))
    suite.execute_evaluation_jobs(jobs(), io.StringIO(), tmp_path)
    assert seen == jobs()


def test_failure_waits_for_other_lane(suite, monkeypatch, tmp_path):
    barrier = threading.Barrier(2, timeout=5)
    finished = threading.Event()

    def execute(command, log, cwd):
        if command[1] != "evaluate_winrate_random10.py":
            barrier.wait()
        if command[-1] == "cuda:0":
            raise RuntimeError("evaluation failed")
        if command[1] == "evaluate_winrate_random10.py":
            finished.set()

    monkeypatch.setattr(suite.runner, "execute", execute)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        suite.execute_evaluation_jobs(jobs(), io.StringIO(), tmp_path, parallel=True)
    assert finished.is_set()
