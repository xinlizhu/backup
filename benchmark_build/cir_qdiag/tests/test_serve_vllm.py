from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_serve_command_sets_encoder_token_budget() -> None:
    project = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/serve_vllm.py",
            "--config",
            "configs/model.yaml",
            "--tensor-parallel-size",
            "1",
            "--dry-run",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--max-model-len 32768" in result.stdout
    assert "--max-num-batched-tokens 32768" in result.stdout
    assert "--mm-processor-kwargs" in result.stdout
    assert "max_pixels" in result.stdout
