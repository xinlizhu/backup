"""Create a separate Qwen3.5 dependency overlay; never upgrade the base environment."""

import argparse
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BASE_PYTHON = PROJECT.parents[1] / "multi_modal_py3.12/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, default=PROJECT / ".venv_qwen35")
    args = parser.parse_args()
    root = args.env_dir.resolve()
    if root.exists():
        cfg = root / "pyvenv.cfg"
        if not cfg.is_file() or f"home = {BASE_PYTHON.parent}" not in cfg.read_text():
            raise ValueError(f"Refusing to modify an unrelated directory/environment: {root}")
    subprocess.run([str(BASE_PYTHON), "-m", "venv", "--system-site-packages", str(root)], check=True)
    python = root / "bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-r",
                    str(PROJECT / "requirements/qwen35.txt")], check=True)
    subprocess.run([str(python), "-c", "import torch,transformers,peft; "
                    "from transformers import Qwen3_5ForConditionalGeneration; "
                    "print('Torch:',torch.__version__,torch.__file__); "
                    "print('Transformers:',transformers.__version__,'PEFT:',peft.__version__)"], check=True)


if __name__ == "__main__":
    main()
