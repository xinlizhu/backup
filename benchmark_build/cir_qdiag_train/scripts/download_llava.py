"""Download the original LLaVA-1.6 Mistral checkpoint, not BGE S1/S2 weights."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "llava-hf/llava-v1.6-mistral-7b-hf"
REVISION = "2424fdd47412fccc66d91719126b420e9fbd7065"
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "llava/llava-v1.6-mistral-7b-hf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    info = HfApi().model_info(REPO, revision=args.revision)
    print(f"Downloading {REPO}@{info.sha} to {args.output_dir}", flush=True)
    path = snapshot_download(
        REPO, revision=info.sha, local_dir=str(args.output_dir),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "README.md"],
        max_workers=args.max_workers,
    )
    from safetensors import safe_open
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
    for shard in set(index.values()):
        with safe_open(Path(path) / shard, framework="pt", device="cpu") as weights:
            if set(weights.keys()) != {k for k, v in index.items() if v == shard}:
                raise RuntimeError(f"Invalid shard index: {shard}")
    print(f"Complete: {path}; revision={info.sha}; verified {len(index)} indexed tensors", flush=True)


if __name__ == "__main__":
    main()
