from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cir_qdiag.config import load_config
from cir_qdiag.data.cirr_loader import load_cirr_annotations
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.pipeline import create_backend


class SmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_summary: str
    target_summary: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-image structured-output vLLM smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_20.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    sample = load_cirr_annotations(
        config.data.annotation_path,
        config.data.image_root,
        config.data.split,
        config.data.image_split_path,
    )[0]
    client = StructuredVLMClient(
        create_backend(config),
        config.runtime.output_dir / "smoke_cache",
        config.runtime.prompt_version,
        max_retries=0,
    )
    response, trace = client.call(
        SmokeResponse,
        images=[sample.reference.image_path, sample.target.image_path],
        system_prompt=(
            "Describe each image independently and return concise JSON matching the schema. "
            "Do not judge whether they form an image-edit transition."
        ),
        user_prompt=json.dumps(
            {
                "Image 1 role": "reference",
                "Image 2 role": "target",
            },
            ensure_ascii=False,
        ),
        stage="smoke_two_image_json",
        temperature=0.0,
        seed=42,
        max_new_tokens=512,
    )
    print(
        json.dumps(
            {"response": response.model_dump(), "trace": trace.model_dump()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
