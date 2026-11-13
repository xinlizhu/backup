from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from pydantic import BaseModel, ConfigDict

from cir_qdiag.config import load_config
from cir_qdiag.data import load_exact_pinpoint_groups
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.pipeline import create_backend


class SmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_summary: str
    positive_summary: str


class TextSmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible gateway smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_50.yaml"))
    parser.add_argument(
        "--mode",
        choices=["text", "vision"],
        default="text",
        help="test text generation or a two-image multimodal request",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    client = StructuredVLMClient(
        create_backend(config),
        config.runtime.output_dir / "smoke_cache",
        config.runtime.prompt_version,
        max_retries=0,
    )
    if args.mode == "text":
        response, trace = client.call(
            TextSmokeResponse,
            images=[],
            system_prompt="Return the requested value.",
            user_prompt="Set value to exactly OK123123OK.",
            stage="smoke_text_json",
            max_output_tokens=128,
        )
    else:
        group = load_exact_pinpoint_groups(
            config.data.parquet_path,
            config.data.image_root,
            min_group_size=config.data.min_group_size,
            max_group_size=config.data.max_group_size,
        )[0]
        response, trace = client.call(
            SmokeResponse,
            images=[
                group.references[0].image_path,
                group.positive_images[0].image_path,
            ],
            system_prompt=(
                "Image 1 is a REFERENCE and Image 2 is a POSITIVE. Summarize both."
            ),
            user_prompt="Return one concise summary for each image.",
            stage="smoke_vision_json",
            max_output_tokens=512,
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
