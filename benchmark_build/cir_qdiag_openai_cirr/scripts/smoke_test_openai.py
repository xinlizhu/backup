from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from pydantic import BaseModel, ConfigDict

from cir_qdiag_openai_cirr.config import load_config
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.pipeline import create_backend


class TextSmokeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only OpenAI gateway smoke test")
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_50.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    client = StructuredLLMClient(
        create_backend(config),
        config.runtime.output_dir / "smoke_cache",
        config.runtime.prompt_version,
        max_retries=0,
    )
    response, trace = client.call(
        TextSmokeResponse,
        images=[],
        system_prompt="Return the requested value.",
        user_prompt="Set value to exactly OK123123OK.",
        stage="smoke_text_json",
        max_output_tokens=128,
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
