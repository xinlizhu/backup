from __future__ import annotations

from pathlib import Path

from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.schemas import CIRRScreenResponse, CallTrace, SampleRecord
from cir_qdiag.utils.prompts import dump_prompt_json, load_prompt


def screen_cirr_triplet(
    record: SampleRecord,
    client: StructuredVLMClient,
    prompt_dir: str | Path,
    prompt_version: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[CIRRScreenResponse, CallTrace]:
    """Decide whether a CIRR triplet is usable before generating negatives."""

    prompt, _ = load_prompt(prompt_dir, "screen_cirr", prompt_version)
    return client.call(
        CIRRScreenResponse,
        images=[record.reference.image_path, record.target.image_path],
        system_prompt=prompt,
        user_prompt=dump_prompt_json(
            {
                "sample_id": record.sample_id,
                "positive_text": record.positive_text,
            }
        ),
        stage="screen_cirr",
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
