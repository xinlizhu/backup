from __future__ import annotations

import json
from pathlib import Path

from cir_qdiag_openai_cirr.config import PipelineConfig
from cir_qdiag_openai_cirr.models.client import StructuredLLMClient
from cir_qdiag_openai_cirr.models.fake import FakeLLMBackend
from cir_qdiag_openai_cirr.paraphrasing import (
    positive_paraphrase_stage,
    write_paraphrase_reports,
)
from cir_qdiag_openai_cirr.schemas import (
    CIRRSampleRecord,
    PositiveParaphraseRecord,
)
from cir_qdiag_openai_cirr.utils.jsonl import read_models, write_jsonl_atomic


PROMPTS = Path(__file__).parents[1] / "prompts"


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "data": {
                "annotation_path": tmp_path / "annotations.json",
                "image_split_path": tmp_path / "split.json",
                "image_root": tmp_path,
            },
            "model": {},
            "paraphrase": {
                "num_variants": 2,
                "generation_max_output_tokens": 256,
                "review_max_output_tokens": 256,
                "max_workers": 1,
            },
            "runtime": {
                "output_dir": tmp_path / "out",
                "cache_dir": tmp_path / "cache",
                "prompt_dir": PROMPTS,
                "prompt_version": "v8",
                "max_workers": 1,
            },
        }
    )


def test_openai_paraphrase_generation_review_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = tmp_path / "source.jsonl"
    output = tmp_path / "positive_paraphrases.jsonl"
    write_jsonl_atomic(
        source,
        [
            CIRRSampleRecord(
                sample_id="cirr_train_7",
                pair_id=7,
                reference={"image_id": "r", "image_path": "r.png"},
                target={"image_id": "t", "image_path": "t.png"},
                positive_text="Make the red dog run",
            )
        ],
    )
    backend = FakeLLMBackend(
        [
            {
                "paraphrases": [
                    {
                        "text": "Have the red dog run",
                        "rewrite_strategy": "Imperative rewording",
                    },
                    {
                        "text": "Make the dog that is red run",
                        "rewrite_strategy": "Relative-clause restructuring",
                    },
                ]
            },
            {
                "reviews": [
                    {
                        "paraphrase_id": "cirr_train_7_p1",
                        "decision": "equivalent",
                        "reason": "Preserves the red dog and running action.",
                    },
                    {
                        "paraphrase_id": "cirr_train_7_p2",
                        "decision": "not_equivalent",
                        "reason": "Awkward scope makes the instruction unreliable.",
                    },
                ]
            },
        ]
    )
    client = StructuredLLMClient(
        backend,
        config.runtime.cache_dir,
        "v8",
        max_retries=0,
    )

    positive_paraphrase_stage(
        config,
        source,
        output,
        client,
        num_variants=2,
    )
    record = list(read_models(output, PositiveParaphraseRecord))[0]
    assert record.metadata.num_generated == 2
    assert record.metadata.num_accepted == 1
    assert [item.text for item in record.paraphrases] == ["Have the red dog run"]
    assert [call["stage"] for call in backend.calls] == [
        "generate_positive_paraphrases",
        "review_positive_paraphrases",
    ]
    assert all(call["images"] == [] for call in backend.calls)

    positive_paraphrase_stage(
        config,
        source,
        output,
        client,
        num_variants=2,
    )
    assert len(backend.calls) == 2

    reports = write_paraphrase_reports(output, tmp_path / "reports")
    summary = json.loads(reports["summary"].read_text())
    assert summary["num_samples"] == 1
    assert summary["accepted_paraphrases"] == 1
