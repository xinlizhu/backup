import json
from pathlib import Path

from cir_qdiag.config import PipelineConfig
from cir_qdiag.models.client import StructuredVLMClient
from cir_qdiag.models.fake import FakeVLMBackend
from cir_qdiag.paraphrasing import positive_paraphrase_stage, write_paraphrase_reports
from cir_qdiag.schemas import (
    ImageRef,
    PositiveParaphraseRecord,
    SampleRecord,
)
from cir_qdiag.utils.jsonl import read_models, write_jsonl_atomic


def _config(tmp_path: Path) -> PipelineConfig:
    annotation = tmp_path / "cap.json"
    split = tmp_path / "split.json"
    annotation.write_text("[]", encoding="utf-8")
    split.write_text("{}", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    return PipelineConfig.model_validate(
        {
            "data": {
                "annotation_path": annotation,
                "image_split_path": split,
                "image_root": tmp_path,
            },
            "model": {"name_or_path": model, "max_retries": 0},
            "paraphrase": {
                "num_variants": 3,
                "generation_temperature": 0.3,
                "generation_max_new_tokens": 512,
                "review_temperature": 0.0,
                "review_max_new_tokens": 512,
            },
            "runtime": {
                "output_dir": tmp_path / "out",
                "cache_dir": tmp_path / "cache",
                "prompt_dir": Path(__file__).parents[1] / "prompts",
                "prompt_version": "v5",
                "max_workers": 1,
            },
        }
    )


def _record() -> SampleRecord:
    return SampleRecord(
        sample_id="cirr_val_1",
        pair_id=1,
        reference=ImageRef(image_id="r", image_path="ref.png"),
        target=ImageRef(image_id="t", image_path="target.png"),
        positive_text="Show a red cat running beside a tree.",
    )


def test_positive_paraphrase_stage_only_calls_text_generation_and_review(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "positive_paraphrases.jsonl"
    write_jsonl_atomic(source_path, [_record()])
    backend = FakeVLMBackend(
        [
            {
                "paraphrases": [
                    {
                        "text": "Display a red cat as it runs next to a tree.",
                        "rewrite_strategy": "lexical and clause restructuring",
                    },
                    {
                        "text": "Show a red dog running beside a tree.",
                        "rewrite_strategy": "noun substitution",
                    },
                    {
                        "text": "Beside a tree, show a running red cat.",
                        "rewrite_strategy": "fronted spatial phrase",
                    },
                ]
            },
            {
                "reviews": [
                    {
                        "paraphrase_id": "cirr_val_1_p1",
                        "decision": "equivalent",
                        "reason": "It preserves the red cat, running action, and tree relation.",
                    },
                    {
                        "paraphrase_id": "cirr_val_1_p2",
                        "decision": "not_equivalent",
                        "reason": "It changes the animal from a cat to a dog.",
                    },
                    {
                        "paraphrase_id": "cirr_val_1_p3",
                        "decision": "equivalent",
                        "reason": "It only fronts the same spatial relation.",
                    },
                ]
            },
        ]
    )
    client = StructuredVLMClient(
        backend,
        config.runtime.cache_dir,
        prompt_version="v5",
        max_retries=0,
    )

    positive_paraphrase_stage(
        config,
        source_path,
        output_path,
        client,
        force=True,
    )

    assert [call["stage"] for call in backend.calls] == [
        "generate_positive_paraphrases",
        "review_positive_paraphrases",
    ]
    assert all(call["images"] == [] for call in backend.calls)
    result = list(read_models(output_path, PositiveParaphraseRecord))[0]
    assert result.metadata.num_generated == 3
    assert result.metadata.num_accepted == 2
    assert [item.paraphrase_id for item in result.paraphrases] == [
        "cirr_val_1_p1",
        "cirr_val_1_p3",
    ]
    assert result.candidates[1].status == "rejected"
    assert result.paraphrases[0].metrics.normalized_token_edit_distance > 0

    reports = write_paraphrase_reports(output_path, tmp_path / "reports")
    summary = json.loads(reports["summary"].read_text(encoding="utf-8"))
    assert summary["num_samples"] == 1
    assert summary["num_generated"] == 3
    assert summary["num_accepted"] == 2
    assert reports["metrics"].read_text(encoding="utf-8").count("\n") == 3
