from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "8.0"
PARAPHRASE_SCHEMA_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SemanticTarget(str, Enum):
    OBJECT = "object"
    ATTRIBUTE = "attribute"
    PREDICATE = "predicate"


class EditOperation(str, Enum):
    ADD = "add"
    DELETE = "delete"
    SUBSTITUTE = "substitute"


class ImageRef(StrictModel):
    """CIRR image metadata used for visual validation and downstream training."""

    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)


class CandidateProposal(StrictModel):
    """One complete text-only counterfactual for one semantic axis."""

    negative_text: str = Field(min_length=1)
    before_span: str | None
    after_span: str | None

    @model_validator(mode="after")
    def validate_nonempty_anchors(self) -> "CandidateProposal":
        for field_name in ("before_span", "after_span"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be null or non-empty")
        return self


class GenerationBatchResponse(StrictModel):
    """Object, attribute, and predicate proposals for one fixed operation."""

    object: CandidateProposal | None
    attribute: CandidateProposal | None
    predicate: CandidateProposal | None


class ReviewFailureCode(str, Enum):
    NOT_SINGLE_EDIT = "not_single_edit"
    WRONG_SEMANTIC_TARGET = "wrong_semantic_target"
    NOT_FLUENT = "not_fluent"
    NOT_CLOSE_OR_HARD = "not_close_or_hard"
    NOT_COUNTERFACTUAL = "not_counterfactual"
    INVALID_DELETE = "invalid_delete"
    OTHER_SEMANTIC_FAILURE = "other_semantic_failure"


class CandidateVerdict(StrictModel):
    negative_id: str = Field(min_length=1)
    decision: Literal["accept", "reject"]
    failure_code: ReviewFailureCode | None
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_failure_code(self) -> "CandidateVerdict":
        if self.decision == "accept" and self.failure_code is not None:
            raise ValueError("accepted verdicts require failure_code=null")
        if self.decision == "reject" and self.failure_code is None:
            raise ValueError("rejected verdicts require a failure_code")
        return self


class SampleReviewResponse(StrictModel):
    """Exactly one multimodal verdict for every candidate in a CIRR sample."""

    verdicts: list[CandidateVerdict]


class ParaphraseProposal(StrictModel):
    text: str = Field(min_length=1)
    rewrite_strategy: str = Field(min_length=1, max_length=200)


class ParaphraseGenerationResponse(StrictModel):
    paraphrases: list[ParaphraseProposal]


class ParaphraseReview(StrictModel):
    paraphrase_id: str = Field(min_length=1)
    decision: Literal["equivalent", "not_equivalent"]
    reason: str = Field(min_length=1, max_length=300)


class ParaphraseReviewResponse(StrictModel):
    reviews: list[ParaphraseReview]


class ParaphraseMetrics(StrictModel):
    source_char_count: int = Field(ge=1)
    paraphrase_char_count: int = Field(ge=1)
    source_token_count: int = Field(ge=1)
    paraphrase_token_count: int = Field(ge=1)
    length_ratio: float = Field(gt=0)
    char_edit_distance: int = Field(ge=0)
    normalized_char_edit_distance: float = Field(ge=0, le=1)
    token_edit_distance: int = Field(ge=0)
    normalized_token_edit_distance: float = Field(ge=0, le=1)
    token_jaccard: float = Field(ge=0, le=1)


class PositiveParaphrase(StrictModel):
    paraphrase_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    rewrite_strategy: str = Field(min_length=1)
    metrics: ParaphraseMetrics
    status: Literal["generated", "accepted", "rejected"] = "generated"
    review_reason: str | None = None


class NegativeCandidate(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    negative_id: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_target: SemanticTarget
    edit_operation: EditOperation
    construction_mode: Literal["text_only_editable"] = "text_only_editable"
    before_span: str | None
    after_span: str | None
    generation_attempts: int = Field(default=1, ge=1)
    status: Literal["generated", "accepted", "rejected", "failed"] = "generated"
    rejection_code: ReviewFailureCode | None = None
    selection_reason: str | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "NegativeCandidate":
        for field_name in ("before_span", "after_span"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be null or non-empty")
        if self.status == "accepted" and self.rejection_code is not None:
            raise ValueError("accepted candidates require rejection_code=null")
        if self.status == "rejected" and self.rejection_code is None:
            raise ValueError("rejected candidates require a rejection_code")
        return self


class CallTrace(StrictModel):
    stage: str
    model_name: str
    model_revision: str | None = None
    provider: str = "openai"
    reasoning_effort: str | None = None
    prompt_version: str = Field(min_length=1)
    prompt_hash: str
    input_hash: str
    raw_response_path: str
    cached: bool = False
    attempts: int = 1
    request_id: str | None = None
    usage: dict[str, Any] | None = None


class SampleMetadata(StrictModel):
    stage: Literal["source", "candidates", "final"] = "source"
    operational_status: Literal["pending", "ok", "failed"] = "pending"
    semantic_status: Literal["pending", "accepted", "partial", "rejected"] = "pending"
    validation_mode: Literal["reference_target_visual"] = "reference_target_visual"
    num_generated_candidates: int = 0
    num_accepted_negatives: int = 0
    generator_model: str | None = None
    validator_model: str | None = None
    prompt_version: str = "v8"
    build_fingerprint: str | None = None
    selection_fingerprint: str | None = None
    random_seed: int = 42
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    call_traces: list[CallTrace] = Field(default_factory=list)


class PositiveParaphraseMetadata(StrictModel):
    requested_variants: int = Field(ge=1, le=20)
    num_generated: int = Field(default=0, ge=0)
    num_accepted: int = Field(default=0, ge=0)
    generator_model: str = Field(min_length=1)
    reviewer_model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    build_fingerprint: str = Field(min_length=1)
    operational_status: Literal["ok", "failed"] = "ok"
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    call_traces: list[CallTrace] = Field(default_factory=list)


class PositiveParaphraseRecord(StrictModel):
    schema_version: Literal["1.0"] = PARAPHRASE_SCHEMA_VERSION
    sample_id: str = Field(pattern=r"^cirr_train_\d+$")
    pair_id: int = Field(ge=0)
    reference: ImageRef
    target: ImageRef
    positive_text: str = Field(min_length=1)
    candidates: list[PositiveParaphrase] = Field(default_factory=list)
    paraphrases: list[PositiveParaphrase] = Field(default_factory=list)
    metadata: PositiveParaphraseMetadata

    @model_validator(mode="after")
    def validate_paraphrases(self) -> "PositiveParaphraseRecord":
        if self.sample_id != f"cirr_train_{self.pair_id}":
            raise ValueError("sample_id must equal cirr_train_<pair_id>")
        _require_unique(
            [item.paraphrase_id for item in self.candidates],
            "paraphrase candidate id",
        )
        _require_unique(
            [" ".join(item.text.casefold().split()) for item in self.candidates],
            "paraphrase candidate text",
        )
        if self.metadata.num_generated != len(self.candidates):
            raise ValueError("generated paraphrase count does not match candidates")
        accepted = [item for item in self.candidates if item.status == "accepted"]
        if [item.model_dump(mode="json") for item in accepted] != [
            item.model_dump(mode="json") for item in self.paraphrases
        ]:
            raise ValueError("paraphrases must contain all and only accepted candidates")
        if self.metadata.num_accepted != len(self.paraphrases):
            raise ValueError("accepted paraphrase count does not match paraphrases")
        return self


class CIRRSampleRecord(StrictModel):
    schema_version: Literal["8.0"] = SCHEMA_VERSION
    sample_id: str = Field(pattern=r"^cirr_train_\d+$")
    source_dataset: Literal["CIRR"] = "CIRR"
    split: Literal["train"] = "train"
    pair_id: int = Field(ge=0)
    reference: ImageRef
    target: ImageRef
    positive_text: str = Field(min_length=1)
    image_set: list[str] = Field(default_factory=list)
    candidates: list[NegativeCandidate] = Field(default_factory=list)
    negatives: list[NegativeCandidate] = Field(default_factory=list)
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)

    @model_validator(mode="after")
    def validate_sample(self) -> "CIRRSampleRecord":
        if self.sample_id != f"cirr_train_{self.pair_id}":
            raise ValueError("sample_id must equal cirr_train_<pair_id>")
        if self.reference.image_id == self.target.image_id:
            raise ValueError("reference and target must differ")
        _require_unique(self.image_set, "image_set member")
        _require_unique([item.negative_id for item in self.candidates], "candidate id")
        _require_unique(
            [
                (item.edit_operation, item.semantic_target)
                for item in self.candidates
            ],
            "candidate taxonomy cell",
        )
        for candidate in self.candidates:
            if candidate.source_text != self.positive_text:
                raise ValueError("candidate source_text must match positive_text")

        _require_unique([item.negative_id for item in self.negatives], "negative id")
        candidate_by_id = {item.negative_id: item for item in self.candidates}
        for negative in self.negatives:
            candidate = candidate_by_id.get(negative.negative_id)
            if candidate is None:
                raise ValueError("every final negative must also appear in candidates")
            if negative.status != "accepted":
                raise ValueError("only accepted candidates may appear in negatives")
            if negative.model_dump(mode="json") != candidate.model_dump(mode="json"):
                raise ValueError("final negatives must exactly match their candidates")

        if self.metadata.stage in {"candidates", "final"}:
            if self.metadata.num_generated_candidates != len(self.candidates):
                raise ValueError("generated candidate count does not match candidates")
        if self.metadata.stage == "final":
            accepted = [item for item in self.candidates if item.status == "accepted"]
            if [item.negative_id for item in accepted] != [
                item.negative_id for item in self.negatives
            ]:
                raise ValueError("negatives must contain all and only accepted candidates")
            if self.metadata.num_accepted_negatives != len(self.negatives):
                raise ValueError("accepted negative count does not match negatives")
        return self


class BackendResult(StrictModel):
    data: dict[str, Any]
    raw_response: dict[str, Any] | str
    model_name: str
    model_revision: str | None = None
    request_id: str | None = None
    provider: str = "openai"
    reasoning_effort: str | None = None
    usage: dict[str, Any] | None = None


def _require_unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique within a sample")
