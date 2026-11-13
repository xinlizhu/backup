from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "5.0"
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


class ConstructionMode(str, Enum):
    TEXT_ONLY = "text_only_editable"
    VISION_GROUNDED = "vision_grounded"


class ImageRef(StrictModel):
    image_id: str
    image_path: str


class CandidateProposal(StrictModel):
    """One complete negative caption produced directly by a generation call."""

    negative_text: str = Field(min_length=1)
    before_span: str | None
    after_span: str | None
    change_summary: str = Field(min_length=1, max_length=400)
    visual_basis: str | None

    @model_validator(mode="after")
    def validate_nonempty_anchors(self) -> "CandidateProposal":
        for field_name in ("before_span", "after_span"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be null or non-empty")
        return self


class GenerationBatchResponse(StrictModel):
    """Three nullable semantic targets for one fixed mode and operation."""

    object: CandidateProposal | None
    attribute: CandidateProposal | None
    predicate: CandidateProposal | None


class CIRRScreenResponse(StrictModel):
    """Keep or remove one source CIRR triplet before negative generation."""

    decision: Literal["keep", "drop"]
    reason: str = Field(min_length=1, max_length=400)


class CandidateReview(StrictModel):
    negative_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=400)


class BatchReviewResponse(StrictModel):
    """A single partition of all generated candidates."""

    selected: list[CandidateReview]
    rejected: list[CandidateReview]


class ParaphraseProposal(StrictModel):
    text: str = Field(min_length=1)
    rewrite_strategy: str = Field(min_length=1, max_length=200)


class ParaphraseGenerationResponse(StrictModel):
    paraphrases: list[ParaphraseProposal] = Field(min_length=1, max_length=20)


class ParaphraseReview(StrictModel):
    paraphrase_id: str = Field(min_length=1)
    decision: Literal["equivalent", "not_equivalent"]
    reason: str = Field(min_length=1, max_length=400)


class ParaphraseReviewResponse(StrictModel):
    reviews: list[ParaphraseReview] = Field(min_length=1, max_length=20)


class ParaphraseMetrics(StrictModel):
    source_char_count: int = Field(ge=1)
    paraphrase_char_count: int = Field(ge=1)
    source_token_count: int = Field(ge=1)
    paraphrase_token_count: int = Field(ge=1)
    length_ratio: float = Field(ge=0.0)
    char_edit_distance: int = Field(ge=0)
    normalized_char_edit_distance: float = Field(ge=0.0, le=1.0)
    token_edit_distance: int = Field(ge=0)
    normalized_token_edit_distance: float = Field(ge=0.0, le=1.0)
    token_jaccard: float = Field(ge=0.0, le=1.0)


class PositiveParaphrase(StrictModel):
    paraphrase_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    rewrite_strategy: str = Field(min_length=1, max_length=200)
    status: Literal["generated", "accepted", "rejected"] = "generated"
    review_reason: str | None = None
    metrics: ParaphraseMetrics


class NegativeCandidate(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    negative_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_target: SemanticTarget
    edit_operation: EditOperation
    construction_mode: ConstructionMode
    before_span: str | None
    after_span: str | None
    change_summary: str = Field(min_length=1, max_length=400)
    visual_basis: str | None = None
    generation_attempts: int = Field(default=1, ge=1)
    status: Literal["generated", "accepted", "rejected", "failed"] = "generated"
    selection_reason: str | None = None

    @model_validator(mode="after")
    def validate_edit_anchors(self) -> "NegativeCandidate":
        before = self.before_span
        after = self.after_span
        if before is not None and not before.strip():
            raise ValueError("before_span must be null or non-empty")
        if after is not None and not after.strip():
            raise ValueError("after_span must be null or non-empty")
        if self.edit_operation == EditOperation.ADD:
            if before is not None or after is None:
                raise ValueError("ADD requires before_span=null and a non-null after_span")
        elif self.edit_operation == EditOperation.DELETE:
            if before is None or after is not None:
                raise ValueError("DELETE requires a non-null before_span and after_span=null")
        elif before is None or after is None:
            raise ValueError("SUBSTITUTE requires non-null before_span and after_span")
        elif before.casefold().strip() == after.casefold().strip():
            raise ValueError("SUBSTITUTE before_span and after_span must differ")
        return self


class CallTrace(StrictModel):
    stage: str
    model_name: str
    model_revision: str | None = None
    temperature: float
    seed: int | None = None
    prompt_version: str = Field(min_length=1)
    prompt_hash: str
    input_hash: str
    raw_response_path: str
    cached: bool = False
    attempts: int = 1
    effective_seed: int | None = None


class PositiveParaphraseMetadata(StrictModel):
    requested_variants: int = Field(ge=1, le=20)
    num_generated: int = Field(default=0, ge=0)
    num_accepted: int = Field(default=0, ge=0)
    model_name: str
    prompt_version: str = Field(min_length=1)
    build_fingerprint: str = Field(min_length=1)
    operational_status: Literal["ok", "failed"] = "ok"
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    call_traces: list[CallTrace] = Field(default_factory=list)


class PositiveParaphraseRecord(StrictModel):
    schema_version: Literal["1.0"] = PARAPHRASE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1)
    pair_id: int
    reference: ImageRef
    target: ImageRef
    positive_text: str = Field(min_length=1)
    candidates: list[PositiveParaphrase] = Field(default_factory=list)
    paraphrases: list[PositiveParaphrase] = Field(default_factory=list)
    metadata: PositiveParaphraseMetadata

    @model_validator(mode="after")
    def validate_paraphrases(self) -> "PositiveParaphraseRecord":
        _require_unique(
            [item.paraphrase_id for item in self.candidates],
            "paraphrase candidate id",
        )
        normalized = [" ".join(item.text.casefold().split()) for item in self.candidates]
        _require_unique(normalized, "paraphrase candidate text")
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


class SampleMetadata(StrictModel):
    stage: Literal["source", "screened", "candidates", "final"] = "source"
    operational_status: Literal["pending", "ok", "failed"] = "pending"
    semantic_status: Literal["pending", "accepted", "partial", "rejected"] = "pending"
    num_generated_candidates: int = 0
    num_accepted_negatives: int = 0
    generator_model: str | None = None
    validator_model: str | None = None
    prompt_version: str = "v5"
    build_fingerprint: str | None = None
    selection_fingerprint: str | None = None
    source_screen_decision: Literal["pending", "keep", "drop"] = "pending"
    source_screen_reason: str | None = None
    random_seed: int = 42
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    call_traces: list[CallTrace] = Field(default_factory=list)


class SampleRecord(StrictModel):
    schema_version: Literal["5.0"] = SCHEMA_VERSION
    sample_id: str
    source_dataset: Literal["CIRR"] = "CIRR"
    split: Literal["train", "val"] = "val"
    pair_id: int
    reference: ImageRef
    target: ImageRef
    positive_text: str = Field(min_length=1)
    image_set: list[str] = Field(default_factory=list)
    candidates: list[NegativeCandidate] = Field(default_factory=list)
    negatives: list[NegativeCandidate] = Field(default_factory=list)
    metadata: SampleMetadata = Field(default_factory=SampleMetadata)

    @model_validator(mode="after")
    def validate_record(self) -> "SampleRecord":
        if self.reference.image_id == self.target.image_id:
            raise ValueError("reference and target must differ")
        _require_unique([item.negative_id for item in self.candidates], "candidate id")
        _require_unique(
            [
                (item.construction_mode, item.edit_operation, item.semantic_target)
                for item in self.candidates
            ],
            "candidate axis",
        )
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


def _require_unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} values must be unique within a sample")
