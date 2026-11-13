from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "7.0"


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
    image_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)


class InstructionRecord(StrictModel):
    instruction_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    query_category: str
    token_count: int = Field(ge=0)
    length_category: str
    is_question: bool
    is_imperative: bool
    l1_interest: str


class CandidateProposal(StrictModel):
    """One complete hard-negative text proposal for one semantic axis."""

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
    """Object, attribute, and predicate proposals for one fixed edit operation."""

    object: CandidateProposal | None
    attribute: CandidateProposal | None
    predicate: CandidateProposal | None


class ReviewFailureCode(str, Enum):
    NOT_SINGLE_EDIT = "not_single_edit"
    WRONG_SEMANTIC_TARGET = "wrong_semantic_target"
    NOT_FLUENT = "not_fluent"
    NOT_CLOSE_OR_HARD = "not_close_or_hard"
    POSITIVE_COMPATIBLE = "positive_compatible"
    NOT_VISUALLY_VERIFIABLE = "not_visually_verifiable"
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


class GroupReviewResponse(StrictModel):
    """Exactly one verdict for every candidate in a PinPoint group."""

    verdicts: list[CandidateVerdict]


class NegativeCandidate(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    negative_id: str = Field(min_length=1)
    instruction_id: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    text: str = Field(min_length=1)
    semantic_target: SemanticTarget
    edit_operation: EditOperation
    before_span: str | None
    after_span: str | None
    generation_attempts: int = Field(default=1, ge=1)
    status: Literal["generated", "accepted", "rejected", "failed"] = "generated"
    rejection_code: ReviewFailureCode | None = None
    selection_reason: str | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> "NegativeCandidate":
        before = self.before_span
        after = self.after_span
        if before is not None and not before.strip():
            raise ValueError("before_span must be null or non-empty")
        if after is not None and not after.strip():
            raise ValueError("after_span must be null or non-empty")
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


class GroupMetadata(StrictModel):
    stage: Literal["source", "candidates", "final"] = "source"
    operational_status: Literal["pending", "ok", "failed"] = "pending"
    semantic_status: Literal["pending", "accepted", "partial", "rejected"] = "pending"
    num_generated_candidates: int = 0
    num_accepted_negatives: int = 0
    generator_model: str | None = None
    validator_model: str | None = None
    prompt_version: str = "v7"
    build_fingerprint: str | None = None
    selection_fingerprint: str | None = None
    random_seed: int = 42
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validated_positive_image_ids: list[str] = Field(default_factory=list, max_length=8)
    call_traces: list[CallTrace] = Field(default_factory=list)


class PinPointGroupRecord(StrictModel):
    schema_version: Literal["7.0"] = SCHEMA_VERSION
    group_id: str = Field(min_length=1)
    source_dataset: Literal["PinPoint"] = "PinPoint"
    references: list[ImageRef] = Field(min_length=1, max_length=2)
    instructions: list[InstructionRecord] = Field(min_length=2, max_length=6)
    positive_images: list[ImageRef] = Field(min_length=1)
    official_negative_image_ids: list[str] = Field(min_length=1)
    candidates: list[NegativeCandidate] = Field(default_factory=list)
    negatives: list[NegativeCandidate] = Field(default_factory=list)
    metadata: GroupMetadata = Field(default_factory=GroupMetadata)

    @model_validator(mode="after")
    def validate_group(self) -> "PinPointGroupRecord":
        _require_unique([item.image_id for item in self.references], "reference image")
        _require_unique([item.image_id for item in self.positive_images], "positive image")
        _require_unique(self.official_negative_image_ids, "official negative image")
        _require_unique(
            self.metadata.validated_positive_image_ids,
            "validated positive image",
        )
        positive_ids = {item.image_id for item in self.positive_images}
        if any(
            item not in positive_ids
            for item in self.metadata.validated_positive_image_ids
        ):
            raise ValueError(
                "validated positive images must belong to the group's positive set"
            )
        instruction_ids = [item.instruction_id for item in self.instructions]
        _require_unique(instruction_ids, "instruction id")
        _require_unique([item.negative_id for item in self.candidates], "candidate id")
        _require_unique(
            [
                (item.instruction_id, item.edit_operation, item.semantic_target)
                for item in self.candidates
            ],
            "instruction candidate axis",
        )
        if any(item.instruction_id not in instruction_ids for item in self.candidates):
            raise ValueError("every candidate must reference an instruction in its group")
        source_by_id = {item.instruction_id: item.text for item in self.instructions}
        for item in self.candidates:
            if item.source_text != source_by_id[item.instruction_id]:
                raise ValueError("candidate source_text must match its instruction")

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
        raise ValueError(f"{label} values must be unique within a group")
