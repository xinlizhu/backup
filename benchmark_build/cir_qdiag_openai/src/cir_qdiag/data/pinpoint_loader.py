from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cir_qdiag.schemas import (
    GroupMetadata,
    ImageRef,
    InstructionRecord,
    PinPointGroupRecord,
)


def load_exact_pinpoint_groups(
    parquet_path: str | Path,
    image_root: str | Path,
    *,
    min_group_size: int = 2,
    max_group_size: int = 6,
) -> list[PinPointGroupRecord]:
    """Group PinPoint rows with identical references and exact positive/negative sets."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("install pandas and pyarrow to read PinPoint parquet") from exc

    frame = pd.read_parquet(parquet_path)
    required = {
        "query_id",
        "query_image_signature",
        "query_image_signature2",
        "instruction",
        "positive_candidates",
        "negative_candidates",
        "token_count",
        "length_category",
        "is_question",
        "is_imperative",
        "query_category",
        "l1_interest",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("PinPoint parquet is missing columns: " + ", ".join(missing))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        reference2 = _optional_signature(row["query_image_signature2"])
        positives = _clean_signatures(row["positive_candidates"])
        negatives = _clean_signatures(row["negative_candidates"])
        key = (
            str(row["query_image_signature"]),
            reference2,
            positives,
            negatives,
        )
        grouped[key].append(row)

    root = Path(image_root).resolve()
    records: list[PinPointGroupRecord] = []
    for key, rows in grouped.items():
        if not min_group_size <= len(rows) <= max_group_size:
            continue
        reference1, reference2, positives, negatives = key
        rows.sort(key=lambda item: _query_sort_key(str(item["query_id"])))
        instruction_ids = [str(item["query_id"]) for item in rows]
        group_id = _group_id(key, instruction_ids)
        references = [_image_ref(reference1, root)]
        if reference2 is not None:
            references.append(_image_ref(reference2, root))
        instructions = [
            InstructionRecord(
                instruction_id=str(item["query_id"]),
                text=" ".join(str(item["instruction"]).split()),
                query_category=str(item["query_category"]),
                token_count=int(item["token_count"]),
                length_category=str(item["length_category"]),
                is_question=bool(item["is_question"]),
                is_imperative=bool(item["is_imperative"]),
                l1_interest=str(item["l1_interest"]),
            )
            for item in rows
        ]
        records.append(
            PinPointGroupRecord(
                group_id=group_id,
                references=references,
                instructions=instructions,
                positive_images=[_image_ref(item, root) for item in positives],
                official_negative_image_ids=list(negatives),
                metadata=GroupMetadata(),
            )
        )
    records.sort(
        key=lambda record: min(
            _query_sort_key(item.instruction_id) for item in record.instructions
        )
    )
    return records


def _clean_signatures(values: Any) -> tuple[str, ...]:
    cleaned = {
        str(item)
        for item in values
        if isinstance(item, str) and item.strip() and item != "None"
    }
    return tuple(sorted(cleaned))


def _optional_signature(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text == "None" else text


def _image_ref(signature: str, root: Path) -> ImageRef:
    return ImageRef(
        image_id=signature,
        image_path=str(root / signature[:2] / f"{signature}.jpg"),
    )


def _group_id(key: tuple[Any, ...], instruction_ids: list[str]) -> str:
    payload = json.dumps([key, instruction_ids], ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"pinpoint_group_{digest}"


def _query_sort_key(query_id: str) -> tuple[int, str]:
    suffix = query_id.rsplit("_", 1)[-1]
    return (int(suffix), query_id) if suffix.isdigit() else (10**12, query_id)
