#!/usr/bin/env python3
"""Build per-Figure/Table retrieval JSON with a local Qwen3.8-VL server."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from visual_recall.evidence import build_evidence, image_quality
from visual_recall.llm_client import QwenOpenAIClient, validate_model_output
from visual_recall.prompts import SYSTEM_PROMPT, build_user_text
from visual_recall.render import render_labeled_image
from visual_recall.schema import VisualEvidence, build_final_record, validate_final_record


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "qwen3.8-27b"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(handle, value: Any) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", value))


def request_signature(request_preview: dict[str, Any]) -> str:
    """Invalidate cached output when prompts, evidence, or visual inputs change."""
    image_inputs: list[dict[str, Any]] = []
    for key in ("target_image", "page_image"):
        raw_path = request_preview.get(key, "")
        path = Path(raw_path) if raw_path else None
        image_inputs.append(
            {
                "key": key,
                "path": raw_path,
                "bytes": path.stat().st_size if path and path.is_file() else None,
                "mtime_ns": path.stat().st_mtime_ns if path and path.is_file() else None,
            }
        )
    signature_payload = {
        "system_prompt": request_preview["system_prompt"],
        "user_text": request_preview["user_text"],
        "image_inputs": image_inputs,
    }
    encoded = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_source_footnote(evidence: VisualEvidence) -> tuple[str, str]:
    """Use an extracted visual source note, otherwise cite report institution."""
    original = evidence.original_footnote.strip()
    if original and re.search(r"\bsource\s*:", original, re.I):
        return original, "original_visual_footnote"
    org = evidence.author_org or "the publishing institution"
    date = evidence.publish_time or "unknown date"
    return f"Source: {org} research report, published {date}.", "institution_report_fallback"


def preliminary_qc(
    evidence: VisualEvidence,
    model_output: dict[str, Any],
    quality: dict[str, Any],
    min_confidence: float,
    *,
    allow_missing_page_image: bool = False,
    allow_existing_table_image: bool = False,
) -> list[str]:
    errors: list[str] = []
    if model_output["decision"] != "keep":
        errors.append(f"model decision is {model_output['decision']}")
    if quality.get("hard_filter_reason"):
        errors.append(str(quality["hard_filter_reason"]))
    if model_output["is_blurry"]:
        errors.append("model judged image blurry")
    if model_output["confidence"] < min_confidence:
        errors.append(f"confidence {model_output['confidence']:.3f} < {min_confidence:.3f}")
    if len(model_output["title"].strip()) < 8:
        errors.append("title is too short")
    if len(model_output["image_content"].strip()) < 40:
        errors.append("image_content is too short")
    for field in ("filter_reason", "title", "image_content"):
        if contains_cjk(str(model_output.get(field, ""))):
            errors.append(f"{field} must be English and contains CJK characters")
    if not evidence.report_title:
        errors.append("report_title missing")
    if not evidence.author_org:
        errors.append("author_org missing")
    if not evidence.publish_time:
        errors.append("publish_time missing")
    if evidence.target_image is None or not evidence.target_image.is_file():
        errors.append("target image missing")
    if not allow_missing_page_image and (evidence.page_image is None or not evidence.page_image.is_file()):
        errors.append("annotated PDF page missing; install PyMuPDF or explicitly allow this fallback")
    if (
        evidence.visual_type == "table"
        and not allow_existing_table_image
        and "PDF renderer unavailable; using existing extracted table image" in evidence.mapping_warnings
    ):
        errors.append("table was not recropped from its PDF bbox")
    return errors


def load_cached_response(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    parsed = value["parsed"]
    errors = validate_model_output(parsed)
    if errors:
        raise ValueError("invalid cached response: " + "; ".join(errors))
    parsed["confidence"] = float(parsed["confidence"])
    return parsed, value.get("raw", ""), value.get("response_metadata", {})


def has_token_usage(response_metadata: dict[str, Any]) -> bool:
    usage = response_metadata.get("usage", {})
    return all(isinstance(usage.get(key), int) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))


def add_token_usage(total: dict[str, int], response_metadata: dict[str, Any]) -> bool:
    """Accumulate standard OpenAI/vLLM token counts; return False when absent."""
    if not has_token_usage(response_metadata):
        return False
    usage = response_metadata["usage"]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] += int(usage[key])
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "dell_visual_example")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name exposed by the vLLM server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--only", default="",
                        help="Comma-separated visual IDs, e.g. main_figure1,main_table4")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build evidence and exact request previews without calling Qwen")
    parser.add_argument("--overwrite", action="store_true", help="Ignore cached model responses and call Qwen again")
    parser.add_argument(
        "--allow-missing-page-image",
        action="store_true",
        help="Allow publishing without the annotated full-page PDF image (not recommended)",
    )
    parser.add_argument(
        "--allow-existing-table-image",
        action="store_true",
        help="Allow an old extracted Table JPG when PDF bbox recropping is unavailable (not recommended)",
    )
    parser.add_argument("--page-dpi", type=int, default=120)
    parser.add_argument("--crop-dpi", type=int, default=220)
    parser.add_argument("--bbox-padding", type=float, default=0.004)
    parser.add_argument("--context-blocks", type=int, default=4)
    parser.add_argument("--nearby-text-items", type=int, default=8)
    parser.add_argument("--nearby-horizontal-margin", type=float, default=0.02)
    parser.add_argument("--nearby-vertical-margin", type=float, default=0.10)
    parser.add_argument("--max-table-chars", type=int, default=40_000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--source-language", default="en",
                        help="Language of the original visual and generated description")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--no-json-mode", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output_dir / "evidence"
    evidences, build_errors = build_evidence(
        args.report_dir,
        evidence_dir,
        context_blocks=args.context_blocks,
        nearby_text_items=args.nearby_text_items,
        nearby_horizontal_margin=args.nearby_horizontal_margin,
        nearby_vertical_margin=args.nearby_vertical_margin,
        page_dpi=args.page_dpi,
        crop_dpi=args.crop_dpi,
        bbox_padding=args.bbox_padding,
    )
    selected_ids = {item.strip() for item in args.only.split(",") if item.strip()}
    if selected_ids:
        evidences = [item for item in evidences if item.visual_id in selected_ids]
        missing_ids = selected_ids - {item.visual_id for item in evidences}
        if missing_ids:
            build_errors.append(f"requested visual IDs not found: {sorted(missing_ids)}")
    if args.max_items > 0:
        evidences = evidences[: args.max_items]

    write_json(
        args.output_dir / "run_manifest.json",
        {
            "report_dir": str(args.report_dir.resolve()),
            "model": args.model,
            "api_base": args.api_base,
            "dry_run": args.dry_run,
            "visual_count": len(evidences),
            "visual_ids": [item.visual_id for item in evidences],
            "build_errors": build_errors,
            "parameters": {
                "page_dpi": args.page_dpi,
                "crop_dpi": args.crop_dpi,
                "bbox_padding": args.bbox_padding,
                "context_blocks": args.context_blocks,
                "nearby_text_items": args.nearby_text_items,
                "nearby_horizontal_margin": args.nearby_horizontal_margin,
                "nearby_vertical_margin": args.nearby_vertical_margin,
                "max_table_chars": args.max_table_chars,
                "min_confidence": args.min_confidence,
                "source_language": args.source_language,
            },
        },
    )

    client = None
    if not args.dry_run:
        client = QwenOpenAIClient(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=not args.no_json_mode,
        )

    requests_path = args.output_dir / "llm_requests.jsonl"
    published_path = args.output_dir / "published_visuals.jsonl"
    audit_path = args.output_dir / "visual_audit.jsonl"
    # Per-item model responses are the resumable cache. Aggregate files describe
    # this invocation only, so always rewrite them instead of accumulating duplicates.
    mode = "w"
    published_count = 0
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_count = 0
    missing_usage_ids: list[str] = []
    api_calls = 0
    cache_hits = 0
    with requests_path.open(mode, encoding="utf-8") as requests_handle, \
         published_path.open(mode, encoding="utf-8") as published_handle, \
         audit_path.open(mode, encoding="utf-8") as audit_handle:
        for evidence in evidences:
            quality = image_quality(evidence.target_image)
            payload = evidence.prompt_payload(args.max_table_chars)
            payload["deterministic_image_quality"] = quality
            request_preview = {
                "id": evidence.global_id,
                "target_image": str(evidence.target_image.resolve()) if evidence.target_image else "",
                "page_image": str(evidence.page_image.resolve()) if evidence.page_image else "",
                "system_prompt": SYSTEM_PROMPT,
                "user_text": build_user_text(payload),
                "payload": payload,
            }
            signature = request_signature(request_preview)
            request_preview["request_signature"] = signature
            append_jsonl(requests_handle, request_preview)
            write_json(args.output_dir / "request_previews" / f"{safe_name(evidence.visual_id)}.json", request_preview)

            if args.dry_run:
                append_jsonl(
                    audit_handle,
                    {
                        "id": evidence.global_id,
                        "stage": "dry_run",
                        "quality": quality,
                        "mapping_warnings": evidence.mapping_warnings,
                    },
                )
                continue
            if quality.get("hard_filter_reason"):
                append_jsonl(
                    audit_handle,
                    {
                        "id": evidence.global_id,
                        "stage": "deterministic_filter",
                        "decision": "drop",
                        "reason": quality["hard_filter_reason"],
                        "quality": quality,
                    },
                )
                continue

            response_path = args.output_dir / "model_responses" / f"{safe_name(evidence.visual_id)}.json"
            try:
                cached = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else None
                cached_metadata = cached.get("response_metadata", {}) if cached else {}
                if (
                    cached
                    and not args.overwrite
                    and cached.get("request_signature") == signature
                    and has_token_usage(cached_metadata)
                ):
                    model_output, raw_response, response_metadata = load_cached_response(response_path)
                    cache_hits += 1
                else:
                    assert client is not None and evidence.target_image is not None
                    model_output, raw_response, response_metadata = client.analyze(
                        payload, evidence.target_image, evidence.page_image
                    )
                    api_calls += 1
                    write_json(
                        response_path,
                        {
                            "request_signature": signature,
                            "parsed": model_output,
                            "raw": raw_response,
                            "response_metadata": response_metadata,
                        },
                    )
            except Exception as exc:
                append_jsonl(
                    audit_handle,
                    {
                        "id": evidence.global_id,
                        "stage": "model_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue

            if add_token_usage(token_totals, response_metadata):
                usage_count += 1
            else:
                missing_usage_ids.append(evidence.global_id)

            qc_errors = preliminary_qc(
                evidence,
                model_output,
                quality,
                args.min_confidence,
                allow_missing_page_image=args.allow_missing_page_image,
                allow_existing_table_image=args.allow_existing_table_image,
            )
            source, source_provenance = select_source_footnote(evidence)
            rendered_path = (
                args.output_dir / "rendered" / evidence.visual_type / f"{safe_name(evidence.visual_id)}.jpg"
            )
            proposed_record = None
            if not qc_errors:
                assert evidence.target_image is not None
                try:
                    render_labeled_image(
                        evidence.target_image,
                        rendered_path,
                        title=model_output["title"],
                        source_footnote=source,
                        font_path=args.font_path,
                        jpeg_quality=args.jpeg_quality,
                    )
                except Exception as exc:
                    qc_errors.append(f"render failed: {type(exc).__name__}: {exc}")

            proposed_record = build_final_record(
                evidence,
                title=model_output["title"],
                image_content=model_output["image_content"],
                image_file=rendered_path,
                source_footnote=source,
                language=args.source_language,
                is_blurry=model_output["is_blurry"],
                publish_ready=not qc_errors,
            )
            if not qc_errors:
                qc_errors.extend(validate_final_record(proposed_record, strict_file=True))
                proposed_record["publish_ready"] = not qc_errors

            audit = {
                "id": evidence.global_id,
                "stage": "completed" if not qc_errors else "quality_rejected",
                "model_output": model_output,
                "quality": quality,
                "qc_errors": qc_errors,
                "source_provenance": source_provenance,
                "mapping_warnings": evidence.mapping_warnings,
                "proposed_record": proposed_record,
            }
            append_jsonl(audit_handle, audit)
            if proposed_record["publish_ready"]:
                append_jsonl(published_handle, proposed_record)
                published_count += 1

    usage_summary = {
        "model": args.model,
        "api_base": args.api_base,
        "selected_visuals": len(evidences),
        "responses_with_usage": usage_count,
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "missing_usage_ids": missing_usage_ids,
        **token_totals,
    }
    write_json(args.output_dir / "token_usage_summary.json", usage_summary)
    run_result = {
        "report_dir": str(args.report_dir),
        "selected_visuals": len(evidences),
        "published": published_count,
        "dry_run": args.dry_run,
        "output_dir": str(args.output_dir.resolve()),
        "build_errors": build_errors,
        "token_usage": usage_summary,
    }
    print(json.dumps(run_result, ensure_ascii=False, indent=2))
    return 0 if not build_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
