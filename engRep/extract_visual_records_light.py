#!/usr/bin/env python3
"""Build original-image + English-text records, using one image per Qwen request."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from extract_visual_records import preliminary_qc, select_source_footnote
from visual_recall.evidence import image_quality
from visual_recall.light import PROFILE, SYSTEM_PROMPT, CONTEXT_RULE, CONTEXT_LIMITS, build_light_evidence
from visual_recall.llm_client import QwenOpenAIClient, validate_retrieval_output
from visual_recall.reliability import model_review_reasons, search_description_errors
from visual_recall.target_evidence import target_subject_errors, target_label_errors
from visual_recall.schema import build_final_record, validate_final_record

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False) + '\n')
    handle.flush()


def run(args):
    started = time.monotonic()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = {x.strip() for x in args.only.split(',') if x.strip()}
    built = time.monotonic()
    entries, errors, selected = build_light_evidence(args.report_dir.resolve(), out / 'evidence',
        only=only, max_items=args.max_items, crop_dpi=args.crop_dpi, padding=args.bbox_padding)
    preprocess_seconds = time.monotonic() - built
    manifest = {'profile': PROFILE, 'report_dir': str(args.report_dir.resolve()),
                'selected_visuals': selected, 'built_visuals': len(entries), 'build_errors': errors,
                'model': args.model, 'api_base': args.api_base, 'dry_run': args.dry_run,
                'context_rule': CONTEXT_RULE, 'context_limits': CONTEXT_LIMITS,
                'crop_dpi': args.crop_dpi, 'bbox_padding': args.bbox_padding,
                'table_groups': [{'id': e.global_id, 'source_visual_ids': e.asset_info['source_visual_ids'],
                                  'method': e.asset_info['table_grouping'], 'bbox': e.bbox, 'page_idx': e.page_idx}
                                 for e, _, _ in entries if len(e.asset_info.get('source_visual_ids', [])) > 1]}
    write_json(out / 'run_manifest.json', manifest)
    client = None
    counters = {'published': 0, 'cache_hits': 0, 'api_attempts': 0, 'model_errors': 0,
                'quality_rejected': 0, 'build_error_count': len(errors)}
    tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    usage_missing = []
    with (out / 'published_visuals.jsonl').open('w') as pub, (out / 'visual_audit.jsonl').open('w') as audit:
        for err in errors:
            emit(audit, {'stage': 'build_error', **err})
        for evidence, payload, method in entries:
            item_start = time.monotonic()
            quality = image_quality(evidence.target_image)
            payload['image_quality'] = {**quality, 'readable': quality.get('readable', False),
                                      'readability_note': 'readable means file decodes; it does not certify legible text'}
            user_text = ('Write a concise English search title and a grounded English description of about 100 words, allowing up to 200 when useful. Evidence:\n'
                         + json.dumps(payload, ensure_ascii=False)
                         + '\n\nAim for about 100 words in image_content; up to 200 is acceptable when needed to cover supported topics and comparisons. '
                         'Do not pad a simple visual with repetition or invented details. Omit observed values and outcomes. '
                         'Use the target company evidence and printed target labels. Report identity describes the document only. '
                         'Do not add the report company to a different target or expand unsupported company names. '
                         'Respect forecast_evidence and keep printed metric acronyms without invented definitions.')
            image_path = evidence.target_image
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            signature_data = {'system': SYSTEM_PROMPT, 'user': user_text, 'image_sha256': image_hash,
                              'model': args.model, 'api_base': args.api_base, 'temperature': args.temperature,
                              'max_tokens': args.max_tokens, 'json_mode': not args.no_json_mode}
            signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
            name = hashlib.sha256(evidence.global_id.encode()).hexdigest()[:24]
            preview = {'id': evidence.global_id, 'target_image': str(image_path), 'image_count': 1,
                       'system_prompt': SYSTEM_PROMPT, 'user_text': user_text, 'payload': payload,
                       'asset_info': evidence.asset_info, 'mapping_method': method, 'mapping_warnings': evidence.mapping_warnings, 'response_format': 'json_schema' if not args.no_json_mode else 'unconstrained', 'request_signature': signature}
            write_json(out / 'request_previews' / f'{name}.json', preview)
            if args.dry_run:
                emit(audit, {'id': evidence.global_id, 'stage': 'dry_run', 'preview_file': f'request_previews/{name}.json'})
                continue
            if payload.get('evidence_review_reasons'):
                counters['quality_rejected'] += 1
                emit(audit, {'id': evidence.global_id, 'stage': 'quality_rejected', 'decision': 'review',
                             'qc_errors': payload['evidence_review_reasons'], 'asset_info': evidence.asset_info,
                             'mapping_warnings': evidence.mapping_warnings, 'api_attempts': 0})
                continue
            if quality.get('hard_filter_reason'):
                counters['quality_rejected'] += 1
                emit(audit, {'id': evidence.global_id, 'stage': 'quality_rejected', 'reason': quality['hard_filter_reason']})
                continue
            cache_path = out / 'model_responses' / f'{name}.json'
            cached = None
            if cache_path.is_file() and not args.overwrite:
                try:
                    value = json.loads(cache_path.read_text())
                    if value.get('request_signature') == signature and not validate_retrieval_output(value['parsed']):
                        cached = value
                except (ValueError, KeyError, TypeError):
                    pass
            request_seconds = 0.0
            attempts = 0
            try:
                if cached:
                    parsed, metadata = cached['parsed'], cached.get('response_metadata', {})
                    counters['cache_hits'] += 1
                else:
                    if client is None:
                        client = QwenOpenAIClient(api_base=args.api_base, api_key=args.api_key, model=args.model,
                            temperature=args.temperature, max_tokens=args.max_tokens, json_mode=not args.no_json_mode)
                        # Keep retries bounded here instead of multiplying SDK retries.
                        client.client = client.client.with_options(max_retries=0, timeout=args.timeout)
                    req_start = time.monotonic()
                    for attempt in range(args.retries + 1):
                        counters['api_attempts'] += 1
                        attempts += 1
                        try:
                            parsed, raw, metadata = client.analyze(payload, image_path, None,
                                system_prompt=SYSTEM_PROMPT, user_text=user_text, allow_json_fallback=False, strict_schema=True, retrieval_checks=True)
                            break
                        except Exception as exc:
                            write_json(out / 'failed_attempts' / f'{name}_{time.time_ns()}.json', {
                                'id': evidence.global_id, 'attempt': attempt + 1, 'request_signature': signature,
                                'error': f'{type(exc).__name__}: {exc}', 'raw': getattr(exc, 'raw', None),
                                'response_metadata': getattr(exc, 'metadata', {})})
                            if attempt == args.retries:
                                raise
                            time.sleep(min(2 ** attempt, 8))
                    request_seconds = time.monotonic() - req_start
                    write_json(cache_path, {'request_signature': signature, 'parsed': parsed, 'raw': raw,
                                           'response_metadata': metadata, 'request_seconds': request_seconds})
                usage = metadata.get('usage') or {}
                if all(isinstance(usage.get(k), int) for k in tokens):
                    for k in tokens:
                        tokens[k] += usage[k]
                else:
                    usage_missing.append(evidence.global_id)
                schema_errors = validate_retrieval_output(parsed)
                if schema_errors:
                    raise ValueError('; '.join(schema_errors))
                qc = preliminary_qc(evidence, parsed, quality, args.min_confidence, allow_missing_page_image=True)
                qc.extend(model_review_reasons(parsed))
                qc.extend(search_description_errors(parsed, payload.get('forecast_evidence')))
                qc.extend(target_subject_errors(parsed, payload))
                qc.extend(target_label_errors(parsed, payload))
                source, provenance = select_source_footnote(evidence)
                record = build_final_record(evidence, title=parsed['title'], image_content=parsed['image_content'],
                    image_file=image_path, source_footnote=source, language='en', is_blurry=parsed['is_blurry'], publish_ready=not qc)
                if not qc:
                    qc.extend(validate_final_record(record, allow_unknown_figure_page=True))
                record['publish_ready'] = not qc
                emit(audit, {'id': evidence.global_id, 'stage': 'quality_rejected' if qc else 'completed',
                             'qc_errors': qc, 'model_output': parsed, 'asset_info': evidence.asset_info, 'mapping_method': method, 'mapping_warnings': evidence.mapping_warnings,
                             'source_provenance': provenance, 'proposed_record': record,
                             'cache_hit': bool(cached), 'api_attempts': attempts, 'request_seconds': request_seconds,
                             'item_seconds': time.monotonic() - item_start})
                if qc:
                    counters['quality_rejected'] += 1
                else:
                    emit(pub, record)
                    counters['published'] += 1
            except Exception as exc:
                counters['model_errors'] += 1
                emit(audit, {'id': evidence.global_id, 'stage': 'model_error', 'error': f'{type(exc).__name__}: {exc}',
                             'api_attempts': attempts, 'item_seconds': time.monotonic() - item_start})
    result = {**manifest, **counters, 'preprocess_seconds': preprocess_seconds,
              'elapsed_seconds': time.monotonic() - started, 'output_dir': str(out),
              'token_usage_including_cache': tokens, 'missing_usage_ids': usage_missing}
    write_json(out / 'run_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 2 if errors or counters['model_errors'] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'dell_visual_light')
    p.add_argument('--model', default='qwen3.8-27b')
    p.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    p.add_argument('--api-key', default='EMPTY')
    p.add_argument('--only', default='')
    p.add_argument('--max-items', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--crop-dpi', type=int, default=220)
    p.add_argument('--bbox-padding', type=float, default=.004)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--max-tokens', type=int, default=1800)
    p.add_argument('--min-confidence', type=float, default=.75)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--no-json-mode', action='store_true')
    args = p.parse_args()
    if args.max_items < 0 or args.retries < 0 or args.crop_dpi <= 0 or args.timeout <= 0:
        p.error('counts must be nonnegative; crop-dpi and timeout must be positive')
    if not 0 <= args.min_confidence <= 1 or not 0 <= args.bbox_padding <= .1:
        p.error('invalid confidence or bbox padding')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
