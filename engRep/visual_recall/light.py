"""Single-image evidence with bounded paragraph context; no page images or HTML."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .evidence import IMAGE_SUFFIXES, PdfAssetRenderer, _clean_text, _find_pdf, normalize_bbox
from .schema import VisualEvidence
from .reliability import needs_figure_recrop, table_review_reasons
from .table_groups import group_tables, recover_row_labels
from .search_evidence import report_identity, figure_forecast_evidence
from .target_evidence import document_company_index, target_labels, resolve_target_companies
from .context import CONTEXT_RULE, CONTEXT_LIMITS, neighboring_blocks, select_context

PROFILE = 'search_keywords_v9_target_evidence_max200'

SYSTEM_PROMPT = '''Write an English search title and an approximately 100-word English description for ONE financial research Figure or Table.
Users search with short topics: company + business/segment + metric + forecast/comparison, not exact data values. Describe what can be found in this visual, not what its pixels look like.
TITLE: a natural English phrase, usually 6-16 words. Name the target company/subject and the main business/metric. Include a ticker only when its identity is supported. Prefer a useful metric title over "financial table", "key data", or "chart".
IMAGE_CONTENT: aim for approximately 100 English words, usually in 4-6 sentences. Up to 200 words is acceptable when useful to cover the supported information. Start with the subject and describe the named segments/products, searchable metric labels, supported comparison relationships, and scope of the visual. Do not repeat the same point or invent details to reach the target length; a simple visual with limited evidence may need fewer words. Preserve useful printed ratios such as P/E, EV/EBITDA, P/CEPS rather than replacing them with verbose definitions. For broad statements, include the main metric families and important named items (e.g. cash, debt, capex, free cash flow) ONLY when actually present. Describe what is covered, with no interpretation of the numerical results. Mention historical data, forecasts, guidance, or consensus comparisons only when explicit target labels support them. A source line saying "estimates" alone does not make the displayed periods forecasts.
GROUNDING: the target image and its own caption identify the subject. Report metadata describes the report, not necessarily this visual. If the visual names another company, NEVER attach the report's ticker or company to it. When no target subject is visible and there is no conflicting subject, report metadata may identify the report's main company. Nearby text can clarify a label but must not add metrics, AI themes, products, forecasts, or causal claims absent from the target. A report mentioning AI does not make every revenue chart an AI revenue chart. The publication date is not a data period.
CONTEXT: context_selection records how nearby prose was selected. preceding_prose_across_visuals is shared background found before a run of other visuals, not a caption for this target. Use it to clarify visible labels only when the subject, metric, and period agree with the target; otherwise ignore it. Never import the intervening visuals' subjects or metrics, or fill missing rows/columns from prose. An ambiguous label such as PREV must not be expanded to prior-year without supporting evidence. If its meaning remains unclear, omit that comparison qualifier. A block marked truncated is only an excerpt, not the full paragraph.
TARGET EVIDENCE: target_labels is a partial transcription from this target's PDF region and table cells. Use it to verify printed spellings, metric acronyms, period labels, and comparison headings; never invent definitions. Missing labels in this partial list do not prove absence from the image. Compare conflicting transcriptions with the image; do not blend their spellings. HTML row/column indices belong to individual source fragments and must not be joined across fragments to infer values. The image remains necessary to establish relationships. Preserve PREV without an inferred meaning unless matching evidence explicitly clarifies it.
target_companies is a partial list of company identities supported by the target caption or target labels matched to a quoted Name (TICKER) pair in the document. Keep other printed peer tickers too, without guessing their names. Use exact supported names; do not add the report company to another company's target. Multiple entries may be peers, not subsidiaries. A name_resolution marked ambiguous permits the ticker only. If target_company_conflicts is nonempty and the subject cannot be resolved from the target, request review. report_identity is background about the document, not an instruction to name that company. If target_companies is empty, identify the subject from the image/caption; use report_identity only when the target clearly concerns the report's main company. Never expand company codes from memory.
The ONLY permitted acronym definitions are EPS = earnings per share, FCF = free cash flow, capex = capital expenditures, P/E = price-to-earnings, YoY/Y/Y = year-over-year. Keep all other metric acronyms exactly as printed, including RSS, P/CEPS, EBITDA, EBIT, ROIC, NCFB, MAC-LD. Company and segment names may be taken from their own labels/caption. No keyword lists, speculative synonyms, analyst contacts, or meta phrases like "supports searches".
OMIT numerical observations: revenue amounts, percentages, prices, price ranges, p-values, confidence intervals and sample sizes. Omit calendar ranges and incidental event dates. Numbers integral to a printed metric/endpoint label may remain, e.g. 52-week range, 12-month rating, 3-year EPS CAGR, Month 13 endpoint, or a product name such as H100. These are labels, not observed results. Also omit colors, fonts, legends, axis ranges and layout narration. Do not explain causes, infer trends, report trial significance/winners, or explain how ratios are calculated. Describe covered metrics and comparisons, not conclusions about the data.
If forecast_evidence.status is periods_without_estimate_labels, do NOT call that visual forecast/projected/estimated, even if its source credits an analyst's estimates. Use neutral "quarterly data". If explicit_estimate_labels, forecast wording is supported. If not_available, use forecast wording only when clearly labeled in the target itself.
EVIDENCE: sufficient means the subject and metric labels support a useful topical description. Unreadable tiny numbers or missing exact years alone do not require rejection if you omit those details. Do not describe a time range/trend or current-versus-prior comparison when period labels are missing. Missing metric labels, an isolated numeric fragment/header, conflicting subjects, or unreadable subject/metric text require review. Use evidence_status missing_period_labels only when the missing period prevents identifying the topic itself. Missing labels cannot be invented from the report title.
If evidence_status is not sufficient, decision MUST be review and give a brief grounded reason. is_blurry is true only when blur prevents identifying the topic/key labels. Keep informative financial/business/trial visuals, including company-specific trading/rating snapshots and price/rating histories. Drop logos, decorations, isolated page furniture, aggregate broker rating-distribution/definition tables, and purely legal disclosure lists.
All generated text must be English.
Return exactly one JSON object, no Markdown, with all fields:
{"decision":"keep|drop|review","category":"chart|table|diagram|map|photo|screenshot|logo|icon|decoration|other","filter_reason":"","title":"","image_content":"","is_blurry":false,"confidence":0.0,"evidence_status":"sufficient|missing_metric_labels|missing_period_labels|unreadable_key_text|conflicting_evidence"}
Choose one enum value; do not output pipe-separated alternatives literally. confidence must be numeric between 0 and 1.
Style example for a detailed financial table (do not copy the placeholder company or any metrics absent from the target): title="ExampleCo financial forecasts, segment performance, and valuation metrics"; image_content="ExampleCo financial summary covering revenue, gross profit, operating income, and earnings per share. Quarterly and annual columns separate reported results from analyst estimates and show the associated comparison periods. Business segment rows include server and networking revenue, storage revenue, and the operating income attributable to each segment. The cash flow section covers operating cash flow, capital expenditures, and free cash flow, alongside balance sheet items such as cash and debt. Additional rows report gross margin, operating margin, and valuation ratios including P/E and EV/EBITDA. Consensus estimates appear beside the analyst forecasts for the financial measures identified in the table."
FINAL CHECK: aim for about 100 words in image_content, with no more than 200 words. Title and image_content must contain grounded topics only, no numerical observations, no invented company/acronym definitions, and no statements about colors, shapes or statistical outcomes. A price-target chart can be described as "Company share price and analyst target history" without writing any prices.'''


def collect_refs(data: dict) -> list[dict]:
    refs = []
    for section in ('main_content', 'appendix_content'):
        seq = data.get(section) or []
        if not isinstance(seq, list):
            raise ValueError(f'{section} must be a list')
        for i, b in enumerate(seq):
            if isinstance(b, dict) and b.get('type') in {'image_ref', 'table_ref'} and b.get('id'):
                before, before_info = select_context(seq, i, True)
                after, after_info = select_context(seq, i, False)
                refs.append({**b, 'before': before, 'after': after,
                             'context_selection': {'before': before_info, 'after': after_info}})
    return refs


def figure_mapping(report_dir: Path, refs: list[dict], items: list[dict]) -> tuple[dict, dict]:
    """Match filtered named assets to layout via original path/hash, bytes, or unique caption.

    Never zip figure refs to layout images: upstream filtering changes that order.
    Ambiguous/conflicting identities are rejected, even if counts happen to agree.
    """
    images = report_dir / 'images'
    files = sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if images.is_dir() else []
    mapping, errors, choices = {}, {}, []
    digests = {}
    def digest(path):
        if path not in digests:
            digests[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digests[path]
    def caption_key(value):
        text = _clean_text(value)
        text = re.split(r"\bsource\s*:", text, flags=re.I)[0]
        return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))
    for ref in refs:
        vid = ref['id']
        named = [p for p in files if p.stem == vid or p.stem.startswith(vid + '_')]
        if len(named) != 1:
            errors[vid] = f'expected one filtered named image, found {len(named)}'
            continue
        target = named[0]
        identity = set()
        for i, item in enumerate(items):
            raw = item.get('img_path') or ''
            if not raw:
                continue
            source = report_dir / raw
            stem = Path(raw).stem
            if source == target or (len(stem) == 64 and stem in target.stem):
                identity.add(i)
            elif source.is_file() and source.stat().st_size == target.stat().st_size and digest(source) == digest(target):
                identity.add(i)
        caption = caption_key(ref.get('context'))
        captions = {i for i, x in enumerate(items) if caption and caption_key(x.get('image_caption')) == caption}
        full_key = lambda value: ' '.join(re.findall(r'[a-z0-9]+', _clean_text(value).casefold()))
        full = full_key(ref.get('context'))
        combined = {i for i, x in enumerate(items) if full and full_key(
            _clean_text(x.get('image_caption')) + ' ' + _clean_text(x.get('image_footnote'))) == full}
        captions = combined or captions
        candidates = identity if identity else captions
        if identity and captions and not identity.intersection(captions):
            errors[vid] = 'image identity and caption disagree'
        elif not candidates:
            errors[vid] = 'no image-identity or normalized-caption layout match'
        else:
            choices.append((vid, target, sorted(candidates), bool(identity)))
    # Resolve repeated captions only when their order has a unique monotonic alignment.
    # Extra unreferenced layout images are allowed; no unconditional ordinal fallback.
    earliest, previous = [], -1
    for _, _, candidates, _ in choices:
        valid = [i for i in candidates if i > previous]
        previous = min(valid) if valid else len(items)
        earliest.append(previous)
    latest, following = [], len(items)
    for _, _, candidates, _ in reversed(choices):
        valid = [i for i in candidates if i < following]
        following = max(valid) if valid else -1
        latest.append(following)
    latest.reverse()
    valid_alignment = all(0 <= lo <= hi < len(items) for lo, hi in zip(earliest, latest))
    for (vid, target, candidates, identity), lo, hi in zip(choices, earliest, latest):
        if not valid_alignment or lo != hi:
            errors[vid] = 'ambiguous or conflicting caption/identity sequence alignment'
        else:
            method = 'image_identity' if identity else ('unique_caption' if len(candidates) == 1 else 'caption_sequence_alignment')
            mapping[vid] = (items[lo], target, method)
    return mapping, errors


def build_light_evidence(report_dir: Path, output_dir: Path, *, only: set[str] | None = None,
                         max_items: int = 0, crop_dpi: int = 220, padding: float = .004):
    report_id = report_dir.name
    report = json.loads((report_dir / f'{report_id}.json').read_text())
    content = json.loads((report_dir / f'{report_id}_content_list.json').read_text())
    if not isinstance(content, list) or any(not isinstance(x, dict) for x in content):
        raise ValueError('content_list must be a flat list of objects')
    refs = collect_refs(report.get('data') or {})
    ids = [r['id'] for r in refs]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate visual IDs')
    selected = [r for r in refs if not only or r['id'] in only]
    if max_items:
        selected = selected[:max_items]
    errors = [{'visual_id': v, 'error': 'requested visual ID not found'} for v in sorted((only or set()) - set(ids))]
    figures, map_errors = figure_mapping(report_dir, [r for r in refs if r['type'] == 'image_ref'] if any(r['type'] == 'image_ref' for r in selected) else [],
                                         [x for x in content if x.get('type') == 'image'])
    table_refs = [r for r in refs if r['type'] == 'table_ref']
    tables = [x for x in content if x.get('type') == 'table']
    table_map = group_tables(table_refs, tables, content=content)
    refs_by_id = {r['id']: r for r in refs}
    identity = report_identity(report, content)
    company_index = document_company_index(report, content, identity)
    renderer = None
    result = []
    processed = set()
    try:
        for ref in selected:
            vid = ref['id']
            try:
                kind = 'figure' if ref['type'] == 'image_ref' else 'table'
                group = table_map.get(vid) if kind == 'table' else None
                if group:
                    vid = group['anchor_id']
                    ref = refs_by_id[vid]
                if vid in processed:
                    continue
                processed.add(vid)
                warnings = []
                review_reasons = []
                evidence_limits = []
                forecast_evidence = {'status': 'not_available'}
                asset_info = {}
                if kind == 'figure':
                    if vid in figures:
                        item, target, method = figures[vid]
                    else:
                        named = [p for p in (report_dir / 'images').iterdir() if p.is_file()
                                 and p.suffix.lower() in IMAGE_SUFFIXES
                                 and (p.stem == vid or p.stem.startswith(vid + '_'))]
                        if len(named) != 1:
                            raise ValueError(f'expected one filtered named image, found {len(named)}')
                        item, target, method = {}, named[0], 'named_image_without_layout'
                        warnings = [map_errors.get(vid, 'figure mapping missing')]
                else:
                    if vid not in table_map:
                        raise ValueError(f'table mapping count mismatch: refs={len(table_refs)}, layout={len(tables)}')
                    item, recovered_labels = recover_row_labels(group['item'], content)
                    method = 'table_order_equal_counts' if group['method'] == 'single_table' else group['method']
                    asset_info = {'source_visual_ids': group['source_visual_ids'], 'table_grouping': group['method'],
                                  'source_bboxes': group['source_bboxes']}
                    if recovered_labels:
                        method = 'recovered_row_label_column'
                        asset_info['recovered_label_blocks'] = recovered_labels
                        asset_info['original_bbox'] = normalize_bbox(group['item'].get('bbox'))
                        asset_info['expanded_bbox'] = normalize_bbox(item['bbox'])
                page, bbox = item.get('page_idx'), normalize_bbox(item.get('bbox'))
                if method != 'named_image_without_layout' and (not isinstance(page, int) or page < 0 or not bbox):
                    raise ValueError('invalid page_idx or bbox')
                if kind == 'table':
                    if renderer is None:
                        renderer = PdfAssetRenderer(_find_pdf(report_dir), output_dir, 120, crop_dpi, padding)
                    if renderer.error:
                        raise ValueError(renderer.error)
                    # Stable filenames based on reference position avoid unsafe IDs and overwrite ambiguity.
                    target = renderer.render_crop(page, bbox, output_dir / 'targets' / f'table_{ids.index(vid):06d}.jpg')
                    if target is None:
                        raise ValueError('PDF bbox crop failed')
                if kind == 'table':
                    for reason in table_review_reasons(item.get('table_body') or ''):
                        if reason == 'table_fragment_missing_metric_labels' and recovered_labels:
                            continue
                        if reason == 'financial_statement_fragment_missing_period_headers':
                            # Search topics can remain grounded without exact periods.
                            evidence_limits.append(reason + ': omit periods, trends, and inter-period comparisons')
                        else:
                            review_reasons.append(reason)
                    asset_info['analysis_image'] = str(target)
                else:
                    original_target = target
                    asset_info = {'original_image': str(original_target), 'recropped': False}
                    if needs_figure_recrop(target):
                        asset_info['recrop_reason'] = 'resolution_proxy'
                        if page is None or not bbox:
                            review_reasons.append('low_resolution_figure_without_reliable_pdf_bbox')
                        else:
                            try:
                                if renderer is None:
                                    renderer = PdfAssetRenderer(_find_pdf(report_dir), output_dir, 120, crop_dpi, padding)
                                if renderer.error:
                                    raise ValueError(renderer.error)
                                crop = renderer.render_crop(page, bbox, output_dir / 'targets' / f'figure_{ids.index(vid):06d}.jpg', dpi=max(300, crop_dpi))
                                if crop is None:
                                    raise ValueError('figure recrop failed')
                                target = crop
                                asset_info['recropped'] = True
                                asset_info['crop_dpi'] = max(300, crop_dpi)
                            except Exception as exc:
                                warnings.append(f'figure recrop failed: {exc}')
                                review_reasons.append('low_resolution_figure_recrop_failed')
                    asset_info['analysis_image'] = str(target)
                    try:
                        if renderer is None and page is not None and bbox:
                            renderer = PdfAssetRenderer(_find_pdf(report_dir), output_dir, 120, crop_dpi, padding)
                        forecast_evidence = figure_forecast_evidence(renderer, page, bbox)
                    except Exception as exc:
                        warnings.append(f'native forecast-label check unavailable: {type(exc).__name__}')
                caption = _clean_text(ref.get('context')) or _clean_text(item.get(f'{"image" if kind == "figure" else "table"}_caption'))
                footnote = _clean_text(item.get(f'{"image" if kind == "figure" else "table"}_footnote'))
                labels = target_labels(renderer, page, bbox, item, vid)
                companies, company_conflicts = resolve_target_companies(caption, labels, company_index)
                evidence = VisualEvidence(
                    report_id, vid, kind,
                    _clean_text(report.get('raw_title') or (report.get('data') or {}).get('title')),
                    _clean_text(report.get('institution')),
                    _clean_text(report.get('raw_date') or report.get('release_timestamp'))[:10],
                    _clean_text(report.get('security_code')), page, bbox, caption, footnote, '',
                    [b['text'] for b in ref['before']], [b['text'] for b in ref['after']],
                    [], [], '', target, None, warnings)
                payload = {
                    'report_metadata': {k: getattr(evidence, k) for k in ('report_id','report_title','author_org','publish_time','security_code')},
                    'report_identity': identity,
                    'target_companies': companies,
                    'target_company_conflicts': company_conflicts,
                    'target_labels': labels,
                    'visual': {'visual_id': vid, 'type': kind, 'original_caption': caption, 'original_footnote': footnote},
                    'context_before': ref['before'], 'context_after': ref['after'],
                    'context_selection': ref['context_selection'],
                    'evidence_review_reasons': review_reasons,
                    'evidence_limits': evidence_limits,
                    'forecast_evidence': forecast_evidence,
                }
                evidence.asset_info = asset_info
                result.append((evidence, payload, method))
            except Exception as exc:
                errors.append({'visual_id': vid, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if renderer:
            renderer.close()
    return result, errors, len(selected)
