"""Target-local labels and explicit company identities, without model/OCR calls.

Labels are partial transcriptions, not an exhaustive inventory or reconstructed
table. Each item retains its source; native text is clipped without crop padding.
"""
from collections import defaultdict
import re

from .evidence import _clean_text, normalize_bbox
from .reliability import is_data_cell, is_period, table_rows


MAX_LABELS = 80
MAX_LABEL_CHARS = 4000
PAIR = re.compile(r"([^:();\n]+?)\s*\(\s*([A-Z][A-Z0-9.-]{0,9})\s*\)")
NON_COMPANY_CODES = {
    'EPS', 'FCF', 'EBIT', 'EBITDA', 'ROIC', 'ROE', 'ROTE', 'ROTCE', 'PTPP', 'NOI',
    'NAV', 'PSF', 'USD', 'GAAP', 'FY', 'CY', 'YOY', 'QOQ', 'CAGR', 'PE', 'PEG',
    'CEO', 'CFO', 'US', 'UK', 'EU', 'AI', 'IT', 'GDP', 'CPI', 'RSS', 'R&D',
}


def _caption_text(value):
    return re.split(r'\bsource\s*:|\bfootnotes?\s*\(?s?\)?\s*[:(]',
                    _clean_text(value), flags=re.I)[0].strip()


def company_pairs(text):
    """Accept explicit Name (TICKER) pairs, never remembered ticker expansions."""
    for match in PAIR.finditer(_caption_text(text)):
        name, ticker = match.groups()
        name = re.sub(r'^\s*(?:figure|table)\s+\d+\s*[-.:]?\s*', '', name, flags=re.I).strip(' -–—,')
        # Headings may contain several pairs joined by commas/semicolons.
        name = re.sub(r'^\s*(?:and|vs\.?)\s+', '', name).strip()
        if (ticker in NON_COMPANY_CODES or not name or not name[0].isupper()
                or len(name) > 75 or len(name.split()) > 8
                or re.search(r'\b(?:figure|table|revenue|income|earnings|margin|ratio|growth|'
                             r'cash flow|return|estimates?|forecast|per share|cap rate|units?)\b', name, re.I)):
            continue
        yield {'name': name, 'ticker': ticker, 'quote': match.group(0).strip()}


def document_company_index(report, content, identity):
    """Only store explicit mappings; later select those actually in the target."""
    index = defaultdict(list)
    sources = []
    for section in ('main_content', 'appendix_content'):
        for block in (report.get('data') or {}).get(section, []):
            if isinstance(block, dict) and block.get('type') in {'heading', 'image_ref', 'table_ref'}:
                sources.append((_clean_text(block.get('context') or block.get('text')),
                                {'source': 'report_heading_or_caption', 'visual_id': block.get('id')}))
    for i, block in enumerate(content):
        text = (block.get('text') if block.get('type') == 'header' else
                block.get('table_caption') if block.get('type') == 'table' else
                block.get('image_caption') if block.get('type') == 'image' else None)
        if text:
            sources.append((_clean_text(text), {'source': 'layout_heading_or_caption',
                                              'content_index': i, 'page_idx': block.get('page_idx')}))
    for text, location in sources:
        for pair in company_pairs(text):
            value = {**pair, **location}
            if not any(p['name'].casefold() == pair['name'].casefold() for p in index[pair['ticker']]):
                index[pair['ticker']].append(value)
    if identity.get('name') and identity.get('ticker'):
        values = index[identity['ticker']]
        if not any(x['name'].casefold() == identity['name'].casefold() for x in values):
            values.append({'name': identity['name'], 'ticker': identity['ticker'],
                           'source': 'report_identity', 'quote': identity['name'],
                           'evidence': identity.get('evidence')})
    return dict(index)


def native_target_lines(renderer, page_idx, bbox):
    if (getattr(renderer, 'document', None) is None or not isinstance(page_idx, int)
            or not bbox or not 0 <= page_idx < len(renderer.document)):
        return []
    page = renderer.document[page_idx]
    bounds = page.rect
    rect = renderer.fitz.Rect(bounds.x0 + bbox[0] * bounds.width, bounds.y0 + bbox[1] * bounds.height,
                              bounds.x0 + bbox[2] * bounds.width, bounds.y0 + bbox[3] * bounds.height)
    # words avoids decoding embedded raster images. Keep cells separated when a
    # PDF text line spans several widely spaced columns.
    by_line = defaultdict(list)
    for word in page.get_text('words', clip=rect):
        by_line[(word[5], word[6])].append(word)
    lines = []
    for words in by_line.values():
        chunks = [[]]
        for word in sorted(words, key=lambda w: w[0]):
            if chunks[-1] and word[0] - chunks[-1][-1][2] > max(8, (word[3] - word[1]) * 1.5):
                chunks.append([])
            chunks[-1].append(word)
        for chunk in chunks:
            box = [min(w[0] for w in chunk), min(w[1] for w in chunk),
                   max(w[2] for w in chunk), max(w[3] for w in chunk)]
            box = [(box[0] - bounds.x0) / bounds.width, (box[1] - bounds.y0) / bounds.height,
                   (box[2] - bounds.x0) / bounds.width, (box[3] - bounds.y0) / bounds.height]
            lines.append({'text': _clean_text(' '.join(w[4] for w in chunk)),
                          'source': 'pdf_text_inside_target', 'bbox': [round(v, 6) for v in box]})
    return sorted(lines, key=lambda x: (x['bbox'][1], x['bbox'][0]))


def _useful_label(text):
    return (bool(text) and len(text) <= 160 and len(text.split()) <= 22
            and not re.fullmatch(r'[a-z]', text)
            and (is_period(text) or re.fullmatch(r'(?:Q[1-4]|[1-4]Q|FY|CY|20\d{2}\s*\([$€£]\))', text)
                 or (re.search(r'[A-Za-z]', text) and not is_data_cell(text)))
            and not re.match(r'\s*(?:source\s*:|notes?\s*:|copyright|please see)', text, re.I)
            and not re.search(r'@|https?://', text))


def target_labels(renderer, page_idx, bbox, item, visual_id):
    candidates = []
    native_error = None
    try:
        candidates.extend(native_target_lines(renderer, page_idx, bbox))
    except Exception as exc:
        # Missing native text is not proof that an image is missing its labels.
        native_error = type(exc).__name__
    native_count = len(candidates)
    fragments = item.get('_table_fragments') or [dict(item, visual_id=visual_id)]
    for fragment in fragments:
        for row_index, row in enumerate(table_rows(fragment.get('table_body'))):
            for column, text in enumerate(row):
                candidates.append({'text': text, 'source': 'table_html_cell',
                                   'source_visual_id': fragment.get('visual_id', visual_id),
                                   'row': row_index, 'column': column,
                                   'source_bbox': normalize_bbox(fragment.get('bbox'))})
    kept, seen, chars = [], set(), 0
    truncated = False
    for candidate in candidates:
        text = _clean_text(candidate['text'])
        if text in seen or not _useful_label(text):
            continue
        seen.add(text)
        if len(kept) >= MAX_LABELS or chars + len(text) > MAX_LABEL_CHARS:
            truncated = True
            continue
        kept.append({**candidate, 'text': text})
        chars += len(text)
    cautions = []
    if any(re.fullmatch(r'PREV\.?', x['text'], re.I) for x in kept):
        cautions.append({'label': 'PREV', 'status': 'meaning_not_resolved_from_label_alone',
                         'instruction': 'Do not equate PREV with a prior calendar year or historical actuals. '
                         'Keep PREV literal or omit its interpretation unless explicit matching evidence defines it.'})
    return {'items': kept, 'page_idx': page_idx, 'target_bbox': bbox, 'label_cautions': cautions,
            'native_text_available': bool(native_count), 'native_text_error': native_error,
            'truncated': truncated,
            'coverage': 'partial transcription; absence is not evidence that a label is absent from the image'}


def resolve_target_companies(caption, labels, index):
    sources = [{'text': _caption_text(caption), 'source': 'target_caption'}, *labels['items']]
    companies = {}
    conflicts = []
    for source in sources:
        # A parenthesized code in a cell is often a unit, broker, or metric:
        # Shares (K), EBIT (UBS), Op Inc (FIFO), Diff. (A-E). Only a target
        # caption introduces a new identity here. Other target labels resolve
        # through the explicit document mappings below and remain available to
        # the model verbatim even when no name can be resolved.
        if source['source'] != 'target_caption':
            continue
        for pair in company_pairs(source['text']):
            ticker = pair['ticker']
            if ticker in companies and companies[ticker].get('name', '').casefold() != pair['name'].casefold():
                # Caption is stronger than a secondary transcription. Preserve
                # the conflict for review instead of overwriting its identity.
                conflicts.append({'ticker': ticker, 'evidence': pair, 'source': source['source']})
                continue
            companies[ticker] = {**pair, 'source': source['source']}
    for ticker, mappings in index.items():
        if ticker in companies:
            continue
        occurrence = next((s for s in sources if re.search(r'(?<![\w.-])' + re.escape(ticker) + r'(?![\w.-])', s['text'])), None)
        if occurrence is None:
            occurrence = next((s for s in sources if any(re.search(
                r'(?<!\w)' + re.escape(m['name']) + r'(?!\w)', s['text'], re.I) for m in mappings)), None)
        if occurrence is None:
            continue
        # A consistent short/full-name pair is okay. Different names for the
        # same code remain unresolved; never select by popularity.
        ordered = sorted(mappings, key=lambda m: len(m['name']), reverse=True)
        best = ordered[0]
        consistent = all(best['name'].casefold() == m['name'].casefold()
                         or best['name'].casefold().startswith(m['name'].casefold().rstrip('.') + ' ')
                         for m in ordered)
        company = {'ticker': ticker, 'source': 'target_label_with_document_mapping',
                   'target_evidence': occurrence, 'name_evidence': best if consistent else ordered}
        if consistent:
            company['name'] = best['name']
        else:
            company['name_resolution'] = 'ambiguous; retain ticker only'
        companies[ticker] = company
    return list(companies.values()), conflicts


def target_subject_errors(parsed, payload):
    """Catch report-company leakage when other target companies are explicit."""
    if parsed.get('decision') != 'keep':
        return []
    companies = payload.get('target_companies') or []
    report = payload.get('report_identity') or {}
    if not companies or not report.get('ticker') or any(c['ticker'] == report['ticker'] for c in companies):
        return []
    visible_text = ' '.join([_caption_text((payload.get('visual') or {}).get('original_caption')),
                            *[x['text'] for x in (payload.get('target_labels') or {}).get('items', [])]])
    if re.search(r'(?<!\w)' + re.escape(report['ticker']) + r'(?!\w)', visible_text):
        return []
    text = parsed['title'] + ' ' + parsed['image_content']
    name = report.get('name', '')
    name_leak = name and re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', text, re.I)
    # Short tickers can be ordinary words (A, IT, ON). Require explicit ticker
    # notation rather than interpreting those words as company assertions.
    ticker = report['ticker']
    ticker_leak = re.search(r'\(' + re.escape(ticker) + r'\)|\$' + re.escape(ticker) + r'\b', text)
    if name_leak or ticker_leak:
        return ['report_company_leaked_into_other_target_subject']
    return []


def target_label_errors(parsed, payload):
    """Withhold explicit, unsupported PREV -> prior-year assertions.

    This is a narrow check, not a general semantic verifier. Negated statements
    and literal PREV descriptions are allowed; the guard does not define PREV
    as previous estimates or infer an annual comparison from nearby years.
    """
    if parsed.get('decision') != 'keep':
        return []
    labels = (payload.get('target_labels') or {}).get('items', [])
    if not any(re.fullmatch(r'PREV\.?', x['text'], re.I) for x in labels):
        return []
    year = r'(?:prior|previous|last)[- ]year(?:\W?s)?'
    sources = [x['text'] for x in labels] + [
        _caption_text((payload.get('visual') or {}).get('original_caption'))]
    definition = r'\bPREV\.?\s*(?:=|:|means|\()\s*(?:the\s+)?' + year
    if any(re.search(definition, s, re.I) for s in sources):
        return []
    text = parsed['title'] + '. ' + parsed['image_content']
    for sentence in re.split(r'[.!?;]|\bbut\b', text, flags=re.I):
        if not re.search(r'\bPREV\b', sentence, re.I):
            continue
        for match in re.finditer(year, sentence, re.I):
            prefix = sentence[:match.start()]
            if re.search(r'\b(?:not(?!\s+only)|no|never|without)\b[^,;]{0,70}$', prefix, re.I):
                continue
            return ['unsupported_prev_as_prior_year']
    return []
