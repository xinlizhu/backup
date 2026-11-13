"""Small, source-grounded hints for names and estimate labels; no extra model calls."""
from collections import Counter
import re

from .evidence import _clean_text


def report_identity(report, content):
    ticker = _clean_text(report.get('security_code')).split('.')[0]
    if not re.fullmatch(r'[A-Za-z0-9-]{1,12}', ticker):
        return {}
    texts = [_clean_text(x.get('text')) for x in content if x.get('type') == 'header']
    texts += [_clean_text(x.get('text') or x.get('context'))
              for section in ('main_content', 'appendix_content')
              for x in (report.get('data') or {}).get(section, [])
              if isinstance(x, dict) and x.get('type') in {'heading', 'image_ref', 'table_ref'}]
    candidates = []
    pattern = re.compile(r'([^:()\n]+?)\s*\(\s*' + re.escape(ticker) + r'(?:\.[A-Z]+)?\s*\)', re.I)
    for text in texts:
        match = pattern.search(text)
        if match:
            name = match[1].strip()
            if (len(name.split()) <= 7 and len(name) <= 75
                    and not re.search(r'\b(figure|table|rating|history|source|estimates)\b', name, re.I)):
                candidates.append(name)
    if candidates:
        name = Counter(candidates).most_common(1)[0][0]
        return {'name': name, 'ticker': ticker, 'evidence': 'explicit_name_ticker_pair_in_report_heading_or_caption'}
    # A short legal company name also present in the report title is unambiguous.
    title = _clean_text(report.get('raw_title')).casefold()
    for text in texts:
        if (text.casefold() in title and len(text.split()) <= 6
                and re.fullmatch(r"[A-Za-z0-9&.,' -]+\b(?:Inc|Corp|Corporation|Ltd|plc|SA)\.?", text, re.I)):
            return {'name': text, 'ticker': ticker, 'evidence': 'company_heading_repeated_in_report_title'}
    return {'ticker': ticker, 'evidence': 'report_security_code_only; do not guess a full company name'}


def figure_forecast_evidence(renderer, page_idx, bbox):
    result = {'status': 'not_available'}
    if getattr(renderer, 'document', None) is None or page_idx is None or not bbox:
        return result
    page, rect = renderer._page_and_rect(page_idx, bbox)
    if page is None:
        return result
    text = page.get_text(clip=rect)
    # Restrict to financial fiscal-period labels, not axis amounts or source notes.
    periods = re.findall(r'\b(?:F?[1-4]Q\d{2,4}|FY\d{2,4}|F20\d{2})(?:[AE])?\b', text, re.I)
    if len(set(periods)) >= 3:
        result = {'status': 'explicit_estimate_labels' if any(p.upper().endswith('E') for p in periods)
                  else 'periods_without_estimate_labels', 'source': 'native_text_inside_target_bbox',
                  'period_labels': list(dict.fromkeys(periods))}
    return result
