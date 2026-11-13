"""Conservative local evidence checks; HTML stays local and is never sent to Qwen."""
import re
from html.parser import HTMLParser

from PIL import Image


class TableCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], [], None
    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        if tag in ('td', 'th'):
            self.cell = []
    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split()))
            self.cell = None
        if tag == 'tr':
            self.rows.append(self.row)
            self.row = []


NUMERIC = re.compile(r'^[\s\d.,%$€£()+−–—/\-]+$')
PLACEHOLDERS = {'nm', 'na', 'n/a', 'nc', 'ne', '-', '—', '–', 'fy'}
PERIOD = re.compile(
    r'(?:FY|CY|F)\s*(?:20\d{2}|\d{2})[AE]?|20\d{2}[AE]?|\d{2}[AE]'
    r'|\d{1,2}/(?:20)?\d{2}[AE]?'
    r'|F?[1-4]Q\s*(?:20)?\d{2}[AE]?(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{2})?'
    r'|Q[1-4]\s*(?:20)?\d{2}[AE]?'
    r'|(?:20)?\d{2}\s*Q[1-4][AE]?', re.I)


def table_rows(html):
    parser = TableCells()
    parser.feed(html or '')
    return parser.rows


def is_period(cell):
    return bool(PERIOD.fullmatch(cell.strip()))


def has_metric_labels(rows):
    return any(not (NUMERIC.fullmatch(c) or is_period(c) or c.casefold() in PLACEHOLDERS)
               for row in rows for c in row if c)


def is_period_header(rows):
    cells = [c for row in rows for c in row if c]
    return bool(cells) and len(rows) <= 3 and sum(is_period(c) for c in cells) >= 2 and all(
        is_period(c) or c.casefold() in PLACEHOLDERS for c in cells)


def is_data_cell(cell):
    """Values/periods used to recognize fragments, not to interpret financial data."""
    cell = cell.strip()
    return (bool(NUMERIC.fullmatch(cell)) or is_period(cell)
            or cell.casefold() in PLACEHOLDERS
            or bool(re.fullmatch(r'[+−-]?[\d,.]+\s*(?:bps|bp|x)', cell, re.I))
            or bool(re.fullmatch(r'[+][\d,.]+%?', cell))
            or bool(re.fullmatch(r'[1-4]Q|Q[1-4]', cell, re.I)))


def is_numeric_body(rows):
    """Column headings alone (Actual, UBS Est, etc.) are not metric row labels.

    Require a substantial numeric body below a short header. A real metric column
    makes its rows nonnumeric even when there are many numeric columns beside it.
    """
    def column_heading(cell):
        if is_data_cell(cell):
            return True
        cell = re.sub(r'^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                      r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                      r'[- /]\d{2,4}[AE]?\s*', '', cell.strip(), flags=re.I)
        return (not cell or cell.casefold() in {'actual', 'est', 'estimates', 'cons', 'consensus',
                'mgmt guide', 'management guidance', 'guidance', 'better / (worse)',
                'better / worse', 'percentage', 'percent', 'difference', 'change', 'vs. consensus'}
                or bool(re.fullmatch(r'[A-Z]{2,5}\s+(?:Est|Estimates|Cons|Consensus)', cell)))
    body = [r for r in rows[3:] if any(c.strip() for c in r)]
    return (all(column_heading(c) for r in rows[:3] for c in r if c.strip())
            and len(body) >= 4 and sum(all(is_data_cell(c) for c in r if c.strip())
                                  for r in body) / len(body) >= .9)


def is_label_column(rows):
    cells = [c for r in rows for c in r if c.strip()]
    return (len(cells) >= 4 and all(sum(bool(c.strip()) for c in r) <= 1 for r in rows)
            and all(not is_data_cell(c) and re.search(r'[A-Za-z]', c) for c in cells))


def table_review_reasons(html):
    rows = table_rows(html)
    cells = [c for row in rows for c in row if c]
    if not cells:
        return []
    # Only flag unambiguous numeric-only fragments. Do not infer missing labels from an empty parser result.
    if not has_metric_labels(rows) or is_numeric_body(rows):
        return ['table_fragment_missing_metric_labels']
    # Scope this to a dense financial row-label column, not qualitative tables
    # or a one-column list of companies, products, or operating guidance.
    if (is_label_column(rows) and sum(bool(re.search(
            r'\b(?:revenue|sales|profit|income|EPS|expenses|tax|shares|margin|goods sold)\b',
            c, re.I)) for c in cells) >= 3):
        return ['financial_table_fragment_missing_data_columns']
    # Strongly scoped to multi-period financial statements, not general comparisons or return summaries.
    text = ' '.join(cells).casefold()
    statement = any(x in text for x in ('total assets', 'total liabilities', 'cash flow from', 'cash flows from'))
    has_period = any(is_period(c) for row in rows[:3] for c in row)
    multi_numeric = any(sum(bool(NUMERIC.fullmatch(c)) for c in row if c) >= 3 for row in rows)
    if statement and multi_numeric and not has_period:
        return ['financial_statement_fragment_missing_period_headers']
    return []


def needs_figure_recrop(path, min_long_edge=1000):
    # Resolution proxy, not an OCR/readability score. Re-rendering cannot fix a low-res raster inside a PDF.
    with Image.open(path) as image:
        width, height = image.size
    return max(width, height) < min_long_edge or min(width, height) < 240


def model_review_reasons(parsed):
    status = parsed.get('evidence_status')
    if status != 'sufficient':
        return ['model_evidence_status:' + str(status)]
    return []


def search_description_errors(parsed, forecast_evidence=None):
    """Enforce inexpensive output constraints; this is not a factual verifier."""
    if parsed.get('decision') != 'keep':
        return []
    errors = []
    if len(parsed['title'].split()) > 24:
        errors.append('retrieval_title_exceeds_24_words')
    # Count whitespace-delimited words, including standalone numbers; not model tokens.
    if len(parsed['image_content'].split()) > 200:
        errors.append('retrieval_content_exceeds_200_words')
    text = parsed['title'] + ' ' + parsed['image_content']
    # A metric label (52-week range, Month 13) is not a measured financial/trial value.
    if re.search(r'[$€£]\s*\d|\b\d+(?:\.\d+)?\s*%|\bp\s*[=<>≤≥]\s*\d'
                 r'|\b\d+(?:\.\d+)?\s+(?:billion|million|dollars|percentage points)\b', text, re.I):
        errors.append('retrieval_description_contains_unnecessary_numeric_values')
    # The model can ignore its period-label hint. With that evidence present,
    # withhold a conflicting description instead of publishing or rewriting it.
    if ((forecast_evidence or {}).get('status') == 'periods_without_estimate_labels'
            and re.search(r'\b(?:forecast\w*|project\w*|estimat\w*|future|forward[- ]looking)\b', text, re.I)):
        errors.append('forecast_claim_without_target_estimate_labels')
    return errors
