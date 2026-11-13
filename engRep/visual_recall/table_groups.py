"""Recover only unambiguous same-page table fragments before a single PDF crop.

HTML is used for structural checks; only extracted text labels reach the model.
"""
from collections import Counter
from html import escape
import re

from .evidence import normalize_bbox, _clean_text
from .reliability import (NUMERIC, has_metric_labels, is_period, is_period_header,
                          is_label_column, is_numeric_body, table_rows)


def _html(rows):
    return '<table>' + ''.join('<tr>' + ''.join('<td>' + escape(c) + '</td>' for c in row)
                              + '</tr>' for row in rows) + '</table>'


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _numeric_columns(rows):
    counts = [sum(bool(NUMERIC.fullmatch(c)) or c.casefold() in {'-', '—', 'nm', 'na', 'n/a'}
                  for c in row[1:] if c) for row in rows if row and not is_period(row[0])]
    counts = [n for n in counts if n >= 2]
    return Counter(counts).most_common(1)[0][0] if counts else 0


def _caption(table):
    return re.split(r'\bsource\s*:', _clean_text(table.get('table_caption')),
                    flags=re.I)[0].strip().casefold()


def _label_column_candidates(tables, rows, boxes, content):
    """Find unique label-column + adjacent numeric-panel chains.

    HTML row counts may differ. Geometry supplies row-band alignment; captions
    and intervening layout blocks bound the table. No cell values are zipped.
    """
    candidates = []
    for left, lr in enumerate(rows):
        a = boxes[left]
        if not a or not is_label_column(lr) or not _caption(tables[left]):
            continue
        members = [left]
        edge = a[2]
        ambiguous = False
        while True:
            adjacent = []
            for right, rr in enumerate(rows):
                b = boxes[right]
                if (right in members or not b or tables[left].get('page_idx') != tables[right].get('page_idx')
                        or not 0 <= b[0] - edge <= .06):
                    continue
                # A fragment may lack the left panel's units/title rows, but its
                # data band must cover nearly the same vertical interval.
                tolerance = min(.025, .12 * min(a[3] - a[1], b[3] - b[1]))
                if abs(a[1] - b[1]) <= tolerance and abs(a[3] - b[3]) <= min(.01, tolerance):
                    adjacent.append(right)
            if not adjacent:
                break
            nearest_x = min(boxes[j][0] for j in adjacent)
            nearest = [j for j in adjacent if abs(boxes[j][0] - nearest_x) < .002]
            if len(nearest) != 1:
                ambiguous = True
                break
            right = nearest[0]
            if not is_numeric_body(rows[right]) or (_caption(tables[right]) and
                                                    _caption(tables[right]) != _caption(tables[left])):
                break
            members.append(right)
            edge = boxes[right][2]
        if ambiguous or len(members) < 2:
            continue
        union = _union([boxes[j] for j in members])
        member_boxes = [boxes[j] for j in members]
        obstructed = False
        for block in content or tables:
            b = normalize_bbox(block.get('bbox'))
            if block.get('page_idx') != tables[left].get('page_idx') or not b or b in member_boxes:
                continue
            overlaps = min(b[2], union[2]) > max(b[0], union[0]) and min(b[3], union[3]) > max(b[1], union[1])
            inside_member = any(m[0] <= b[0] and m[1] <= b[1] and b[2] <= m[2] and b[3] <= m[3]
                                for m in member_boxes)
            if (overlaps and not inside_member and (block.get('type') in {'image', 'table', 'heading'}
                    or re.match(r'(?:figure|table)\s+\d', _clean_text(block.get('text')), re.I)
                    or len(_clean_text(block.get('text')).split()) > 15)):
                obstructed = True
                break
        if not obstructed:
            candidates.append(members)
    # Two possible label columns for the same data are not safe to resolve by order.
    return [m for m in candidates if all(sum(j in other for other in candidates) == 1 for j in m)]


def group_tables(refs, tables, *, content=()):
    """Return per-source groups, keyed by source ID, with a stable data-block ID.

    Ambiguous matches stay separate. Equal counts alone are not merge evidence.
    """
    if len(refs) != len(tables):
        return {}
    rows = [table_rows(t.get('table_body')) for t in tables]
    boxes = [normalize_bbox(t.get('bbox')) for t in tables]
    groups = {i: {'anchor': i, 'members': [i], 'rows': rows[i], 'method': 'single_table'}
              for i in range(len(tables))}
    consumed = set()

    for members in _label_column_candidates(tables, rows, boxes, content):
        anchor = members[0]
        groups[anchor] = {'anchor': anchor, 'members': members,
                          'rows': [r for j in members for r in rows[j]],
                          'method': 'label_column_with_numeric_panels'}
        consumed.update(members[1:])

    def same_page(i, j):
        return boxes[i] and boxes[j] and tables[i].get('page_idx') == tables[j].get('page_idx')

    # A labels+historical-values left half and a numeric-only forecast right half.
    candidates = []
    for right, rr in enumerate(rows):
        if right in consumed or len(groups[right]['members']) > 1 or not rr or has_metric_labels(rr) or is_period_header(rr):
            continue
        for left, lr in enumerate(rows):
            if left in consumed or len(groups[left]['members']) > 1:
                continue
            if not same_page(left, right) or not has_metric_labels(lr) or len(lr) != len(rr):
                continue
            left_caption = re.split(r'\bsource\s*:', _clean_text(tables[left].get('table_caption')), flags=re.I)[0].strip().casefold()
            right_caption = re.split(r'\bsource\s*:', _clean_text(tables[right].get('table_caption')), flags=re.I)[0].strip().casefold()
            if right_caption and right_caption != left_caption:
                continue
            a, b = boxes[left], boxes[right]
            tolerance = min(.012, .12 * min(a[3] - a[1], b[3] - b[1]))
            if (0 <= b[0] - a[2] <= .06 and abs(a[1] - b[1]) <= tolerance
                    and abs(a[3] - b[3]) <= tolerance
                    and any(is_period(c) for c in lr[0])
                    and any(is_period(c) for c in rr[0])):
                candidates.append((left, right))
    for left, right in candidates:
        if sum(left == a for a, b in candidates) != 1 or sum(right == b for a, b in candidates) != 1:
            continue
        groups[left] = {'anchor': left, 'members': [left, right],
                        'rows': [a + b for a, b in zip(rows[left], rows[right])],
                        'method': 'horizontal_fragments'}
        consumed.add(right)

    # A detached year header directly above a same-width financial statement.
    candidates = []
    for header, hr in enumerate(rows):
        if header in consumed or not is_period_header(hr) or not boxes[header] or boxes[header][3] - boxes[header][1] > .06:
            continue
        periods = max(sum(is_period(c) for c in row) for row in hr)
        for body, br in enumerate(rows):
            if body in consumed or header == body or not same_page(header, body) or not has_metric_labels(br):
                continue
            if any(is_period(c) for row in br[:3] for c in row):
                continue
            h, b = boxes[header], boxes[body]
            if (0 <= b[1] - h[3] <= .03 and abs(h[2] - b[2]) <= .015
                    and b[0] <= h[0] < b[2] and _numeric_columns(br) == periods):
                candidates.append((header, body))
    for header, body in candidates:
        if sum(header == a for a, b in candidates) != 1 or sum(body == b for a, b in candidates) != 1:
            continue
        if header in consumed or body in consumed or len(groups[body]['members']) > 1:
            continue
        members = [header, body]
        # Contiguous assets / liabilities sections may share that single year header.
        balance = any('assets' in c.casefold() for row in rows[body] for c in row)
        if balance:
            continuations = []
            for j, jr in enumerate(rows):
                if j in consumed or j in members or not same_page(body, j):
                    continue
                a, b = boxes[body], boxes[j]
                if (0 <= b[1] - a[3] <= .015 and abs(a[0] - b[0]) <= .015
                        and abs(a[2] - b[2]) <= .015 and _numeric_columns(jr) == _numeric_columns(rows[body])
                        and not any(is_period(c) for row in jr[:3] for c in row)
                        and any('liabilities' in c.casefold() for row in jr for c in row)):
                    continuations.append(j)
            if len(continuations) == 1:
                members += continuations
                consumed.add(continuations[0])
        groups[body] = {'anchor': body, 'members': members,
                        'rows': [row for member in members for row in rows[member]],
                        'method': 'detached_period_header'}
        consumed.add(header)

    result = {}
    for i, group in groups.items():
        if i in consumed:
            continue
        members = group['members']
        item = dict(tables[i])
        if len(members) > 1:
            item['bbox'] = _union([boxes[j] for j in members])
            item['table_body'] = _html(group['rows'])
            # Preserve each original HTML separately: unequal row counts must
            # never invent label/value associations in downstream evidence.
            item['_table_fragments'] = [dict(tables[j], visual_id=refs[j]['id']) for j in members]
            for field in ('table_caption', 'table_footnote'):
                item[field] = list(dict.fromkeys(_clean_text(tables[j].get(field)) for j in members
                                                if _clean_text(tables[j].get(field))))
        value = {'anchor_id': refs[i]['id'], 'source_visual_ids': [refs[j]['id'] for j in members],
                 'source_bboxes': [boxes[j] for j in members],
                 'item': item, 'method': group['method']}
        for j in members:
            result[refs[j]['id']] = value
    return result


def recover_row_labels(item, content):
    """Extend a numeric matrix to a dense, aligned left label column, if unique.

    This handles labels extracted as text blocks rather than as another table.
    Short blocks must cover most of the data height; nearby prose is insufficient.
    """
    rows = table_rows(item.get('table_body'))
    box = normalize_bbox(item.get('bbox'))
    data = [r for r in rows if len(r) >= 3 and sum(bool(NUMERIC.fullmatch(c)) for c in r if c) >= 3]
    if not box or len(data) < 4 or sum(bool(NUMERIC.fullmatch(r[0])) for r in data) / len(data) < .8:
        return item, []
    labels = []
    for index, block in enumerate(content):
        b = normalize_bbox(block.get('bbox'))
        text = _clean_text(block.get('text'))
        if (block.get('page_idx') == item.get('page_idx') and block.get('type') == 'text'
                and b and text and len(text.split()) <= 12 and any(c.isalpha() for c in text)
                and box[1] <= b[1] < b[3] <= box[3] + .005
                and 0 <= box[0] - b[2] <= .25 and b[3] - b[1] <= .035):
            labels.append({'content_index': index, 'bbox': b, 'text': text})
    if len(labels) < 4:
        return item, []
    if max(x['bbox'][0] for x in labels) - min(x['bbox'][0] for x in labels) > .04:
        return item, []
    extent = _union([x['bbox'] for x in labels])
    if (extent[3] - extent[1]) / (box[3] - box[1]) < .65:
        return item, []
    expanded = _union([box, extent])
    for block in content:
        b = normalize_bbox(block.get('bbox'))
        if (block.get('page_idx') != item.get('page_idx') or block.get('type') not in {'table', 'image'}
                or not b or b == box):
            continue
        if min(b[2], expanded[2]) > max(b[0], expanded[0]) and min(b[3], expanded[3]) > max(b[1], expanded[1]):
            return item, []
    return {**item, 'bbox': expanded}, labels
