"""Bounded report prose around a visual, with explicit selection provenance."""
import re

from .evidence import _clean_text


CONTEXT_LIMITS = {
    'paragraphs_per_direction': 2,
    'max_preceding_visuals': 4,
    'max_scanned_blocks': 16,
    'max_chars_per_direction': 3000,
}
CONTEXT_RULE = ('before: skip at most four leading visual refs, then up to two paragraphs; '
                'after: stop at any visual; both: stop at heading, omit administrative headings/contact lines; '
                'at most sixteen blocks and 3000 text characters per direction')

# These are boundaries, not a reason to skip farther into another section.
ADMIN_HEADING = re.compile(
    r'^(?:related\s+(?:research|reports?|publications?)'
    r'|(?:important\s+)?disclosures?'
    r'|(?:legal\s+)?disclaimers?'
    r'|analyst\s+(?:certifications?|disclosures?|contacts?)'
    r'|other\s+companies\s+mentioned\s+in\s+this\s+report)\b', re.I)
AUTHOR_LINE = re.compile(r'.{1,120}\|\s*(?:Equity|Research)\s+(?:Analyst|Associate)\s*\**', re.I)


def select_context(sequence, position, before, limit=2):
    """Bridge only the initial upstream run of visuals, never a second run.

There is no semantic relevance claim here: the selected prose may describe a
group of visuals. The payload marks that distinction for the model and audits.
"""
    result, skipped = [], []
    paragraphs = chars = scanned = 0
    truncated = False
    stop = 'sequence_boundary'
    indexes = range(position - 1, -1, -1) if before else range(position + 1, len(sequence))
    for index in indexes:
        if paragraphs >= limit:
            stop = 'paragraph_limit'
            break
        if scanned >= CONTEXT_LIMITS['max_scanned_blocks']:
            stop = 'scan_limit'
            break
        scanned += 1
        block = sequence[index]
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if kind in {'image_ref', 'table_ref'}:
            if before and not result:
                if len(skipped) < CONTEXT_LIMITS['max_preceding_visuals']:
                    skipped.append(block.get('id', ''))
                    continue
                stop = 'visual_skip_limit'
            else:
                stop = 'visual_boundary'
            break
        if kind not in {'heading', 'paragraph'}:
            # An unsupported nonempty block is not a transparent part of a run.
            if _clean_text(block.get('text') or block.get('context')):
                stop = 'unsupported_block'
                break
            continue
        text = _clean_text(block.get('text'))
        if kind == 'heading' and ADMIN_HEADING.match(text):
            stop = 'administrative_heading'
            break
        if kind == 'paragraph' and (AUTHOR_LINE.fullmatch(text)
                                    or re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', text)):
            stop = 'contact_line'
            break
        if text:
            remaining = CONTEXT_LIMITS['max_chars_per_direction'] - chars
            selected = {'id': block.get('id', ''), 'type': kind, 'text': text}
            if len(text) > remaining:
                prefix = text[:remaining]
                # Avoid splitting an English word. The flag makes omitted text explicit.
                selected['text'] = prefix.rsplit(' ', 1)[0] if ' ' in prefix else prefix
                selected['truncated'] = True
                truncated = True
            result.append(selected)
            chars += len(selected['text'])
            if truncated or chars >= CONTEXT_LIMITS['max_chars_per_direction']:
                stop = 'character_limit'
                break
            if kind == 'paragraph':
                paragraphs += 1
        if kind == 'heading':
            stop = 'heading_boundary'
            break
        if paragraphs >= limit:
            stop = 'paragraph_limit'
            break
    if before:
        result.reverse()
        skipped.reverse()
    info = {
        'source': ('preceding_prose_across_visuals' if skipped else 'adjacent_text') if result else 'none',
        'skipped_visual_ids': skipped,
        'stop_reason': stop,
        'truncated': truncated,
    }
    return result, info


def neighboring_blocks(sequence, position, before, limit=2):
    """List-only interface retained for callers that do not need provenance."""
    return select_context(sequence, position, before, limit)[0]
