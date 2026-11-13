import unittest

from visual_recall.context import CONTEXT_LIMITS, select_context
from visual_recall.light import collect_refs


def block(kind, text, identity=None):
    return {'type': kind, 'id': identity or text, 'text': text}


class ContextTests(unittest.TestCase):
    def test_eps_table_recovers_estimate_update_across_three_tables(self):
        seq = [block('heading', 'Things to think about:'),
               block('paragraph', 'Our 2026 adj EPS estimate increases to 12.85.', 'p4'),
               *[block('table_ref', '', f't{i}') for i in range(1, 5)],
               block('heading', 'Related Research'), block('paragraph', 'Unrelated report title')]
        ref = collect_refs({'main_content': seq})[-1]
        self.assertEqual([x['id'] for x in ref['before']], ['Things to think about:', 'p4'])
        self.assertEqual(ref['after'], [])
        self.assertEqual(ref['context_selection']['before']['skipped_visual_ids'], ['t1', 't2', 't3'])
        self.assertEqual(ref['context_selection']['before']['source'], 'preceding_prose_across_visuals')

    def test_visual_skip_limit_prevents_unbounded_backtracking(self):
        for n in (4, 5):
            seq = [block('paragraph', 'Shared prose'),
                   *[block('image_ref', '', f'f{i}') for i in range(n)], block('table_ref', '', 'target')]
            result, info = select_context(seq, len(seq)-1, True)
            self.assertEqual(bool(result), n == 4)
            if n == 5:
                self.assertEqual(info['stop_reason'], 'visual_skip_limit')

    def test_do_not_cross_second_visual_group_after_finding_prose(self):
        seq = [block('paragraph', 'Another company'), block('image_ref', '', 'old'),
               block('paragraph', 'Current discussion'), block('table_ref', '', 't1'),
               block('table_ref', '', 'target')]
        result, info = select_context(seq, 4, True)
        self.assertEqual([x['text'] for x in result], ['Current discussion'])
        self.assertEqual(info['stop_reason'], 'visual_boundary')

    def test_heading_is_a_boundary_even_inside_a_visual_run(self):
        seq = [block('paragraph', 'Older section'), block('image_ref', '', 'f1'),
               block('heading', 'New company'), block('table_ref', '', 't1'), block('table_ref', '', 'target')]
        result, info = select_context(seq, 4, True)
        self.assertEqual([x['text'] for x in result], ['New company'])
        self.assertEqual(info['stop_reason'], 'heading_boundary')

    def test_administrative_headings_stop_without_leaking_their_text(self):
        for heading in ('Related Research', 'RELATED RESEARCH:', 'Important Disclosures', 'Analyst Certification'):
            seq = [block('paragraph', 'Old section'), block('heading', heading),
                   block('table_ref', '', 'target'), block('heading', heading), block('paragraph', 'Other reports')]
            for before in (True, False):
                result, info = select_context(seq, 2, before)
                self.assertEqual(result, [])
                self.assertEqual(info['stop_reason'], 'administrative_heading')

    def test_forward_context_does_not_bridge_visuals(self):
        seq = [block('table_ref', '', 'target'), block('table_ref', '', 'next'), block('paragraph', 'Next table discussion')]
        result, info = select_context(seq, 0, False)
        self.assertEqual(result, [])
        self.assertEqual(info['skipped_visual_ids'], [])
        self.assertEqual(info['stop_reason'], 'visual_boundary')

    def test_author_signature_is_not_body_context_or_a_bridge_to_later_text(self):
        for text in ('Matthew Taylor, CFA * | Equity Analyst', 'Name | Equity Associate', 'analyst@example.com'):
            seq = [block('paragraph', 'Earlier page'), block('paragraph', text),
                   block('table_ref', '', 'target'), block('paragraph', text), block('paragraph', 'Later page')]
            for before in (True, False):
                result, info = select_context(seq, 2, before)
                self.assertEqual(result, [])
                self.assertEqual(info['stop_reason'], 'contact_line')

    def test_scan_and_character_budgets_are_explicit(self):
        seq = [block('paragraph', 'Too far'), *[{} for _ in range(16)], block('table_ref', '', 'target')]
        result, info = select_context(seq, len(seq)-1, True)
        self.assertEqual(result, [])
        self.assertEqual(info['stop_reason'], 'scan_limit')
        seq = [block('paragraph', 'Older prose'), block('paragraph', 'Revenue ' * 600), block('table_ref', '', 'target')]
        result, info = select_context(seq, 2, True)
        self.assertLessEqual(sum(len(x['text']) for x in result), CONTEXT_LIMITS['max_chars_per_direction'])
        self.assertTrue(info['truncated'])
        self.assertTrue(result[0]['truncated'])
        self.assertEqual(info['stop_reason'], 'character_limit')

    def test_main_and_appendix_never_share_prose(self):
        refs = collect_refs({'main_content': [block('paragraph', 'Main section')],
                             'appendix_content': [block('table_ref', '', 'appendix_table')]})
        self.assertEqual(refs[0]['before'], [])


if __name__ == '__main__':
    unittest.main()
