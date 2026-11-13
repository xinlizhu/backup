import json
from pathlib import Path
import tempfile
import unittest

from visual_recall.evidence import PdfAssetRenderer
from visual_recall.reliability import table_review_reasons
from visual_recall.table_groups import group_tables
from visual_recall.target_evidence import (
    company_pairs, document_company_index, native_target_lines, resolve_target_companies,
    target_labels, target_subject_errors, target_label_errors,
)


def table(box, rows, caption='', page=0):
    body = '<table>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</table>'
    return {'type': 'table', 'page_idx': page, 'bbox': box,
            'table_body': body, 'table_caption': [caption] if caption else []}


def group(items, content=()):
    return group_tables([{'id': f't{i}'} for i in range(len(items))], items, content=content)


class TargetEvidenceTests(unittest.TestCase):
    def panels(self):
        return [table([70, 300, 270, 600], [[x] for x in (
                    'Revenue', 'Product', 'Service', 'Gross Profit', 'Income', 'Tax', 'EPS', 'Shares')],
                      'Figure 1: Company financial comparison'),
                table([280, 310, 650, 600], [['Actual', 'Estimates'], ['2025A', '2026E'],
                                           *[['1', '2'] for _ in range(7)]]),
                table([660, 310, 920, 600], [['Better / Worse'], ['$', 'Percentage'],
                                           *[['1', '2%'] for _ in range(6)]])]

    def test_label_column_and_multiple_data_panels_keep_one_complete_image(self):
        items = self.panels()
        original = json.dumps(items)
        result = group(items)
        self.assertEqual({r['anchor_id'] for r in result.values()}, {'t0'})
        self.assertEqual(result['t0']['source_visual_ids'], ['t0', 't1', 't2'])
        merged = result['t0']['item']
        self.assertEqual(merged['bbox'], [.07, .3, .92, .6])
        self.assertEqual([t['table_body'] for t in merged['_table_fragments']], [t['table_body'] for t in items])
        self.assertEqual(json.dumps(items), original)
        self.assertEqual(table_review_reasons(merged['table_body']), [])
        labels = target_labels(None, 0, merged['bbox'], merged, 't0')['items']
        self.assertTrue(any(x['text'] == 'Revenue' and x['source_visual_id'] == 't0' for x in labels))
        self.assertTrue(any(x['text'] == 'Actual' and x['source_visual_id'] == 't1' for x in labels))

    def test_distinct_titles_pages_and_intervening_figures_prevent_merge(self):
        for change in ('caption', 'page', 'gap', 'vertical_shift', 'obstruction'):
            items = self.panels()[:2]
            content = []
            if change == 'caption': items[1]['table_caption'] = ['A separate table']
            if change == 'page': items[1]['page_idx'] = 1
            if change == 'gap': items[1]['bbox'][0] = 350
            if change == 'vertical_shift': items[1]['bbox'][1] = 380
            if change == 'obstruction':
                content = [*items, {'type': 'image', 'page_idx': 0, 'bbox': [271, 350, 279, 500]}]
            with self.subTest(change=change):
                self.assertTrue(all(len(x['source_visual_ids']) == 1 for x in group(items, content).values()))

    def test_ambiguous_label_columns_and_complete_neighbors_are_not_merged(self):
        items = self.panels()[:2]
        items.insert(1, dict(items[0]))
        self.assertTrue(all(len(x['source_visual_ids']) == 1 for x in group(items).values()))
        items = self.panels()[:2]
        items[1] = table(items[1]['bbox'], [['Metric', '2026E'], *[[f'Product {i}', '5'] for i in range(8)]])
        self.assertTrue(all(len(x['source_visual_ids']) == 1 for x in group(items).values()))

    def test_financial_fragments_are_reviewed_but_qualitative_tables_are_allowed(self):
        labels, numbers, _ = self.panels()
        self.assertEqual(table_review_reasons(labels['table_body']), ['financial_table_fragment_missing_data_columns'])
        self.assertEqual(table_review_reasons(numbers['table_body']), ['table_fragment_missing_metric_labels'])
        qualitative = table([0, 0, 1, 1], [['Revenue', 'Growing'], ['Margin', 'Stable'], ['EPS', 'Higher']])
        self.assertEqual(table_review_reasons(qualitative['table_body']), [])
        company_list = table([0, 0, 1, 1], [[n] for n in ('Alpha', 'Beta', 'Gamma', 'Delta')])
        self.assertEqual(table_review_reasons(company_list['table_body']), [])
        transposed = table([0, 0, 1, 1], [['Year', 'Revenue', 'EPS'],
                                          *[[str(y), '100', '5'] for y in range(2020, 2027)]])
        self.assertEqual(table_review_reasons(transposed['table_body']), [])

    def test_native_labels_are_clipped_and_spelling_is_preserved(self):
        import fitz
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((80, 100), 'CTSH')
            page.insert_text((80, 120), 'PTPP')
            page.insert_text((80, 140), 'PREV')
            page.insert_text((140, 100), '2026 ($)')
            page.insert_text((140, 120), 'Q1')
            page.insert_text((400, 100), 'NEIGHBOR')
            path = root / 'test.pdf'
            doc.save(path); doc.close()
            renderer = PdfAssetRenderer(path, root, 120, 220, .1)
            try:
                labels = target_labels(renderer, 0, [.1, .1, .3, .2], {}, 'f1')
                self.assertEqual({x['text'] for x in labels['items']}, {'CTSH', 'PTPP', 'PREV', '2026 ($)', 'Q1'})
                self.assertEqual(labels['label_cautions'][0]['label'], 'PREV')
                self.assertTrue(all(x['source'] == 'pdf_text_inside_target' for x in labels['items']))
                self.assertEqual(native_target_lines(renderer, -1, [.1, .1, .3, .2]), [])
            finally:
                renderer.close()

    def test_caption_company_overrides_report_and_only_visible_peers_get_names(self):
        identity = {'name': 'Vornado Realty', 'ticker': 'VNO'}
        report = {'data': {'main_content': [
            {'type': 'table_ref', 'id': 't1', 'context': 'Figure 7: Brandywine (BDN) - Implied Value PSF'},
            {'type': 'table_ref', 'id': 't2', 'context': 'Figure 12: Empire State Realty Trust (ESRT) - Implied Value PSF'},
        ]}}
        index = document_company_index(report, [], identity)
        companies, conflicts = resolve_target_companies(report['data']['main_content'][0]['context'], {'items': []}, index)
        self.assertEqual([(x['name'], x['ticker']) for x in companies], [('Brandywine', 'BDN')])
        self.assertEqual(conflicts, [])
        peers, _ = resolve_target_companies('NYC office cap rates', {'items': [
            {'text': 'VNO', 'source': 'pdf_text_inside_target'},
            {'text': 'ESRT', 'source': 'pdf_text_inside_target'}]}, index)
        self.assertEqual({x['ticker'] for x in peers}, {'VNO', 'ESRT'})
        self.assertEqual(next(x for x in peers if x['ticker'] == 'ESRT')['name'], 'Empire State Realty Trust')
        self.assertEqual(resolve_target_companies('US electricity demand', {'items': []}, index)[0], [])

    def test_acronyms_and_conflicting_name_mappings_are_not_company_guesses(self):
        for caption in ('Earnings Per Share (EPS)', 'Net Operating Income (NOI)', 'Free Cash Flow (FCF)'):
            self.assertEqual(list(company_pairs(caption)), [])
        index = {'ABC': [{'name': 'Alpha Co', 'ticker': 'ABC'}, {'name': 'Beta Co', 'ticker': 'ABC'}]}
        companies, _ = resolve_target_companies('Peer comparison', {'items': [{'text': 'ABC', 'source': 'pdf_text_inside_target'}]}, index)
        self.assertNotIn('name', companies[0])
        self.assertIn('ambiguous', companies[0]['name_resolution'])
        labels = {'items': [{'text': s, 'source': 'table_html_cell'} for s in (
            'Diluted weighted avg shares (K)', 'EBIT (UBS)', 'Op Inc (FIFO)',
            'Buy (UR)', 'Market Capitalization (M)', 'Rev. (MM)', 'Diff. (A-E)')]}
        self.assertEqual(resolve_target_companies('Financial summary', labels, {})[0], [])

    def test_company_guard_allows_peers_but_withholds_report_company_leakage(self):
        output = {'decision': 'keep', 'title': 'Vornado Realty Brandywine implied value', 'image_content': 'BDN asset values.'}
        payload = {'report_identity': {'name': 'Vornado Realty', 'ticker': 'VNO'},
                   'target_companies': [{'name': 'Brandywine', 'ticker': 'BDN'}]}
        self.assertEqual(target_subject_errors(output, payload), ['report_company_leaked_into_other_target_subject'])
        output['title'] = 'Brandywine implied value'
        self.assertEqual(target_subject_errors(output, payload), [])
        payload['target_companies'].append({'ticker': 'VNO'})
        output['title'] = 'Vornado Realty and Brandywine implied value'
        self.assertEqual(target_subject_errors(output, payload), [])

    def test_prev_is_not_assumed_to_mean_prior_year_and_negations_are_allowed(self):
        payload = {'target_labels': {'items': [{'text': 'EPS'}, {'text': 'PREV'}]}, 'visual': {}}
        output = {'decision': 'keep', 'title': 'EPS comparison',
                  'image_content': 'A PREV row provides the corresponding prior-year EPS values.'}
        self.assertEqual(target_label_errors(output, payload), ['unsupported_prev_as_prior_year'])
        for content in ('EPS figures are displayed with a PREV comparison row.',
                        'PREV is not defined as prior-year EPS.',
                        'There is no evidence that PREV means previous year.',
                        'A PREV row shows previous estimates.'):
            self.assertEqual(target_label_errors({**output, 'image_content': content}, payload), [])
        defined = {**payload, 'visual': {'original_caption': 'PREV: prior-year values'}}
        self.assertEqual(target_label_errors(output, defined), [])


if __name__ == '__main__':
    unittest.main()
