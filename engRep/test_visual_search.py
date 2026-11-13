import math
import unittest

from visual_recall.reliability import is_period, search_description_errors, table_review_reasons
from visual_recall.table_groups import group_tables, recover_row_labels
from visual_recall.llm_client import validate_retrieval_output
from visual_recall.search_evidence import report_identity, figure_forecast_evidence


def html(rows):
    return '<table>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</table>'


def table(box, rows, page=0):
    return {'type': 'table', 'page_idx': page, 'bbox': box, 'table_body': html(rows)}


def groups(tables):
    return group_tables([{'id': f't{i}'} for i in range(len(tables))], tables)


class SearchEvidenceTests(unittest.TestCase):
    def test_financial_period_formats_do_not_reject_complete_statements(self):
        for periods in [('FY24A', 'FY25A', 'FY26E'), ('F2024', 'F2025E', 'F2026E'),
                        ('F1Q22Apr-21', 'F2Q22Jul-21', 'F3Q22Oct-21'),
                        ('1Q26A', '2Q26E', '3Q26E'), ('Q1 2026', 'Q2 2026', 'Q3 2026')]:
            with self.subTest(periods=periods):
                self.assertTrue(all(is_period(c) for c in periods))
                self.assertEqual(table_review_reasons(html([['Balance Sheet', *periods],
                                                           ['Total assets', '100', '200', '300']])), [])
        self.assertEqual(table_review_reasons(html([['Total assets', '100', '200', '300']])),
                         ['financial_statement_fragment_missing_period_headers'])

    def test_horizontal_halves_have_one_stable_id_and_complete_bbox(self):
        left = table([45, 60, 520, 300], [['Revenue', '12/24', '12/25'], ['Product A', '12', '13'], ['Total', '12', '13']])
        right = table([550, 60, 950, 300], [['12/26E', '12/27E'], ['14', '15'], ['14', '15']])
        result = groups([left, right])
        self.assertEqual(result['t0'], result['t1'])
        self.assertEqual(result['t1']['anchor_id'], 't0')
        self.assertEqual(result['t0']['source_visual_ids'], ['t0', 't1'])
        self.assertEqual(result['t0']['item']['bbox'], [.045, .06, .95, .3])
        self.assertEqual(table_review_reasons(result['t0']['item']['table_body']), [])

    def test_neighboring_complete_tables_and_ambiguous_halves_stay_separate(self):
        left = table([45, 60, 520, 300], [['Revenue', '12/24', '12/25'], ['Product A', '12', '13'], ['Total', '12', '13']])
        right = table([550, 60, 950, 300], [['EPS', '12/26E'], ['Product B', '14'], ['Total', '14']])
        self.assertNotEqual(groups([left, right])['t0']['anchor_id'], groups([left, right])['t1']['anchor_id'])
        right['table_body'] = html([['12/26E', '12/27E'], ['14', '15'], ['14', '15']])
        ambiguous = groups([left, dict(left), right])
        self.assertTrue(all(len(g['source_visual_ids']) == 1 for g in ambiguous.values()))
        right['table_caption'] = ['An unrelated metric forecast']
        self.assertTrue(all(len(g['source_visual_ids']) == 1 for g in groups([left, right]).values()))
        right['table_caption'] = []
        right['page_idx'] = 1
        self.assertTrue(all(len(g['source_visual_ids']) == 1 for g in groups([left, right]).values()))

    def test_detached_header_rejoins_assets_and_liabilities(self):
        header = table([440, 150, 900, 180], [['2025A', '2026E', '2027E'], ['FY', 'FY', 'FY']])
        assets = table([70, 200, 900, 430], [['Assets', '', '', ''], ['Cash', '1', '2', '3'], ['Total assets', '4', '5', '6']])
        debt = table([70, 440, 900, 750], [['Liabilities', '', '', ''], ['Debt', '1', '2', '3']])
        result = groups([header, assets, debt])
        self.assertEqual({g['anchor_id'] for g in result.values()}, {'t1'})
        self.assertEqual(result['t1']['source_visual_ids'], ['t0', 't1', 't2'])
        self.assertEqual(table_review_reasons(result['t1']['item']['table_body']), [])
        debt['bbox'] = [70, 480, 900, 750]
        result = groups([header, assets, debt])
        self.assertNotEqual(result['t1']['anchor_id'], result['t2']['anchor_id'])

    def test_header_with_wrong_column_count_is_not_attached(self):
        header = table([440, 150, 900, 180], [['2025A', '2026E']])
        body = table([70, 200, 900, 430], [['Total assets', '1', '2', '3']])
        self.assertTrue(all(len(g['source_visual_ids']) == 1 for g in groups([header, body]).values()))

    def test_numeric_crop_recovers_dense_left_labels_but_not_prose(self):
        target = table([280, 160, 950, 660], [['2025A', '2026E', '2027E'],
                                              *[['100', '200', '300'] for _ in range(5)]])
        labels = [{'type': 'text', 'page_idx': 0, 'bbox': [45, y, 220, y+15], 'text': 'Segment revenue'}
                  for y in (200, 300, 400, 500, 630)]
        crop, recovered = recover_row_labels(target, [target, *labels])
        self.assertEqual(len(recovered), 5)
        self.assertEqual(crop['bbox'][0], .045)
        self.assertEqual(recover_row_labels(target, [target, labels[0]])[1], [])
        obstruction = {'type': 'image', 'page_idx': 0, 'bbox': [100, 250, 200, 280]}
        self.assertEqual(recover_row_labels(target, [target, *labels, obstruction])[1], [])
        for label in labels:
            label['text'] = 'unrelated paragraph ' * 20
        self.assertEqual(recover_row_labels(target, [target, *labels])[1], [])

    def test_schema_types_and_finite_confidence(self):
        output = {'decision': 'keep', 'category': 'chart', 'filter_reason': '', 'title': 'Company revenue forecast',
                  'image_content': 'Company revenue forecasts and earnings estimates.', 'is_blurry': False,
                  'confidence': .95, 'evidence_status': 'sufficient'}
        self.assertEqual(validate_retrieval_output(output), [])
        for value in ('0.95', True, math.nan, math.inf):
            self.assertTrue(validate_retrieval_output({**output, 'confidence': value}))
        self.assertTrue(validate_retrieval_output({**output, 'filter_reason': 45}))

    def test_search_description_rejects_values_but_keeps_product_names(self):
        output = {'decision': 'keep', 'title': '3M revenue forecasts',
                  'image_content': 'Company revenue estimates and H100 server demand.'}
        self.assertEqual(search_description_errors(output), [])
        for text in ('Revenue reaches $10 billion.', 'Margins reach 30%.', 'The endpoint has p=0.0299.'):
            self.assertTrue(search_description_errors({**output, 'image_content': text}))
        for text in ('52-week range and 12-month rating.', '3-year EPS CAGR.', 'RSS change at Month 13.'):
            self.assertEqual(search_description_errors({**output, 'image_content': text}), [])

    def test_company_identity_comes_from_source_not_ticker_expansion(self):
        report = {'security_code': 'INSM.O', 'raw_title': 'ENCORE Clean Win', 'data': {}}
        content = [{'type': 'header', 'text': 'Insmed Inc (INSM)'}]
        self.assertEqual(report_identity(report, content)['name'], 'Insmed Inc')
        self.assertNotIn('name', report_identity(report, []))
        # A different company in an image caption must not become the report identity.
        report['data'] = {'main_content': [{'type': 'image_ref', 'context': 'Other Corp (OTHR) revenue'}]}
        self.assertNotIn('name', report_identity(report, []))
        report = {'security_code': 'DHR.N', 'raw_title': 'Danaher Corp: Company update', 'data': {
            'main_content': [{'type': 'heading', 'text': 'Danaher Corp'}]}}
        self.assertEqual(report_identity(report, [])['name'], 'Danaher Corp')

    def test_estimate_source_note_does_not_turn_actual_periods_into_forecasts(self):
        from types import SimpleNamespace
        bbox = [.1, .2, .6, .7]
        texts = iter(('F1Q24 F2Q24 F3Q24\nSource: Analyst estimates',
                      'F1Q24 F2Q24 F3Q24E\nSource: Analyst estimates',
                      '0 10 20\nSource: Analyst estimates'))
        clips = []
        def extract(*, clip):
            clips.append(clip)
            return next(texts)
        page = SimpleNamespace(get_text=extract)
        renderer = SimpleNamespace(document=True, _page_and_rect=lambda p, b: (page, tuple(b)))
        self.assertEqual(figure_forecast_evidence(renderer, 0, bbox)['status'], 'periods_without_estimate_labels')
        self.assertEqual(figure_forecast_evidence(renderer, 0, bbox)['status'], 'explicit_estimate_labels')
        self.assertEqual(figure_forecast_evidence(renderer, 0, bbox)['status'], 'not_available')
        self.assertEqual(clips, [tuple(bbox)] * 3)
        self.assertEqual(figure_forecast_evidence(None, 0, bbox)['status'], 'not_available')

    def test_conflicting_forecast_claim_is_withheld_using_target_period_evidence(self):
        output = {'decision': 'keep', 'title': 'Company operating income and margin',
                  'image_content': 'Quarterly operating income with historical and future fiscal quarter data.'}
        self.assertEqual(search_description_errors(output, {'status': 'periods_without_estimate_labels'}),
                         ['forecast_claim_without_target_estimate_labels'])
        for status in ('not_available', 'explicit_estimate_labels'):
            self.assertEqual(search_description_errors(output, {'status': status}), [])
        output['image_content'] = 'Quarterly operating income and operating margin.'
        self.assertEqual(search_description_errors(output, {'status': 'periods_without_estimate_labels'}), [])


if __name__ == '__main__':
    unittest.main()
