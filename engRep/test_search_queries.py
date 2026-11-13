"""Check durable resume behavior without model/network calls."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import generate_search_queries as gen


GOOD = {"status": "ok", "reason": "", "queries": [
    {"type": "core", "text": "ExampleCo segment revenue growth", "evidence": [
        {"source": "image", "text": "Segment Revenue Growth"}]}]}


class QueryTests(unittest.TestCase):
    def test_no_generated_text_in_evidence(self):
        value = gen.payload({"title": "HALLUCINATION", "image_content": "HALLUCINATION",
                             "original_caption": "Revenue", "report_title": "Company"})
        self.assertNotIn("HALLUCINATION", json.dumps(value))

    def test_duplicate_or_ungrounded_status_schema(self):
        bad = {**GOOD, "queries": GOOD['queries'] * 2}
        with self.assertRaises(ValueError):
            gen.validate(bad)
        with self.assertRaises(ValueError):
            gen.validate({"status": "review", "reason": "missing rows", "queries": GOOD['queries']})

    def test_fragment_guard_preserves_labeled_tables(self):
        labels = ['2Q26E', 'New', 'Old', 'EPS Impact', '600 bp', '-64 bp']
        self.assertIsNotNone(gen.fragment_reason({'type': 'table', '_target_table_labels': labels}))
        self.assertIsNone(gen.fragment_reason({'type': 'table', '_target_table_labels': labels + ['Revenue']}))
        self.assertIsNone(gen.fragment_reason({'type': 'figure', '_target_table_labels': labels}))

    def test_torn_tail_recovery_and_interior_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'results.jsonl'
            p.write_bytes(b'{"visual_id":"saved"}\n{"visual_id":')
            self.assertEqual(set(gen.load_results(p)), {'saved'})
            self.assertEqual(p.read_bytes(), b'{"visual_id":"saved"}\n')
            p.write_bytes(b'broken\n{"visual_id":"saved"}\n')
            with self.assertRaises(ValueError):
                gen.load_results(p)

    def test_resume_repairs_flat_export_and_rejects_prompt_change(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            image = p / 'image.jpg'
            image.write_bytes(b'test fixture, not sent to a real model')
            st = image.stat()
            manifest = p / 'manifest.jsonl'
            row = {'id': 'r:figure:f1', 'type': 'figure', 'image_file': str(image),
                   '_image_size': st.st_size, '_image_mtime_ns': st.st_mtime_ns}
            manifest.write_text(gen.dump_line(row))
            prompt = p / 'prompt.txt'
            prompt.write_text('test prompt')
            args = argparse.Namespace(manifest=manifest, output=p/'out', prompt=prompt,
                                      api_base='http://test/v1', model='test', workers=1, retries=0)
            calls = []
            def respond(request):
                calls.append(request)
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                                      'message': {'content': json.dumps(GOOD)}}]})
            real_client = httpx.Client
            def client(**kwargs):
                return real_client(transport=httpx.MockTransport(respond), **kwargs)
            gen.STOP = False
            with patch.object(gen.httpx, 'Client', side_effect=client):
                gen.run(args)
                self.assertEqual(len(calls), 1)
                (args.output/'queries.jsonl').write_text('partial or corrupt export')
                gen.run(args)
                self.assertEqual(len(calls), 1)
                flat = [json.loads(x) for x in (args.output/'queries.jsonl').read_text().splitlines()]
                self.assertEqual(len(flat), 1)
                self.assertEqual(flat[0]['query_id'], 'r:figure:f1:core')
                self.assertEqual(json.loads((args.output/'progress.json').read_text())['status'], 'complete')
                prompt.write_text('changed')
                with self.assertRaises(ValueError):
                    gen.run(args)


if __name__ == '__main__':
    unittest.main()
