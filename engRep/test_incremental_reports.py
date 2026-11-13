import argparse
from contextlib import contextmanager, nullcontext
import json
import signal
import shutil
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import incremental_reports as inc


class IncrementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.mid = inc.mid_root(self.data)
        self.source = self.root / 'source'
        inc.save(self.mid / '_state/migration_complete.json', {})
        inc.save(self.mid / '_state/legacy_reports.json', {})
        inc.save_rows(self.data / 'jsonl' / (inc.batch_key(inc.LEGACY_BATCH) + '.jsonl'), [])
        inc.save_rows(self.mid / inc.LEGACY_BATCH / 'question.jsonl', [])
        self.prompt = self.root / 'prompt.txt'
        self.prompt.write_text('test prompt')
        self.db = self.root / 'index.db'
        with sqlite3.connect(self.db) as c:
            c.execute('create table research_reports(pdf_id text,json_relative_path text,raw_date text)')
        self.args = argparse.Namespace(data=self.data, db=self.db, source_root=self.source,
            start_date='2026-06-01', end_date=None, batch_date='2026_10_13',
            prompt=self.prompt, model='test', api_base='http://127.0.0.1:9/v1',
            workers=2, max_reports=0)
        factory=patch.object(inc,'embedding_worker',return_value=self.fake_embeddings)
        self.embedding_factory=factory.start()
        self.addCleanup(factory.stop)

    @staticmethod
    def fake_embeddings(records,images,work):
        return [{**r,'data':{**r['data'],'embedding':[1.0]+[0.0]*2047}} for r in records]

    def test_embedding_starts_only_after_extraction_server_exits(self):
        self.report('a')
        events=[]
        @contextmanager
        def server(*args):
            events.append('server_started')
            try:yield
            finally:events.append('server_stopped')
        def embed(*args):
            self.assertEqual(events[-1],'server_stopped')
            events.append('embedding')
            return self.fake_embeddings(*args)
        self.embedding_factory.return_value=embed
        with patch.object(inc,'server',side_effect=server),patch.object(inc,'command',side_effect=self.fake_command):
            self.assertEqual(inc.run(self.args),0)
        self.assertEqual(events,['server_started','server_stopped','embedding'])
        status=inc.read_json(self.mid/self.args.batch_date/'publication.json')
        self.assertEqual(status['status'],'complete')

    def test_failed_embedding_retries_without_reextracting_completed_reports(self):
        self.report('a')
        def fail(*args):raise RuntimeError('embedding failed')
        self.embedding_factory.return_value=fail
        with patch.object(inc,'server',return_value=nullcontext()),patch.object(inc,'command',side_effect=self.fake_command):
            with self.assertRaisesRegex(RuntimeError,'embedding failed'):inc.run(self.args)
        self.assertEqual(inc.read_json(self.mid/self.args.batch_date/'publication.json')['status'],'failed')
        self.assertFalse((self.root/'upload_data/research-report-images/meta.json').exists())
        self.embedding_factory.return_value=self.fake_embeddings
        self.args.batch_date='2026_10_14'
        with patch.object(inc,'server') as server,patch.object(inc,'command') as command:
            self.assertEqual(inc.run(self.args),0)
            server.assert_not_called();command.assert_not_called()
        self.assertEqual(inc.read_json(self.mid/self.args.batch_date/'publication.json')['records'],1)

    def report(self, name, published_date='2026-07-02', exists=True):
        rel = f'part1/{name}/{name}.json'
        with sqlite3.connect(self.db) as c:
            c.execute('insert into research_reports values(?,?,?)', (name, rel, published_date))
        if exists:
            inc.save(self.source / rel, {'data': {'tables': {}}})
        return dict(id=name, relative_path=rel, date=published_date, reason='new')

    def test_discovery_respects_dates_empty_success_retry_and_missing(self):
        for name in ['done', 'empty', 'failed', 'new']:
            self.report(name)
        self.report('old', '2026-06-30')
        self.report('missing', exists=False)
        inc.save(self.mid / '_state/legacy_reports.json', {
            'done': {'status': 'imported'}, 'empty': {'status': 'empty'}, 'failed': {'status': 'failed'}})
        self.args.start_date = '2026-07-01'
        ready, missing, summary = inc.discover(self.args)
        self.assertEqual({r['id'] for r in ready}, {'failed', 'new'})
        self.assertEqual([r['id'] for r in missing], ['missing'])
        self.assertEqual(summary['already_processed'], 2)
        self.assertEqual(summary['reasons'], {'retry': 1, 'new': 2})
        self.args.end_date = '2026-07-01'
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)

    def test_source_traversal_rejected(self):
        with self.assertRaises(ValueError):
            inc.source_path({'id': 'x', 'relative_path': '../../x/x.json'}, self.source)

    def test_check_and_run_with_only_upload_package_remaining(self):
        self.report('historical')
        self.report('new')
        image = self.data / 'picture/20261013/old.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'historical-image')
        inc.save_rows(self.data / 'jsonl/20261013.jsonl', [dict(
            id='historical:figure:1', report_id='historical', title='Original',
            image_content='Preserve content', image_file='../picture/20261013/old.jpg')])
        inc.prepare_upload(self.data, '20261013')
        upload = self.root / 'upload_data'
        historical = {p: p.read_bytes() for p in (upload / 'research-report-images/batches').rglob('*') if p.is_file()}
        shutil.rmtree(self.data)
        shutil.rmtree(self.mid)
        ready, _, summary = inc.discover(self.args)
        self.assertEqual([r['id'] for r in ready], ['new'])
        self.assertEqual(summary['already_processed'], 1)
        self.assertFalse(self.data.exists())
        self.assertFalse(self.mid.exists())  # check remains read-only
        self.args.batch_date = '2026_10_27'
        with patch.object(inc, 'server', return_value=nullcontext()), \
             patch.object(inc, 'command', side_effect=self.fake_command):
            self.assertEqual(inc.run(self.args), 0)
        docs = inc.rows(self.data / 'full.jsonl')
        self.assertEqual({d['report_id'] for d in docs}, {'historical', 'new'})
        for d in docs:
            self.assertTrue((self.data / d['image_file']).is_file())
        self.assertEqual(historical, {p: p.read_bytes() for p in historical})
        self.assertIsNone(inc.prepare_upload(self.data, '20261013'))
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)

    def test_local_indexes_work_without_migration_marker_and_keep_empty_receipt(self):
        (self.mid / '_state/migration_complete.json').unlink()
        self.report('indexed')
        self.report('empty')
        inc.save_rows(self.data / 'jsonl/20261013.jsonl', [dict(
            id='indexed:figure:1', report_id='indexed', image_file='../picture/20261013/x.jpg')])
        inc.save(self.mid / '_state/reports/empty.json', dict(id='empty', status='empty'))
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)

    def fake_command(self, cmd, log):
        if '--output-dir' in cmd:
            out = Path(cmd[cmd.index('--output-dir') + 1])
            src = Path(cmd[cmd.index('--report-dir') + 1])
            self.assertEqual(out.name, src.name)
            out.mkdir(parents=True, exist_ok=True)
            image = out / 'figure.jpg'
            image.write_bytes(b'unchanged-image-bytes')
            inc.save(out / 'run_result.json', dict(report_dir=str(src), build_error_count=0,
                     model_errors=0, dry_run=False))
            inc.save_rows(out / 'published_visuals.jsonl', [dict(id=f'{src.name}:figure:1',
                publish_ready=True, image_file=str(image), title='Original title', image_content='Original content',
                type='figure', publish_time='2026-07-02')])
        else:
            out = Path(cmd[cmd.index('--output') + 1])
            manifest = inc.rows(Path(cmd[cmd.index('--manifest') + 1]))
            r = manifest[0]
            inc.save(out / 'progress.json', {'status': 'complete'})
            inc.save_rows(out / 'queries.jsonl', [dict(query_id=r['id'] + ':core', visual_id=r['id'],
                report_id=r['id'].split(':')[0], image_file=r['image_file'], query='Original query')])
        return 0

    def test_process_resume_export_and_recover_indexes(self):
        row = self.report('a')
        with patch.object(inc, 'command', side_effect=self.fake_command) as cmd:
            receipt = inc.process_report(row, self.args)
            self.assertEqual(receipt['status'], 'complete', receipt)
            self.assertEqual(cmd.call_count, 2)
            # Retrying after a crash before receipt commit reuses completed extraction.
            receipt = inc.process_report(row, self.args)
            self.assertEqual(receipt['status'], 'complete', receipt)
            self.assertEqual(cmd.call_count, 3)
        image = self.data / 'picture' / inc.batch_key(receipt['batch']) / receipt['documents'][0]['image_file']
        self.assertEqual(image.read_bytes(), b'unchanged-image-bytes')
        self.assertEqual(receipt['documents'][0]['title'], 'Original title')
        inc.save(self.mid / '_state/reports/a.json', receipt)
        # Simulate stale/corrupted derived indexes after commit; rebuild from receipt.
        (self.data / 'full.jsonl').write_text('interrupted')
        inc.combine(self.data)
        inc.combine(self.data)
        docs = inc.rows(self.data / 'full.jsonl')
        qs = inc.rows(self.mid / 'question.jsonl')
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(qs), 1)
        self.assertEqual((self.mid / qs[0]['image_file']).resolve(),
                         (self.data / docs[0]['image_file']).resolve())
        self.assertTrue((self.data / docs[0]['image_file']).is_file())
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)

    def test_partial_extraction_keeps_published_visuals(self):
        for field in ('build_error_count', 'model_errors'):
            with self.subTest(error=field):
                row = self.report(field)
                def command(cmd, log):
                    self.fake_command(cmd, log)
                    if '--output-dir' in cmd:
                        out = Path(cmd[cmd.index('--output-dir') + 1])
                        result = inc.read_json(out / 'run_result.json')
                        result[field] = 6
                        inc.save(out / 'run_result.json', result)
                        return 2
                    return 0
                with patch.object(inc, 'command', side_effect=command):
                    result = inc.process_report(row, self.args)
                self.assertEqual(result['status'], 'complete', result)
                self.assertEqual(len(result['documents']), 1)
                self.assertEqual(len(result['questions']), 1)
                self.assertEqual(result['documents'][0]['title'], 'Original title')

    def test_extraction_failure_without_published_visuals_stays_retryable(self):
        row = self.report('failed')
        def command(cmd, log):
            self.fake_command(cmd, log)
            out = Path(cmd[cmd.index('--output-dir') + 1])
            result = inc.read_json(out / 'run_result.json')
            result['build_error_count'] = 6
            inc.save(out / 'run_result.json', result)
            inc.save_rows(out / 'published_visuals.jsonl', [])
            return 2
        with patch.object(inc, 'command', side_effect=command) as cmd:
            result = inc.process_report(row, self.args)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(cmd.call_count, 1)

    def test_query_failure_retains_extraction_for_retry(self):
        row = self.report('a')
        def command(cmd, log):
            if '--output-dir' in cmd:
                return self.fake_command(cmd, log)
            out = Path(cmd[cmd.index('--output') + 1])
            inc.save(out / 'progress.json', {'status': 'complete_with_errors'})
            return 0
        with patch.object(inc, 'command', side_effect=command):
            result = inc.process_report(row, self.args)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.data / 'picture' / inc.batch_key(self.args.batch_date)).exists())
        with patch.object(inc, 'command', side_effect=self.fake_command) as cmd:
            result = inc.process_report(row, self.args)
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(cmd.call_count, 1)

    def test_empty_report_terminal_and_no_query_call(self):
        row = self.report('empty')
        def command(cmd, log):
            out = Path(cmd[cmd.index('--output-dir') + 1])
            src = Path(cmd[cmd.index('--report-dir') + 1])
            inc.save(out / 'run_result.json', dict(report_dir=str(src), build_error_count=0,
                     model_errors=0, dry_run=False))
            inc.save_rows(out / 'published_visuals.jsonl', [])
            return 0
        with patch.object(inc, 'command', side_effect=command) as cmd:
            result = inc.process_report(row, self.args)
        self.assertEqual(result['status'], 'empty', result)
        self.assertEqual(cmd.call_count, 1)
        inc.save(self.mid / '_state/reports/empty.json', result)
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)

    def test_run_idempotent_and_does_not_start_server_for_no_work(self):
        self.report('a')
        with patch.object(inc, 'server', return_value=nullcontext()) as server, \
             patch.object(inc, 'command', side_effect=self.fake_command):
            self.assertEqual(inc.run(self.args), 0)
            self.assertEqual(inc.run(self.args), 0)
            self.assertEqual(server.call_count, 1)
        self.assertEqual(len(inc.rows(self.data / 'full.jsonl')), 1)
        upload = self.root / 'upload_data'
        meta = inc.read_json(upload / 'research-report-images/meta.json')
        self.assertEqual(len(meta['batches']), 1)
        self.assertEqual(meta['retentionHours'], 4320)
        self.assertEqual(len(list((upload / 'pic/20261013').glob('*.jpg'))), 1)
        self.assertEqual({p.name for p in self.data.iterdir()}, {'full.jsonl', 'jsonl', 'picture'})
        self.assertTrue(all(p.is_file() and p.suffix == '.jpg'
                            for p in (self.data / 'picture' / inc.batch_key(self.args.batch_date)).iterdir()))
        batch_index = self.data / 'jsonl' / (inc.batch_key(self.args.batch_date) + '.jsonl')
        for record in inc.rows(batch_index):
            self.assertTrue((batch_index.parent / record['image_file']).is_file())
        for record in inc.rows(self.mid / self.args.batch_date / 'question.jsonl'):
            self.assertTrue((self.mid / self.args.batch_date / record['image_file']).is_file())
        self.assertFalse((self.mid / self.args.batch_date / 'data.jsonl').exists())

    def test_changed_input_blocks_cache_reuse(self):
        row = self.report('a')
        with patch.object(inc, 'command', side_effect=self.fake_command):
            self.assertEqual(inc.process_report(row, self.args)['status'], 'complete')
        self.prompt.write_text('different prompt')
        with patch.object(inc, 'command') as cmd:
            result = inc.process_report(row, self.args)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Inputs changed', result['error'])
        cmd.assert_not_called()

    def test_legacy_stop_files_do_not_block_resume(self):
        self.report('a')
        (self.data / 'STOP').touch()
        query_stop = self.mid / '_state/work/a/queries/STOP'
        query_stop.parent.mkdir(parents=True)
        query_stop.symlink_to(self.data / 'STOP')
        def command(cmd, log):
            if '--output' in cmd:
                self.assertFalse(query_stop.exists())
                self.assertFalse(query_stop.is_symlink())
            return self.fake_command(cmd, log)
        with patch.object(inc, 'server', return_value=nullcontext()), \
             patch.object(inc, 'command', side_effect=command):
            self.assertEqual(inc.run(self.args), 0)
        self.assertEqual(len(inc.rows(self.data / 'full.jsonl')), 1)

    def test_interrupt_only_stops_current_run(self):
        self.report('a')
        self.report('b')
        self.args.workers = 1
        def interrupted_report(row, args):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return {**row, 'status': 'empty', 'batch': args.batch_date,
                    'documents': [], 'questions': []}
        with patch.object(inc, 'server', return_value=nullcontext()), \
             patch.object(inc, 'process_report', side_effect=interrupted_report) as process:
            self.assertEqual(inc.run(self.args), 130)
            self.assertEqual(process.call_count, 1)
        self.assertFalse((self.data / 'STOP').exists())
        with patch.object(inc, 'server', return_value=nullcontext()), \
             patch.object(inc, 'command', side_effect=self.fake_command):
            self.assertEqual(inc.run(self.args), 0)
        self.assertEqual(inc.discover(self.args)[2]['pending'], 0)


if __name__ == '__main__':
    unittest.main()
