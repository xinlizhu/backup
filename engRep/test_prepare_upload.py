import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_upload as pub


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.upload = self.root / 'upload_data'
        self.business = self.upload / pub.BUSINESS

    def index(self, day, ids):
        rows = []
        for identity in ids:
            picture = self.data / 'picture' / day / f'{identity}.jpg'
            picture.parent.mkdir(parents=True, exist_ok=True)
            picture.write_bytes(b'image:' + identity.encode())
            rows.append(dict(id=identity, title='图表标题', image_content='Title then content stays unchanged',
                             image_file=f'../picture/{day}/{identity}.jpg', page_idx=None))
        path = self.data / 'jsonl' / f'{day}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        return path

    def decoded(self, manifest):
        rows = []
        folder = self.business / 'batches' / manifest['batchVersion']
        for info in manifest['files']:
            packed = (folder / info['fileName']).read_bytes()
            self.assertEqual(info['checksum'], 'sha256:' + hashlib.sha256(packed).hexdigest())
            self.assertEqual(len(packed), info['sizeBytes'])
            self.assertLessEqual(len(packed), pub.MAX_BYTES)
            raw = gzip.decompress(packed)
            self.assertFalse((folder / info['fileName']).with_suffix('').exists())
            self.assertEqual(len(raw), info['uncompressedSizeBytes'])
            lines = raw.splitlines()
            self.assertEqual(len(lines), info['recordCount'])
            rows.extend(json.loads(line) for line in lines)
        self.assertEqual(len(rows), manifest['totalRecordCount'])
        return rows

    @staticmethod
    def fake_embeddings(records, images, work):
        return [{**r, 'data': {**r['data'], 'embedding': [1.0] + [0.0]*2047}} for r in records]

    def test_increment_adds_vectors_without_rewriting_history_or_reencoding(self):
        self.index('20260928', ['a'])
        pub.prepare(self.data, '20260928')
        self.index('20261013', ['b'])
        pub.prepare(self.data, '20261013')
        # Simulate the full-vector snapshot after the two original batches.
        snapshot = pub.prepare(self.data, '20261028', embedder=self.fake_embeddings, all_pending=True)
        self.assertEqual(snapshot['totalRecordCount'], 2)
        frozen = {p:p.read_bytes() for p in (self.business/'batches').rglob('*') if p.is_file()}
        self.index('20261106', ['c'])
        with patch.object(self, 'fake_embeddings', wraps=self.fake_embeddings) as embed:
            m = pub.prepare(self.data, '20261106', embedder=embed, all_pending=True)
            self.assertEqual([r['id'] for r in embed.call_args.args[0]], ['ResearchReportImages:c'])
            result = self.decoded(m)
            self.assertEqual(len(result[0]['data']['embedding']), 2048)
            self.assertIsNone(pub.prepare(self.data, '20261106', embedder=embed, all_pending=True))
            self.assertEqual(embed.call_count, 1)
        self.assertEqual(frozen, {p:p.read_bytes() for p in frozen})
        self.assertEqual(len(pub.read_json(self.business/'meta.json')['batches']), 4)

    def test_embedding_failure_and_next_day_resume(self):
        self.index('20261105', ['a'])
        with self.assertRaisesRegex(RuntimeError,'encoder failed'):
            def fail(*args): raise RuntimeError('encoder failed')
            pub.prepare(self.data, '20261105', embedder=fail, all_pending=True)
        self.assertFalse((self.business/'meta.json').exists())
        self.assertFalse(list(self.business.rglob('manifest.json')))
        m=pub.prepare(self.data,'20261106',embedder=self.fake_embeddings,all_pending=True)
        self.assertEqual(m['batchVersion'],'20261106000000000-0000')
        self.assertEqual(self.decoded(m)[0]['data']['image_file'],'pic/20261105/a.jpg')

    def test_invalid_or_misassociated_vectors_never_publish(self):
        self.index('20261106',['a'])
        for mistake in ('nan','dimension','identity','title','missing'):
            def invalid(records,images,work):
                enriched=self.fake_embeddings(records,images,work)
                if mistake=='nan':enriched[0]['data']['embedding'][0]=float('nan')
                elif mistake=='dimension':enriched[0]['data']['embedding']=[1.0]
                elif mistake=='identity':enriched[0]['id']='wrong'
                elif mistake=='title':enriched[0]['data']['title']='changed'
                elif mistake=='missing':return []
                return enriched
            with self.subTest(mistake=mistake), self.assertRaises(ValueError):
                pub.prepare(self.data,'20261106',embedder=invalid)
            self.assertFalse((self.business/'meta.json').exists())
            self.assertFalse(list(self.business.rglob('manifest.json')))

    def test_embedding_subprocess_bridge_preserves_input_and_protocol(self):
        self.index('20261106',['a'])
        def child(cmd,**kwargs):
            self.assertTrue(kwargs['check'])
            self.assertEqual(cmd[cmd.index('--gpu-memory-utilization')+1],'0.95')
            self.assertNotIn('--rewrite-unpublished',cmd)
            source=Path(cmd[cmd.index('--input-records')+1])
            output=Path(cmd[cmd.index('--output-records')+1])
            records=[json.loads(l) for l in source.read_text().splitlines()]
            for r in records:
                self.assertEqual(Path(r.pop('_source_image')),self.data/'picture/20261106/a.jpg')
            enriched=self.fake_embeddings(records,{},source.parent)
            output.write_text(''.join(json.dumps(r)+'\n' for r in enriched))
        with patch.object(pub.subprocess,'run',side_effect=child) as process:
            result=pub.prepare(self.data,'20261106',embedder=pub.embedding_worker(self.data))
            self.assertEqual(process.call_count,1)
        self.assertEqual(len(self.decoded(result)[0]['data']['embedding']),2048)

    def test_meta_retry_does_not_reencode_completed_vector_batch(self):
        self.index('20261106',['a'])
        with patch.object(pub,'update_meta',side_effect=OSError('meta failed')):
            with self.assertRaises(OSError):
                pub.prepare(self.data,'20261106',embedder=self.fake_embeddings)
        frozen={p:p.read_bytes() for p in (self.business/'batches').rglob('*') if p.is_file()}
        with patch.object(self,'fake_embeddings',wraps=self.fake_embeddings) as embed:
            self.assertIsNone(pub.prepare(self.data,'20261106',embedder=embed))
            embed.assert_not_called()
        self.assertTrue((self.business/'meta.json').exists())
        self.assertEqual(frozen,{p:p.read_bytes() for p in frozen})

    def test_package_idempotence_and_same_day_delta(self):
        index = self.index('20261027', ['b', 'a'])
        source_before = index.read_bytes()
        m = pub.prepare(self.data, '2026_10_27')
        self.assertEqual(m['batchVersion'], '20261027000000000-0000')
        records = self.decoded(m)
        self.assertEqual([r['id'] for r in records], ['ResearchReportImages:a', 'ResearchReportImages:b'])
        self.assertEqual(records[0]['data']['image_file'], 'pic/20261027/a.jpg')
        self.assertEqual(records[0]['data']['id'], records[0]['id'])
        self.assertEqual((self.upload / records[0]['data']['image_file']).read_bytes(), b'image:a')
        self.assertEqual(index.read_bytes(), source_before)
        frozen = {p: p.read_bytes() for p in self.business.rglob('*') if p.is_file()}
        self.assertIsNone(pub.prepare(self.data, '20261027'))
        self.assertEqual(frozen, {p: p.read_bytes() for p in frozen})
        self.index('20261027', ['a', 'b', 'c'])
        second = pub.prepare(self.data, '20261027')
        self.assertEqual(second['batchVersion'], '20261027000000000-0001')
        self.assertEqual([r['id'] for r in self.decoded(second)], ['ResearchReportImages:c'])
        for path, value in frozen.items():
            if path.name != 'meta.json':
                self.assertEqual(path.read_bytes(), value)
        meta = pub.read_json(self.business / 'meta.json')
        self.assertEqual(meta['retentionHours'], 4320)
        self.assertEqual(len(meta['batches']), 2)
        self.assertEqual(meta['batches'][-1]['producedAt'], second['producedAt'])

    def test_date_batches_and_metadata_update_are_separate(self):
        self.index('20260928', ['a'])
        first = pub.prepare(self.data, '20260928')
        self.index('20261013', ['b'])
        second = pub.prepare(self.data, '20261013')
        self.assertEqual(first['totalRecordCount'], 1)
        self.assertEqual(second['totalRecordCount'], 1)
        # A changed business field emits another upsert, retaining explicit null.
        p = self.data / 'jsonl/20261013.jsonl'
        row = json.loads(p.read_text())
        row['title'] = None
        p.write_text(json.dumps(row) + '\n')
        third = pub.prepare(self.data, '20261013')
        self.assertIsNone(self.decoded(third)[0]['data']['title'])
        # Backdated changes cannot silently create a version smaller than history.
        self.index('20260928', ['a', 'c'])
        with self.assertRaisesRegex(ValueError, 'exceed all existing'):
            pub.prepare(self.data, '20260928')

    def test_meta_failure_recovers_without_rewriting_batch(self):
        self.index('20261027', ['a'])
        with patch.object(pub, 'update_meta', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                pub.prepare(self.data, '20261027')
        folder = self.business / 'batches/20261027000000000-0000'
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        self.assertFalse((self.business / 'meta.json').exists())
        self.assertIsNone(pub.prepare(self.data, '20261027'))
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
        self.assertEqual(len(pub.read_json(self.business / 'meta.json')['batches']), 1)

    def test_image_failure_does_not_commit_and_empty_does_not_publish(self):
        self.assertIsNone(pub.prepare(self.data, '20261027'))
        self.assertFalse(self.business.exists())
        self.index('20261027', ['a'])
        with patch.object(pub, 'copy_image', side_effect=OSError('copy failed')):
            with self.assertRaises(OSError):
                pub.prepare(self.data, '20261027')
        self.assertFalse(list(self.business.rglob('manifest.json')))
        self.assertFalse((self.business / 'meta.json').exists())
        self.assertEqual(pub.prepare(self.data, '20261027')['totalRecordCount'], 1)

    def test_compressed_size_split_and_single_record_limit(self):
        import random
        rng = random.Random(1)
        records = [dict(id=str(i), content=rng.randbytes(300).hex()) for i in range(20)]
        folder = self.root / 'shards'
        folder.mkdir()
        with patch.object(pub, 'MAX_BYTES', 1024), patch.object(pub, 'TARGET_BYTES', 800):
            files = pub.write_shards(folder, records)
            self.assertGreater(len(files), 1)
            self.assertTrue(all(f['sizeBytes'] <= 1024 for f in files))
            self.assertEqual(sum(f['recordCount'] for f in files), 20)
        with patch.object(pub, 'MAX_BYTES', 32):
            with self.assertRaisesRegex(ValueError, 'Single compressed record'):
                pub.write_shards(folder, records[:1])


if __name__ == '__main__':
    unittest.main()
