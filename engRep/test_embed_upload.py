import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import errno

import numpy as np
import embed_upload as emb
import prepare_upload as pub


class EmbedUploadTests(unittest.TestCase):
    def test_incremental_worker_cache_reuse_input_changes_and_failure_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';upload=root/'upload_data';model=root/'model';model.mkdir()
            (model/'config.json').write_text('{}')
            source=root/'input.jsonl';output=root/'output.jsonl';cache=root/'cache'
            def record(identity,title):
                image=data/'picture/20261106'/f'{identity}.jpg';image.parent.mkdir(parents=True,exist_ok=True)
                image.write_bytes(identity.encode())
                key='ResearchReportImages:'+identity
                return dict(op='upsert',id=key,data=dict(id=key,title=title,image_content='Content',image_file=f'pic/20261106/{identity}.jpg'),_source_image=str(image))
            records=[record('a','A'),record('b','B')]
            calls=[];fail_on={'text':'Title: B\nContent: Content'}
            class FakeEncoder:
                last_timings={'embed_seconds':0.0,'render_seconds':0.0}
                def __init__(self,args):pass
                def prepare_items(self,items):return items,0.0
                def encode_prepared(self,items):
                    vectors=[]
                    for item in items:
                        calls.append(item['text'])
                        if item['text']==fail_on['text']:raise RuntimeError('test interrupted encoding')
                        v=np.zeros(2048,dtype='<f4');v[0 if 'Title: A' in item['text'] else 1]=1.0
                        vectors.append(v)
                    return np.asarray(vectors),[10]*len(items)
            argv=['embed_upload.py','--input-records',str(source),'--output-records',str(output),
                  '--source-data',str(data),'--upload-dir',str(upload),'--model',str(model),
                  '--cache-root',str(cache),'--batch-size','1']
            def run():
                source.write_text(''.join(json.dumps(r)+'\n' for r in records))
                with patch.object(sys,'argv',argv),patch.object(emb,'Encoder',FakeEncoder),patch.object(emb,'smoke'):
                    emb.main()
            with self.assertRaisesRegex(RuntimeError,'interrupted'):run()
            self.assertFalse(output.exists())
            self.assertFalse((upload/pub.BUSINESS/'meta.json').exists())
            fail_on['text']=None;calls.clear();run()
            self.assertEqual(calls,['Title: B\nContent: Content'])
            result=[json.loads(l) for l in output.read_text().splitlines()]
            self.assertEqual([r['id'] for r in result],[r['id'] for r in records])
            self.assertEqual([r['data']['embedding'][:2] for r in result],[[1.0,0.0],[0.0,1.0]])
            self.assertTrue(all('_source_image' not in r for r in result))
            calls.clear();records[0]['data']['report_title']='Metadata-only update';run()
            self.assertEqual(calls,[])
            records[0]['data']['title']='A changed';run()
            self.assertEqual(calls,['Title: A changed\nContent: Content'])

    def test_directory_exchange_fallback_keeps_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'staging';b=Path(tmp)/'published'
            a.mkdir();b.mkdir()
            (a/'value').write_text('new');(b/'value').write_text('old')
            with patch.object(emb, '_atomic_exchange', side_effect=OSError(errno.EINVAL,'unsupported')):
                emb.exchange_directories(a,b)
            self.assertEqual((b/'value').read_text(),'new')
            self.assertEqual((a/'value').read_text(),'old')

    def test_directory_exchange_fallback_rolls_back_failed_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'staging';b=Path(tmp)/'published'
            a.mkdir();b.mkdir()
            (a/'value').write_text('new');(b/'value').write_text('old')
            rename=Path.rename
            def fail_staging(path,target):
                if path==a: raise OSError('simulated move failure')
                return rename(path,target)
            with patch.object(emb, '_atomic_exchange', side_effect=OSError(errno.EINVAL,'unsupported')), patch.object(Path,'rename',fail_staging):
                with self.assertRaisesRegex(OSError,'simulated'):
                    emb.exchange_directories(a,b)
            self.assertEqual((b/'value').read_text(),'old')
            self.assertEqual((a/'value').read_text(),'new')

    def test_input_matches_documented_template(self):
        item = {'image': '/external/chart.jpg', 'text': 'Title: Example\nContent: Description'}
        actual = emb.conversation(item)
        expected = [
            {'role': 'system', 'content': [{'type': 'text', 'text': "Represent the user's input."}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': '/external/chart.jpg', 'min_pixels': 4096, 'max_pixels': 1048576},
                {'type': 'text', 'text': 'Title: Example\nContent: Description'}]}]
        self.assertEqual(actual, expected)
        self.assertEqual(emb.text_for({'title': '  Example  ', 'image_content': '\nDescription\n'}),
                         'Title: Example\nContent: Description')

    def test_invalid_vectors_rejected(self):
        for a in [np.zeros((1,2048)),np.ones((1,7)),np.full((1,2048),np.nan)]:
            with self.assertRaises(ValueError): emb.checked_vectors(a)

    def test_rewrite_preserves_fields_and_retains_original_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); business=root/'upload_data'/pub.BUSINESS
            folder=business/'batches/20261028000000000-0000';folder.mkdir(parents=True)
            original={'op':'upsert','id':'ResearchReportImages:x','data':{'id':'ResearchReportImages:x','title':'t','image_content':'c','image_file':'pic/20261028/x.jpg'}}
            files=pub.write_shards(folder,[original])
            m=dict(protocolVersion='1.0',businessCode=pub.BUSINESS,batchVersion=folder.name,batchType='incremental',producedAt='2026-10-28T00:00:00Z',format='jsonl',compression='gzip',contentEncoding='utf-8',files=files,totalFileCount=len(files),totalRecordCount=1)
            pub.save_json(folder/'manifest.json',m)
            pub.update_meta(business,[m])
            before=emb.source_hashes(business)
            cache=root/'cache';cache.mkdir()
            with sqlite3.connect(cache/'test.sqlite3') as db:
                db.execute('CREATE TABLE vectors (key TEXT PRIMARY KEY, vector BLOB)')
                v=np.zeros(2048,dtype='<f4');v[0]=1
                db.execute('INSERT INTO vectors VALUES (?,?)',('key',v.tobytes()))
                emb.rewrite(business,[(m,[original])],db,{original['id']:'key'},cache,before)
            result=pub.read_json(cache/'completed.json')
            self.assertEqual(emb.source_hashes(Path(result['backup'])),before)
            manifest=pub.read_json(folder/'manifest.json')
            path=folder/manifest['files'][0]['fileName']
            record=json.loads(gzip.decompress(path.read_bytes()))
            self.assertEqual(record['data'].pop('embedding'),v.tolist())
            self.assertEqual(record,original)
            pub.history(business)
            # Plain local extraction records must not overwrite/re-publish
            # the enriched package solely because they omit embedding.
            data=root/'data';(data/'jsonl').mkdir(parents=True)
            image=root/'upload_data/pic/20261028/x.jpg'
            image.parent.mkdir(parents=True);image.write_bytes(b'test-image')
            local={**original['data'],'id':'x','image_file':'../../upload_data/pic/20261028/x.jpg'}
            (data/'jsonl/20261028.jsonl').write_text(json.dumps(local)+'\n')
            frozen=emb.source_hashes(business)
            self.assertIsNone(pub.prepare(data,'20261028'))
            self.assertEqual(emb.source_hashes(business),frozen)


if __name__=='__main__': unittest.main()
