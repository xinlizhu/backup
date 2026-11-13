import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

from PIL import Image
from visual_recall.light import neighboring_blocks, figure_mapping, build_light_evidence
from visual_recall.llm_client import build_messages
from extract_visual_records_light import run


def b(kind, text=''):
    return {'type': kind, 'text': text, 'id': text}


class LightTests(unittest.TestCase):
    def test_heading_boundaries_and_order(self):
        seq = [b('paragraph','old'),b('heading','h'),b('paragraph','p1'),b('image_ref','fig'),b('paragraph','p2'),b('paragraph','p3')]
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,3,True)], ['h','p1'])
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,3,False)], ['p2','p3'])
        seq.pop(2)
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,2,True)], ['h'])

    def test_leading_visuals_share_prose_but_heading_stops_search(self):
        seq = [b('heading','Estimates'),b('paragraph','EPS estimates increased'),b('table_ref','t1'),b('table_ref','t2'),b('heading','next'),b('paragraph','unrelated')]
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,3,True)], ['Estimates','EPS estimates increased'])
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,3,False)], ['next'])

    def test_paragraph_quota(self):
        seq = [b('heading','h'),b('paragraph','p0'),b('paragraph','p1'),b('image_ref','f')]
        self.assertEqual([x['text'] for x in neighboring_blocks(seq,3,True)], ['p0','p1'])

    def fixture(self, root):
        d = root/'report'; (d/'images').mkdir(parents=True)
        Image.new('RGB',(1200,800),'blue').save(d/'images/main_figure1_plot.jpg')
        report={'raw_title':'Test report','institution':'Test Bank','raw_date':'2026-01-01', 'data':{'main_content':[
            b('heading','Nearby heading'),b('paragraph','Useful paragraph'),
            {'id':'main_figure1','type':'image_ref','context':'Plot'},b('paragraph','After')]}}
        items=[{'type':'image','image_caption':['Logo'],'page_idx':0,'bbox':[0,0,10,10]},
               {'type':'image','image_caption':['Plot'],'page_idx':1,'bbox':[100,100,900,800],'table_body':'NEVER SEND HTML'}]
        (d/'report.json').write_text(json.dumps(report));(d/'report_content_list.json').write_text(json.dumps(items))
        return d,items

    def test_filtered_mapping_not_ordinal_and_no_extra_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,items=self.fixture(root)
            entries,errs,n=build_light_evidence(d,root/'out')
            self.assertEqual(errs,[]);self.assertEqual(n,1)
            e,payload,method=entries[0]
            self.assertEqual(e.page_idx,1);self.assertEqual(method,'unique_caption')
            self.assertIsNone(e.page_image);self.assertEqual(e.table_body,'')
            self.assertNotIn('table_body',payload);self.assertNotIn('section_title',payload['visual'])
            msgs=build_messages(payload,e.target_image,None,system_prompt='test',user_text=json.dumps(payload))
            self.assertEqual(sum(x['type']=='image_url' for x in msgs[1]['content']),1)
            self.assertFalse((root/'out/pages').exists())
            items.append(dict(items[1]))
            mapping,errors=figure_mapping(d,[{'id':'main_figure1','context':'Plot'}],items)
            self.assertEqual(mapping,{});self.assertIn('main_figure1',errors)

    def test_repeated_captions_require_unique_sequence(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,items=self.fixture(root)
            Image.new('RGB',(1200,800),'red').save(d/'images/main_figure2_plot.jpg')
            items.append(dict(items[1],page_idx=2))
            refs=[{'id':'main_figure1','context':'Plot Source: Bank'}, {'id':'main_figure2','context':'Plot'}]
            mapping,errors=figure_mapping(d,refs,items)
            self.assertEqual(errors,{})
            self.assertEqual(mapping['main_figure1'][0]['page_idx'],1)
            self.assertEqual(mapping['main_figure2'][0]['page_idx'],2)
            items.append(dict(items[1],page_idx=3))
            mapping,errors=figure_mapping(d,refs,items)
            self.assertEqual(mapping,{})
            self.assertEqual(len(errors),2)

    def test_footnote_only_mapping_and_unknown_page(self):
        from visual_recall.schema import build_final_record, validate_final_record
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,items=self.fixture(root)
            items[1]['image_caption']=[]
            items[1]['image_footnote']=['Source: IDC']
            mapping,errors=figure_mapping(d,[{'id':'main_figure1','context':'Source: IDC'}],items)
            self.assertEqual(errors,{})
            self.assertEqual(mapping['main_figure1'][0]['page_idx'],1)
            report=json.loads((d/'report.json').read_text())
            report['data']['main_content'][2]['context']=''
            (d/'report.json').write_text(json.dumps(report))
            entries,errors,n=build_light_evidence(d,root/'out')
            self.assertEqual(errors,[])
            e,_,method=entries[0]
            self.assertEqual(method,'named_image_without_layout')
            self.assertIsNone(e.page_idx)
            self.assertTrue(e.mapping_warnings)
            record=build_final_record(e,title='Useful title',image_content='Description',image_file=e.target_image,
                source_footnote='Source: Bank',language='en',is_blurry=False,publish_ready=True)
            self.assertEqual(validate_final_record(record,allow_unknown_figure_page=True),[])
            record['type']='table'
            self.assertTrue(validate_final_record(record,allow_unknown_figure_page=True))

    def test_schema_request_and_failed_response_preserved(self):
        from visual_recall.llm_client import QwenOpenAIClient, MODEL_FIELDS, ModelOutputError
        client=QwenOpenAIClient.__new__(QwenOpenAIClient)
        client.model='test';client.temperature=.1;client.max_tokens=1800;client.json_mode=True;client.client=Mock()
        raw='{"decision":"keep"}'
        client.client.chat.completions.create.return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw),finish_reason='stop')],
            usage=None,id='test',model='test',created=1)
        with tempfile.TemporaryDirectory() as t:
            image=Path(t)/'test.jpg';Image.new('RGB',(10,10)).save(image)
            with self.assertRaises(ModelOutputError) as caught:
                client.analyze({},image,None,strict_schema=True,allow_json_fallback=False)
        self.assertEqual(caught.exception.raw,raw)
        schema=client.client.chat.completions.create.call_args.kwargs['response_format']
        self.assertEqual(schema['type'],'json_schema')
        self.assertEqual(set(schema['json_schema']['schema']['required']),MODEL_FIELDS)
        self.assertFalse(schema['json_schema']['schema']['additionalProperties'])

    def test_table_crops_without_page_or_html(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,_=self.fixture(root)
            report=json.loads((d/'report.json').read_text()); report['data']['main_content']=[{'id':'main_table1','type':'table_ref','context':''}]
            (d/'report.json').write_text(json.dumps(report))
            (d/'report_content_list.json').write_text(json.dumps([{'type':'table','table_body':'<table>secret</table>','bbox':[100,100,800,800],'page_idx':0}]))
            class Renderer:
                error=''
                def __init__(self,*a):pass
                def close(self):pass
                def render_crop(self,page,bbox,path):
                    path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(300,300)).save(path);return path
            with patch('visual_recall.light.PdfAssetRenderer',Renderer):
                entries,errs,_=build_light_evidence(d,root/'out')
            self.assertEqual(errs,[]);self.assertNotIn('secret',json.dumps(entries[0][1]))
            self.assertTrue(entries[0][0].target_image.is_file())

    def test_table_fragment_rules_and_legitimate_comparisons(self):
        from visual_recall.reliability import table_review_reasons
        self.assertEqual(table_review_reasons('<table><tr><td>12/26E</td><td>12/27E</td></tr><tr><td>25,657</td><td>27,105</td></tr></table>'),['table_fragment_missing_metric_labels'])
        self.assertEqual(table_review_reasons('<table><tr><td>Total assets</td><td>2264.6</td><td>2055.8</td><td>2311.7</td></tr></table>'),['financial_statement_fragment_missing_period_headers'])
        self.assertEqual(table_review_reasons('<table><tr><td>Forecast return</td><td>24.4%</td></tr></table>'),[])
        self.assertEqual(table_review_reasons(''),[])

    def test_small_figure_is_recropped_from_pdf_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,_=self.fixture(root)
            source=d/'images/main_figure1_plot.jpg'
            Image.new('RGB',(411,225),'blue').save(source)
            original=source.read_bytes()
            class Renderer:
                error=''
                def __init__(self,*a):pass
                def close(self):pass
                def render_crop(self,page,bbox,path,*,dpi):
                    self.dpi=dpi
                    path.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(1200,700),'blue').save(path)
                    return path
            with patch('visual_recall.light.PdfAssetRenderer',Renderer):
                entries,errs,_=build_light_evidence(d,root/'out')
            e,payload,_=entries[0]
            self.assertEqual(errs,[])
            self.assertNotEqual(e.target_image,source)
            self.assertEqual(source.read_bytes(),original)
            self.assertTrue(e.asset_info['recropped'])
            self.assertEqual(e.asset_info['crop_dpi'],300)
            self.assertEqual(payload['evidence_review_reasons'],[])
            self.assertIsNone(e.page_image)
            class Broken(Renderer):
                error='PDF renderer unavailable'
            with patch('visual_recall.light.PdfAssetRenderer',Broken):
                entries,errs,_=build_light_evidence(d,root/'failed')
            self.assertEqual(entries[0][0].target_image,source)
            self.assertEqual(entries[0][1]['evidence_review_reasons'],['low_resolution_figure_recrop_failed'])

    def test_high_confidence_does_not_override_missing_labels(self):
        from visual_recall.reliability import model_review_reasons
        from visual_recall.llm_client import validate_retrieval_output
        output={'decision':'keep','confidence':1.0,'evidence_status':'missing_period_labels'}
        self.assertEqual(model_review_reasons(output),['model_evidence_status:missing_period_labels'])
        self.assertTrue(validate_retrieval_output({'decision':'keep'}))

    def test_publish_original_and_resume_by_content(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d,_=self.fixture(root)
            args=argparse.Namespace(report_dir=d,output_dir=root/'out',only='',max_items=0,crop_dpi=220,bbox_padding=.004,
                model='qwen3.8-27b',api_base='http://unused',dry_run=False,temperature=.1,max_tokens=1800,
                no_json_mode=False,overwrite=False,api_key='EMPTY',timeout=180,retries=0,min_confidence=.75)
            parsed={'decision':'keep','category':'chart','filter_reason':'Useful chart','title':'Test company revenue chart',
                    'image_content':'The chart shows revenue for the company across the displayed periods, with values in millions.',
                    'is_blurry':False,'confidence':.95,'evidence_status':'sufficient'}
            with patch('extract_visual_records_light.QwenOpenAIClient') as cls, patch('builtins.print'):
                cls.return_value.analyze.return_value=(parsed,json.dumps(parsed),{'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}})
                self.assertEqual(run(args),0)
                record=json.loads((root/'out/published_visuals.jsonl').read_text())
                self.assertEqual(record['image_file'],str(d/'images/main_figure1_plot.jpg'))
                self.assertEqual(run(args),0)
                self.assertEqual(cls.return_value.analyze.call_count,1)
                self.assertEqual(len((root/'out/published_visuals.jsonl').read_text().splitlines()),1)
                self.assertFalse((root/'out/rendered').exists())
                args.temperature=.2
                self.assertEqual(run(args),0)
                self.assertEqual(cls.return_value.analyze.call_count,2)
                parsed['evidence_status']='missing_metric_labels'
                args.temperature=.3
                self.assertEqual(run(args),0)
                self.assertEqual((root/'out/published_visuals.jsonl').read_text(),'')
                audit=json.loads((root/'out/visual_audit.jsonl').read_text())
                self.assertIn('model_evidence_status:missing_metric_labels',audit['qc_errors'])


if __name__=='__main__':unittest.main()
