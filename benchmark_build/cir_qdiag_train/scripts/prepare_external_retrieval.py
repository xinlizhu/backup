"""Prepare manifests using the original local CIRR/FashionIQ/PinPoint protocol."""
import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT.parent/'diag_model'))
from cir_qdiag_train.on_policy_data import load_image_map,load_cirr_positive_records
from cir_qdiag_train.fashioniq_evaluation import load_fashioniq_val,FASHIONIQ_CATEGORIES
from common import write_json_atomic,sha256_file
from pinpoint_full_diagnostics import read_available_signatures,_resolve_image_path
from run_clip_pinpoint_official import load_official_queries,normalize_query_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=PROJECT/'outputs/external_retrieval_20261109/manifests')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifests=[]
    cirr=WORKSPACE/'CIR_new/datasets/CIRR'
    annotation=cirr/'cirr/captions/cap.rc2.val.json'
    split=cirr/'cirr/image_splits/split.rc2.val.json'
    ids,paths=load_image_map(split,cirr)
    records=load_cirr_positive_records(annotation,ids,paths)
    annotations=json.loads(annotation.read_text())
    manifests.append(dict(name='cirr_val',gallery=[dict(id=i,path=str(p)) for i,p in zip(ids,paths)],
        queries=[dict(id=str(r.pair_id),text=r.text,references=[r.reference_id],target=r.target_id,
                      group_members=a['img_set']['members']) for r,a in zip(records,annotations)],
        protocol=dict(split='val',exclude_reference=True,recall_k=[1,5,10,50],group_recall_k=[1,2,3]),
        sources={str(p):sha256_file(p) for p in [annotation,split]}))
    assert len(records)==4181 and len(ids)==2297
    fiq=WORKSPACE/'CIR_new/datasets/fashion-iq'
    expected={'dress':(2017,3817),'shirt':(2038,6346),'toptee':(1961,5373)}
    for category in FASHIONIQ_CATEGORIES:
        ids,paths,records=load_fashioniq_val(fiq,category)
        assert (len(records),len(ids))==expected[category]
        files=[fiq/'captions'/f'cap.{category}.val.json',fiq/'image_splits'/f'split.{category}.val.json']
        manifests.append(dict(name='fashioniq_'+category,gallery=[dict(id=i,path=str(p)) for i,p in zip(ids,paths)],
            queries=[dict(id=str(i),text=r.text,captions=list(r.captions),references=[r.reference_id],target=r.target_id)
                     for i,r in enumerate(records)],
            protocol=dict(split='val',exclude_reference=False,category=category,caption_composition="strip('.?, ') then caption_1 + ' and ' + caption_2"),
            sources={str(p):sha256_file(p) for p in files}))
    pp=WORKSPACE/'pinpoint-dataset'
    frame,queries=load_official_queries(pp/'pinpoint_licensed.parquet')
    ids=read_available_signatures(pp/'index_signatures.txt',pp/'image_cache',limit=None,
                                  report_path=args.output/'missing_pinpoint_corpus.txt')
    known=set(ids)
    missing_refs=sorted({r for q in queries for r in q.reference_ids}-known)
    assert not missing_refs,missing_refs
    assert len(queries)==7635
    manifests.append(dict(name='pinpoint',gallery=[dict(id=i,path=str(_resolve_image_path(i,pp/'image_cache'))) for i in ids],
        queries=[dict(id=normalize_query_id(row.query_id),text=q.text,references=list(q.reference_ids))
                 for row,q in zip(frame.itertuples(),queries)],
        protocol=dict(split='official',exclude_reference=False,top_k=50,official_evaluator=str(pp/'evaluate.py'),
                      official_evaluator_sha256=sha256_file(pp/'evaluate.py')),
        sources={str(p):sha256_file(p) for p in [pp/'pinpoint_licensed.parquet',pp/'index_signatures.txt']}))
    report=[]
    for m in manifests:
        ids=[x['id'] for x in m['gallery']]
        assert len(ids)==len(set(ids))
        assert len(m['queries'])==len({q['id'] for q in m['queries']})
        assert {r for q in m['queries'] for r in q['references']}<=set(ids)
        write_json_atomic(args.output/(m['name']+'.json'),m)
        report.append(dict(dataset=m['name'],queries=len(m['queries']),gallery=len(ids),
                           reference_images=len({r for q in m['queries'] for r in q['references']}),
                           two_reference_queries=sum(len(q['references'])==2 for q in m['queries']),
                           **m['protocol']))
        print(report[-1],flush=True)
    write_json_atomic(args.output/'coverage.json',report)


if __name__=='__main__':main()
