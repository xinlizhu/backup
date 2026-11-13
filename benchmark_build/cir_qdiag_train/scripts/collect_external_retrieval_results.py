"""Collect complete benchmark results and verify overlap with prior QDiag runs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT.parent/'diag_model'))
from common import write_json_atomic,sha256_file
from pinpoint_full_diagnostics import load_v7_dataset

DATASETS=['cirr_val','fashioniq_dress','fashioniq_shirt','fashioniq_toptee','pinpoint']


def verify_qdiag_overlap(model_root,qdiag_root,manifest_root):
    previous=qdiag_root/model_root.name
    if not previous.exists():return {'status':'prior QDiag run unavailable'}
    identity=json.loads((previous/'identity.json').read_text())
    for dataset in DATASETS:
        current=json.loads((model_root/dataset/'identity.json').read_text())
        assert current['checkpoint_sha256']==identity['checkpoint_sha256']
        if 'clip_checkpoint_sha256' in current:
            assert current['clip_checkpoint_sha256']==identity.get('clip_checkpoint_sha256')
    pp=json.loads((manifest_root/'pinpoint.json').read_text())
    ids=[v['id'] for v in pp['gallery']];idx={v:i for i,v in enumerate(ids)}
    old_ids=json.loads((previous/'cache/target_ids.json').read_text())
    old_targets=np.load(previous/'cache/target_features.npy',mmap_mode='r')
    targets=np.load(model_root/'pinpoint/gallery_features.npy',mmap_mode='r')
    image_error=float(np.max(np.abs(np.array(targets[[idx[v] for v in old_ids]])-old_targets)))
    _,old_queries,_=load_v7_dataset(PROJECT.parent/'cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl',limit_groups=None)
    old_q=np.load(previous/'cache/query_features.npy',mmap_mode='r')
    current_q=np.load(model_root/'pinpoint/query_features.npy',mmap_mode='r')
    by_input={ (q['text'],tuple(q['references'])):i for i,q in enumerate(pp['queries']) }
    pairs=[(q.query_index,by_input[(q.text,tuple(q.reference_ids))]) for q in old_queries
           if q.query_kind=='positive' and (q.text,tuple(q.reference_ids)) in by_input]
    assert len(pairs)==1933,len(pairs)
    old_indices,current_indices=zip(*pairs)
    query_error=float(np.max(np.abs(np.array(current_q[list(current_indices)])-old_q[list(old_indices)])))
    # fp16 CLIP convolutions can differ slightly with batch/tail sizes. Large
    # deviations would indicate a changed model, preprocessing, or reference policy.
    assert image_error<2e-3 and query_error<2e-3,(image_error,query_error)
    result=dict(same_checkpoint_hashes=True,target_overlap=len(old_ids),positive_query_overlap=len(pairs),
                target_max_abs_error=image_error,query_max_abs_error=query_error)
    write_json_atomic(model_root/'qdiag_overlap_verification.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=PROJECT/'outputs/external_retrieval_20261109')
    p.add_argument('--verify-overlap',action='store_true')
    args=p.parse_args()
    summary=[];full={};details=[]
    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir() or folder.name=='manifests':continue
        metadatas=[folder/d/'run_metadata.json' for d in DATASETS]
        if not all(p.exists() and json.loads(p.read_text()).get('complete') for p in metadatas):
            print('[pending]',folder.name,flush=True);continue
        cirr=json.loads((folder/'cirr_val/val_metrics.json').read_text())
        fiq=json.loads((folder/'fashioniq_metrics.json').read_text())
        pp=json.loads((folder/'pinpoint/metrics.json').read_text())
        assert cirr['num_queries']==4181 and fiq['num_queries']==6016 and int(pp['total_queries'])==7635
        if args.verify_overlap:
            overlap=verify_qdiag_overlap(folder,PROJECT/'outputs/external_qdiag_20261109',args.root/'manifests')
        else:overlap=None
        full[folder.name]=dict(cirr=cirr,fashioniq=fiq,pinpoint=pp,qdiag_overlap=overlap)
        row={'model':folder.name}
        row.update({f'CIRR_{k}':v for k,v in cirr.items() if 'recall_at_' in k})
        row.update({f'FIQ_macro_{k}':v for k,v in fiq['macro_average'].items()})
        row.update({f'FIQ_micro_{k}':v for k,v in fiq['micro_average'].items()})
        row.update({f'PinPoint_{k}':v for k,v in pp.items() if k not in ['model','total_queries']})
        summary.append(row)
        for category,values in fiq['categories'].items():details.append(dict(model=folder.name,category=category,**values))
        print('[complete]',folder.name,'CIRR R1',cirr['recall_at_1'],'FIQ R10',fiq['macro_average']['recall_at_10'],'PinPoint mAP10',pp['mAP@10'],flush=True)
    if summary:
        pd.DataFrame(summary).to_csv(args.root/'comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(details).to_csv(args.root/'fashioniq_categories.csv',index=False,encoding='utf-8-sig')
        write_json_atomic(args.root/'all_metrics.json',full)
        write_json_atomic(args.root/'evaluation_protocol.json',{
            'coverage':json.loads((args.root/'manifests/coverage.json').read_text()),
            'evaluators':{str(p):sha256_file(p) for p in [PROJECT/'scripts/evaluate_cirr_original_adapter.py',
                PROJECT/'src/cir_qdiag_train/fashioniq_evaluation.py',PROJECT.parents[1]/'pinpoint-dataset/evaluate.py']},
            'weights':'Same checkpoint files as external_qdiag_20261109; no dataset-specific checkpoint switching.',
            'multi_reference':'Mean native reference visual features before fusion, as in the completed QDiag runs.',
            'units':'CIRR/FashionIQ Recall are percentages; PinPoint metrics retain official 0-1 units.'})


if __name__=='__main__':main()
