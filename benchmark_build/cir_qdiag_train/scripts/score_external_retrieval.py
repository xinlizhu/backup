"""Score external features using the original local benchmark evaluators."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
WORKSPACE=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT.parent/'diag_model'))
from common import write_json_atomic,sha256_file
from cir_qdiag_train.on_policy_data import load_image_map,load_cirr_positive_records
from cir_qdiag_train.fashioniq_evaluation import load_fashioniq_val,evaluate_feature_files,aggregate_category_metrics
from evaluate_cirr_original_adapter import _evaluate as evaluate_cirr
from pinpoint_full_diagnostics import exact_topk_torch
from run_clip_pinpoint_official import load_official_queries,write_standard_results,run_official_evaluator


def score_dataset(manifest_path,root,model_name,checkpoint,device='cuda:0'):
    # The official PinPoint evaluator changes cwd to the dataset directory.
    # Resolve output/input paths before passing them to that subprocess.
    manifest_path=Path(manifest_path).resolve()
    root=Path(root).resolve()
    manifest=json.loads(manifest_path.read_text())
    name=manifest['name'];folder=root/name
    identity=json.loads((folder/'identity.json').read_text())
    assert identity['manifest_sha256']==sha256_file(manifest_path)
    gallery=folder/'gallery_features.npy';queries=folder/'query_features.npy'
    g=np.load(gallery,mmap_mode='r');q=np.load(queries,mmap_mode='r')
    assert len(g)==len(manifest['gallery']) and len(q)==len(manifest['queries'])
    assert np.isfinite(g).all() and np.isfinite(q).all()
    assert np.max(np.abs(np.linalg.norm(g,axis=1)-1))<1e-4
    assert np.max(np.abs(np.linalg.norm(q,axis=1)-1))<1e-4
    ids=[v['id'] for v in manifest['gallery']]
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if name=='cirr_val':
        cirr=WORKSPACE/'CIR_new/datasets/CIRR'
        annotation=cirr/'cirr/captions/cap.rc2.val.json'
        old_ids,paths=load_image_map(cirr/'cirr/image_splits/split.rc2.val.json',cirr)
        records=load_cirr_positive_records(annotation,old_ids,paths)
        assert ids==old_ids
        metrics=evaluate_cirr(model_name=model_name,model_path=Path(checkpoint),records=records,
                             annotations=json.loads(annotation.read_text()),image_ids=ids,
                             gallery_path=gallery,query_path=queries,device_name=device,score_batch_size=512)
        write_json_atomic(folder/'val_metrics.json',metrics)
    elif name.startswith('fashioniq_'):
        category=name.split('_',1)[1]
        old_ids,_,records=load_fashioniq_val(WORKSPACE/'CIR_new/datasets/fashion-iq',category)
        assert ids==old_ids
        metrics=evaluate_feature_files(gallery_path=gallery,query_path=queries,gallery_ids=ids,records=records,
                                       device=torch.device(device),score_batch_size=512)
        write_json_atomic(folder/'metrics.json',metrics)
        category_paths={c:root/f'fashioniq_{c}'/'metrics.json' for c in ['dress','shirt','toptee']}
        if all(p.exists() for p in category_paths.values()):
            combined=aggregate_category_metrics(model_name=model_name,model_source=str(checkpoint),
                       category_metrics={c:json.loads(p.read_text()) for c,p in category_paths.items()})
            write_json_atomic(root/'fashioniq_metrics.json',combined)
    elif name=='pinpoint':
        pp=WORKSPACE/'pinpoint-dataset'
        frame,_=load_official_queries(pp/'pinpoint_licensed.parquet')
        top_path=folder/'top50.npz'
        if top_path.exists():
            top=np.load(top_path)['indices']
        else:
            scores,top=exact_topk_torch(q,g,top_k=50,device=device,query_batch_size=512,corpus_chunk_size=16384)
            np.savez(top_path,scores=scores,indices=top)
        assert top.shape==(7635,50)
        result_path=folder/(model_name+'.json')
        write_standard_results(result_path,frame,top,ids)
        metrics=run_official_evaluator(pp,result_path,folder/'official_metrics.csv')
        assert int(metrics['total_queries'])==7635
        write_json_atomic(folder/'metrics.json',metrics)
    else:raise ValueError(name)
    metadata=json.loads((folder/'run_metadata.json').read_text())
    metadata.update(complete=True,metrics=metrics,scoring_protocol=manifest['protocol'])
    write_json_atomic(folder/'run_metadata.json',metadata)
    print('[metrics]',name,json.dumps(metrics,ensure_ascii=False),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--model-name',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(8)
    score_dataset(args.manifest,args.model_root,args.model_name,args.checkpoint,args.device)
