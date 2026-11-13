"""Run validated PyTorch CIR adapters on original full retrieval benchmarks."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import subprocess

import numpy as np
from PIL import Image
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT.parent/'diag_model'))
from common import write_json_atomic,save_npy_atomic,sha256_file
from external_qdiag_adapters import make_adapter
from score_external_retrieval import score_dataset

DATASETS=['cirr_val','fashioniq_dress','fashioniq_shirt','fashioniq_toptee','pinpoint']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['tema','clip4cir','oprd_head'],required=True)
    p.add_argument('--model-name',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--clip-checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifests',type=Path,default=PROJECT/'outputs/external_retrieval_20261109/manifests')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--image-batch-size',type=int,default=256)
    p.add_argument('--query-batch-size',type=int,default=256)
    p.add_argument('--workers',type=int,default=24)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    adapter=None
    weight_sha=sha256_file(args.checkpoint)
    clip_sha=sha256_file(args.clip_checkpoint) if args.clip_checkpoint else None
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    for name in args.datasets:
        started=time.time()
        mp=args.manifests/(name+'.json');manifest=json.loads(mp.read_text())
        folder=root/name;folder.mkdir(exist_ok=True)
        identity=dict(manifest_sha256=sha256_file(mp),checkpoint_sha256=weight_sha,clip_checkpoint_sha256=clip_sha,
                      adapter_sha256=sha256_file(Path(__file__).with_name('external_qdiag_adapters.py')),
                      protocol='native-reference-feature-mean-v1')
        if args.model == 'oprd_head':
            identity['head_adapter_sha256'] = sha256_file(Path(__file__).with_name('oprd_head_adapter.py'))
        identity_path=folder/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise RuntimeError(f'Identity mismatch: {folder}')
        write_json_atomic(identity_path,identity)
        if (folder/'run_metadata.json').exists() and json.loads((folder/'run_metadata.json').read_text()).get('complete'):
            print('[skip completed]',name,flush=True);continue
        gallery=manifest['gallery'];queries=manifest['queries']
        ids=[x['id'] for x in gallery];index={v:i for i,v in enumerate(ids)}
        ref_ids=sorted({r for q in queries for r in q['references']});ref_index={r:i for i,r in enumerate(ref_ids)}
        assert set(ref_ids)<=set(ids)
        gallery_path=folder/'gallery_features.npy';query_path=folder/'query_features.npy'
        metadata=dict(model=args.model_name,checkpoint=str(args.checkpoint),gallery_count=len(ids),query_count=len(queries),
                      reference_feature_pooling='mean of native visual features/tokens before fusion',
                      TEMA_equal_length_batches=args.model=='tema',environment=sys.executable,
                      image_batch_size=args.image_batch_size,query_batch_size=args.query_batch_size,complete=False)
        write_json_atomic(folder/'run_metadata.json',metadata)
        if not gallery_path.exists() or not query_path.exists():
            if adapter is None:
                print('[load model]',args.model_name,flush=True)
                adapter=make_adapter(args.model,args.checkpoint,args.device,args.clip_checkpoint)
                print('[loaded] strict weights',flush=True)
            def prepare(i):
                with Image.open(gallery[i]['path']) as im:
                    return adapter.preprocess(im.convert('RGB'))
            ref_path=folder/'reference_visual_features.npy'
            if not gallery_path.exists():
                tmp=folder/'gallery_features.tmp.npy';rtmp=folder/'reference_visual_features.tmp.npy'
                progress_path=folder/'image_progress.json'
                start=json.loads(progress_path.read_text())['next_index'] if progress_path.exists() and tmp.exists() and rtmp.exists() else 0
                gf=np.load(tmp,mmap_mode='r+') if start else None
                rf=np.load(rtmp,mmap_mode='r+') if start else None
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for i in range(start,len(gallery),args.image_batch_size):
                        end=min(i+args.image_batch_size,len(gallery))
                        images=torch.stack(list(pool.map(prepare,range(i,end))))
                        base=adapter.image_base(images)
                        target=adapter.targets(base).float().cpu().numpy()
                        assert np.isfinite(target).all()
                        if gf is None:
                            gf=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(ids),target.shape[-1]))
                            rf=np.lib.format.open_memmap(rtmp,mode='w+',dtype=np.float32,shape=(len(ref_ids),*base.shape[1:]))
                        gf[i:end]=target
                        selected=[(j,ref_index[ids[i+j]]) for j in range(end-i) if ids[i+j] in ref_index]
                        if selected:
                            local,destination=zip(*selected)
                            rf[list(destination)]=base[list(local)].float().cpu().numpy()
                        gf.flush();rf.flush()
                        write_json_atomic(progress_path,dict(next_index=end,total=len(gallery)))
                        print(f'[{name} images] {end}/{len(gallery)} elapsed={time.time()-started:.1f}s',flush=True)
                del gf,rf
                tmp.replace(gallery_path);rtmp.replace(ref_path)
            if not query_path.exists():
                refs=np.load(ref_path,mmap_mode='r')
                buckets=defaultdict(list)
                for i,q in enumerate(queries):
                    key=adapter.text_length(q['text']) if adapter.equal_text_lengths else 0
                    buckets[key].append(i)
                tmp=folder/'query_features.tmp.npy';done_path=folder/'query_done.npy'
                done=np.load(done_path) if tmp.exists() and done_path.exists() else np.zeros(len(queries),bool)
                qf=np.load(tmp,mmap_mode='r+') if done.any() else None
                for bucket in buckets.values():
                    pending=[i for i in bucket if not done[i]]
                    for begin in range(0,len(pending),args.query_batch_size):
                        indices=pending[begin:begin+args.query_batch_size]
                        values=np.stack([np.array(refs[[ref_index[r] for r in queries[i]['references']]]).mean(0) for i in indices])
                        vectors=adapter.queries(torch.from_numpy(values),[queries[i]['text'] for i in indices]).float().cpu().numpy()
                        assert np.isfinite(vectors).all()
                        if qf is None:qf=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(queries),vectors.shape[-1]))
                        qf[indices]=vectors;qf.flush();done[indices]=True;save_npy_atomic(done_path,done)
                        print(f'[{name} queries] {done.sum()}/{len(queries)} elapsed={time.time()-started:.1f}s',flush=True)
                assert done.all()
                del qf,refs
                tmp.replace(query_path)
        metadata.update(features_complete=True,feature_seconds=time.time()-started)
        write_json_atomic(folder/'run_metadata.json',metadata)
        score_dataset(mp,root,args.model_name,args.checkpoint,args.device)
    print('[done model]',args.model_name,flush=True)


if __name__=='__main__':main()
