import argparse
from pathlib import Path
import time
import _bootstrap
import numpy as np
import torch
from alignment_data import digest,load_config,prepare,read_json,read_jsonl,sha,write_json


def rank_batch(scores,target_indices,target_counts):
    """Full stable ordering; gallery IDs must already be lexicographically sorted."""
    order=torch.argsort(scores,dim=1,descending=True,stable=True)
    inverse=torch.empty_like(order,dtype=torch.int32)
    inverse.scatter_(1,order,torch.arange(1,scores.shape[1]+1,device=scores.device,dtype=torch.int32)[None].expand_as(order))
    indices=torch.as_tensor(target_indices,device=scores.device,dtype=torch.int64)
    ranks=inverse.gather(1,indices);values=scores.gather(1,indices)
    top_ids=order[:,:50];top_scores=scores.gather(1,top_ids)
    kth9=top_scores[:,9:10];kth10=top_scores[:,10:11]
    boundary=torch.where(ranks<=10,kth10,kth9)
    valid=torch.arange(indices.shape[1],device=scores.device)[None]<torch.as_tensor(target_counts,device=scores.device)[:,None]
    return {'target_scores':values.masked_fill(~valid,float('nan')).cpu().numpy(),
        'target_ranks':ranks.masked_fill(~valid,0).cpu().numpy(),
        'margin10':(values-boundary).masked_fill(~valid,float('nan')).cpu().numpy(),
        'gallery_mean':scores.mean(1).cpu().numpy(),'top50_indices':top_ids.to(torch.int32).cpu().numpy(),
        'top50_scores':top_scores.cpu().numpy()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full_four_models.yaml');p.add_argument('--model',required=True)
    args=p.parse_args();config=load_config(args.config);meta=prepare(config)
    root=Path(config['runtime']['output_dir']);out=root/args.model
    completed=read_json(out/'encoding_complete.json');assert completed['complete']
    q=np.load(out/'query_features.npy',mmap_mode='r');g=np.load(out/'gallery_features.npy',mmap_mode='r')
    queries=read_jsonl(root/'queries.jsonl');assert len(q)==len(queries) and len(g)==meta['gallery_images']
    identity={'encoding_identity':completed['identity_signature'],'dataset_signature':meta['signature'],
        'score_code_sha256':sha(__file__),'gallery_features_sha256':sha(out/'gallery_features.npy'),
        'query_features_sha256':sha(out/'query_features.npy'),'score_dtype':'FP32, TF32 disabled',
        'tie_break':'image ID ascending, stable argsort'}
    result_dir=out/'ranks';result_dir.mkdir(exist_ok=True)
    marker=result_dir/'identity.json'
    if marker.exists() and read_json(marker)!=identity:raise ValueError('Ranking inputs/code changed')
    write_json(marker,identity)
    if (result_dir/'complete.json').exists():print('[reuse ranks]',args.model);return
    torch.set_num_threads(config['runtime']['torch_threads']);torch.backends.cuda.matmul.allow_tf32=False
    device=config['runtime']['device'];bank=torch.from_numpy(np.asarray(g).copy()).to(device)
    counts=np.asarray([len(x['target_indices']) for x in queries]);width=int(counts.max())
    targets=np.zeros((len(q),width),dtype=np.int64)
    for i,row in enumerate(queries):targets[i,:counts[i]]=row['target_indices']
    progress=result_dir/'progress.json';start=read_json(progress)['next_index'] if progress.exists() else 0
    arrays={}
    for name,shape,dtype in [('target_scores',(len(q),width),np.float32),('target_ranks',(len(q),width),np.int32),
        ('margin10',(len(q),width),np.float32),('gallery_mean',(len(q),),np.float32),
        ('top50_indices',(len(q),50),np.int32),('top50_scores',(len(q),50),np.float32)]:
        path=result_dir/(name+'.npy');arrays[name]=np.lib.format.open_memmap(path,mode='r+' if start else 'w+',dtype=dtype,shape=shape)
    bs=config['runtime']['search_batch_size'];begun=time.time()
    with torch.inference_mode():
        for begin in range(start,len(q),bs):
            end=min(len(q),begin+bs);query=torch.from_numpy(np.asarray(q[begin:end]).copy()).to(device)
            scores=query@bank.T
            values=rank_batch(scores,targets[begin:end],counts[begin:end])
            for name,value in values.items():arrays[name][begin:end]=value;arrays[name].flush()
            write_json(progress,{'next_index':end,'total':len(q),'elapsed':time.time()-begun})
            if (begin-start)//bs%10==0 or end==len(q):print(f'[ranks] {args.model} {end}/{len(q)} elapsed={time.time()-begun:.1f}s',flush=True)
    write_json(result_dir/'complete.json',{'complete':True,'signature':digest(identity),'query_count':len(q),'gallery_count':len(g)})


if __name__=='__main__':main()
