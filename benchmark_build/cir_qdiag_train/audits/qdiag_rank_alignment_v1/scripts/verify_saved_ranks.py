"""Independently verify saved ranks by counting competitors, not sorting them."""
import argparse
from pathlib import Path
import _bootstrap
import numpy as np
import torch
from alignment_data import load_config,read_json,read_jsonl,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full_four_models.yaml');p.add_argument('--model',required=True);a=p.parse_args()
 c=load_config(a.config);root=Path(c['runtime']['output_dir']);out=root/a.model
 complete=read_json(out/'ranks/complete.json');assert complete['complete']
 rows=read_jsonl(root/'queries.jsonl');q=np.load(out/'query_features.npy',mmap_mode='r');g=np.load(out/'gallery_features.npy',mmap_mode='r')
 saved={name:np.load(out/'ranks'/f'{name}.npy',mmap_mode='r') for name in ['target_scores','target_ranks','margin10','top50_indices']}
 rng=np.random.default_rng(991);selected=[]
 for kind in ['positive','negative']:
  selected.extend(rng.choice([r['query_index'] for r in rows if r['kind']==kind],16,replace=False).tolist())
 selected=sorted(set(selected+[0,len(q)-1]));bs=c['runtime']['search_batch_size'];by_batch={}
 for index in selected:by_batch.setdefault(index//bs*bs,[]).append(index)
 torch.set_num_threads(c['runtime']['torch_threads']);torch.backends.cuda.matmul.allow_tf32=False
 bank=torch.from_numpy(np.asarray(g).copy()).to(c['runtime']['device']);image_indices=np.arange(len(g));checked=0
 with torch.inference_mode():
  for start,indices in by_batch.items():
   scores=(torch.from_numpy(np.asarray(q[start:min(start+bs,len(q))]).copy()).to(bank.device)@bank.T).cpu().numpy()
   for index in indices:
    row=scores[index-start]
    for j,target in enumerate(rows[index]['target_indices']):
     value=row[target];rank=int(1+np.count_nonzero(row>value)+np.count_nonzero((row==value)&(image_indices<target)))
     assert rank==saved['target_ranks'][index,j],(index,target,rank,int(saved['target_ranks'][index,j]))
     assert np.isclose(value,saved['target_scores'][index,j],atol=2e-7,rtol=0)
     competitors=np.delete(row,target);tenth=np.partition(competitors,len(competitors)-10)[-10]
     assert np.isclose(value-tenth,saved['margin10'][index,j],atol=2e-7,rtol=0)
     checked+=1
    expected=np.lexsort((image_indices,-row))[:50]
    assert np.array_equal(expected,saved['top50_indices'][index])
 result={'complete':True,'model':a.model,'ranking_signature':complete['signature'],'sampled_queries':selected,'target_ranks_checked':checked,'top50_lists_checked':len(selected),'method':'Direct counts of higher scores and earlier tied IDs; independent NumPy lexsort for Top-50; partition over gallery excluding each target for its tenth-competitor margin.'}
 write_json(out/'ranks/verification.json',result);print(result,flush=True)
if __name__=='__main__':main()
