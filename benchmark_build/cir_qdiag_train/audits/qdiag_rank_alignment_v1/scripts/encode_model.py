import argparse
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import _bootstrap
import numpy as np
import torch
from alignment_data import ROOT,digest,load_config,prepare,read_json,read_jsonl,sha,write_json
from alignment_models import make_adapter,rgb


def check_vectors(values,dim):
    values=np.asarray(values,dtype=np.float32)
    assert values.ndim==2 and values.shape[1]==dim and np.isfinite(values).all()
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    assert np.all(norms>0)
    return values/norms


def encode_batches(path,total,dim,batch_size,prepare_batch,encode_batch):
    progress=path.with_suffix('.progress.json')
    if path.exists():
        arr=np.load(path,mmap_mode='r');assert arr.shape==(total,dim)
        print('[reuse]',path.name,arr.shape,flush=True);return arr
    tmp=path.with_suffix('.tmp.npy');start=read_json(progress)['next_index'] if progress.exists() else 0
    if start and not tmp.exists():raise ValueError('Progress without partial array')
    arr=np.lib.format.open_memmap(tmp,mode='r+' if start else 'w+',dtype=np.float32,shape=(total,dim))
    begun=time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(prepare_batch,start,min(total,start+batch_size)) if start<total else None
        for begin in range(start,total,batch_size):
            end=min(total,begin+batch_size);prepared=future.result()
            future=pool.submit(prepare_batch,end,min(total,end+batch_size)) if end<total else None
            values=check_vectors(encode_batch(prepared),dim);assert len(values)==end-begin
            arr[begin:end]=values
            if end==total or (begin-start)//batch_size%5==0:
                arr.flush();write_json(progress,{'next_index':end,'total':total,'elapsed_this_invocation':time.time()-begun})
                print(f'[{path.name}] {end}/{total} elapsed={time.time()-begun:.1f}s',flush=True)
    arr.flush();del arr;tmp.replace(path);write_json(progress,{'next_index':total,'total':total,'complete':True})
    return np.load(path,mmap_mode='r')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full_four_models.yaml')
    p.add_argument('--model',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();c=load_config(args.config);meta=prepare(c);spec=c['models'][args.model]
    out=Path(c['runtime']['output_dir'])/args.model;out.mkdir(exist_ok=True)
    gallery=read_json(Path(c['runtime']['output_dir'])/'gallery.json')
    queries=read_jsonl(Path(c['runtime']['output_dir'])/'queries.jsonl');idmap={x['id']:i for i,x in enumerate(gallery)}
    torch.set_num_threads(c['runtime']['torch_threads'])
    if not torch.cuda.is_available():raise RuntimeError('A GPU-enabled process is required')
    adapter=make_adapter(spec,c['runtime']['device'])
    identity={'dataset_signature':meta['signature'],'model_spec':spec,
        'weights_sha256':{p.name:sha(p) for p in sorted(Path(spec['path']).glob('*.safetensors'))},
        'pytorch_model_sha256':sha(Path(spec['path'])/'pytorch_model.bin') if (Path(spec['path'])/'pytorch_model.bin').exists() else None,
        'adapter_class_sha256':digest(inspect.getsource(type(adapter))),'encoder_sha256':sha(Path(__file__)),
        'model_metadata_sha256':{p.name:sha(p) for p in sorted(Path(spec['path']).glob('*.json'))},
        'normalization':'All stored embeddings normalized in FP32; CLIP reference mean is normalized before residual fusion; MLLM multi-reference queries use normalized mean of native per-reference query embeddings.'}
    marker=out/'encoding_identity.json'
    if marker.exists() and read_json(marker)!=identity:
        if any(out.glob('*features*.npy')):raise ValueError('Encoding identity changed')
        write_json(out/'encoding_identity.previous_smoke.json',read_json(marker))
    write_json(marker,identity)
    if args.smoke:
        ims=[rgb(x['path']) for x in gallery[:2]];t=time.time()
        vectors=check_vectors(adapter.encode_images(adapter.prepare_images(ims)),adapter.dim)
        result={'model':args.model,'image_shape':list(vectors.shape),'seconds':time.time()-t,'finite':True,
                'torch':torch.__version__,'allocated_gb':torch.cuda.max_memory_allocated()/1e9}
        if hasattr(adapter,'prepare_queries'):
            q=check_vectors(adapter.encode_images(adapter.prepare_queries(ims,['Find the same object.']*2)),adapter.dim)
            result['query_shape']=list(q.shape)
        else:
            qs=[{'text':'Find the same object.','reference_ids':[gallery[0]['id']]},
                {'text':'Combine these references.','reference_ids':[gallery[0]['id'],gallery[1]['id']]}]
            q=check_vectors(adapter.encode_queries_from_gallery(qs,vectors,{r['id']:i for i,r in enumerate(gallery[:2])}),adapter.dim)
            result['query_shape']=list(q.shape)
        write_json(out/'smoke.json',result);print(result,flush=True);return
    with ThreadPoolExecutor(max_workers=c['runtime']['image_workers']) as images_pool:
        def image_batch(begin,end):return adapter.prepare_images(list(images_pool.map(rgb,[g['path'] for g in gallery[begin:end]])))
        gf=encode_batches(out/'gallery_features.npy',len(gallery),adapter.dim,spec['image_batch_size'],image_batch,adapter.encode_images)
        if hasattr(adapter,'encode_queries_from_gallery'):
            qf=encode_batches(out/'query_features.npy',len(queries),adapter.dim,spec['query_batch_size'],
                lambda b,e:queries[b:e],lambda rows:adapter.encode_queries_from_gallery(rows,gf,idmap))
        else:
            def query_batch(begin,end):
                rows=queries[begin:end];paths=[];texts=[];counts=[]
                for q in rows:
                    counts.append(len(q['reference_ids']))
                    for rid in q['reference_ids']:paths.append(gallery[idmap[rid]]['path']);texts.append(q['text'])
                ims=list(images_pool.map(rgb,paths))
                return adapter.prepare_queries(ims,texts),counts
            def query_encode(value):
                prepared,counts=value;flat=adapter.encode_images(prepared);rows=[];offset=0
                for n in counts:rows.append(flat[offset:offset+n].mean(0));offset+=n
                return np.stack(rows)
            qf=encode_batches(out/'query_features.npy',len(queries),adapter.dim,spec['query_batch_size'],query_batch,query_encode)
    write_json(out/'encoding_complete.json',{'model':args.model,'gallery_shape':list(gf.shape),'query_shape':list(qf.shape),
        'identity_signature':digest(identity),'complete':True})


if __name__=='__main__':main()
