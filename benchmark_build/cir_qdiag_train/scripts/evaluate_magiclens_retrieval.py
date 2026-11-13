"""Native JAX MagicLens inference with the original local retrieval evaluators."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

from export_magiclens_qdiag import (
    PROJECT,WORKSPACE,REPO,MagicLens,clip_tokenizer,clip,serialization,
    image_method,text_method,fusion_method,sha,dump,save,jax,jnp,np,torch,F,Image,
)

DATASETS=['cirr_val','fashioniq_dress','fashioniq_shirt','fashioniq_toptee','pinpoint']


class Backend:
    def __init__(self,size,image_batch,text_batch):
        self.image_batch=image_batch;self.text_batch=text_batch
        self.model=MagicLens(size)
        vocab=Path(clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'
        self.tokenize=clip_tokenizer.build_tokenizer(bpe_path=str(vocab),truncate=True)
        empty=self.tokenize('').astype(np.int32)
        print('[model] initialize official parameter tree',flush=True)
        initial=self.model.init(jax.random.PRNGKey(0),{'ids':jnp.asarray(empty),'image':jnp.ones((1,224,224,3),jnp.float32)})
        with (REPO/'models'/f'magic_lens_clip_{size}.pkl').open('rb') as f:payload=pickle.load(f)
        loaded=serialization.from_bytes(initial,payload)
        a,ta=jax.tree_util.tree_flatten(initial);b,tb=jax.tree_util.tree_flatten(loaded)
        assert ta==tb and [x.shape for x in a]==[x.shape for x in b]
        self.params=jax.device_put(loaded)
        del initial,loaded,payload,a,b
        self.encode_image=jax.jit(lambda p,x:self.model.apply(p,x,method=image_method))
        self.encode_text=jax.jit(lambda p,x:self.model.apply(p,x,method=text_method))
        self.fuse=jax.jit(lambda p,x,y:self.model.apply(p,x,y,method=fusion_method))
        self.empty_feature=np.asarray(self.encode_text(self.params,empty))
        self.empty_tokens=empty
        print('[model] checkpoint tree and shape verification passed',flush=True)

    @staticmethod
    def pad(x,n):
        return np.concatenate([x,np.repeat(x[-1:],n-len(x),axis=0)]) if len(x)<n else x

    @staticmethod
    def prepare(path):
        with Image.open(path) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)
        a=a/(a.max()+1e-12)
        x=torch.from_numpy(a).permute(2,0,1)[None]
        return F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0].permute(1,2,0).numpy()

    def image_base(self,pixels):
        return np.asarray(self.encode_image(self.params,self.pad(pixels,self.image_batch)))[:len(pixels)]

    def targets(self,base):
        values=[]
        for i in range(0,len(base),self.text_batch):
            b=base[i:i+self.text_batch]
            empty=np.repeat(self.empty_feature,self.text_batch,axis=0)
            values.append(np.asarray(self.fuse(self.params,self.pad(b,self.text_batch),empty))[:len(b)])
        return np.concatenate(values)

    def queries(self,base,texts):
        ids=self.tokenize(texts).astype(np.int32)
        text_features=self.encode_text(self.params,self.pad(ids,self.text_batch))
        return np.asarray(self.fuse(self.params,self.pad(base,self.text_batch),text_features))[:len(texts)]

    def text_length(self,text):
        return len(self.tokenize.keywords['tokenizer'].encode(text))+2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',choices=['base','large'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifests',type=Path,default=PROJECT/'outputs/external_retrieval_20261109/manifests')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--image-batch-size',type=int,default=128)
    p.add_argument('--text-batch-size',type=int,default=512)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    jax.config.update('jax_default_matmul_precision','highest')
    assert any(x.platform=='gpu' for x in jax.devices())
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    name='MagicLens-CLIP-'+args.size.title()
    checkpoint=REPO/'models'/f'magic_lens_clip_{args.size}.pkl'
    weight_sha=sha(checkpoint)
    backend=None
    for dataset in args.datasets:
        started=time.time()
        mp=args.manifests/(dataset+'.json');manifest=json.loads(mp.read_text())
        folder=root/dataset;folder.mkdir(exist_ok=True)
        identity=dict(manifest_sha256=sha(mp),checkpoint_sha256=weight_sha,exporter_sha256=sha(Path(__file__)),
                      helper_sha256=sha(Path(__file__).with_name('export_magiclens_qdiag.py')),
                      protocol='native-MagicLens-reference-mean-before-transformer-v1')
        if (folder/'identity.json').exists() and json.loads((folder/'identity.json').read_text())!=identity:
            raise RuntimeError(f'Identity mismatch: {folder}')
        dump(folder/'identity.json',identity)
        if (folder/'run_metadata.json').exists() and json.loads((folder/'run_metadata.json').read_text()).get('complete'):
            print('[skip completed]',dataset,flush=True);continue
        gallery=manifest['gallery'];queries=manifest['queries']
        ids=[x['id'] for x in gallery];index={v:i for i,v in enumerate(ids)}
        gallery_path=folder/'gallery_features.npy';query_path=folder/'query_features.npy'
        metadata=dict(model=name,checkpoint=str(checkpoint),gallery_count=len(ids),query_count=len(queries),
                      environment=sys.executable,jax_version=jax.__version__,jax_devices=str(jax.devices()),
                      image_batch_size=args.image_batch_size,text_batch_size=args.text_batch_size,
                      preprocessing='native MagicLens max-normalization + antialiased bilinear square resize',
                      reference_feature_pooling='mean raw CLIP image embeddings before multimodal Transformer',
                      target_encoding='image + empty text through full multimodal model',complete=False)
        dump(folder/'run_metadata.json',metadata)
        if not gallery_path.exists() or not query_path.exists():
            if backend is None:
                backend=Backend(args.size,args.image_batch_size,args.text_batch_size)
                sample_queries=queries[:3]
                pixels=np.stack([backend.prepare(gallery[index[q['references'][0]]]['path']) for q in sample_queries])
                texts=[q['text'] for q in sample_queries]
                base=backend.image_base(pixels)
                adapted=backend.queries(base,texts);adapted_target=backend.targets(base)
                native=jax.jit(lambda p,x,t:backend.model.apply(p,{'image':x,'ids':t})['multimodal_embed_norm'])
                native_q=np.asarray(native(backend.params,pixels,backend.tokenize(texts).astype(np.int32)))
                native_t=np.asarray(native(backend.params,pixels,np.repeat(backend.empty_tokens,3,axis=0)))
                np.testing.assert_allclose(adapted,native_q,atol=3e-5,rtol=1e-4)
                np.testing.assert_allclose(adapted_target,native_t,atol=3e-5,rtol=1e-4)
                smoke=dict(native_query_max_abs_error=float(abs(adapted-native_q).max()),
                           native_target_max_abs_error=float(abs(adapted_target-native_t).max()))
                dump(root/'smoke.json',smoke);print('[smoke]',smoke,flush=True)
            base_path=folder/'image_base.npy'
            if not base_path.exists():
                tmp=folder/'image_base.tmp.npy';progress=folder/'image_progress.json'
                start=json.loads(progress.read_text())['next_index'] if progress.exists() and tmp.exists() else 0
                features=np.load(tmp,mmap_mode='r+') if start else None
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for i in range(start,len(gallery),args.image_batch_size):
                        rows=gallery[i:i+args.image_batch_size]
                        pixels=np.stack(list(pool.map(backend.prepare,[r['path'] for r in rows])))
                        result=backend.image_base(pixels)
                        assert np.isfinite(result).all()
                        if features is None:features=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(gallery),result.shape[-1]))
                        features[i:i+len(rows)]=result;features.flush()
                        dump(progress,dict(next_index=i+len(rows),total=len(gallery)))
                        print(f'[{dataset} images] {i+len(rows)}/{len(gallery)} elapsed={time.time()-started:.1f}s',flush=True)
                del features
                tmp.replace(base_path)
            base_features=np.load(base_path,mmap_mode='r')
            if not gallery_path.exists():
                tmp=folder/'gallery_features.tmp.npy'
                gf=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=base_features.shape)
                for i in range(0,len(gallery),args.text_batch_size):
                    gf[i:i+args.text_batch_size]=backend.targets(np.array(base_features[i:i+args.text_batch_size]))
                gf.flush();del gf;tmp.replace(gallery_path)
                print(f'[{dataset} targets] complete',flush=True)
            if not query_path.exists():
                tmp=folder/'query_features.tmp.npy';progress=folder/'query_progress.json'
                start=json.loads(progress.read_text())['next_index'] if tmp.exists() and progress.exists() else 0
                qf=np.lib.format.open_memmap(tmp,mode='r+' if start else 'w+',dtype=np.float32,
                                            shape=(len(queries),base_features.shape[-1]))
                for i in range(start,len(queries),args.text_batch_size):
                    rows=queries[i:i+args.text_batch_size]
                    refs=np.stack([np.array(base_features[[index[r] for r in q['references']]]).mean(0) for q in rows])
                    vectors=backend.queries(refs,[q['text'] for q in rows])
                    assert np.isfinite(vectors).all()
                    qf[i:i+len(rows)]=vectors;qf.flush()
                    dump(progress,dict(next_index=i+len(rows),total=len(queries)))
                    print(f'[{dataset} queries] {i+len(rows)}/{len(queries)} elapsed={time.time()-started:.1f}s',flush=True)
                del qf;tmp.replace(query_path)
            metadata['truncated_query_count']=sum(backend.text_length(q['text'])>77 for q in queries)
        metadata.update(features_complete=True,feature_seconds=time.time()-started)
        dump(folder/'run_metadata.json',metadata)
        subprocess.run([str(WORKSPACE/'multi_modal_py3.12/bin/python'),'-u',str(PROJECT/'scripts/score_external_retrieval.py'),
                        '--manifest',str(mp),'--model-root',str(root),'--model-name',name,'--checkpoint',str(checkpoint)],check=True)
    print('[done model]',name,flush=True)


if __name__=='__main__':main()
