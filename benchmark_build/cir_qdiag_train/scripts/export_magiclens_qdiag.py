"""Export native MagicLens scores in its JAX environment; statistics run separately.

Single-reference outputs are checked against the unmodified official forward.
Multiple references use the mean of unnormalized CLIP image embeddings before
the official multimodal Transformer. Targets use image + empty-text encoding.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import shutil
import sys
import time
import types

import numpy as np
from PIL import Image
import jax
import jax.numpy as jnp
from flax import serialization
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
REPO = WORKSPACE / 'models/MagicLens'
sys.path.insert(0, str(REPO))


def local_scenic_file_io():
    """Scenic uses TensorFlow only for checkpoint file I/O in these modules.

    Execute their original source with a local stdlib gfile implementation.
    All neural layers, configuration constants and preprocessing are unchanged.
    This avoids importing a 1+ GB TensorFlow library for already-local weights.
    """
    gfile = types.SimpleNamespace(GFile=open, exists=os.path.exists, isdir=os.path.isdir,
                                  makedirs=lambda p:os.makedirs(p,exist_ok=True),
                                  remove=os.remove, copy=lambda a,b,overwrite=False:shutil.copyfile(a,b))
    for name in ['scenic.projects.baselines.clip.download', 'scenic.projects.baselines.clip.model']:
        spec = importlib.util.find_spec(name)
        module = importlib.util.module_from_spec(spec)
        module.gfile = gfile
        source = Path(spec.origin).read_text()
        assert 'from tensorflow.io import gfile' in source
        source = source.replace('from tensorflow.io import gfile', '# local gfile supplied by QDiag runner')
        sys.modules[name] = module
        exec(compile(source,spec.origin,'exec'),module.__dict__)


local_scenic_file_io()
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
import clip


def dump(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(4*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp.npy')
    np.save(temp, value, allow_pickle=False)
    temp.replace(path)


def image_method(module, pixels):
    return module.clip.encode_image(module._preprocess_images(pixels), normalize=False)


def text_method(module, ids):
    return module.clip.encode_text(ids, normalize=False)


def fusion_method(module, images, texts):
    sequence = jnp.stack([images, texts], axis=1)
    output = module.multimodal_encoder(sequence)
    output = module.contrastive_multimodal_pooler(output)[:, 0]
    return module._normalize_embed(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=['base','large'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-batch-size', type=int, default=128)
    parser.add_argument('--text-batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    jax.config.update('jax_default_matmul_precision', 'highest')
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('A CUDA JAX device is required')
    print('[devices]',jax.devices(),flush=True)
    started = time.time()
    root = args.output
    cache = root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    source = PROJECT.parent / 'cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl'
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    targets = sorted({str(t['image_id']) for r in records for t in r['positive_images']})
    image_ids = sorted(set(targets) | {str(t['image_id']) for r in records for t in r['references']})
    queries = []
    for record in records:
        refs = [str(x['image_id']) for x in record['references']]
        for instruction in record['instructions']:
            queries.append({'key':'positive::'+instruction['instruction_id'], 'text':instruction['text'], 'refs':refs})
        for negative in record['negatives']:
            queries.append({'key':'negative::'+negative['negative_id'], 'text':negative['text'], 'refs':refs})
    index = {value:i for i,value in enumerate(image_ids)}
    paths = {}
    image_root = WORKSPACE/'pinpoint-dataset/image_cache'
    for image_id in image_ids:
        paths[image_id] = next((image_root/image_id[:2]/(image_id+ext) for ext in ['.jpg','.jpeg','.png','.webp']
                               if (image_root/image_id[:2]/(image_id+ext)).is_file()),None)
    missing = [k for k,v in paths.items() if v is None]
    preflight = {'groups':len(records),'queries':len(queries),'negative_count':sum(len(r['negatives']) for r in records),
                 'images':len(image_ids),'target_images':len(targets),'missing_images':missing}
    dump(root/'preflight.json',preflight)
    print('[preflight]',preflight,flush=True)
    if missing:
        raise FileNotFoundError(missing)
    checkpoint = REPO/'models'/f'magic_lens_clip_{args.size}.pkl'
    identity = {'size':args.size,'checkpoint_sha256':sha(checkpoint),'input_sha256':sha(source),
                'script_sha256':sha(Path(__file__)),'protocol':'native-MagicLens-reference-mean-before-transformer-v1'}
    if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text())!=identity:
        raise RuntimeError('Output directory identity mismatch')
    dump(root/'identity.json',identity)
    if (cache/'query_target_scores.npy').exists() and not args.smoke_only:
        print('[cached] native scores already exported',flush=True)
        return
    vocab = Path(clip.__file__).parent/'bpe_simple_vocab_16e6.txt.gz'
    tokenize = clip_tokenizer.build_tokenizer(bpe_path=str(vocab),truncate=False)
    tokens = tokenize([q['text'] for q in queries]).astype(np.int32)
    empty = tokenize('').astype(np.int32)
    model = MagicLens(args.size)
    print('[model] initializing official parameter tree',flush=True)
    initial = model.init(jax.random.PRNGKey(0),{'ids':jnp.asarray(empty),'image':jnp.ones((1,224,224,3),jnp.float32)})
    with checkpoint.open('rb') as f:
        payload = pickle.load(f)
    loaded = serialization.from_bytes(initial,payload)
    original_leaves, original_tree = jax.tree_util.tree_flatten(initial)
    leaves, tree = jax.tree_util.tree_flatten(loaded)
    assert tree==original_tree
    assert [x.shape for x in leaves]==[x.shape for x in original_leaves]
    params = jax.device_put(loaded)
    del initial,loaded,payload,original_leaves,leaves
    print('[model] checkpoint tree and shapes verified',flush=True)
    encode_image = jax.jit(lambda p,x:model.apply(p,x,method=image_method))
    encode_text = jax.jit(lambda p,x:model.apply(p,x,method=text_method))
    fuse = jax.jit(lambda p,i,t:model.apply(p,i,t,method=fusion_method))
    native = jax.jit(lambda p,i,t:model.apply(p,{'image':i,'ids':t})['multimodal_embed_norm'])

    def prepare(image_id):
        # CPU antialiased bilinear interpolation is checked below against the
        # official JAX process_img, including its per-image maximum scaling.
        with Image.open(paths[image_id]) as im:
            a=np.asarray(im.convert('RGB'),dtype=np.float32)
        a=a/(a.max()+1e-12)
        x=torch.from_numpy(a).permute(2,0,1)[None]
        return F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0].permute(1,2,0).numpy()

    sample = [q for q in queries if len(q['refs'])==1][:3]
    fast_pixels=np.stack([prepare(q['refs'][0]) for q in sample])
    official_pixels=[]
    for q in sample:
        with Image.open(paths[q['refs'][0]]) as im:
            pixels=np.asarray(im.convert('RGB'))
        # Small number of reference checks: use the original JAX expression.
        with jax.default_device(jax.devices('cpu')[0]):
            a=jnp.asarray(pixels)[None]
            a=a/(a.max()+1e-12)
            official_pixels.append(np.asarray(jax.image.resize(a,(1,224,224,3),method='bilinear'))[0])
    official_pixels=np.stack(official_pixels)
    pdiff=float(np.max(np.abs(fast_pixels-official_pixels)))
    np.testing.assert_allclose(fast_pixels,official_pixels,atol=3e-5,rtol=1e-5)
    sample_tokens=tokenize([q['text'] for q in sample]).astype(np.int32)
    img=encode_image(params,fast_pixels)
    txt=encode_text(params,sample_tokens)
    adapted=np.asarray(fuse(params,img,txt))
    reference=np.asarray(native(params,official_pixels,sample_tokens))
    qdiff=float(np.max(np.abs(adapted-reference)))
    np.testing.assert_allclose(adapted,reference,atol=3e-5,rtol=1e-4)
    empty_embedding=np.asarray(encode_text(params,empty))
    adapted_target=np.asarray(fuse(params,img,np.repeat(empty_embedding,len(sample),axis=0)))
    native_target=np.asarray(native(params,official_pixels,np.repeat(empty,len(sample),axis=0)))
    tdiff=float(np.max(np.abs(adapted_target-native_target)))
    np.testing.assert_allclose(adapted_target,native_target,atol=3e-5,rtol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(adapted,axis=-1),1,atol=1e-5)
    smoke={'native_query_max_abs_error':qdiff,'native_target_max_abs_error':tdiff,
           'preprocess_max_abs_error':pdiff,'checkpoint_tree_verified':True,'query_shape':list(adapted.shape)}
    dump(root/'smoke.json',smoke)
    print('[smoke]',smoke,flush=True)
    if args.smoke_only:return
    metadata={'model_name':'MagicLens-CLIP-'+args.size.title(),'checkpoint':str(checkpoint),
              'repository_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
              'environment':{'python':sys.version,'jax':jax.__version__,'devices':str(jax.devices())},
              'image_preprocessing':'official process_img: RGB, divide by per-image max, bilinear antialiased resize to 224; CPU implementation numerically verified',
              'multi_reference':'mean unnormalized CLIP reference embeddings before official multimodal encoder',
              'target':'official multimodal encoding of target image and empty text',
              'image_batch_size':args.image_batch_size,'text_batch_size':args.text_batch_size,
              'precision':'float32; JAX highest matmul precision','coverage':preflight,'complete':False}
    dump(root/'run_metadata.json',metadata)
    dump(cache/'image_ids.json',image_ids)
    dump(cache/'target_ids.json',targets)
    dump(cache/'query_keys.json',[q['key'] for q in queries])

    def padded(x,size):
        return np.concatenate([x,np.repeat(x[-1:],size-len(x),axis=0)]) if len(x)<size else x

    image_path=cache/'image_base.npy'
    dim=512 if args.size=='base' else 768
    if not image_path.exists():
        temp=cache/'image_base.tmp.npy'
        progress=cache/'image_progress.json'
        begin=json.loads(progress.read_text())['next_index'] if temp.exists() and progress.exists() else 0
        image_features=np.lib.format.open_memmap(temp,mode='r+' if begin else 'w+',dtype=np.float32,shape=(len(image_ids),dim))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in range(begin,len(image_ids),args.image_batch_size):
                ids=image_ids[i:i+args.image_batch_size]
                pixels=np.stack(list(pool.map(prepare,ids)))
                features=np.asarray(encode_image(params,padded(pixels,args.image_batch_size)))[:len(ids)]
                image_features[i:i+len(ids)]=features
                image_features.flush()
                dump(progress,{'next_index':i+len(ids)})
                print(f'[images] {i+len(ids)}/{len(image_ids)} elapsed={time.time()-started:.1f}s',flush=True)
        del image_features
        temp.replace(image_path)
    image_features=np.load(image_path)
    target_path=cache/'target_features.npy'
    if not target_path.exists():
        target_features=np.asarray(fuse(params,image_features[[index[x] for x in targets]],np.repeat(empty_embedding,len(targets),axis=0)))
        save(target_path,target_features)
        print('[targets] complete',flush=True)
    query_path=cache/'query_features.npy'
    if not query_path.exists():
        all_features=[]
        for i in range(0,len(queries),args.text_batch_size):
            batch=queries[i:i+args.text_batch_size]
            text_features=encode_text(params,padded(tokens[i:i+len(batch)],args.text_batch_size))
            refs=np.stack([image_features[[index[x] for x in q['refs']]].mean(0) for q in batch])
            features=np.asarray(fuse(params,padded(refs,args.text_batch_size),text_features))[:len(batch)]
            all_features.append(features)
            print(f'[queries] {i+len(batch)}/{len(queries)} elapsed={time.time()-started:.1f}s',flush=True)
        save(query_path,np.concatenate(all_features))
    q=np.load(query_path)
    t=np.load(target_path)
    scores=np.asarray(jax.jit(lambda a,b:a@b.T)(jnp.asarray(q),jnp.asarray(t)))
    assert scores.shape==(len(queries),len(targets)) and np.isfinite(scores).all()
    save(cache/'query_target_scores.npy',scores)
    metadata.update(scores_complete=True,elapsed_seconds=time.time()-started)
    dump(root/'run_metadata.json',metadata)
    print(f'[done scores] {root} elapsed={metadata["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':main()
