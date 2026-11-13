import argparse,gc,time
from pathlib import Path
import _bootstrap
import numpy as np
import torch
from alignment_data import load_config,read_json,write_json
from alignment_models import make_adapter,rgb

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);args=p.parse_args()
    c=load_config('configs/full_four_models.yaml');s=c['models'][args.model]
    torch.set_num_threads(c['runtime']['torch_threads'])
    root=Path(c['runtime']['output_dir']);gallery=read_json(root/'gallery.json')
    adapter=make_adapter(s,'cuda:0')
    n=s['image_batch_size'];t=time.time()
    prepared=adapter.prepare_images([rgb(x['path']) for x in gallery[:n]])
    print({'model':args.model,'batch_size':n,'cpu_prepare_seconds':time.time()-t,'input_shape':list(prepared['input_ids'].shape)},flush=True)
    values={};timings={}
    for backend in ['sdpa','flash_attention_2']:
        adapter.model.set_attn_implementation(backend)
        adapter.encode_images(adapter.prepare_images([rgb(gallery[0]['path'])]))
        torch.cuda.synchronize();t=time.time()
        values[backend]=adapter.encode_images(prepared);torch.cuda.synchronize()
        timings[backend]=time.time()-t
        print({'model':args.model,'backend':backend,'seconds':timings[backend],'images_per_second':n/timings[backend],'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9},flush=True)
    a=values['sdpa'];b=values['flash_attention_2'];a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    cos=(a*b).sum(1)
    result={'model':args.model,'batch_size':n,'seconds':timings,'cosine_min':float(cos.min()),'cosine_mean':float(cos.mean()),'max_absolute_difference':float(np.abs(a-b).max())}
    print(result,flush=True);write_json(root/f'{args.model}_attention_benchmark.json',result)
if __name__=='__main__':main()
