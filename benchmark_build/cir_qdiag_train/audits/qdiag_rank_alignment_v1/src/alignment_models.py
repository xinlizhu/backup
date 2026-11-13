import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def rgb(path):
    with Image.open(path) as im:return im.convert('RGB')


class ClipAdapter:
    def __init__(self,spec,device):
        from transformers import CLIPModel,CLIPProcessor
        from oprd.model import CLIPComposer,from_release
        self.device=device;self.dtype=torch.float16 if spec['dtype']=='fp16' else torch.float32
        if spec['kind']=='clip_release':self.model,self.processor=from_release(spec['path'])
        else:
            self.model=CLIPComposer(CLIPModel.from_pretrained(spec['path'],local_files_only=True))
            self.processor=CLIPProcessor.from_pretrained(spec['path'],local_files_only=True,use_fast=False)
        self.model.to(device=device,dtype=self.dtype).eval();self.dim=self.model.backbone.config.projection_dim
    def prepare_images(self,images):return self.processor(images=images,return_tensors='pt')['pixel_values']
    @torch.inference_mode()
    def encode_images(self,prepared):
        return self.model.encode_image(prepared.to(device=self.device,dtype=self.dtype)).float().cpu().numpy()
    @torch.inference_mode()
    def encode_queries_from_gallery(self,queries,gallery_vectors,id_to_index):
        refs=[]
        for q in queries:
            v=torch.from_numpy(np.asarray(gallery_vectors[[id_to_index[r] for r in q['reference_ids']]],dtype=np.float32).copy()).to(self.device,dtype=self.dtype)
            refs.append(F.normalize(v.mean(0),dim=-1))
        inp=self.processor(text=[q['text'] for q in queries],return_tensors='pt',padding=True,truncation=True)
        text=self.model.encode_text(inp['input_ids'].to(self.device),inp['attention_mask'].to(self.device))
        return self.model.compose(torch.stack(refs),text).float().cpu().numpy()


class BgeAdapter:
    def __init__(self,spec,device):
        from oprd.teacher import S2Teacher,TASK_INSTRUCTION
        self.teacher=S2Teacher(spec['path'],device=device,dtype=spec['dtype'])
        self.model=self.teacher.model;self.model.config.use_cache=False
        self.device=device;self.dtype=next(self.model.parameters()).dtype;self.dim=4096;self.instruction=TASK_INSTRUCTION
    def prepare_images(self,images,texts=None):
        images=[im.resize((512,512)) for im in images]
        query=texts is not None
        prompts=[self.model.prepare_text_input(im,t,'q' if query else 'c',self.instruction if query else None)
            for im,t in zip(images,texts if query else [None]*len(images))]
        return dict(self.model.processor(images=images,text=prompts,padding=True,return_tensors='pt'))
    @torch.inference_mode()
    def encode_images(self,prepared):
        inputs={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in prepared.items()}
        output=self.model(**inputs,use_cache=False,output_hidden_states=False)
        if not torch.is_tensor(output):output=output.last_hidden_state
        return F.normalize(output[:,-1,:],dim=-1).float().cpu().numpy()
    def prepare_queries(self,images,texts):return self.prepare_images(images,texts)


class QwenAdapter:
    def __init__(self,spec,device):
        path=Path(spec['path'])/'scripts/qwen3_vl_embedding.py'
        module_spec=importlib.util.spec_from_file_location('rank_alignment_qwen_native',path)
        module=importlib.util.module_from_spec(module_spec);sys.modules[module_spec.name]=module;module_spec.loader.exec_module(module)
        self.embedder=module.Qwen3VLEmbedder(spec['path'],max_length=8192,min_pixels=4096,max_pixels=262144,
            device=device,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa')
        self.model=self.embedder.model;self.model.config.use_cache=False;self.device=device;self.dim=2048
    def prepare_images(self,images,texts=None):
        query=texts is not None
        instruction='Represent the composed image retrieval query.' if query else 'Represent the image for retrieval.'
        conv=[self.embedder.format_model_input(image=im,text=t,instruction=instruction)
            for im,t in zip(images,texts if query else [None]*len(images))]
        inputs=self.embedder._preprocess_inputs(conv)
        if 'pixel_values' not in inputs or len(inputs['input_ids'])!=len(images) or len(inputs['image_grid_thw'])!=len(images):
            raise ValueError('Qwen vision preprocessing failed; refusing the native NULL fallback')
        return inputs
    @torch.inference_mode()
    def encode_images(self,prepared):
        inputs={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in prepared.items()}
        inputs['use_cache']=False
        output=self.embedder.forward(inputs)
        hidden=output['last_hidden_state'];mask=output['attention_mask']
        positions=(torch.arange(mask.shape[1],device=mask.device)[None]*mask).max(1).values
        values=hidden[torch.arange(len(hidden),device=hidden.device),positions]
        return F.normalize(values,dim=-1).float().cpu().numpy()
    def prepare_queries(self,images,texts):return self.prepare_images(images,texts)


def make_adapter(spec,device):
    if spec['kind'].startswith('clip_'):return ClipAdapter(spec,device)
    if spec['kind']=='qwen':return QwenAdapter(spec,device)
    if spec['kind']=='bge_s2':return BgeAdapter(spec,device)
    raise ValueError(spec['kind'])
