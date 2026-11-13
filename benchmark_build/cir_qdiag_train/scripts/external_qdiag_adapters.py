"""Inference adapters for the published CLIP4CIR, TEMA and QuRe checkpoints.

Each process loads one official repository. Compatibility changes only bypass
unrelated LAVIS imports, obsolete version guards, and redundant base downloads;
strict final-state loading is required. No retrieval layers are replaced.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MODELS = Path('/mnt/data/datasets/search_dataset/zxl/models')


class TargetPad:
    def __init__(self, ratio=1.25):
        self.ratio = ratio

    def __call__(self, image):
        w, h = image.size
        if max(w, h) / min(w, h) < self.ratio:
            return image
        side = max(w, h) / self.ratio
        hp, vp = max(int((side-w)/2), 0), max(int((side-h)/2), 0)
        return transforms.functional.pad(image, [hp, vp, hp, vp], 0, 'constant')


def preprocess(size):
    return transforms.Compose([
        TargetPad(), transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_lavis(repo):
    """Load only the inference dependency tree, not the entire LAVIS model zoo."""
    for name in ['lavis', 'lavis.common', 'lavis.models', 'lavis.models.blip_models', 'lavis.models.blip2_models']:
        module = types.ModuleType(name)
        module.__path__ = [str(repo.joinpath(*name.split('.')))]
        sys.modules[name] = module
    from lavis.common.registry import registry
    registry.register_path('library_root', str(repo / 'lavis'))
    registry.register_path('cache_root', str(MODELS / '.lavis_cache'))
    import lavis.models.base_model as base
    sys.modules['lavis.models'].BaseModel = base.BaseModel
    import transformers.modeling_utils as mu
    import transformers.pytorch_utils as pu
    for name in ['apply_chunking_to_forward', 'find_pruneable_heads_and_indices', 'prune_linear_layer']:
        if not hasattr(mu, name):
            setattr(mu, name, getattr(pu, name))


def load_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    for key in ['model', 'state_dict']:
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if all(k.startswith('module.') for k in obj):
        obj = {k[7:]: v for k, v in obj.items()}
    return obj


class Tema:
    name = 'TEMA-MCIRR'
    multi_vector = False
    equal_text_lengths = True

    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        self.repo = MODELS / 'TEMA/repo'
        setup_lavis(self.repo)
        from transformers import BertTokenizer
        from lavis.models.base_model import BaseModel
        from lavis.models.blip_models.blip import BlipBase
        # The custom MED BERT implementation uses the compatible 4.x API.
        BlipBase.__init__ = BaseModel.__init__

        def tokenizer(cls):
            tok = BertTokenizer.from_pretrained(str(MODELS / 'bert-base-uncased'), local_files_only=True)
            tok.add_special_tokens({'bos_token': '[DEC]'})
            tok.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            tok.enc_token_id = tok.convert_tokens_to_ids('[ENC]')
            return tok

        BlipBase.init_tokenizer = classmethod(tokenizer)
        from lavis.models.blip_models.blip_cir import BlipCir
        from lavis.models.med import XBertEncoder
        from lavis.models.vit import VisionTransformerEncoder
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(self.repo / 'lavis/configs/models/blip_cir.yaml').model
        cfg.pretrained = None
        cfg.vit_grad_ckpt = False
        state = load_state(checkpoint)
        dim = state['vision_proj.weight'].shape[0]
        # The official implementation calls .cuda() on these Parameters before
        # assigning them: they become unregistered zero tensors. Accordingly the
        # official checkpoint has no ini_ground_* keys. Preserve its q=1 default.
        q = state['ini_ground_img'].shape[1] if 'ini_ground_img' in state else 1
        self.model = BlipCir(VisionTransformerEncoder.from_config(cfg), XBertEncoder.from_config(cfg),
                             embed_dim=dim, cfg=cfg, max_txt_len=512, q=q)
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.to(device).eval()
        self.preprocess = preprocess(224)

    def text_length(self, text):
        return len(self.model.tokenizer(text)['input_ids'])

    @torch.inference_mode()
    def image_base(self, images):
        return self.model.visual_encoder.forward_features(images.to(self.device)).float()

    @torch.inference_mode()
    def targets(self, base):
        m = self.model
        base = base.to(self.device).float()
        local, glob = m.vision_proj(base[:, 1:]), m.vision_proj(base[:, 0])
        ground = m.transformer1(torch.cat([glob.unsqueeze(1).expand(-1, m.q, -1), local], 1))[:, :m.q]
        return F.normalize(torch.cat([glob.unsqueeze(1), ground], 1).mean(1), dim=-1)

    @torch.inference_mode()
    def queries(self, base, captions):
        m = self.model
        base = base.to(self.device).float()
        local, glob = m.vision_proj(base[:, 1:]), m.vision_proj(base[:, 0])
        b, d = glob.shape
        text = m.tokenizer(captions, return_tensors='pt', padding=True, max_length=m.max_txt_len).to(self.device)
        text.input_ids[:, 0] = m.tokenizer.enc_token_id
        encoded = m.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_local = m.text_proj(encoded.last_hidden_state[:, 1:])
        text_glob = m.text_proj(encoded.last_hidden_state[:, 0])
        ig = m.transformer1(torch.cat([m.ini_ground_img.expand(b, m.q, d), local], 1))[:, :m.q]
        tg = m.transformer2(torch.cat([m.ini_ground_txt.expand(b, m.q, d), text_local], 1))[:, :m.q]
        fusion = m.affine(torch.cat([glob.unsqueeze(1), ig], 1), torch.cat([text_glob.unsqueeze(1), tg], 1))
        return F.normalize(fusion.mean(1), dim=-1)

    @torch.inference_mode()
    def native_smoke(self, images, captions):
        # Official TEMA leaves pad tokens unmasked in its entity mapper, so
        # compare individual queries and batch only equal token lengths later.
        return (torch.cat([self.model.extract_retrieval_compose(im[None].to(self.device), [cap])
                           for im, cap in zip(images, captions)]),
                self.model.extract_retrieval_target(images.to(self.device)))


class Clip4Cir:
    name = 'CLIP4CIR-CIRR'
    multi_vector = False
    equal_text_lengths = False

    def __init__(self, checkpoint, device, clip_checkpoint=None):
        import clip
        self.clip = clip
        self.device = torch.device(device)
        self.repo = MODELS / 'CLIP4CIR/repo'
        state = torch.load(clip_checkpoint, map_location='cpu', weights_only=False)
        state = state.get('CLIP', state)
        self.model = clip.model.build_model(state).to(device).float().eval()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        weights = weights.get('Combiner', weights)
        combiner_module = module_from_file('official_clip4cir_combiner', self.repo / 'src/combiner.py')
        d = weights['output_layer.weight'].shape[0]
        projection = weights['text_projection_layer.weight'].shape[0]
        hidden = weights['output_layer.weight'].shape[1]
        self.combiner = combiner_module.Combiner(d, projection, hidden).to(device).eval()
        self.combiner.load_state_dict(weights, strict=True)
        self.preprocess = preprocess(int(self.model.visual.input_resolution))

    @torch.inference_mode()
    def image_base(self, images):
        with torch.autocast('cuda', dtype=torch.float16):
            return self.model.encode_image(images.to(self.device)).float()

    @torch.inference_mode()
    def targets(self, base):
        return F.normalize(base.to(self.device).float(), dim=-1)

    @torch.inference_mode()
    def queries(self, base, captions):
        tokens = self.clip.tokenize(captions, truncate=True).to(self.device)
        with torch.autocast('cuda', dtype=torch.float16):
            text = self.model.encode_text(tokens).float()
        return self.combiner.combine_features(base.to(self.device).float(), text)


class Qure:
    name = 'QuRe-CIRR'
    multi_vector = True
    equal_text_lengths = False

    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        self.repo = MODELS / 'QuRe/repo'
        setup_lavis(self.repo)
        sys.path.insert(0, str(self.repo))
        from transformers import BertTokenizer
        from lavis.models.blip2_models.blip2 import Blip2Base, LayerNorm
        from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
        from lavis.models.clip_vit import VisionTransformer

        def tokenizer(cls, truncation_side='right'):
            tok = BertTokenizer.from_pretrained(str(MODELS / 'bert-base-uncased'), local_files_only=True,
                                                truncation_side=truncation_side)
            tok.add_special_tokens({'bos_token': '[DEC]'})
            return tok

        def qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
            cfg = BertConfig.from_json_file(str(MODELS / 'bert-base-uncased/config.json'))
            cfg.encoder_width, cfg.add_cross_attention = vision_width, True
            cfg.cross_attention_freq, cfg.query_length = cross_attention_freq, num_query_token
            return BertLMHeadModel(cfg), torch.nn.Parameter(torch.zeros(1, num_query_token, cfg.hidden_size))

        def vision(self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision):
            assert model_name == 'clip_L'
            v = VisionTransformer(input_resolution=img_size, patch_size=14, width=1024,
                                  layers=23, heads=16, use_grad_checkpointing=False)
            self.vit_name = model_name
            return v, LayerNorm(v.num_features)

        Blip2Base.init_tokenizer = classmethod(tokenizer)
        Blip2Base.init_Qformer = classmethod(qformer)
        Blip2Base.init_vision_encoder = vision
        from lavis.models.blip2_models.blip2_qure import Blip2QuRe
        self.model = Blip2QuRe(vit_model='clip_L', img_size=224, freeze_vit=True).to(device).eval()
        self.model.load_state_dict(load_state(checkpoint), strict=True)
        self.preprocess = preprocess(224)

    @torch.inference_mode()
    def image_base(self, images):
        with self.model.maybe_autocast():
            value = self.model.visual_encoder(images.to(self.device))
        return self.model.ln_vision(value).float()

    @torch.inference_mode()
    def targets(self, base):
        m = self.model
        base = base.to(self.device).float()
        query = m.query_tokens.expand(base.shape[0], -1, -1)
        output = m.Qformer.bert(query_embeds=query, encoder_hidden_states=base,
                               encoder_attention_mask=torch.ones(base.shape[:-1], device=self.device, dtype=torch.long),
                               return_dict=True)
        return F.normalize(m.vision_proj(output.last_hidden_state), dim=-1)

    @torch.inference_mode()
    def queries(self, base, captions):
        import re
        m = self.model
        captions = [' '.join(re.sub(r'([.!"()*#:;~])', ' ', s.lower()).split()) for s in captions]
        captions = [' '.join(s.split()[:50]) for s in captions]
        text = m.tokenizer(captions, padding='max_length', truncation=True,
                           max_length=m.max_txt_len, return_tensors='pt').to(self.device)
        base = base.to(self.device).float()
        query = m.query_tokens.expand(base.shape[0], -1, -1)
        qatts = torch.ones(query.shape[:-1], device=self.device, dtype=torch.long)
        output = m.Qformer.bert(text.input_ids, query_embeds=query,
                               attention_mask=torch.cat([qatts, text.attention_mask], 1),
                               encoder_hidden_states=base,
                               encoder_attention_mask=torch.ones(base.shape[:-1], device=self.device, dtype=torch.long),
                               return_dict=True)
        return F.normalize(m.text_proj(output.last_hidden_state[:, :query.shape[1]]).mean(1), dim=-1)


def make_adapter(name, checkpoint, device, clip_checkpoint=None):
    if name == 'oprd_head':
        from oprd_head_adapter import OprdHeadAdapter
        return OprdHeadAdapter(checkpoint, device)
    if name == 'tema':
        return Tema(checkpoint, device)
    if name == 'qure':
        return Qure(checkpoint, device)
    if name == 'clip4cir':
        return Clip4Cir(checkpoint, device, clip_checkpoint)
    raise ValueError(name)
