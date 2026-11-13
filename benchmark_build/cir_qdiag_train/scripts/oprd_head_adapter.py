"""Inference adapter for the OPRD head experiments, with checkpoint-defined heads."""
from pathlib import Path
import sys
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from cir_qdiag_train.config import ModelConfig
from cir_qdiag_train.model import ComposedRetrievalModel


class OprdHeadAdapter:
    equal_text_lengths = False
    multi_vector = False

    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = state['config']
        m = ModelConfig.model_validate(cfg['model']).model_dump()
        m['gradient_checkpointing'] = False
        pretrained = m.pop('pretrained_model_name_or_path')
        self.model, self.processor = ComposedRetrievalModel.from_pretrained(
            pretrained, **m, temperature=cfg['optimizer']['temperature'],
            learnable_temperature=cfg['optimizer']['learnable_temperature'])
        self.model.load_trainable_state_dict(state['model'])
        self.model.to(self.device).eval()
        self.name = 'CLIP-L-OPRD-' + m['composition_head']
        self.repo = PROJECT
        self.precision = cfg['optimizer']['mixed_precision']

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == 'cuda' and self.precision == 'bf16')

    def preprocess(self, image):
        return self.processor(images=image, return_tensors='pt')['pixel_values'][0]

    def feature_to_device(self, features):
        dtype = torch.bfloat16 if self.device.type == 'cuda' and self.precision == 'bf16' else torch.float32
        return features.to(device=self.device, dtype=dtype)

    @torch.inference_mode()
    def image_base(self, images):
        with self.autocast():
            return self.model._encode_image_raw(images.to(self.device)).float()

    @torch.inference_mode()
    def targets(self, base):
        with self.autocast():
            return self.model.target_from_raw(self.feature_to_device(base)).float()

    @torch.inference_mode()
    def queries(self, base, captions):
        tokens = self.processor(text=captions, padding=True, truncation=True,
                                max_length=77, return_tensors='pt').to(self.device)
        with self.autocast():
            text = self.model.encode_text(tokens['input_ids'], tokens.get('attention_mask'))
            ref = self.model.reference_from_raw(self.feature_to_device(base))
            return self.model.compose(ref, text).float()

    @torch.inference_mode()
    def native_smoke(self, images, captions):
        queries = []
        with self.autocast():
            ref = self.model.encode_reference(images.to(self.device))
            for i, caption in enumerate(captions):
                tokens = self.processor(text=[caption], padding=True, truncation=True,
                                        max_length=77, return_tensors='pt').to(self.device)
                text = self.model.encode_text(tokens['input_ids'], tokens.get('attention_mask'))
                queries.append(self.model.compose(ref[i:i+1], text))
            targets = self.model.encode_image(images.to(self.device))
        return torch.cat(queries).float(), targets.float()
