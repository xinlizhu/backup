# SPDX-License-Identifier: GPL-3.0-or-later
"""TG-CIR port of iLearn-Lab/MM23-TG-CIR/model.py, supporting ViT-B/16 and L/14.

Original authors: Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, Liqiang Nie.
Source: https://github.com/iLearn-Lab/MM23-TG-CIR/blob/main/model.py
Ported 2026-09-10. Uses equivalent Hugging Face CLIP weights/token states;
no released TG-CIR checkpoint, OPRD composer, or external teacher is used.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel


class SpatialAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(dim, 1, 1), nn.Sigmoid())

    def forward(self, tokens):
        tokens = tokens.transpose(1, 2)
        return (tokens * self.conv(tokens)).mean(dim=-1)


class TokenLearner(nn.Module):
    def __init__(self, count: int, dim: int):
        super().__init__()
        self.tokenizers = nn.ModuleList([SpatialAttention(dim) for _ in range(count)])

    def forward(self, tokens):
        return torch.stack([layer(tokens) for layer in self.tokenizers], dim=1)


class AttributeBackbone(nn.Module):
    def __init__(self, clip: CLIPModel, global_tokens: int = 4, local_tokens: int = 8):
        super().__init__()
        self.clip = clip
        dim = clip.config.projection_dim
        self.fc = nn.Linear(clip.config.vision_config.hidden_size, dim)
        self.text_fc = nn.Linear(clip.config.text_config.hidden_size, dim)
        self.tokenlearn = TokenLearner(local_tokens, dim)
        self.masks = nn.Embedding(global_tokens, dim)
        with torch.no_grad():
            self.masks.weight.fill_(0.1)
            length = dim // global_tokens
            for index in range(global_tokens):
                self.masks.weight[index, index * length : (index + 1) * length] = 1

    def _attributes(self, global_feature, local_states, projection):
        global_attributes = global_feature[:, None, :] * F.relu(self.masks.weight)[None]
        local_attributes = self.tokenlearn(projection(local_states.float()))
        return torch.cat((global_attributes, local_attributes), dim=1)

    def extract_img_fea(self, pixels):
        output = self.clip.vision_model(pixel_values=pixels)
        # HF last_hidden_state is BEFORE ln_post, exactly as upstream x.
        # CLS is intentionally included in the local tokens (upstream does so).
        global_feature = self.clip.visual_projection(output.pooler_output)
        return self._attributes(global_feature, output.last_hidden_state, self.fc)

    def extract_text_fea(self, input_ids):
        # OpenAI CLIP attends causally to ALL 77 tokens, including zero padding.
        # An HF padding mask or variable-length padding would change local tokens.
        output = self.clip.text_model(input_ids=input_ids)
        states = output.last_hidden_state
        eos = input_ids.argmax(dim=-1)
        global_feature = self.clip.text_projection(
            states[torch.arange(len(states), device=states.device), eos]
        )
        return self._attributes(global_feature, states, self.text_fc)


class TGCIR(nn.Module):
    def __init__(
        self,
        clip: CLIPModel,
        global_tokens: int = 4,
        local_tokens: int = 8,
        temperature: float = 0.05,
    ):
        super().__init__()
        if temperature <= 0 or min(global_tokens, local_tokens) < 1:
            raise ValueError("Temperature and attribute counts must be positive")
        self.backbone = AttributeBackbone(clip, global_tokens, local_tokens)
        dim = clip.config.projection_dim
        self.loss_T = nn.Parameter(torch.tensor([10.0]))
        self.local_weight = nn.Parameter(torch.ones(global_tokens + local_tokens))
        self.s_remain_map = self._mask_map(dim)
        self.t_remain_map = self._mask_map(dim)
        self.t_replace_map = self._mask_map(dim)
        self.temperature = temperature

    @staticmethod
    def _mask_map(dim):
        return nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())

    def student_attributes(self, reference, modification):
        remain = self.s_remain_map(torch.cat((reference, modification), dim=-1))
        return remain * reference + (1 - remain) * modification, remain

    @staticmethod
    def pool(attributes):
        return F.normalize(attributes.float().mean(dim=1), dim=-1)

    def extract_retrieval_compose(self, pixels, input_ids):
        reference = self.backbone.extract_img_fea(pixels)
        modification = self.backbone.extract_text_fea(input_ids)
        return self.pool(self.student_attributes(reference, modification)[0])

    def extract_retrieval_target(self, pixels):
        return self.pool(self.backbone.extract_img_fea(pixels))

    @staticmethod
    def orthogonal_regularization(attributes):
        normalized = F.normalize(attributes.float(), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(attributes.shape[1], device=attributes.device)
        return F.mse_loss(similarity, identity.expand_as(similarity))

    def losses_from_attributes(self, reference, modification, target):
        student, s_remain = self.student_attributes(reference, modification)
        # Upstream teacher detaches ref/text but NOT target attributes.
        ref_detached, mod_detached = reference.detach(), modification.detach()
        t_remain = self.t_remain_map(torch.cat((ref_detached, target), dim=-1))
        t_replace = self.t_replace_map(torch.cat((mod_detached, target), dim=-1))
        teacher = t_remain * ref_detached + t_replace * mod_detached
        weights = self.local_weight[None, :, None]
        teacher_flat = (F.normalize(teacher.float(), dim=-1) * weights).flatten(1)
        target_flat = (F.normalize(target.float(), dim=-1) * weights).flatten(1)
        query, gallery = self.pool(student), self.pool(target)
        labels = torch.arange(len(query), device=query.device)
        # These are batch-internal negatives, NOT mined/random gallery Top-K.
        # Upstream CE has a trainable multiplier initially 10, separate from KL T.
        student_logits = (query @ gallery.T) * self.loss_T
        teacher_logits = (teacher_flat @ target_flat.T) * self.loss_T
        target_unit = F.normalize(target_flat, dim=-1)
        target_distribution = F.softmax(
            (target_unit @ target_unit.T).detach() / self.temperature, dim=-1
        )
        return {
            "stu_rank": F.cross_entropy(student_logits, labels),
            "tea_rank": F.cross_entropy(teacher_logits, labels),
            "tea_mask": F.mse_loss(t_remain + t_replace, torch.ones_like(t_remain)),
            "ortho": sum(
                self.orthogonal_regularization(x) for x in (reference, target, modification)
            )
            / 3,
            "ckd": F.mse_loss(s_remain, t_remain.detach())
            + F.mse_loss(1 - s_remain, t_replace.detach()),
            # Upstream Variable(x2_y2) detaches the teacher distribution.
            # No T**2 multiplier in the released code.
            "kl": F.kl_div(
                F.log_softmax((query @ gallery.T) / self.temperature, dim=-1),
                target_distribution,
                reduction="batchmean",
            ),
        }

    def forward(self, reference_pixels, input_ids, target_pixels):
        # These CLIP ViT backbones have no stochastic dropout; reuse ref/text instead of
        # redundantly encoding them in each branch as upstream does.
        reference = self.backbone.extract_img_fea(reference_pixels)
        modification = self.backbone.extract_text_fea(input_ids)
        target = self.backbone.extract_img_fea(target_pixels)
        return self.losses_from_attributes(reference, modification, target)


def total_loss(losses, weights):
    return sum(losses[name] * weights[name] for name in losses)
