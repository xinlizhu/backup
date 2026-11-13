"""Official composition architectures; raw CLIP features are the inputs.

MagicLens is a layout-preserving PyTorch port of google-deepmind/magiclens
layers.py, revision 5135f58be020614010104c3542d26b9e269d5e9f (Apache-2.0).
Weights, including initialization, are imported from that Flax implementation.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MMRetAddition(nn.Module):
    def forward(self, image, text):
        return F.normalize(image + text, dim=-1)


class MagicLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).square().mean(-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-6) * (1 + self.scale) + self.bias


class MagicProjection(nn.Module):
    def __init__(self, dim, heads, head_dim, output=False):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim, heads, head_dim))
        self.b = nn.Parameter(torch.zeros(dim) if output else torch.zeros(heads, head_dim))
        self.output = output

    def forward(self, x):
        eqn = '...nh,dnh->...d' if self.output else '...d,dnh->...nh'
        return torch.einsum(eqn, x, self.w) + self.b


class MagicPerDimScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.per_dim_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * (1.442695041 / (x.shape[-1] ** 0.5)) * F.softplus(self.per_dim_scale)


class MagicAttention(nn.Module):
    def __init__(self, dim, heads, hidden_dim, per_dim_scale=False):
        super().__init__()
        self.head_dim = hidden_dim // heads
        assert self.head_dim * heads == hidden_dim
        self.key = MagicProjection(dim, heads, self.head_dim)
        self.query = MagicProjection(dim, heads, self.head_dim)
        self.value = MagicProjection(dim, heads, self.head_dim)
        self.post = MagicProjection(dim, heads, self.head_dim, output=True)
        self.per_dim_scale = MagicPerDimScale(self.head_dim) if per_dim_scale else None

    def forward(self, query, key, value):
        q, k, v = self.query(query), self.key(key), self.value(value)
        q = self.per_dim_scale(q) if self.per_dim_scale is not None else q * self.head_dim ** -0.5
        logits = torch.einsum('btnh,bsnh->bnts', q, k)
        logits = 50.0 * torch.tanh(logits / 50.0)
        probs = logits.softmax(-1).to(k.dtype)
        return self.post(torch.einsum('bnts,bsnh->btnh', probs, v))


class MagicWeight(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d_in, d_out))

    def forward(self, x):
        return x @ self.w


class MagicBias(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x + self.b


class MagicFFN(nn.Module):
    def __init__(self, d_in, d_out, relu=True):
        super().__init__()
        self.linear = MagicWeight(d_in, d_out)
        self.bias = MagicBias(d_out)
        self.relu = relu

    def forward(self, x):
        x = self.bias(self.linear(x))
        return F.relu(x) if self.relu else x


class MagicTransformerFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = MagicLayerNorm(dim)
        self.ffn_layer1 = MagicFFN(dim, dim * 4)
        self.ffn_layer2 = MagicFFN(dim * 4, dim, relu=False)

    def forward(self, x):
        return x + self.ffn_layer2(self.ffn_layer1(self.layer_norm(x)))


class MagicTransformer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ff_layer = MagicTransformerFFN(dim)
        self.self_attention = MagicAttention(dim, heads, dim)
        self.layer_norm = MagicLayerNorm(dim)

    def forward(self, x):
        z = self.layer_norm(x)
        return self.ff_layer(x + self.self_attention(z, z, z))


class MagicStack(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        for i in range(4):
            self.add_module(f'x_layers_{i}', MagicTransformer(dim, heads))

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class MagicPooler(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        # Official hidden_dim defaults to 4 * input_dim HERE too; not dim.
        self.pool_attn = MagicAttention(dim, heads, 4 * dim, per_dim_scale=True)
        self.pool_attn_ln = MagicLayerNorm(dim)
        self.pooling_attn_query = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        q = self.pooling_attn_query[None].expand(x.shape[0], -1, -1)
        return self.pool_attn_ln(self.pool_attn(q, x, x))


class MagicLensHead(nn.Module):
    def __init__(self, dim, init_path):
        super().__init__()
        if dim not in (512, 768):
            raise ValueError('Official MagicLens-CLIP supports d=512 or d=768')
        heads = 8 if dim == 512 else 16
        self.multimodal_encoder = MagicStack(dim, heads)
        self.contrastive_multimodal_pooler = MagicPooler(dim, heads)
        if init_path is None:
            raise ValueError('MagicLens requires an initialization exported by the official Flax code')
        with np.load(Path(init_path), allow_pickle=False) as archive:
            state = {k: torch.from_numpy(archive[k].copy()) for k in archive.files}
        self.load_state_dict(state, strict=True)

    def forward(self, image, text):
        # Flax promotes to FP32 via its FP32 weights; preserve that under AMP.
        with torch.autocast(device_type=image.device.type, enabled=False):
            x = self.multimodal_encoder(torch.stack((image.float(), text.float()), dim=1))
            x = self.contrastive_multimodal_pooler(x)[:, 0].float()
            return x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-12)


def make_head(name, dim, init_path=None):
    if name == 'mmret_add':
        return MMRetAddition()
    if name == 'clip4cir_combiner':
        from cir_qdiag_train.official_clip4cir_combiner import Combiner
        return Combiner(dim, 2560, 5120)
    if name == 'magiclens':
        return MagicLensHead(dim, init_path)
    raise ValueError(f'Unknown official composition head: {name}')
