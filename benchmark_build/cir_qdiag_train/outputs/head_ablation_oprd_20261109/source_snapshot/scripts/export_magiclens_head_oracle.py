"""Run with magic_lens311 on CPU: export official init, forward and VJP oracle."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'models/MagicLens'))
from layers import StackedTransformer, AttenTokenPoolingLayer


class OfficialHead(nn.Module):
    @nn.compact
    def __call__(self, image, text):
        x = jnp.stack((image, text), axis=1)
        x = StackedTransformer(num_layers=4, num_heads=16, input_dim=768,
            hidden_dim=3072, use_bias=True, add_skip_connection=True,
            use_per_dim_scale=False, name='multimodal_encoder')(x)
        x = AttenTokenPoolingLayer(input_dim=768, query_dim=768, num_heads=16,
            num_query_tokens=1, use_bias=True, use_per_dim_scale=True,
            name='contrastive_multimodal_pooler')(x)[:, 0].astype(jnp.float32)
        return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    rng = np.random.default_rng(42)
    image, text, probe = [rng.normal(size=(3, 768)).astype(np.float32) for _ in range(3)]
    m = OfficialHead()
    state = m.init(jax.random.PRNGKey(42), image, text)['params']
    result = m.apply({'params': state}, image, text)
    def loss(params, image, text):
        return (m.apply({'params': params}, image, text) * probe).sum()
    gp, gi, gt = jax.grad(loss, argnums=(0, 1, 2))(state, image, text)
    flat = lambda tree: {'.'.join(k): np.asarray(v) for k, v in flatten_dict(tree).items()}
    np.savez(a.output / 'magiclens_large_init_seed42.npz', **flat(state))
    np.savez(a.output / 'magiclens_large_oracle.npz', image=image, text=text, probe=probe,
             result=np.asarray(result), image_grad=np.asarray(gi), text_grad=np.asarray(gt),
             **{'grad.'+k: v for k, v in flat(gp).items()})
    meta = {'seed': 42, 'jax': jax.__version__, 'configuration': 'official large',
            'parameter_count': sum(v.size for v in flat(state).values()),
            'sources': {f: hashlib.sha256((ROOT / 'models/MagicLens' / f).read_bytes()).hexdigest()
                        for f in ['model.py', 'layers.py']}}
    (a.output / 'magiclens_head_oracle.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
