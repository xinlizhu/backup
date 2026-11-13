"""Compare head values and derivatives against the original published code."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from cir_qdiag_train.composition_heads import make_head


def close(actual, expected, atol=3e-6, rtol=3e-5):
    expected = torch.as_tensor(expected, dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual.detach() - expected.detach()).abs().max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    sources = ROOT / 'models/head_architecture_sources'
    report = {}
    # Execute the actual official function with raw-feature getters as its inputs.
    source = sources / 'modeling_MMRet_CLIP.py'
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'encode_multimodal')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    image, text = [torch.randn(3, 768, requires_grad=True) for _ in range(2)]
    obj = SimpleNamespace(get_image_features=lambda x: x, get_text_features=lambda **kw: kw['features'])
    official = scope['encode_multimodal'](obj, image, {'features': text})
    actual = make_head('mmret_add', 768)(image, text)
    probe = torch.randn_like(image)
    errors = [close(actual, official)]
    ga = torch.autograd.grad((actual * probe).sum(), (image, text))
    go = torch.autograd.grad((official * probe).sum(), (image, text))
    errors += [close(a, b) for a, b in zip(ga, go)]
    report['mmret_add'] = {'maximum_error': max(errors), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'parameters': 0}
    source = ROOT / 'models/CLIP4CIR/repo/src/combiner.py'
    local = PROJECT / 'src/cir_qdiag_train/official_clip4cir_combiner.py'
    assert source.read_bytes() == local.read_bytes()
    spec = importlib.util.spec_from_file_location('official_combiner_oracle', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    official = mod.Combiner(768, 2560, 5120)
    actual = make_head('clip4cir_combiner', 768)
    actual.load_state_dict(official.state_dict(), strict=True)
    errors = []
    for training in (True, False):
        actual.train(training); official.train(training)
        torch.manual_seed(7)
        a = actual.combine_features(image, text)
        torch.manual_seed(7)
        o = official.combine_features(image, text)
        errors.append(close(a, o, 0, 0))
        ga = torch.autograd.grad((a * probe).sum(), (image, text, *actual.parameters()))
        go = torch.autograd.grad((o * probe).sum(), (image, text, *official.parameters()))
        errors += [close(a, b, 0, 0) for a, b in zip(ga, go)]
    report['clip4cir_combiner'] = {'maximum_error': max(errors), 'parameters': sum(p.numel() for p in actual.parameters()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    del actual, official, ga, go
    actual = make_head('magiclens', 768, sources / 'magiclens_large_init_seed42.npz')
    oracle = np.load(sources / 'magiclens_large_oracle.npz')
    image, text = [torch.tensor(oracle[k], requires_grad=True) for k in ('image', 'text')]
    result = actual(image, text)
    forward_error = close(result, oracle['result'])
    (result * torch.tensor(oracle['probe'])).sum().backward()
    input_errors = [close(t.grad, oracle[k+'_grad'], atol=1e-5, rtol=1e-4) for k, t in [('image', image), ('text', text)]]
    gradient_errors = {k: close(p.grad, oracle['grad.'+k], atol=5e-5, rtol=1e-4) for k, p in actual.named_parameters()}
    report['magiclens'] = {'forward_maximum_error': forward_error,
        'input_gradient_maximum_error': max(input_errors),
        'parameter_gradient_maximum_error': max(gradient_errors.values()),
        'gradient_tensor_count': len(gradient_errors),
        'parameters': sum(p.numel() for p in actual.parameters()),
        'official_configuration': {'layers': 4, 'heads': 16, 'dim': 768, 'ffn': 3072,
            'pool_attention_hidden_dim': 3072, 'pool_query_tokens': 1, 'dropout': 0,
            'layernorm_eps': 1e-6, 'attention_logit_cap': 50,
            'target_encoding': 'same head(image, CLIP(empty text))'}}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
