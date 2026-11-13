"""Evaluate external CIR models with the original QDiag statistics and native scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'diag_model'))
from common import write_json_atomic, save_npy_atomic, sha256_file
from pinpoint_full_diagnostics import load_v7_dataset, _resolve_image_path, group_balanced_quantile, bootstrap_group_balanced_quantile
from evaluate_winrate_random10 import _sample_indices, summarize
from external_qdiag_adapters import make_adapter

DEFAULT_INPUT = PROJECT.parent / 'cir_qdiag_openai/outputs/pinpoint_pilot_500_v7/final.jsonl'


def calibrate_scores(groups, queries, scores, image_ids, *, replicates=2000, seed=42):
    qi = {(q.group_id, q.instruction_id): q.query_index for q in queries if q.query_kind == 'positive'}
    ii = {value: i for i, value in enumerate(image_ids)}
    perturbations = {}
    for g in groups:
        block = scores[np.ix_([qi[g.group_id, i] for i in g.instruction_ids], [ii[t] for t in g.positive_ids])]
        values = [np.abs(block[a] - block[b]) for a, b in combinations(range(len(block)), 2)]
        perturbations[g.group_id] = np.concatenate(values) if values else np.empty(0)
    return {
        'method': 'group-balanced 95th percentile of absolute native positive-paraphrase target-score differences',
        'normal_perturbation_threshold': group_balanced_quantile(perturbations, 0.95),
        'threshold_bootstrap_ci': list(bootstrap_group_balanced_quantile(perturbations, quantile=0.95, replicates=replicates, seed=seed)),
        'calibration_group_count': len(perturbations),
        'calibration_observation_count': sum(len(v) for v in perturbations.values()),
        'group_weighting': 'each PinPoint rewrite group has total weight 1',
    }


def negative_frame(groups, queries, scores, image_ids, *, threshold, random_samples=10, seed=42):
    gm = {g.group_id: g for g in groups}
    ii = {value: i for i, value in enumerate(image_ids)}
    qi = {(q.group_id, q.instruction_id): q.query_index for q in queries if q.query_kind == 'positive'}
    positive = [q.query_index for q in queries if q.query_kind == 'positive']
    negative = [q.query_index for q in queries if q.query_kind == 'negative']
    rows = []
    for q in queries:
        if q.query_kind != 'negative':
            continue
        own = [ii[t] for t in gm[q.group_id].positive_ids]
        own_set = set(own)
        source = qi[q.group_id, q.source_instruction_id]
        rand_i = _sample_indices([i for i in range(len(image_ids)) if i not in own_set], count=random_samples,
                                  seed=seed, key=f'{q.negative_id}:random-targets')
        rand_p = _sample_indices([i for i in positive if queries[i].group_id != q.group_id], count=random_samples,
                                  seed=seed, key=f'{q.negative_id}:random-positive-queries')
        rand_n = _sample_indices([i for i in negative if queries[i].group_id != q.group_id], count=random_samples,
                                  seed=seed, key=f'{q.negative_id}:random-negative-queries')
        pos_scores, neg_scores = scores[source, own], scores[q.query_index, own]
        delta = pos_scores - neg_scores
        pos, neg = float(pos_scores.mean()), float(neg_scores.mean())
        pos_rand, neg_rand = float(scores[source, rand_i].mean()), float(scores[q.query_index, rand_i].mean())
        rand_pos = float(scores[np.ix_(rand_p, own)].mean())
        rand_neg = float(scores[np.ix_(rand_n, own)].mean())
        rows.append({
            'group_id': q.group_id, 'instruction_id': q.instruction_id, 'negative_id': q.negative_id,
            'source_text': queries[source].text, 'negative_text': q.text,
            'semantic_target': q.semantic_target, 'edit_operation': q.edit_operation,
            'positive_target_count': len(own), 'random_sample_count': random_samples,
            'random_target_ids': '|'.join(image_ids[i] for i in rand_i),
            'mean_positive_query_own_target_score': pos, 'mean_negative_query_own_target_score': neg,
            'mean_target_delta': float(delta.mean()), 'target_raw_win_rate': float(np.mean(delta > 0)),
            'target_stable_win_rate': float(np.mean(delta > threshold)),
            'mean_positive_query_random_target_score': pos_rand,
            'positive_query_own_minus_random_target': pos-pos_rand,
            'mean_negative_query_random_target_score': neg_rand,
            'negative_query_own_minus_random_target': neg-neg_rand,
            'mean_random_positive_query_own_target_score': rand_pos,
            'positive_query_own_minus_random_positive_query': pos-rand_pos,
            'mean_random_negative_query_own_target_score': rand_neg,
            'negative_query_own_minus_random_negative_query': neg-rand_neg,
            'normal_perturbation_threshold': threshold,
        })
    return pd.DataFrame(rows)


def write_metrics(groups, queries, matrix, image_ids, root, args):
    calibration = calibrate_scores(groups, queries, matrix, image_ids, replicates=args.bootstrap_replicates, seed=args.seed)
    frame = negative_frame(groups, queries, matrix, image_ids,
                           threshold=calibration['normal_perturbation_threshold'], seed=args.seed)
    table = summarize(frame, calibration=calibration, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    frame.to_csv(root / 'per_negative_random10.csv', index=False)
    table.to_csv(root / 'summary.csv', index=False)
    write_json_atomic(root / 'calibration.json', calibration)
    single = {g.group_id for g in groups if len(g.reference_ids) == 1}
    summarize(frame[frame.group_id.isin(single)], calibration=calibration,
              bootstrap_replicates=args.bootstrap_replicates, seed=args.seed).to_csv(root / 'summary_single_reference.csv', index=False)
    display = table[['subset', 'group_count', 'instruction_count', 'negative_count', 'mean_target_delta',
                     'target_raw_win_rate', 'target_stable_win_rate', 'stable_win_rate_ci_low',
                     'stable_win_rate_ci_high', 'normal_perturbation_threshold']].copy()
    display.columns = ['subset', 'groups', 'instructions', 'negatives', 'mean_delta', 'DRR_percent',
                       'CRR_percent', 'CRR_CI_low_percent', 'CRR_CI_high_percent', 'b_psi']
    for c in ['DRR_percent', 'CRR_percent', 'CRR_CI_low_percent', 'CRR_CI_high_percent']:
        display[c] *= 100
    display.to_csv(root / 'paper_metrics.csv', index=False)
    print(display[display.subset.isin(['all', 'strict', 'add', 'substitute', 'delete'])].to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['clip4cir', 'tema', 'qure', 'oprd_head'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--clip-checkpoint', type=Path)
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--image-cache', type=Path, default=WORKSPACE / 'pinpoint-dataset/image_cache')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--image-batch-size', type=int, default=128)
    p.add_argument('--query-batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--bootstrap-replicates', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    groups, queries, coverage = load_v7_dataset(args.input, limit_groups=None)
    target_ids = sorted({t for g in groups for t in g.positive_ids})
    image_ids = sorted(set(target_ids) | {i for g in groups for i in g.reference_ids})
    paths = {i: _resolve_image_path(i, args.image_cache) for i in image_ids}
    missing = [i for i, v in paths.items() if v is None]
    preflight = {**coverage, 'unique_images': len(image_ids), 'target_images': len(target_ids),
                 'reference_count_groups': {str(n): sum(len(g.reference_ids)==n for g in groups) for n in [1,2]},
                 'missing_images': missing}
    write_json_atomic(root / 'preflight.json', preflight)
    print('[preflight]', json.dumps(preflight), flush=True)
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing images; see preflight.json. No groups were dropped.')
    if args.preflight_only:
        return
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    fingerprint = {'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
                   'clip_checkpoint_sha256': sha256_file(args.clip_checkpoint) if args.clip_checkpoint else None,
                   'model': args.model, 'protocol': 'native-score-v1-reference-feature-mean-before-fusion',
                   'adapter_sha256': sha256_file(Path(__file__).with_name('external_qdiag_adapters.py'))}
    if args.model == 'oprd_head':
        fingerprint['head_adapter_sha256'] = sha256_file(Path(__file__).with_name('oprd_head_adapter.py'))
    metadata_file = root / 'identity.json'
    if metadata_file.exists() and json.loads(metadata_file.read_text()) != fingerprint:
        raise RuntimeError('Existing output belongs to a different model/protocol; use a new output directory.')
    write_json_atomic(metadata_file, fingerprint)
    cache = root / 'cache'
    cache.mkdir(exist_ok=True)
    matrix_path = cache / 'query_target_scores.npy'
    if matrix_path.exists() and not args.smoke_only:
        write_metrics(groups, queries, np.load(matrix_path, mmap_mode='r'), target_ids, root, args)
        return
    started = time.time()
    adapter = make_adapter(args.model, args.checkpoint, args.device, args.clip_checkpoint)
    print('[model] strict checkpoint load succeeded', flush=True)
    example_queries = [q for q in queries if len(q.reference_ids)==1][:3]

    def prepare(i):
        with Image.open(paths[i]) as im:
            return adapter.preprocess(im.convert('RGB'))

    images = torch.stack([prepare(q.reference_ids[0]) for q in example_queries])
    base = adapter.image_base(images)
    captions = [q.text for q in example_queries]
    smoke_queries = torch.cat([adapter.queries(base[i:i+1], [cap]) for i, cap in enumerate(captions)])
    smoke_targets = adapter.targets(base)
    assert torch.isfinite(smoke_queries).all() and torch.isfinite(smoke_targets).all()
    smoke = {'query_shape': list(smoke_queries.shape), 'target_shape': list(smoke_targets.shape), 'finite': True}
    if hasattr(adapter, 'native_smoke'):
        native_q, native_t = adapter.native_smoke(images, captions)
        qdiff = (native_q-smoke_queries).abs().max().item()
        tdiff = (native_t-smoke_targets).abs().max().item()
        smoke.update(native_query_max_abs_error=qdiff, native_target_max_abs_error=tdiff)
        torch.testing.assert_close(native_q, smoke_queries, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(native_t, smoke_targets, atol=2e-5, rtol=2e-5)
    write_json_atomic(root / 'smoke.json', smoke)
    print('[smoke]', json.dumps(smoke), flush=True)
    if args.smoke_only:
        return

    def repo_commit():
        if args.model == 'oprd_head':
            return None  # Local training project is not a Git checkout; source hashes are recorded separately.
        return subprocess.check_output(['git','-C',str(adapter.repo),'rev-parse','HEAD'], text=True).strip()

    metadata = {**vars(args), 'model_name': adapter.name, 'repository_commit': repo_commit(),
                'multi_reference': 'arithmetic mean of native reference visual features/tokens BEFORE multimodal fusion; identical references reduce to the single-image path',
                'scoring': 'max over target query-token cosine similarities' if adapter.multi_vector else 'normalized query-target dot product',
                'tema_padding': 'batch only equal token lengths to avoid unmasked padding in official entity mapper' if adapter.equal_text_lengths else None,
                'metrics': 'original group-balanced QDiag; all includes DELETE; original bootstrap and random-10 seeds',
                'environment': {'python':sys.version, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name()},
                'coverage':preflight}
    metadata = json.loads(json.dumps(metadata, default=str))
    write_json_atomic(root / 'run_metadata.json', metadata)
    write_json_atomic(cache / 'image_ids.json', image_ids)
    write_json_atomic(cache / 'target_ids.json', target_ids)
    write_json_atomic(cache / 'queries.json', [asdict(q) for q in queries])
    base_path = cache / 'image_base.npy'
    if not base_path.exists():
        tmp = cache / 'image_base.tmp.npy'
        done_path = cache / 'image_progress.json'
        start = json.loads(done_path.read_text())['next_index'] if done_path.exists() and tmp.exists() else 0
        mmap = np.load(tmp, mmap_mode='r+') if start else None
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in range(start, len(image_ids), args.image_batch_size):
                batch = image_ids[i:i+args.image_batch_size]
                imgs = torch.stack(list(pool.map(prepare, batch)))
                features = adapter.image_base(imgs).cpu().numpy()
                if mmap is None:
                    mmap = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(len(image_ids),*features.shape[1:]))
                mmap[i:i+len(batch)] = features
                mmap.flush()
                write_json_atomic(done_path, {'next_index':i+len(batch)})
                print(f'[images] {i+len(batch)}/{len(image_ids)} elapsed={time.time()-started:.1f}s', flush=True)
        del mmap
        tmp.replace(base_path)
    base_cache = np.load(base_path, mmap_mode='r')
    index = {name:i for i,name in enumerate(image_ids)}
    target_path = cache / 'target_features.npy'
    if not target_path.exists():
        targets = []
        for i in range(0,len(target_ids),args.query_batch_size):
            b = torch.from_numpy(np.array(base_cache[[index[x] for x in target_ids[i:i+args.query_batch_size]]]))
            targets.append(adapter.targets(b).cpu().numpy())
        save_npy_atomic(target_path, np.concatenate(targets))
        print('[targets] complete', flush=True)
    query_path = cache / 'query_features.npy'
    if not query_path.exists():
        buckets = defaultdict(list)
        for q in queries:
            key = adapter.text_length(q.text) if adapter.equal_text_lengths else 0
            buckets[key].append(q)
        tmp = cache / 'query_features.tmp.npy'
        done_path = cache / 'query_done.npy'
        done = np.load(done_path) if tmp.exists() and done_path.exists() else np.zeros(len(queries),dtype=bool)
        mmap = np.load(tmp,mmap_mode='r+') if done.any() else None
        for bucket in buckets.values():
            pending = [q for q in bucket if not done[q.query_index]]
            for i in range(0,len(pending),args.query_batch_size):
                batch = pending[i:i+args.query_batch_size]
                pooled = np.stack([np.asarray(base_cache[[index[x] for x in q.reference_ids]]).mean(0) for q in batch])
                features = adapter.queries(torch.from_numpy(pooled), [q.text for q in batch]).cpu().numpy()
                if mmap is None:
                    mmap = np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(len(queries),features.shape[-1]))
                indices = [q.query_index for q in batch]
                mmap[indices] = features
                mmap.flush()
                done[indices] = True
                save_npy_atomic(done_path, done)
                print(f'[queries] {done.sum()}/{len(queries)} elapsed={time.time()-started:.1f}s',flush=True)
        assert done.all()
        del mmap
        tmp.replace(query_path)
    qfeatures = np.load(query_path,mmap_mode='r')
    tfeatures = torch.from_numpy(np.load(target_path)).to(args.device)
    matrix = np.lib.format.open_memmap(cache / 'query_target_scores.tmp.npy',mode='w+',dtype=np.float32,
                                     shape=(len(queries),len(target_ids)))
    for i in range(0,len(queries),128):
        q = torch.from_numpy(np.array(qfeatures[i:i+128])).to(args.device)
        with torch.inference_mode():
            scores = torch.einsum('bd,nkd->bnk',q,tfeatures).max(-1).values if adapter.multi_vector else q @ tfeatures.T
        matrix[i:i+len(q)] = scores.cpu().numpy()
    matrix.flush()
    assert np.isfinite(matrix).all()
    del matrix, adapter, tfeatures
    (cache / 'query_target_scores.tmp.npy').replace(matrix_path)
    torch.cuda.empty_cache()
    write_metrics(groups, queries, np.load(matrix_path,mmap_mode='r'),target_ids,root,args)
    metadata.update(complete=True, elapsed_seconds=time.time()-started)
    write_json_atomic(root / 'run_metadata.json', metadata)
    print(f'[done] {root} elapsed={metadata["elapsed_seconds"]:.1f}s',flush=True)


if __name__ == '__main__':
    main()
