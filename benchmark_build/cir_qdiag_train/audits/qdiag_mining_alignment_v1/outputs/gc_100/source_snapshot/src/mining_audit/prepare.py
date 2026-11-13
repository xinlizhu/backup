import json
from pathlib import Path
import numpy as np
from mining_audit.common import ROOT, digest, read_json, read_jsonl, sha, write_json, write_jsonl


def prepare(config):
    out = Path(config['runtime']['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    run = Path(config['data']['run_dir'])
    effective = read_json(run / 'effective_config.json')
    if effective['model']['composition_head'] != 'clip4cir_combiner':
        raise ValueError('This configuration requires the recorded GC experiment')
    data = effective['data']
    annotation = Path(data['train_annotation_path'])
    split = Path(data['train_image_split_path'])
    teacher = Path(data['teacher_cache_dir'])
    plan_path = Path(config['data']['sampling_plan'])
    plan_meta = read_json(config['data']['sampling_metadata'])
    assert sha(annotation) == plan_meta['annotation_sha256']
    assert sha(plan_path) == plan_meta['sampling_plan_sha256']
    plan = read_jsonl(plan_path)
    annotations = read_json(annotation)
    image_map = read_json(split)
    ids = sorted(image_map)
    index = {v: i for i, v in enumerate(ids)}
    assert ids == read_json(teacher / 'image_ids.json')
    assert [r['pairid'] for r in annotations] == read_json(teacher / 'pair_ids.json')
    assert read_json(teacher / 'metadata.json')['complete']
    assert len(plan) == len({r['pair_id'] for r in plan}) == 100
    assert all(sum(r['epoch'] == e for r in plan) == 25 for e in range(1, 5))
    source_paths = [run/'effective_config.json', annotation, split, plan_path,
                    teacher/'metadata.json', teacher/'gallery_features.npy', teacher/'query_features.npy']
    epoch_cache = {}
    for epoch in range(1, 5):
        folder = run / 'on_policy' / f'epoch_{epoch:03d}'
        meta = read_json(folder / 'metadata.json')
        assert meta['complete'] and meta['candidate_source'] == 'student'
        assert meta['target_forced_into_candidates'] and meta['top_k'] == 32
        candidates = np.load(folder / 'candidate_indices.npy', mmap_mode='r')
        scores = np.load(folder / 'teacher_scores.npy', mmap_mode='r')
        assert candidates.shape == scores.shape == (len(annotations), 32)
        epoch_cache[epoch] = (candidates, scores, meta)
        source_paths += [folder/n for n in ['metadata.json','candidate_indices.npy','teacher_scores.npy']]
    provenance = {'version': 'gc-candidate-audit-v1', 'include_random_control': config['data']['include_random_control'],
        'random_seed': config['data']['random_seed'], 'source_sha256': {str(p): sha(p) for p in source_paths}}
    signature = digest(provenance)
    marker = out / 'prepared_metadata.json'
    if marker.exists():
        previous = read_json(marker)
        if previous['signature'] != signature:
            raise RuntimeError('Data, sampling or teacher cache changed; use another output directory')
        if previous.get('manifest_sha256') and (out/'image_inventory.json').exists():
            for name,expected in previous['manifest_sha256'].items():
                if sha(out/name)!=expected:raise RuntimeError('Prepared manifest changed: '+name)
            for path,expected in read_json(out/'image_inventory.json').items():
                stat=Path(path).stat()
                if [stat.st_size,stat.st_mtime_ns]!=expected:
                    raise RuntimeError('An input image changed since preparation: '+path)
            print('[reuse prepared]', previous['query_count'], 'queries,',previous['unique_case_count'],'unique pairs',flush=True)
            return previous
    gallery = np.load(teacher/'gallery_features.npy', mmap_mode='r')
    teacher_queries = np.load(teacher/'query_features.npy', mmap_mode='r')
    memberships, queries, cases = [], [], {}
    image_root = Path(data['image_root']).resolve()
    def image_path(i):
        path = (image_root / image_map[ids[int(i)]]).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError(f'Invalid image path: {path}')
        return str(path)
    for item in plan:
        row_index, epoch = item['training_row_index'], item['epoch']
        row = annotations[row_index]
        assert row['pairid'] == item['pair_id'] and row['caption'].strip() == item['source_instruction']
        assert row['reference'] == item['reference_id']
        target_id = row.get('target_hard', row.get('target'))
        assert target_id == item['target_id']
        ref, target = index[row['reference']], index[target_id]
        candidate_matrix, score_matrix, meta = epoch_cache[epoch]
        candidate_ids = np.asarray(candidate_matrix[row_index], dtype=np.int64)
        saved_scores = np.asarray(score_matrix[row_index], dtype=np.float32)
        assert candidate_ids[0] == target and ref not in candidate_ids
        assert len(set(candidate_ids.tolist())) == 32
        query = {**item, 'reference_path': image_path(ref), 'target_path': image_path(target),
                 'mining_source_global_step':meta['source_global_step'], 'run_dir':str(run)}
        queries.append(query)
        query_embedding = np.asarray(teacher_queries[row_index], dtype=np.float32)
        groups = {'mined':candidate_ids[1:]}
        if config['data']['include_random_control']:
            rng = np.random.default_rng(np.random.SeedSequence([config['data']['random_seed'], epoch, row_index]))
            eligible = np.delete(np.arange(len(ids)), sorted([ref, target]))
            groups['random'] = rng.choice(eligible, 31, replace=False)
        for origin, members in groups.items():
            full = np.r_[target, members]
            recomputed = np.asarray(gallery[full], dtype=np.float32) @ query_embedding
            for position, ci in enumerate(members, 1):
                case_id = 'case_' + digest([item['audit_query_id'],ids[int(ci)]])[:20]
                cases.setdefault(case_id, {'case_id':case_id,'audit_query_id':item['audit_query_id'],
                    'source_instruction':item['source_instruction'],'images':[image_path(ref),image_path(target),image_path(ci)]})
                memberships.append({'case_id':case_id,'audit_query_id':item['audit_query_id'],'epoch':epoch,
                    'pair_id':item['pair_id'],'origin':origin,'candidate_position':position,'candidate_image_id':ids[int(ci)],
                    'teacher_score_saved':float(saved_scores[position]) if origin=='mined' else None,
                    'teacher_target_score_saved':float(saved_scores[0]) if origin=='mined' else None,
                    'teacher_margin_saved':float(saved_scores[0]-saved_scores[position]) if origin=='mined' else None,
                    'teacher_margin_recomputed_fp32':float(recomputed[0]-recomputed[position]),
                    'teacher_recomputed_source':'FP32 dot product of the same frozen normalized FP16 embedding cache'})
    # Deduplicate overlaps between the mined and random arms; do not exclude them.
    shuffled = list(cases.values())
    np.random.default_rng(config['data']['random_seed']).shuffle(shuffled)
    write_jsonl(out/'queries.private.jsonl', queries)
    write_jsonl(out/'cases.private.jsonl', shuffled)
    write_jsonl(out/'memberships.private.jsonl', memberships)
    inventory={}
    for path in sorted({p for case in shuffled for p in case['images']}):
        stat=Path(path).stat();inventory[path]=[stat.st_size,stat.st_mtime_ns]
    write_json(out/'image_inventory.json',inventory)
    meta = {'signature':signature, **provenance,'query_count':len(queries),'unique_case_count':len(cases),
        'membership_count':len(memberships),'mined_pair_count':sum(x['origin']=='mined' for x in memberships),
        'random_pair_count':sum(x['origin']=='random' for x in memberships),'max_initial_api_calls':2*len(cases),
        'source_run':'GC head ablation, recorded training caches; NOT historical original RIM',
        'annotation_calls_per_pair':2,'teacher_temperature':effective['objective']['rank_temperature'],
        'no_images_or_teacher_scores_used_to_select_queries':True,
        'manifest_sha256':{name:sha(out/name) for name in ['queries.private.jsonl','cases.private.jsonl','memberships.private.jsonl','image_inventory.json']}}
    write_json(marker,meta)
    print('[prepared]',json.dumps({k:meta[k] for k in ['query_count','mined_pair_count','random_pair_count','unique_case_count','max_initial_api_calls']}),flush=True)
    return meta
