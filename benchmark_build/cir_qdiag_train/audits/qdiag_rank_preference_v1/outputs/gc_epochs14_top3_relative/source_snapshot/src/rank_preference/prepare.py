from collections import defaultdict
from pathlib import Path
import gc
import hashlib
import numpy as np
from PIL import Image
from rank_preference.common import ROOT,digest,read_json,read_jsonl,sha,write_json,write_jsonl


def score_order(scores,image_ids,query_id,seed):
    # Equal scores never favor the forcibly inserted target at column zero.
    return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),digest([seed,query_id,image_ids[i]])))


def pixel_hash(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB')
        h=hashlib.sha256(str(rgb.size).encode());h.update(rgb.tobytes());return h.hexdigest()


def build_comparisons(rankings,ranks,seed):
    hashes={};cases={};comparisons=[]
    def visual(path):
        if path not in hashes:hashes[path]=pixel_hash(path)
        return hashes[path]
    for row in rankings:
        for rank in ranks:
            ti=row['teacher_order'][rank-1];si=row['student_order'][rank-1]
            teacher=row['candidates'][ti];student=row['candidates'][si]
            th,sh=visual(teacher['path']),visual(student['path'])
            same=teacher['id']==student['id'] or th==sh
            item={'comparison_id':f"{row['audit_query_id']}_rank{rank}",'audit_query_id':row['audit_query_id'],
                'epoch':row['epoch'],'rank':rank,'source_instruction':row['source_instruction'],
                'reference_path':row['reference_path'],'teacher_image_id':teacher['id'],'student_image_id':student['id'],
                'teacher_image_path':teacher['path'],'student_image_path':student['path'],
                'teacher_score':row['teacher_scores'][ti],'student_score':row['student_scores'][si],
                'teacher_tie_size':sum(v==row['teacher_scores'][ti] for v in row['teacher_scores']),
                'student_tie_size':sum(v==row['student_scores'][si] for v in row['student_scores']),
                'same_image_skipped':same,'teacher_side':None,'student_side':None,'case_id':None,
                'student_checkpoint':row['student_checkpoint'],'source_global_step':row['source_global_step']}
            if not same:
                key='pair_'+digest([row['audit_query_id'],sorted([th,sh])])[:20]
                paths=sorted([(th,teacher['path']),(sh,student['path'])])
                cases.setdefault(key,{'case_id':key,'audit_query_id':row['audit_query_id'],
                    'source_instruction':row['source_instruction'],'reference_path':row['reference_path'],
                    'canonical_images':[p for _,p in paths],'canonical_pixel_hashes':[h for h,_ in paths]})
                item.update(case_id=key,teacher_pixel_hash=th,student_pixel_hash=sh)
            comparisons.append(item)
    rng=np.random.default_rng(seed)
    case_list=[cases[k] for k in sorted(cases)]
    # Randomize once per distinct unordered image pair. Repeated pairs at other ranks
    # reuse this judgment instead of issuing a reversed-order request.
    for case in case_list:
        reverse=bool(rng.integers(0,2));order=[1,0] if reverse else [0,1]
        case['images']=[case['reference_path']]+[case['canonical_images'][i] for i in order]
        case['A_pixel_hash']=case['canonical_pixel_hashes'][order[0]]
        case['B_pixel_hash']=case['canonical_pixel_hashes'][order[1]]
    for c in comparisons:
        if c['same_image_skipped']:continue
        case=cases[c['case_id']]
        c['teacher_side']='A' if c.pop('teacher_pixel_hash')==case['A_pixel_hash'] else 'B'
        c.pop('student_pixel_hash');c['student_side']='B' if c['teacher_side']=='A' else 'A'
    rng.shuffle(case_list)
    return comparisons,case_list,hashes


def replay_student(run,checkpoint,epoch,selected,annotations,image_ids,image_paths,device):
    import torch
    from cir_qdiag_train.config import ModelConfig
    from cir_qdiag_train.model import ComposedRetrievalModel
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta=read_json(run/'on_policy'/f'epoch_{epoch:03d}'/'metadata.json')
    if state['global_step']!=meta['source_global_step']:
        raise ValueError('Checkpoint is not the student that mined this epoch')
    cfg=state['config'];kw=ModelConfig.model_validate(cfg['model']).model_dump()
    if kw['composition_head']!='clip4cir_combiner':raise ValueError('Expected GC checkpoint')
    kw['gradient_checkpointing']=False;pretrained=kw.pop('pretrained_model_name_or_path')
    model,processor=ComposedRetrievalModel.from_pretrained(pretrained,**kw,
        temperature=cfg['optimizer']['temperature'],learnable_temperature=cfg['optimizer']['learnable_temperature'])
    model.load_trainable_state_dict(state['model']);del state
    model.to(device).eval();torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    refs=[]
    id_to_pos={v:i for i,v in enumerate(image_ids)}
    for q in selected:
        with Image.open(image_paths[id_to_pos[q['reference_id']]]) as im:refs.append(im.convert('RGB'))
    image_inputs=processor(images=refs,return_tensors='pt')['pixel_values'].to(device)
    texts=processor(text=[q['source_instruction'] for q in selected],padding=True,truncation=True,return_tensors='pt')
    dtype=torch.bfloat16 if cfg['optimizer']['mixed_precision']=='bf16' else torch.float16
    enabled=str(device).startswith('cuda') and cfg['optimizer']['mixed_precision']!='no'
    with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,dtype=dtype,enabled=enabled):
        # Match the mining path: raw reference bank is promoted to FP32 before fusion.
        reference=model.reference_from_raw(model._encode_image_raw(image_inputs)).float()
        text=model.encode_text(texts['input_ids'].to(device),texts.get('attention_mask').to(device) if texts.get('attention_mask') is not None else None)
        query=model.compose(reference,text).float()
    folder=run/'on_policy'/f'epoch_{epoch:03d}'
    gallery=np.load(folder/'gallery_features.npy',mmap_mode='r')
    bank=torch.from_numpy(np.asarray(gallery,dtype=np.float32).copy()).to(device)
    bank=torch.nn.functional.normalize(bank,dim=-1)
    with torch.inference_mode():scores=(query@bank.T).float().cpu().numpy()
    del model,processor,query,bank,image_inputs,reference,text;gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return scores


def prepare(config):
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    run=Path(config['data']['run_dir']);effective=read_json(run/'effective_config.json')
    plan_path=Path(config['data']['sampling_plan']);plan_meta=read_json(config['data']['sampling_metadata'])
    data=effective['data'];annotation=Path(data['train_annotation_path']);split=Path(data['train_image_split_path'])
    assert sha(plan_path)==plan_meta['sampling_plan_sha256'] and sha(annotation)==plan_meta['annotation_sha256']
    plan=[q for q in read_jsonl(plan_path) if q['epoch'] in config['data']['epochs']]
    assert len(plan)==50 and all(sum(q['epoch']==e for q in plan)==25 for e in [1,4])
    annotations=read_json(annotation);mapping=read_json(split);image_ids=sorted(mapping)
    image_paths=[str((Path(data['image_root'])/mapping[i]).resolve()) for i in image_ids]
    teacher=Path(data['teacher_cache_dir']);assert image_ids==read_json(teacher/'image_ids.json')
    assert [r['pairid'] for r in annotations]==read_json(teacher/'pair_ids.json')
    source=[annotation,split,plan_path,run/'effective_config.json',teacher/'metadata.json']
    checkpoints={e:run/config['data']['student_checkpoints'][e] for e in [1,4]}
    for epoch in [1,4]:
        folder=run/'on_policy'/f'epoch_{epoch:03d}'
        source.extend([checkpoints[epoch],*[folder/n for n in ['metadata.json','candidate_indices.npy','teacher_scores.npy','gallery_features.npy']]])
    identity={'version':'rank-preference-v1','epochs':[1,4],'ranks':[1,2,3],
        'seed':config['data']['random_seed'],'sources':{str(p):sha(p) for p in source},
        'ranking_code_sha256':sha(Path(__file__)),
        'student_code_sha256':{n:sha(ROOT.parents[1]/'src/cir_qdiag_train'/n) for n in ['model.py','composition_heads.py','official_clip4cir_combiner.py']}}
    signature=digest(identity);marker=out/'prepared_metadata.json'
    if marker.exists():
        old=read_json(marker)
        if old['signature']!=signature:raise RuntimeError('Ranking inputs/code changed; use a new output directory')
        for name,value in old['artifacts'].items():
            if sha(out/name)!=value:raise RuntimeError('Prepared ranking artifact changed: '+name)
        for path,stats in read_json(out/'image_inventory.json').items():
            st=Path(path).stat()
            if [st.st_size,st.st_mtime_ns]!=stats:raise RuntimeError('An input image changed')
        print('[reuse rankings]',old['planned_comparisons'],'comparisons,',old['gpt_case_count'],'GPT cases',flush=True)
        return old
    import torch
    device=config['runtime']['device']
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU required to prepare student ranking replay; use a GPU-enabled session')
    rankings=[];checks={}
    for epoch in [1,4]:
        selected=[q for q in plan if q['epoch']==epoch]
        folder=run/'on_policy'/f'epoch_{epoch:03d}'
        meta=read_json(folder/'metadata.json');assert meta['complete'] and meta['top_k']==32
        assert meta['target_forced_into_candidates'] and meta['candidate_source']=='student'
        candidates=np.load(folder/'candidate_indices.npy',mmap_mode='r')
        teacher_scores=np.load(folder/'teacher_scores.npy',mmap_mode='r')
        print('[student replay]',epoch,checkpoints[epoch].name,flush=True)
        scores=replay_student(run,checkpoints[epoch],epoch,selected,annotations,image_ids,image_paths,device)
        diagnostics=[]
        for i,q in enumerate(selected):
            source_row=annotations[q['training_row_index']]
            assert q['pair_id']==source_row['pairid'] and q['source_instruction']==source_row['caption'].strip()
            indices=np.asarray(candidates[q['training_row_index']],dtype=np.int64)
            ids=[image_ids[j] for j in indices]
            assert ids[0]==q['target_id'] and q['reference_id'] not in ids and len(set(ids))==32
            ts=np.asarray(teacher_scores[q['training_row_index']],dtype=np.float32)
            ss=np.asarray(scores[i,indices],dtype=np.float32)
            assert np.isfinite(ts).all() and np.isfinite(ss).all()
            to=score_order(ts,ids,q['audit_query_id'],config['data']['random_seed'])
            so=score_order(ss,ids,q['audit_query_id'],config['data']['random_seed'])
            non_target=[j for j in so if j!=0]
            diagnostics.append({'audit_query_id':q['audit_query_id'],
                'non_target_top1_matches_saved_order':non_target[0]==1,
                'non_target_top3_same_set_as_saved':set(non_target[:3])=={1,2,3},
                'adjacent_saved_order_score_inversions_over_1e4':int(np.sum(np.diff(ss[1:])>1e-4))})
            rankings.append({**q,'reference_path':image_paths[image_ids.index(q['reference_id'])],
                'student_checkpoint':str(checkpoints[epoch]),'source_global_step':meta['source_global_step'],
                'candidates':[{'id':image_ids[j],'path':image_paths[j]} for j in indices],
                'teacher_scores':ts.tolist(),'student_scores':ss.tolist(),'teacher_order':to,'student_order':so,
                'teacher_target_rank_within_32':to.index(0)+1,'student_target_rank_within_32':so.index(0)+1})
        checks[str(epoch)]=diagnostics
    comparisons,cases,hashes=build_comparisons(rankings,config['data']['ranks'],config['data']['random_seed'])
    write_jsonl(out/'rankings.private.jsonl',rankings)
    write_jsonl(out/'comparisons.private.jsonl',comparisons)
    write_jsonl(out/'judge_cases.private.jsonl',cases)
    write_json(out/'ranking_replay_checks.json',checks)
    paths={q['reference_path'] for q in rankings}|set(hashes)
    inventory={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in paths}
    write_json(out/'image_inventory.json',inventory)
    artifact_names=['rankings.private.jsonl','comparisons.private.jsonl','judge_cases.private.jsonl','ranking_replay_checks.json','image_inventory.json']
    meta={'signature':signature,'identity':identity,'query_count':len(plan),'planned_comparisons':len(comparisons),
        'same_image_skipped':sum(x['same_image_skipped'] for x in comparisons),'gpt_case_count':len(cases),
        'duplicate_unordered_pairs_reused':sum(not x['same_image_skipped'] for x in comparisons)-len(cases),
        'api_calls_per_case':1,'swapped_order_repeats':0,
        'candidate_scope':'Same recorded target-plus-31 GC training candidates; NOT full-gallery Top-3',
        'teacher_scores':'Actual saved training scores; ties use a model-independent hash tie-break, never array position',
        'student_scores':'Mining-snapshot query inference replay against normalized saved FP16 student gallery vectors; not saved per-step scores',
        'same_image_policy':'Same image ID or identical decoded RGB pixels and dimensions; skipped images are not automatically judged correct',
        'epoch_comparison_limit':'Epochs 1 and 4 use different frozen query samples; this is not a within-query learning trajectory',
        'artifacts':{n:sha(out/n) for n in artifact_names}}
    write_json(marker,meta)
    print('[prepared]',{k:meta[k] for k in ['query_count','planned_comparisons','same_image_skipped','gpt_case_count','duplicate_unordered_pairs_reused']},flush=True)
    return meta
