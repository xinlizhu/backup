"""Import identical case annotations into a declared smaller sampling scope."""
import copy
import os
from pathlib import Path
import re
import shutil
from mining_audit.common import ROOT,digest,read_json,read_jsonl,sha,write_json
from mining_audit.responses import (ProposalResponse,ReviewResponse,accepted_witnesses,
    proposal_partition_error,review_partition_error)

SEMANTIC_FILES=['prompts/propose_witnesses.txt','prompts/review_witnesses.txt',
                'proposal.schema.json','review.schema.json','src/mining_audit/responses.py']


def source_directory(config):
    value=config['runtime'].get('reuse_from')
    return (ROOT/value).resolve() if value else None


def offline_backend_identity(config):
    """No API key or request is needed to reuse old records from the same backend."""
    source=source_directory(config)
    previous=read_json(source/'annotation_identity.json')
    if previous['mock']:raise ValueError('Cannot import mock annotations into a real audit')
    backend=previous['backend']
    for key in ['generation_model','validation_model','generation_reasoning_effort',
                'validation_reasoning_effort','image_detail','structured_output_mode']:
        if config['model'].get(key)!=backend.get(key):
            raise ValueError('Configured model/decoding differs from reusable records: '+key)
    base=config['model']['api_base']
    if base=='${OPENAI_BASE_URL}':base=os.environ.get('OPENAI_BASE_URL') or backend['api_base']
    if base.rstrip('/')!=backend['api_base']:
        raise ValueError('Configured API endpoint differs from reusable records')
    return backend


def initialize_annotations(config,meta,backend_identity,mock):
    out=Path(config['runtime']['output_dir'])
    paths=[ROOT/n for n in SEMANTIC_FILES[:4]]+list((ROOT/'src/mining_audit').glob('*.py'))
    identity={'prepared_signature':meta['signature'],'backend':backend_identity,'mock':mock,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'generation':config['generation'],'validation':config['validation']}
    signature=digest(identity)
    marker=out/'annotation_identity.json'
    if marker.exists() and read_json(marker)!=identity:
        raise RuntimeError('Annotation prompts/model/code changed; use another output directory instead of mixing results')
    write_json(marker,identity)
    for path in paths:
        dest=out/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(path,dest)
    write_json(out/'effective_config.json',{**config,'model':{**config['model'],'api_key':'${OPENAI_API_KEY}'}})
    if source_directory(config):import_records(config,identity,signature)
    return identity,signature


def import_records(config,identity,signature):
    source=source_directory(config);out=Path(config['runtime']['output_dir'])
    if source==out:raise ValueError('Reuse source and output must differ')
    old_identity=read_json(source/'annotation_identity.json')
    for key in ['backend','mock','generation','validation']:
        if old_identity[key]!=identity[key]:raise ValueError('Incompatible reuse source: '+key)
    for key in SEMANTIC_FILES:
        if old_identity['source_hashes'].get(key)!=identity['source_hashes'].get(key):
            raise ValueError('Prompts/schema/acceptance rules changed; cannot reuse '+key)
    old_meta=read_json(source/'prepared_metadata.json')
    for name,expected in old_meta['manifest_sha256'].items():
        if sha(source/name)!=expected:raise ValueError('Reuse source manifest changed: '+name)
    old_cases={x['case_id']:x for x in read_jsonl(source/'cases.private.jsonl')}
    cases={x['case_id']:x for x in read_jsonl(out/'cases.private.jsonl')}
    old_images=read_json(source/'image_inventory.json');images=read_json(out/'image_inventory.json')
    old_signature=digest(old_identity);imported=[];cached=[]
    for case_id,case in cases.items():
        if old_cases.get(case_id)!=case:raise ValueError('Selected case is not an identical subset: '+case_id)
        if any(old_images.get(p)!=images[p] for p in case['images']):
            raise ValueError('Selected image changed since old annotation')
        file=source/'results'/f'{case_id}.json';dest=out/'results'/file.name
        if dest.exists() or not file.exists():continue
        value=read_json(file)
        if value.get('status')!='complete':continue
        if value['signature']!=old_signature or bool(value.get('simulated'))!=bool(identity['mock']):
            raise ValueError('Invalid source annotation provenance')
        ProposalResponse.model_validate(value['proposal']);ReviewResponse.model_validate(value['review'])
        if proposal_partition_error(value['proposal'],case_id) or review_partition_error(value['review'],case_id,value['proposal']['witnesses']):
            raise ValueError('Invalid source response IDs')
        accepted,checks=accepted_witnesses(case['source_instruction'],value['proposal'],value['review'])
        copied=copy.deepcopy(value)
        copied.update(signature=signature,accepted_witnesses=accepted,local_checks=checks,
            imported_from={'result_file':str(file),'result_sha256':sha(file),
                           'original_signature':old_signature,'reason':'User restricted audit to final epoch; identical query/images/prompts/backend'})
        write_json(dest,copied);imported.append(case_id)
    for stage in ['propose_witnesses','review_witnesses']:
        for file in (source/'api_cache'/stage).glob('*.json'):
            if not re.fullmatch(r'[0-9a-f]{64}\.json',file.name):continue
            dest=out/'api_cache'/stage/file.name
            if dest.exists():continue
            value=read_json(file);case_id=value.get('parsed',{}).get('case_id')
            if case_id not in cases or value.get('images')!=cases[case_id]['images']:continue
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);cached.append(str(dest))
    report=out/'reuse_report.json'
    previous=read_json(report) if report.exists() else {}
    write_json(report,{'source_output':str(source),'source_annotation_signature':old_signature,
        'imported_complete_cases':sorted(set(previous.get('imported_complete_cases',[]))|set(imported)),
        'copied_request_caches':sorted(set(previous.get('copied_request_caches',[]))|set(cached)),
        'api_calls_to_import':0,'original_results_preserved':True})
    print('[reuse annotations]',len(imported),'completed cases,',len(cached),'request caches imported',flush=True)
