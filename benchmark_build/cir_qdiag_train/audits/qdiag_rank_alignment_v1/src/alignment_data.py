import hashlib
import json
from pathlib import Path
import threading
import yaml

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.tmp.{threading.get_ident()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def write_jsonl(path,rows):Path(path).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def load_config(path):
    c=yaml.safe_load(Path(path).read_text());c['runtime']['output_dir']=str((ROOT/c['runtime']['output_dir']).resolve());return c


def prepare(config):
    out=Path(config['runtime']['output_dir']);out.mkdir(parents=True,exist_ok=True)
    data=config['data'];identity={k:sha(v) for k,v in data.items()}
    marker=out/'preparation.json'
    if marker.exists():
        old=read_json(marker)
        if old['source_hashes']!=identity:raise ValueError('Dataset/gallery changed')
        for name,h in old['artifacts'].items():
            if sha(out/name)!=h:raise ValueError('Prepared artifact changed: '+name)
        return old
    rows=read_jsonl(data['dataset']);manifest=read_json(data['gallery_manifest'])
    gallery=sorted(manifest['gallery'],key=lambda r:r['id']);lookup={r['id']:i for i,r in enumerate(gallery)}
    assert len(lookup)==len(gallery)==109598
    queries=[];groups=[]
    for row in rows:
        refs=row['reference_image_ids'];targets=row['positive_image_ids']
        missing=(set(refs)|set(targets))-set(lookup)
        if missing:raise ValueError(f'Required images absent from the fixed gallery: {missing}')
        group={'group_id':row['group_id'],'reference_ids':refs,'positive_ids':targets,
            'reviewed_positive_ids':row['reviewed_positive_image_ids'],
            'target_indices':[lookup[t] for t in targets],'positive_query_indices':[],'negative_query_indices':[]}
        originals={}
        for inst in row['instructions']:
            qi=len(queries);originals[inst['instruction_id']]=qi;group['positive_query_indices'].append(qi)
            queries.append({'query_index':qi,'query_key':'positive::'+inst['instruction_id'],'kind':'positive',
                'group_id':row['group_id'],'instruction_id':inst['instruction_id'],'text':inst['text'],
                'reference_ids':refs,'target_indices':group['target_indices']})
        for edit in row['negatives']:
            qi=len(queries);group['negative_query_indices'].append(qi)
            source=originals[edit['instruction_id']]
            assert queries[source]['text']==edit['source_text']
            queries.append({'query_index':qi,'query_key':'negative::'+edit['negative_id'],'kind':'negative',
                'group_id':row['group_id'],'instruction_id':edit['instruction_id'],'negative_id':edit['negative_id'],
                'text':edit['text'],'source_query_index':source,'reference_ids':refs,'target_indices':group['target_indices'],
                'edit_operation':edit['edit_operation'],'semantic_target':edit['semantic_target']})
        groups.append(group)
    assert len(groups)==931 and len(queries)==13551
    assert sum(q['kind']=='negative' for q in queries)==10131
    assert len({q['query_key'] for q in queries})==len(queries)
    write_json(out/'gallery.json',gallery);write_jsonl(out/'queries.jsonl',queries);write_json(out/'groups.json',groups)
    write_json(out/'pilot_group_ids.json',Path(data['pilot_group_ids']).read_text().split())
    artifacts=['gallery.json','queries.jsonl','groups.json','pilot_group_ids.json']
    value={'source_hashes':identity,'groups':len(groups),'queries':len(queries),'gallery_images':len(gallery),
        'original_queries':3420,'edits':10131,'strict_add_sub_edits':5735,'delete_edits':4396,
        'reference_images':len({x for g in groups for x in g['reference_ids']}),
        'max_targets_per_group':max(len(g['target_indices']) for g in groups),
        'gallery_policy':'Same full available PinPoint gallery for q+ and q-; references retained, matching existing official evaluation.',
        'rank_ties':'Stable descending FP32 cosine score; ties resolved by ascending image ID, independent of target labels.',
        'artifacts':{n:sha(out/n) for n in artifacts}}
    value['signature']=digest(value);write_json(marker,value);return value
