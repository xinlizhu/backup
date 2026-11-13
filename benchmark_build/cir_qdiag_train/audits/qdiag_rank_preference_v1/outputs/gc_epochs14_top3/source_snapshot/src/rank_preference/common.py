import hashlib
import json
import os
from pathlib import Path
import threading
import yaml

ROOT=Path(__file__).resolve().parents[2]


def read_json(path):return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.tmp.{threading.get_ident()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def write_jsonl(path,rows):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)


def load_config(file,output=None):
    c=yaml.safe_load(Path(file).read_text())
    for k in ['run_dir','sampling_plan','sampling_metadata']:
        c['data'][k]=str((ROOT/c['data'][k]).resolve())
    c['runtime']['output_dir']=str(Path(output).resolve() if output else (ROOT/c['runtime']['output_dir']).resolve())
    if c['data']['epochs']!=[1,4] or c['data']['ranks']!=[1,2,3]:raise ValueError('This audit requires epochs 1/4 and ranks 1/2/3')
    if c['runtime']['max_workers']<1:raise ValueError('max_workers must be positive')
    return c


def redact(value):
    text=str(value);key=os.environ.get('OPENAI_API_KEY')
    return text.replace(key,'[REDACTED]') if key else text
