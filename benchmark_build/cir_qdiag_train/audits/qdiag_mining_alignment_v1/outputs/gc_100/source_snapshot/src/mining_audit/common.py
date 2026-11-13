import hashlib
import json
import os
from pathlib import Path
import threading
import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp.{threading.get_ident()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_jsonl(path, values):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path, output=None):
    config = yaml.safe_load(Path(path).read_text())
    for key in ['run_dir', 'sampling_plan', 'sampling_metadata']:
        config['data'][key] = str((ROOT / config['data'][key]).resolve())
    config['runtime']['output_dir'] = str(Path(output).resolve() if output else
                                        (ROOT / config['runtime']['output_dir']).resolve())
    if config['runtime']['max_workers'] < 1:
        raise ValueError('max_workers must be positive')
    return config


def redact(value):
    text = str(value)
    key = os.environ.get('OPENAI_API_KEY')
    return text.replace(key, '[REDACTED]') if key else text
