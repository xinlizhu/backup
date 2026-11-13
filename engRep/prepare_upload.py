#!/usr/bin/env python3
"""Prepare new local OSS batches, optionally enriching records via an embedding subprocess. No OSS calls."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zlib

BUSINESS = 'research-report-images'
ID_PREFIX = 'ResearchReportImages:'
RETENTION_HOURS = 4320
MAX_BYTES = 10 * 1024 * 1024
TARGET_BYTES = 8 * 1024 * 1024


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


@contextmanager
def export_lock(data):
    state = data.with_name(data.name + '_mid') / '_state'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'upload.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def history(business, documents=None, ignore_embedding=False, embedded_ids=None):
    """Also retain versions outside rolling Meta, and recover completed orphan batches."""
    latest = {}
    manifests = []
    for path in sorted((business / 'batches').glob('*/manifest.json')):
        m = read_json(path)
        version = path.parent.name
        if not re.fullmatch(r'\d{17}-\d{4}', version) or m['batchVersion'] != version:
            raise ValueError(f'Invalid batch version: {path}')
        if m['businessCode'] != BUSINESS or m['protocolVersion'] != '1.0':
            raise ValueError(f'Invalid manifest: {path}')
        count = 0
        for sequence, info in enumerate(m['files']):
            name = f'data_part{sequence:04d}.jsonl.gz'
            if info['sequence'] != sequence or info['fileName'] != name:
                raise ValueError(f'Invalid shard sequence: {path}')
            packed = (path.parent / name).read_bytes()
            if (len(packed) != info['sizeBytes'] or len(packed) > MAX_BYTES
                    or 'sha256:' + digest(packed) != info['checksum']):
                raise ValueError(f'Shard checksum/size mismatch: {path.parent / name}')
            raw = gzip.decompress(packed)
            lines = raw.splitlines()
            if (len(raw) != info['uncompressedSizeBytes'] or len(lines) != info['recordCount']
                    or not lines or not raw.endswith(b'\n')):
                raise ValueError(f'Shard content mismatch: {path.parent / name}')
            for line in lines:
                record = json.loads(line)
                if embedded_ids is not None:
                    if record['op'] == 'delete':
                        embedded_ids.discard(record['id'])
                    elif 'embedding' in record['data']:
                        vector = record['data']['embedding']
                        if isinstance(vector, list) and len(vector) == 2048:
                            embedded_ids.add(record['id'])
                        else:
                            embedded_ids.discard(record['id'])
                if ignore_embedding and record.get('op') == 'upsert':
                    record['data'].pop('embedding', None)
                latest[record['id']] = digest(encoded(record))
                if documents is not None:
                    if record['op'] == 'upsert':
                        documents[record['id']] = {**documents.get(record['id'], {}), **record['data']}
                    elif record['op'] == 'delete':
                        documents.pop(record['id'], None)
                    else:
                        raise ValueError(f'Invalid operation in {path}: {record["op"]}')
            count += len(lines)
        if count != m['totalRecordCount'] or len(m['files']) != m['totalFileCount']:
            raise ValueError(f'Manifest totals mismatch: {path}')
        manifests.append(m)
    meta_path = business / 'meta.json'
    if meta_path.exists():
        meta = read_json(meta_path)
        known = {m['batchVersion']: m for m in manifests}
        if meta['businessCode'] != BUSINESS:
            raise ValueError(f'Wrong business: {meta_path}')
        for entry in meta['batches']:
            m = known.get(entry['batchVersion'])
            if (m is None or m['producedAt'] != entry['producedAt']
                    or entry['manifestPath'] != f"batches/{entry['batchVersion']}/manifest.json"):
                raise ValueError(f'Meta/Manifest mismatch: {entry["batchVersion"]}')
    return latest, manifests


def write_shards(folder, records, *, preview=True):
    files = []

    def emit(lines):
        raw = b''.join(lines)
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        if len(packed) > MAX_BYTES:
            if len(lines) == 1:
                raise ValueError('Single compressed record exceeds 10 MiB')
            middle = len(lines) // 2
            emit(lines[:middle])
            emit(lines[middle:])
            return
        sequence = len(files)
        name = f'data_part{sequence:04d}.jsonl.gz'
        (folder / name).write_bytes(packed)
        if preview:
            (folder / name).with_suffix('').write_bytes(raw)
        files.append(dict(sequence=sequence, fileName=name, sizeBytes=len(packed),
                          uncompressedSizeBytes=len(raw), recordCount=len(lines),
                          checksum='sha256:' + digest(packed)))

    compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
    pending, size = [], 0
    for record in records:
        line = encoded(record)
        pending.append(line)
        size += len(compressor.compress(line))
        if size >= TARGET_BYTES:
            emit(pending)
            pending, size = [], 0
            compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
    if pending:
        emit(pending)
    return files


def copy_image(source, destination):
    """Never replace bytes at an existing image key referenced by older batches."""
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size or digest(source.read_bytes()) != digest(destination.read_bytes()):
            raise ValueError(f'Existing image has different bytes: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + '.tmp')
    shutil.copy2(source, temp)
    if digest(source.read_bytes()) != digest(temp.read_bytes()):
        raise ValueError(f'Image copy verification failed: {source}')
    temp.replace(destination)


def update_meta(business, manifests):
    if not manifests:
        return
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=RETENTION_HOURS)
    entries = [dict(batchVersion=m['batchVersion'], producedAt=m['producedAt'],
                    manifestPath=f"batches/{m['batchVersion']}/manifest.json")
               for m in manifests if datetime.fromisoformat(m['producedAt'].replace('Z', '+00:00')) >= cutoff]
    if not entries:
        return  # No new batch; keep the last usable discovery document.
    value = dict(protocolVersion='1.0', businessCode=BUSINESS,
                 generatedAt=now.isoformat(timespec='seconds').replace('+00:00', 'Z'),
                 retentionHours=RETENTION_HOURS, firstBatchVersion=entries[0]['batchVersion'],
                 latestBatchVersion=entries[-1]['batchVersion'], batches=entries)
    path = business / 'meta.json'
    if path.exists():
        old = read_json(path)
        if {k: v for k, v in old.items() if k != 'generatedAt'} == {k: v for k, v in value.items() if k != 'generatedAt'}:
            return
    temp = path.with_suffix('.json.tmp')
    save_json(temp, value)
    temp.replace(path)


def validate_enriched(records, enriched):
    """Check the subprocess result before making any batch discoverable."""
    for expected, actual in zip(records, enriched, strict=True):
        if not isinstance(actual.get('data'), dict):
            raise ValueError('Embedding result is missing business data')
        plain = {**actual, 'data': {k:v for k,v in actual['data'].items() if k != 'embedding'}}
        if plain != expected:
            raise ValueError(f'Embedding output changed ID/order/business fields: {expected["id"]}')
        vector = actual['data'].get('embedding')
        if (not isinstance(vector, list) or len(vector) != 2048
                or not all(type(v) in (float, int) and math.isfinite(v) for v in vector)
                or abs(math.fsum(v*v for v in vector) - 1.0) > 1e-4):
            raise ValueError(f'Invalid normalized 2048-dimensional embedding: {expected["id"]}')
        yield actual


def embedding_worker(data, upload_dir=None, *, model=None, gpu_memory_utilization=0.95, batch_size=256):
    """Return an embedding callback. GPU imports/live model stay in the child."""
    data = Path(data).resolve()
    upload = Path(upload_dir).resolve() if upload_dir else data.parent / 'upload_data'
    script = Path(__file__).resolve().parent / 'embed_upload.py'

    def enrich(records, images, work):
        source, output = work / 'embedding_input.jsonl', work / 'embedding_output.jsonl'
        with source.open('w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps({**r, '_source_image': str(images[r['data']['image_file']])}, ensure_ascii=False) + '\n')
        cmd = [sys.executable, '-u', str(script), '--input-records', str(source),
               '--output-records', str(output), '--source-data', str(data),
               '--cache-root', str(data.with_name(data.name + '_mid') / 'embedding'),
               '--upload-dir', str(upload), '--gpu-memory-utilization', str(gpu_memory_utilization),
               '--batch-size', str(batch_size)]
        if model:
            cmd.extend(['--model', str(model)])
        env = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env.setdefault(key, '1')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        print(f'Embedding {len(records)} new/changed records (GPU budget {gpu_memory_utilization}); cache: {data.with_name(data.name + "_mid") / "embedding"}', flush=True)
        subprocess.run(cmd, env=env, check=True)
        # The subprocess has exited, releasing its GPU resources before packing.
        with output.open(encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    return enrich


def prepare(data, batch_date, upload_dir=None, *, embedder=None, all_pending=False):
    data = Path(data).resolve()
    upload = Path(upload_dir).resolve() if upload_dir else data.parent / 'upload_data'
    day = datetime.strptime(batch_date.replace('_', ''), '%Y%m%d').strftime('%Y%m%d')
    indices = sorted((data / 'jsonl').glob('*.jsonl')) if all_pending else [data / 'jsonl' / f'{day}.jsonl']
    with export_lock(data):
        business = upload / BUSINESS
        # Base extraction indexes omit vectors. Do not emit redundant upserts
        # merely because the published package has since received embeddings.
        embedded_ids = set()
        latest, manifests = history(business, ignore_embedding=True, embedded_ids=embedded_ids)
        records, images, seen = [], {}, set()
        for index in indices:
            if not index.exists():
                continue
            source_day = datetime.strptime(index.stem, '%Y%m%d').strftime('%Y%m%d')
            with index.open(encoding='utf-8') as handle:
                for line in handle:
                    original = json.loads(line)
                    identity = original['id']
                    if not isinstance(identity, str) or not identity or identity in seen:
                        raise ValueError(f'Invalid/duplicate ID in {index}: {identity}')
                    seen.add(identity)
                    source = (index.parent / original['image_file']).resolve()
                    expected = (data / 'picture' / source_day / source.name).resolve()
                    packaged = (upload / 'pic' / source_day / source.name).resolve()
                    if source not in (expected, packaged) or not source.is_file():
                        raise ValueError(f'Invalid or missing source image: {source}')
                    identity = ID_PREFIX + identity
                    key = str(PurePosixPath('pic', source_day, source.name))
                    record = dict(op='upsert', id=identity,
                                  data={**original, 'id': identity, 'image_file': key})
                    comparable = {**record, 'data': {k:v for k,v in record['data'].items() if k != 'embedding'}}
                    if latest.get(identity) != digest(encoded(comparable)) or (embedder is not None and identity not in embedded_ids):
                        records.append(comparable if embedder is not None else record)
                        images[key] = source
        if not records:
            update_meta(business, manifests)
            print(f'Upload package: no new/changed records for {day}.', flush=True)
            return None
        records.sort(key=lambda r: r['id'])
        prefix = day + '000000000-'
        sequence = max((int(m['batchVersion'][-4:]) for m in manifests
                        if m['batchVersion'].startswith(prefix)), default=-1) + 1
        version = prefix + f'{sequence:04d}'
        if sequence > 9999 or (manifests and version <= manifests[-1]['batchVersion']):
            raise ValueError('New upload version must exceed all existing batches; process increments with a current --batch-date')
        destination = business / 'batches' / version
        if destination.exists():
            raise ValueError(f'Unrecognized existing batch directory: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Staging stays outside upload_data, and Meta is replaced only after completion.
        with tempfile.TemporaryDirectory(prefix='upload_', dir=data.with_name(data.name + '_mid')) as temp:
            staging = Path(temp) / version
            staging.mkdir()
            enriched = validate_enriched(records, embedder(records, images, Path(temp))) if embedder is not None else records
            files = write_shards(staging, enriched, preview=False)
            for key, source in images.items():
                copy_image(source, upload / key)
            produced = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
            manifest = dict(protocolVersion='1.0', businessCode=BUSINESS, batchVersion=version,
                            batchType='incremental', producedAt=produced, format='jsonl',
                            compression='gzip', contentEncoding='utf-8', files=files,
                            totalFileCount=len(files), totalRecordCount=len(records))
            save_json(staging / 'manifest.json', manifest)
            staging.rename(destination)
        manifests.append(manifest)
        update_meta(business, manifests)
        print(f'Upload package: {destination}; records={len(records)}, parts={len(files)} (local only).', flush=True)
        return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--batch-date', required=True, help='YYYYMMDD or YYYY_MM_DD')
    parser.add_argument('--upload-dir', type=Path, help='Defaults to upload_data next to data')
    parser.add_argument('--embedding-model-path', type=Path)
    parser.add_argument('--embedding-gpu-memory-utilization', type=float, default=0.95)
    parser.add_argument('--embedding-batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.embedding_batch_size < 1 or not 0 < args.embedding_gpu_memory_utilization <= 1:
        parser.error('Invalid embedding batch size or GPU memory utilization')
    prepare(args.data, args.batch_date, args.upload_dir, all_pending=True,
            embedder=embedding_worker(args.data, args.upload_dir, model=args.embedding_model_path,
                                      gpu_memory_utilization=args.embedding_gpu_memory_utilization,
                                      batch_size=args.embedding_batch_size))
