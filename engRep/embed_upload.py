#!/usr/bin/env python3
"""Embed unpublished local upload batches using Qwen3-VL-Embedding and vLLM.

Caches and original package backups live in data_mid/embedding. No OSS calls.
Only --rewrite-unpublished replaces local batches, after every vector is ready.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import ctypes
import errno
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np
import prepare_upload as pub

ROOT = Path(__file__).resolve().parent
DIM = 2048
INSTRUCTION = "Represent the user's input."
MIN_PIXELS = 4096
MAX_PIXELS = 1048576


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    pub.save_json(tmp, value)
    tmp.replace(path)


def text_for(data):
    return f"Title: {data['title'].strip()}\nContent: {data['image_content'].strip()}"


def conversation(item):
    content = []
    if item.get('image'):
        content.append(dict(type='image', image=str(item['image']),
                            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS))
    if item['text']:
        content.append(dict(type='text', text=item['text']))
    return [dict(role='system', content=[dict(type='text', text=item.get('instruction', INSTRUCTION))]),
            dict(role='user', content=content)]


def checked_vectors(values):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DIM or not np.isfinite(array).all():
        raise ValueError(f'Invalid embedding dimensions or finite values: {array.shape}')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 0.95) or np.any(norms > 1.05):
        raise ValueError(f'Unexpected pooling normalization: {norms.min()}..{norms.max()}')
    return (array / norms).astype('<f4')


class Encoder:
    def __init__(self, args):
        import torch
        torch.set_num_threads(1)
        from transformers import Qwen3VLProcessor
        from vllm import LLM
        from vllm.config import PoolerConfig
        self.processor = Qwen3VLProcessor.from_pretrained(str(args.model), local_files_only=True, padding_side='right')
        self.pool = ThreadPoolExecutor(max_workers=args.workers)
        self.llm = LLM(model=str(args.model), runner='pooling', convert='embed',
                       dtype='bfloat16', trust_remote_code=True, tensor_parallel_size=1,
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                       max_num_batched_tokens=args.max_num_batched_tokens,
                       enable_prefix_caching=True, seed=0,
                       limit_mm_per_prompt={'image': 1, 'video': 0},
                       renderer_num_workers=args.workers, mm_processor_cache_gb=0,
                       mm_processor_kwargs={'do_resize': False},
                       pooler_config=PoolerConfig(seq_pooling_type='LAST', use_activation=True))
        self.max_model_len = args.max_model_len
        self.render_loop = asyncio.new_event_loop()

    def prepare(self, item):
        from qwen_vl_utils import process_vision_info
        messages = conversation(item)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        request = {'prompt': prompt}
        if item.get('image'):
            images, videos = process_vision_info(messages, image_patch_size=16)
            if videos is not None or len(images) != 1:
                raise ValueError('Exactly one image is required')
            request['multi_modal_data'] = {'image': images[0]}
        return request

    def prepare_items(self, items):
        from vllm.renderers.inputs.preprocess import parse_model_prompt
        begin = time.monotonic()
        requests = list(self.pool.map(self.prepare, items))
        try:
            # Same parse/tokenize/vision operations as EmbedIOProcessor, using
            # vLLM's supported async renderer rather than serial render_cmpl.
            renderer = self.llm.renderer
            parsed = [parse_model_prompt(renderer.model_config, request) for request in requests]
            prepared = self.render_loop.run_until_complete(renderer.render_cmpl_async(
                parsed, renderer.default_cmpl_tok_params))
            return prepared, time.monotonic() - begin
        finally:
            for request in requests:
                for image in request.get('multi_modal_data', {}).values():
                    image.close()

    def encode(self, items):
        requests, _ = self.prepare_items(items)
        return self.encode_prepared(requests)

    def encode_prepared(self, requests):
        from vllm import PoolingParams
        from vllm.outputs import EmbeddingRequestOutput, PoolingRequestOutput
        begin = time.monotonic()
        try:
            results = self.llm._render_and_run_requests(
                prompts=iter(requests), params=[PoolingParams(task='embed') for _ in requests],
                output_type=PoolingRequestOutput, use_tqdm=False)
            outputs = [EmbeddingRequestOutput.from_base(result) for result in results]
            if len(outputs) != len(requests):
                raise ValueError('Embedding count mismatch')
            lengths = [len(o.prompt_token_ids) for o in outputs]
            if max(lengths) > self.max_model_len:
                raise ValueError('Input exceeds context budget')
            return checked_vectors([o.outputs.embedding for o in outputs]), lengths
        finally:
            self.last_timings = dict(embed_seconds=time.monotonic()-begin, render_seconds=0.0)


def load_batches(business):
    _, manifests = pub.history(business)
    batches = []
    for manifest in manifests:
        records = []
        for info in manifest['files']:
            path = business / 'batches' / manifest['batchVersion'] / info['fileName']
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    if record['op'] != 'upsert' or record['id'] != record['data']['id']:
                        raise ValueError(f'Expected full upsert records: {record["id"]}')
                    record['data'].pop('embedding', None)
                    records.append(record)
        batches.append((manifest, records))
    return batches


def source_hashes(business):
    return {str(p.relative_to(business)): sha_file(p) for p in sorted(business.rglob('*'))
            if p.is_file() and (p.suffix == '.gz' or p.name in ('meta.json', 'manifest.json'))}


def smoke(encoder, cache, sample_path=None):
    if sample_path is None:
        print('External consistency sample not supplied; dimension, finiteness and normalization checks remain enabled.', flush=True)
        return
    sample_path = Path(sample_path).resolve()
    sample = pub.read_json(sample_path)
    query, _ = encoder.encode([sample['query']])
    items = [{**r['target'], 'image': str(sample_path.parent / r['target']['image'])} for r in sample['top10']]
    vectors, lengths = encoder.encode(items)
    scores = (vectors @ query[0]).tolist()
    references = [r['similarity'] for r in sample['top10']]
    differences = [abs(a - b) for a, b in zip(scores, references)]
    report = dict(dim=DIM, norms=np.linalg.norm(vectors, axis=1).tolist(), token_lengths=lengths,
                  scores=scores, reference_scores=references, max_score_difference=max(differences),
                  note='vLLM backend vs historical Transformers BF16 reference; not an online-service comparison')
    previous = cache / 'sample_validation.json'
    if previous.exists():
        old_scores = pub.read_json(previous)['scores']
        report['difference_from_previous_vllm_run'] = max(abs(a-b) for a,b in zip(scores,old_scores,strict=True))
        if report['difference_from_previous_vllm_run'] > 0.001:
            raise ValueError(f'Optimized encoder changed sample scores: {report}')
    atomic_json(cache / 'sample_validation.json', report)
    print('Sample validation:', json.dumps(report), flush=True)
    if max(differences) > 0.02:
        raise ValueError('Sample scores differ by >0.02; investigate inputs before full encoding')


def _atomic_exchange(a, b):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = libc.renameat2
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(a))


def exchange_directories(a, b):
    """Keep the original at a, including on filesystems without RENAME_EXCHANGE.

    Caller holds the extraction and publication locks. The fallback uses
    ordinary renames and restores the original destination on failure.
    """
    try:
        _atomic_exchange(a, b)
        return
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS):
            raise
    previous = a.with_name(a.name + '.previous')
    if previous.exists():
        raise FileExistsError(f'Previous interrupted replacement needs inspection: {previous}')
    b.rename(previous)
    try:
        a.rename(b)
    except BaseException:
        previous.rename(b)
        raise
    try:
        previous.rename(a)
    except BaseException:
        b.rename(a)
        previous.rename(b)
        raise


def rewrite(business, batches, db, keys, cache, before):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = cache / f'package_{stamp}'
    staging = work / pub.BUSINESS
    staging.mkdir(parents=True)
    summaries = []
    for old, records in batches:
        folder = staging / 'batches' / old['batchVersion']
        folder.mkdir(parents=True)

        def enriched():
            for record in records:
                key = keys[record['id']]
                row = db.execute('SELECT vector FROM vectors WHERE key=?', (key,)).fetchone()
                if row is None:
                    raise ValueError('Missing vector: ' + record['id'])
                vector = np.frombuffer(row[0], dtype='<f4')
                checked_vectors([vector])
                yield {**record, 'data': {**record['data'], 'embedding': vector.tolist()}}

        files = pub.write_shards(folder, enriched())
        manifest = {**old, 'files': files, 'totalFileCount': len(files),
                    'totalRecordCount': len(records)}
        pub.save_json(folder / 'manifest.json', manifest)
        index = 0
        for info in files:
            packed = (folder / info['fileName']).read_bytes()
            if len(packed) != info['sizeBytes'] or len(packed) > pub.MAX_BYTES or 'sha256:' + pub.digest(packed) != info['checksum']:
                raise ValueError('Generated shard failed checksum/size validation')
            raw = gzip.decompress(packed)
            if raw != (folder / info['fileName']).with_suffix('').read_bytes() or len(raw) != info['uncompressedSizeBytes']:
                raise ValueError('Preview or uncompressed size mismatch')
            lines = raw.splitlines()
            if len(lines) != info['recordCount']:
                raise ValueError('Shard record count mismatch')
            for line in lines:
                actual = json.loads(line)
                vector = actual['data'].pop('embedding')
                checked_vectors([vector])
                original = records[index]
                if actual != original:
                    raise ValueError('Business data changed: ' + original['id'])
                stored = db.execute('SELECT vector FROM vectors WHERE key=?', (keys[original['id']],)).fetchone()[0]
                if np.asarray(vector, dtype='<f4').tobytes() != stored:
                    raise ValueError('Embedding associated with wrong record: ' + original['id'])
                index += 1
        if index != len(records):
            raise ValueError('Batch total mismatch')
        summaries.append(dict(batch=old['batchVersion'], records=index, parts=len(files)))
        print('Validated batch:', summaries[-1], flush=True)
    meta = pub.read_json(business / 'meta.json')
    meta['generatedAt'] = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    pub.save_json(staging / 'meta.json', meta)
    if source_hashes(business) != before:
        raise ValueError('Original package changed during encoding; refusing to replace it')
    # After exchange, staging contains the original package (retained as backup).
    atomic_json(cache / 'replacement_plan.json', dict(destination=str(business), backup=str(staging), batches=summaries))
    exchange_directories(staging, business)
    atomic_json(cache / 'completed.json', dict(finished_at=meta['generatedAt'], backup=str(staging),
                                               batches=summaries, records=sum(s['records'] for s in summaries), dim=DIM))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload-dir', type=Path, default=ROOT / 'upload_data')
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('EMBEDDING_MODEL_PATH', str(ROOT / 'models/Qwen3-VL-Embedding-2B'))))
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.95)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--max-num-batched-tokens', type=int, default=32768)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--validation-sample', type=Path, default=os.environ.get('EMBEDDING_VALIDATION_SAMPLE'), help='Optional external consistency-test JSON; no dataset is bundled')
    parser.add_argument('--limit', type=int, default=0, help='Cache only the first N uncached records; do not rewrite packages')
    parser.add_argument('--rewrite-unpublished', action='store_true', help='Authorized replacement of local unpublished batches')
    parser.add_argument('--input-records', type=Path, help='Internal incremental input; encode these records only, without editing any published batches')
    parser.add_argument('--output-records', type=Path, help='Internal enriched JSONL output in data_mid')
    parser.add_argument('--source-data', type=Path, default=ROOT / 'data')
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'data_mid/embedding')
    args = parser.parse_args()
    if min(args.batch_size, args.workers, args.max_num_seqs) < 1 or args.limit < 0:
        parser.error('Invalid batch size/workers/limit')
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error('GPU memory utilization must be in (0, 1]')
    incremental = args.input_records is not None
    if incremental and (args.output_records is None or args.limit or args.rewrite_unpublished):
        parser.error('--input-records requires --output-records and cannot be used with --limit or --rewrite-unpublished')
    if args.output_records is not None and not incremental:
        parser.error('--output-records requires --input-records')
    if incremental and args.input_records.resolve() == args.output_records.resolve():
        parser.error('Input and output records must be different files')
    args.upload_dir, args.model = args.upload_dir.resolve(), args.model.resolve()
    business = args.upload_dir / pub.BUSINESS
    args.source_data = args.source_data.resolve()
    state = args.source_data.with_name(args.source_data.name + '_mid') / '_state'
    state.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        # Incremental mode is invoked while the parent owns both publication
        # locks. It only writes the vector cache and the parent's staging output.
        for name in (() if incremental else ('active.lock', 'upload.lock')):
            handle = stack.enter_context((state / name).open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {}
        if incremental:
            records = []
            with args.input_records.open(encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    sources[record['id']] = Path(record.pop('_source_image')).resolve()
                    if record.get('op') != 'upsert' or record['id'] != record['data']['id']:
                        raise ValueError('Invalid incremental upsert')
                    records.append(record)
            batches, before = [], None
        else:
            print('Reading and validating source packages...', flush=True)
            before = source_hashes(business)
            batches = load_batches(business)
            records = [r for _, rs in batches for r in rs]
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('This backfill expects unique IDs across the date batches')
        if not records:
            raise ValueError('No embedding records supplied')
        print(f'Loaded {len(records)} records ({"incremental" if incremental else "backfill"}).', flush=True)
        config = dict(model=str(args.model), dim=DIM, dtype='bfloat16', pooling='LAST', normalize=True,
                      text_template='Title: {title.strip()}\nContent: {image_content.strip()}',
                      content_order=['image', 'text'], instruction=INSTRUCTION,
                      min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, patch_size=16,
                      add_generation_prompt=True, do_resize=False, max_model_len=args.max_model_len,
                      libraries={n: importlib.metadata.version(n) for n in ['vllm', 'torch', 'transformers', 'qwen-vl-utils', 'pillow']},
                      model_files={p.name:sha_file(p) for p in sorted(args.model.iterdir())
                                   if p.is_file() and p.suffix in ('.json', '.jinja', '.safetensors')})
        fingerprint = pub.digest(pub.encoded(config))
        cache = args.cache_root.resolve() / fingerprint[:20]
        cache.mkdir(parents=True, exist_ok=True)
        atomic_json(cache / 'config.json', config)
        db = sqlite3.connect(cache / 'vectors.sqlite3', check_same_thread=False)
        stack.callback(db.close)
        # Single writer on this host. Append-only WAL avoids repeated random
        # fsyncs on the shared filesystem; interrupted transactions are retried.
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('PRAGMA wal_autocheckpoint=4096')
        db.execute('PRAGMA cache_size=-1048576')  # 1 GiB; avoids per-vector shared-disk page reads/spills.
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL, tokens INTEGER NOT NULL)')

        def describe(record):
            image_key = Path(record['data']['image_file'])
            if len(image_key.parts) != 3 or image_key.parts[0] != 'pic' or '..' in image_key.parts:
                raise ValueError('Invalid image key: ' + str(image_key))
            packaged = (args.upload_dir / image_key).resolve()
            path = sources[record['id']] if incremental else packaged
            local = (args.source_data / 'picture' / image_key.parts[1] / image_key.parts[2]).resolve()
            if path not in (packaged, local) or not path.is_file():
                raise ValueError('Invalid/missing image: ' + str(path))
            text = text_for(record['data'])
            image_hash = sha_file(path)
            key = pub.digest(pub.encoded(dict(config=fingerprint, image=image_hash, text=text)))
            return record['id'], key, dict(image=str(path), text=text), image_hash

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            descriptors = list(pool.map(describe, records))
        keys = {identity:key for identity,key,_,_ in descriptors}
        existing = {r[0] for r in db.execute('SELECT key FROM vectors')}
        pending = [(identity,key,item,image_hash) for identity,key,item,image_hash in descriptors if key not in existing]
        print(f'Cache: {cache}; cached={len(records)-len(pending)} pending={len(pending)}', flush=True)
        start = time.monotonic()
        max_tokens = 0
        if pending:
            encoder = Encoder(args)
            smoke(encoder, cache, args.validation_sample)
            if args.limit:
                pending = pending[:args.limit]
            # Prepare the next image batch while vLLM encodes and we commit this
            # batch. Only one future is queued, keeping CPU RAM bounded.
            def persist(chunk, vectors, lengths):
                writing = time.monotonic()
                with db:
                    db.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?,?)',
                                   [(key, vector.tobytes(), tokens)
                                    for (_,key,_,_),vector,tokens in zip(chunk,vectors,lengths,strict=True)])
                return time.monotonic()-writing

            def committed(future, progress):
                progress['timings']['write'] = future.result()
                progress['elapsed_seconds'] = time.monotonic()-start
                progress['records_per_second'] = progress['processed']/progress['elapsed_seconds']
                atomic_json(cache / 'progress.json', progress)
                t = progress['timings']
                print(f'Embedding {progress["processed"]}/{len(pending)} '
                      f'{progress["records_per_second"]:.2f} images/s '
                      f'wait={t["wait"]:.2f}s prep={t["prepare"]:.2f}s '
                      f'embed={t["embed_seconds"]:.2f}s write={t["write"]:.2f}s', flush=True)

            # DB access has a single writer; main-thread reads resume only
            # after the writer joins. At most one vector batch waits for disk.
            with ThreadPoolExecutor(max_workers=1) as prefetch, ThreadPoolExecutor(max_workers=1) as writer:
                future = prefetch.submit(encoder.prepare_items, [x[2] for x in pending[:args.batch_size]])
                start = time.monotonic()
                write_future = None
                last_progress = None
                for offset in range(0, len(pending), args.batch_size):
                    chunk = pending[offset:offset+args.batch_size]
                    waiting = time.monotonic()
                    requests, prep_seconds = future.result()
                    wait_seconds = time.monotonic()-waiting
                    following = pending[offset+args.batch_size:offset+2*args.batch_size]
                    if following:
                        future = prefetch.submit(encoder.prepare_items, [x[2] for x in following])
                    vectors, lengths = encoder.encode_prepared(requests)
                    if write_future is not None:
                        committed(write_future, last_progress)
                    write_future = writer.submit(persist, chunk, vectors, lengths)
                    max_tokens = max(max_tokens, max(lengths))
                    completed = min(offset+len(chunk), len(pending))
                    elapsed = time.monotonic()-start
                    last_progress = dict(status='encoding', processed=completed, pending_at_start=len(pending),
                                    completed_total=len(records)-len(pending)+completed,
                                    total_records=len(records), elapsed_seconds=elapsed,
                                    records_per_second=completed/elapsed, max_observed_tokens=max_tokens,
                                    gpu_memory_utilization=args.gpu_memory_utilization,
                                    timings=dict(prepare=prep_seconds, wait=wait_seconds,
                                                 **encoder.last_timings))
                if write_future is not None:
                    committed(write_future, last_progress)
        if args.limit:
            print('Limited cache run complete; source packages unchanged.', flush=True)
            return
        # Verify inputs did not change while encoding; cached vectors are content-keyed.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            now = list(pool.map(describe, records))
        if now != descriptors:
            raise ValueError('Embedding inputs changed during execution')
        if incremental:
            temporary = args.output_records.with_suffix(args.output_records.suffix + '.tmp')
            with temporary.open('wb') as f:
                for record in records:
                    row = db.execute('SELECT vector FROM vectors WHERE key=?', (keys[record['id']],)).fetchone()
                    if row is None:
                        raise ValueError('Missing incremental vector: ' + record['id'])
                    vector = np.frombuffer(row[0], dtype='<f4')
                    checked_vectors([vector])
                    f.write(pub.encoded({**record, 'data': {**record['data'], 'embedding': vector.tolist()}}))
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(args.output_records)
            atomic_json(cache / 'progress.json', dict(status='complete', mode='incremental', records=len(records), dim=DIM))
            print(f'Incremental embeddings ready: {len(records)} records; {args.output_records}', flush=True)
        elif args.rewrite_unpublished:
            result = rewrite(business, batches, db, keys, cache, before)
            atomic_json(cache / 'progress.json', dict(status='complete', records=len(records), batches=result))
            print('COMPLETE:', json.dumps(result), flush=True)
        else:
            print('All vectors cached; packages unchanged. Use --rewrite-unpublished to apply.', flush=True)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    main()
