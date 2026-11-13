#!/usr/bin/env python3
"""Weekly report discovery and resumable extraction; indexes are derived from receipts."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import date, datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
from threading import Event
import time
from zoneinfo import ZoneInfo

from prepare_upload import BUSINESS, ID_PREFIX, embedding_worker, history, prepare as prepare_upload

ROOT = Path(__file__).resolve().parent
TERMINAL = {'imported', 'complete', 'empty'}
LEGACY_BATCH = '2026_09_28'


def mid_root(data):
    return data.with_name(data.name + '_mid')


def batch_key(batch):
    return batch.replace('_', '')


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def save(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def save_rows(path, values):
    atomic(path, ''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def lock(data):
    state = mid_root(data) / '_state'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'active.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another migration/incremental run is active') from exc
        yield


def database_rows(db, start, end=None):
    # SQLite backup gives a coherent snapshot even if the source has an active WAL.
    with tempfile.TemporaryDirectory(prefix='engreport_index_') as tmp:
        source = sqlite3.connect(f'file:{db.resolve()}?mode=ro', uri=True, timeout=30)
        target = sqlite3.connect(str(Path(tmp) / 'index.db'))
        try:
            source.backup(target)
            values = target.execute(
                'SELECT pdf_id,json_relative_path,raw_date FROM research_reports '
                'WHERE raw_date>=? AND (? IS NULL OR raw_date<=?) ORDER BY raw_date DESC,pdf_id',
                (start, end, end)).fetchall()
        finally:
            source.close()
            target.close()
    return [dict(id=i, relative_path=p, date=d) for i, p, d in values]


def existing_documents(data, upload_dir=None):
    """Read durable outputs even when old extraction caches/receipts were removed."""
    upload = Path(upload_dir).resolve() if upload_dir else data.parent / 'upload_data'
    exported = {}
    history(upload / BUSINESS, exported, ignore_embedding=True)  # Local extraction indexes do not duplicate vectors.
    result = {}
    for identity, record in exported.items():
        if not identity.startswith(ID_PREFIX):
            raise ValueError(f'Unexpected exported ID: {identity}')
        identity = identity[len(ID_PREFIX):]
        image = Path(record['image_file'])
        if len(image.parts) != 3 or image.parts[0] != 'pic' or '..' in image.parts:
            raise ValueError(f'Invalid exported image key: {image}')
        day = datetime.strptime(image.parts[1], '%Y%m%d').strftime('%Y%m%d')
        document = {**record, 'id': identity,
                    'image_file': os.path.relpath(upload / image, data)}
        result[identity] = (day, document)
    for index in sorted((data / 'jsonl').glob('*.jsonl')):
        day = datetime.strptime(index.stem, '%Y%m%d').strftime('%Y%m%d')
        seen = set()
        for record in rows(index):
            if record['id'] in seen:
                raise ValueError(f'Duplicate ID in {index}: {record["id"]}')
            seen.add(record['id'])
            image = (index.parent / record['image_file']).resolve()
            result[record['id']] = (day, {**record, 'image_file': os.path.relpath(image, data)})
    return result


def states(data, upload_dir=None):
    state = mid_root(data) / '_state'
    initial = state / 'legacy_reports.json'
    result = read_json(initial) if initial.exists() else {}
    for _, document in existing_documents(data, upload_dir).values():
        report_id = document.get('report_id') or document['id'].split(':', 1)[0]
        result[report_id] = {'status': 'imported'}
    for p in sorted((state / 'reports').glob('*.json')):
        row = read_json(p)
        result[row['id']] = row
    return result


def source_path(row, root):
    root = root.resolve()
    path = (root / row['relative_path']).resolve()
    if not path.is_relative_to(root) or path.parent.name != row['id']:
        raise ValueError('Invalid source path: ' + row['relative_path'])
    return path


def discover(args):
    old = states(args.data, getattr(args, 'upload_dir', None))
    inventory = database_rows(args.db, args.start_date, args.end_date)
    pending = []
    for r in inventory:
        prior = old.get(r['id'], {})
        if prior.get('status') not in TERMINAL:
            pending.append({**r, 'reason': 'retry' if prior else 'new'})
    with ThreadPoolExecutor(max_workers=16) as pool:
        available = list(pool.map(lambda r: source_path(r, args.source_root).is_file(), pending))
    ready = [r for r, ok in zip(pending, available) if ok]
    missing = [r for r, ok in zip(pending, available) if not ok]
    weeks = Counter()
    for r in pending:
        d = date.fromisoformat(r['date'][:10])
        weeks[(d - timedelta(days=d.weekday())).isoformat()] += 1
    summary = dict(start_date=args.start_date, end_date=args.end_date,
                   scoped_reports=len(inventory), already_processed=len(inventory) - len(pending),
                   pending=len(pending), ready=len(ready), missing_source=len(missing),
                   reasons=dict(Counter(r['reason'] for r in pending)),
                   weeks=dict(sorted(weeks.items())))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return ready, missing, summary


def combine(data, upload_dir=None):
    """Recoverable export: a report receipt commits images, metadata and questions together."""
    mid = mid_root(data)
    by_id = existing_documents(data, upload_dir)
    by_query = {}
    for index in sorted(mid.glob('????_??_??/question.jsonl')):
        day = batch_key(index.parent.name)
        for q in rows(index):
            by_query[q['query_id']] = (day, q)
    grouped = defaultdict(list)
    for p in sorted((mid / '_state/reports').glob('*.json')):
        r = read_json(p)
        if r['status'] in TERMINAL:
            grouped[r['batch']].append(r)
    for batch, receipts in sorted(grouped.items()):
        if batch == LEGACY_BATCH:
            raise ValueError('Historical batch is immutable')
        day = batch_key(batch)
        receipt_ids = set()
        for r in receipts:
            for d in r['documents']:
                if d['id'] in receipt_ids:
                    raise ValueError(f'Duplicate receipt document: {d["id"]}')
                receipt_ids.add(d['id'])
                by_id[d['id']] = (day, {**d, 'image_file': f'picture/{day}/' + Path(d['image_file']).name})
            for q in r['questions']:
                by_query[q['query_id']] = (day, q)
        save_rows(mid / batch / 'reports.jsonl', [
            {k: v for k, v in r.items() if k not in {'documents', 'questions'}} for r in receipts])
    documents_by_day, questions_by_day = defaultdict(list), defaultdict(list)
    for identity in sorted(by_id):
        day, d = by_id[identity]
        documents_by_day[day].append(d)
    for identity in sorted(by_query):
        day, q = by_query[identity]
        if q['visual_id'] not in by_id:
            raise ValueError(f'Question has no visual: {identity}')
        questions_by_day[day].append({**q, 'image_file': by_id[q['visual_id']][1]['image_file']})
    for day in sorted(set(documents_by_day) | {batch_key(b) for b in grouped}):
        def relative(values, folder):
            return [{**v, 'image_file': os.path.relpath(data / v['image_file'], folder)} for v in values]
        save_rows(data / 'jsonl' / f'{day}.jsonl', relative(documents_by_day[day], data / 'jsonl'))
        batch = datetime.strptime(day, '%Y%m%d').strftime('%Y_%m_%d')
        if day in questions_by_day or batch in grouped:
            save_rows(mid / batch / 'question.jsonl', relative(questions_by_day[day], mid / batch))
    documents = [d for day in sorted(documents_by_day) for d in documents_by_day[day]]
    questions = [q for day in sorted(questions_by_day) for q in questions_by_day[day]]
    save_rows(data / 'full.jsonl', documents)
    save_rows(mid / 'question.jsonl', [{**q, 'image_file': os.path.relpath(data / q['image_file'], mid)}
                                     for q in questions])
    save(mid / 'summary.json', dict(images=len(documents), questions=len(questions),
         reports=len({d.get('report_id') or d['id'].split(':', 1)[0] for d in documents}),
         batches=[datetime.strptime(day, '%Y%m%d').strftime('%Y_%m_%d') for day in sorted(documents_by_day)],
         updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))


def command(cmd, log):
    with log.open('a') as f:
        f.write('\nCOMMAND ' + repr(cmd) + '\n')
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def process_report(row, args):
    work = mid_root(args.data) / '_state/work' / row['id']
    work.mkdir(parents=True, exist_ok=True)
    # report_records derives report_id from the output directory name.
    visual = work / 'visuals/reports' / row['id']
    # Receipts are committed only after both extraction and query generation finish.
    # Failed reports keep all intermediate caches in this stable per-report directory.
    try:
        source = source_path(row, args.source_root)
        identity = dict(source=str(source), source_sha256=sha(source),
                        model=args.model, api_base=args.api_base,
                        prompt_sha256=sha(args.prompt),
                        code={str(p.relative_to(ROOT)): sha(p) for p in sorted([
                            ROOT / 'extract_visual_records_light.py', ROOT / 'extract_visual_records.py',
                            ROOT / 'generate_search_queries.py', *list((ROOT / 'visual_recall').glob('*.py'))])})
        identity_path = work / 'identity.json'
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError(f'Inputs changed; inspect/rename work directory before retry: {work}')
        save(identity_path, identity)
        complete = work / 'extracted.json'
        if not complete.exists():
            rc = command([sys.executable, str(ROOT / 'extract_visual_records_light.py'),
                          '--report-dir', str(source.parent), '--output-dir', str(visual),
                          '--model', args.model, '--api-base', args.api_base], work / 'extract.log')
            result = read_json(visual / 'run_result.json')
            # As in the original batch flow, retain published visuals despite
            # per-visual extraction errors (exit 2).
            if (rc not in (0, 2) or result['dry_run']
                    or (rc == 2 and not rows(visual / 'published_visuals.jsonl'))):
                raise RuntimeError(f'Extraction incomplete (exit={rc}); see {work / "extract.log"}')
            save(complete, result)
        import generate_search_queries as gen
        manifest = work / 'query_manifest.jsonl'
        if not manifest.exists():
            save_rows(manifest, gen.report_records(visual / 'published_visuals.jsonl'))
        published = rows(manifest)
        query_out = work / 'queries'
        query_out.mkdir(exist_ok=True)
        # Remove legacy markers/links so cached query jobs can resume normally.
        (query_out / 'STOP').unlink(missing_ok=True)
        if published:
            rc = command([sys.executable, str(ROOT / 'generate_search_queries.py'), 'run',
                          '--manifest', str(manifest), '--output', str(query_out),
                          '--prompt', str(args.prompt), '--model', args.model,
                          '--api-base', args.api_base, '--workers', '1'], work / 'queries.log')
            progress = read_json(query_out / 'progress.json')
            if rc != 0 or progress['status'] != 'complete':
                raise RuntimeError(f'Queries incomplete; see {work / "queries.log"}')
        exported = []
        images = args.data / 'picture' / batch_key(args.batch_date)
        images.mkdir(parents=True, exist_ok=True)
        for r in published:
            src = Path(r['image_file'])
            name = r['id'].replace(':', '__') + src.suffix.lower()
            if Path(name).name != name:
                raise ValueError('Invalid visual ID')
            dst = images / name
            if not dst.exists():
                tmp = dst.with_suffix(dst.suffix + '.tmp')
                shutil.copy2(src, tmp)
                tmp.replace(dst)
            if sha(src) != sha(dst):
                raise ValueError(f'Image conflict: {dst}')
            exported.append({**{k: v for k, v in r.items() if not k.startswith('_')},
                             'report_id': row['id'], 'image_file': name})
        mapping = {r['id']: r['image_file'] for r in exported}
        questions = [{**q, 'image_file': mapping[q['visual_id']]}
                     for q in rows(query_out / 'queries.jsonl')]
        return {**row, 'status': 'complete' if exported else 'empty',
                'batch': args.batch_date, 'documents': exported, 'questions': questions,
                'processed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    except Exception as exc:
        return {**row, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                'attempted_batch': args.batch_date}


@contextmanager
def server(args, interrupted=None):
    import httpx
    proc = None
    handle = None
    try:
        if not args.reuse_server:
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', args.port)) == 0:
                    raise RuntimeError(f'Port {args.port} is occupied; use --reuse-server to use an existing matching model')
            executable = Path(sys.executable).parent / 'vllm'
            if not executable.is_file():
                raise FileNotFoundError(f'vLLM executable missing: {executable}; set PYTHON_BIN to your vLLM environment')
            log = mid_root(args.data) / '_state' / f'vllm_{args.batch_date}.log'
            handle = log.open('a')
            cmd = [str(executable), 'serve', str(args.model_path), '--served-model-name', args.model,
                   '--tensor-parallel-size', str(args.tensor_parallel_size), '--dtype', 'bfloat16',
                   '--max-model-len', str(args.max_model_len),
                   '--gpu-memory-utilization', str(args.gpu_memory_utilization),
                   '--limit-mm-per-prompt', '{"image": 2}', '--trust-remote-code',
                   '--host', '127.0.0.1', '--port', str(args.port)]
            proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            print(f'Starting vLLM; log: {log}', flush=True)
        deadline = time.monotonic() + args.startup_timeout
        with httpx.Client(timeout=3, trust_env=False) as client:
            while True:
                if interrupted is not None and interrupted.is_set():
                    raise InterruptedError('Stopped while waiting for vLLM')
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError('vLLM exited during startup; check its log')
                try:
                    response = client.get(args.api_base + '/models')
                    response.raise_for_status()
                    models = {m['id'] for m in response.json()['data']}
                    if args.model not in models:
                        raise ValueError(f'Expected model {args.model}, server advertises {models}')
                    break
                except (httpx.HTTPError, KeyError):
                    if args.reuse_server or time.monotonic() >= deadline:
                        raise RuntimeError('vLLM endpoint is not ready')
                    time.sleep(2)
        yield
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if handle:
            handle.close()


def publish_increment(args):
    upload = getattr(args, 'upload_dir', None)
    embedder = embedding_worker(args.data, upload,
        model=getattr(args, 'embedding_model_path', None),
        gpu_memory_utilization=getattr(args, 'embedding_gpu_memory_utilization', 0.95),
        batch_size=getattr(args, 'embedding_batch_size', 256))
    batch = mid_root(args.data) / args.batch_date
    status = batch / 'publication.json'
    save(status, dict(status='preparing', batch_date=args.batch_date))
    try:
        result = prepare_upload(args.data, args.batch_date, upload, embedder=embedder, all_pending=True)
    except KeyboardInterrupt:
        save(status, dict(status='pending', note='Embedding/publication interrupted; cached vectors and extraction receipts retained.'))
        raise
    except Exception as exc:
        save(status, dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                         note='Extraction receipts and embedding cache retained; retry publication after resolving the error.'))
        raise
    save(status, dict(status='complete' if result else 'no_changes',
                      batch_version=result['batchVersion'] if result else None,
                      records=result['totalRecordCount'] if result else 0))


def run(args):
    with lock(args.data):
        combine(args.data, getattr(args, 'upload_dir', None))  # recover indexes from durable data and receipts
        ready, missing, summary = discover(args)
        batch = mid_root(args.data) / args.batch_date
        batch.mkdir(exist_ok=True)
        save(batch / 'latest_discovery.json', summary)
        save_rows(batch / 'missing_sources.jsonl', missing)
        if not ready:
            publish_increment(args)
            print('No ready reports; extraction vLLM was not started.', flush=True)
            return 0
        if args.max_reports:
            ready = ready[:args.max_reports]
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        save_rows(batch / f'inventory_{stamp}.jsonl', ready)
        shutil.copy2(args.prompt, batch / 'query_prompt.txt')
        interrupted = Event()
        def stop_requested(*_):
            interrupted.set()
            print('Stopping this run; waiting for active reports. Rerun the same command to resume.', flush=True)
        previous = {s: signal.signal(s, stop_requested) for s in (signal.SIGINT, signal.SIGTERM)}
        failures = 0
        done = 0
        try:
            with server(args, interrupted), ThreadPoolExecutor(max_workers=args.workers) as pool:
                iterator = iter(ready)
                active = {}
                exhausted = False
                while active or not exhausted:
                    while len(active) < args.workers and not exhausted and not interrupted.is_set():
                        r = next(iterator, None)
                        if r is None:
                            exhausted = True
                            break
                        active[pool.submit(process_report, r, args)] = r['id']
                    if not active:
                        break
                    completed, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        result = future.result()
                        del active[future]
                        save(mid_root(args.data) / '_state/reports' / (result['id'] + '.json'), result)
                        done += 1
                        failures += result['status'] == 'failed'
                        print(f'[{done}/{len(ready)}] {result["id"]} {result["status"]} '
                              f'images={len(result.get("documents", []))} {result.get("error", "")}', flush=True)
                        save(batch / 'progress.json', dict(planned=len(ready), processed=done,
                             failures=failures, status='running'))
        except InterruptedError:
            if not interrupted.is_set():
                raise
        finally:
            combine(args.data, getattr(args, 'upload_dir', None))
            save(batch / 'progress.json', dict(planned=len(ready), processed=done, failures=failures,
                 status='stopped' if done < len(ready) else ('complete_with_errors' if failures else 'complete')))
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        # server() has now exited and closed the vLLM process it owned.
        # Ctrl+C must not unexpectedly start a second GPU job during cleanup.
        if not interrupted.is_set():
            publish_increment(args)
        else:
            save(batch / 'publication.json', dict(status='pending', note='Stopped before embedding/publication; rerun to resume.'))
        return 130 if interrupted.is_set() else (2 if failures or done < len(ready) else 0)


def parse_date(value):
    return date.fromisoformat(value).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run'])
    parser.add_argument('--start-date', type=parse_date, required=True, help='Inclusive report publication date, YYYY-MM-DD')
    parser.add_argument('--end-date', type=parse_date, help='Optional inclusive publication date')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--upload-dir', type=Path, help='Local protocol package directory; defaults to upload_data next to data. No OSS upload.')
    parser.add_argument('--db', type=Path, default=Path(os.environ.get('REPORT_DB', str(ROOT / 'external_data/research_index.db'))))
    parser.add_argument('--source-root', type=Path, default=Path(os.environ.get('REPORT_SOURCE_ROOT', str(ROOT / 'external_data/json_output'))))
    parser.add_argument('--batch-date', default=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y_%m_%d'),
                        help='Storage batch date, YYYYMMDD or YYYY_MM_DD; defaults to today in Asia/Shanghai')
    parser.add_argument('--workers', type=int, default=8, help='Concurrent reports / model requests')
    parser.add_argument('--max-reports', type=int, default=0, help='0 means all ready reports')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '7888')))
    parser.add_argument('--model', default=os.environ.get('SERVED_MODEL_NAME', 'qwen3.8-27b'))
    parser.add_argument('--model-path', type=Path, default=Path(os.environ.get('MODEL_PATH',
                        str(ROOT / 'models/qwen3.8-27b'))))
    parser.add_argument('--prompt', type=Path, default=ROOT / 'search_query_prompt.txt')
    parser.add_argument('--tensor-parallel-size', type=int, default=int(os.environ.get('TENSOR_PARALLEL_SIZE', '1')))
    parser.add_argument('--max-model-len', type=int, default=int(os.environ.get('MAX_MODEL_LEN', '32768')))
    parser.add_argument('--gpu-memory-utilization', type=float, default=float(os.environ.get('GPU_MEMORY_UTILIZATION', '0.85')))
    parser.add_argument('--embedding-model-path', type=Path, default=Path(os.environ.get('EMBEDDING_MODEL_PATH', str(ROOT / 'models/Qwen3-VL-Embedding-2B'))))
    parser.add_argument('--embedding-gpu-memory-utilization', type=float, default=0.95,
                        help='Separate embedding GPU budget, after extraction (default: 0.95)')
    parser.add_argument('--embedding-batch-size', type=int, default=256)
    parser.add_argument('--startup-timeout', type=int, default=900)
    parser.add_argument('--reuse-server', action='store_true', help='Use an already running local server; do not start/stop it')
    args = parser.parse_args()
    args.data = args.data.resolve()
    try:
        batch = datetime.strptime(batch_key(args.batch_date), '%Y%m%d').strftime('%Y_%m_%d')
    except ValueError:
        parser.error('--batch-date must be YYYYMMDD or YYYY_MM_DD')
    args.batch_date = batch
    if args.command == 'run' and batch == LEGACY_BATCH:
        parser.error('Historical batch is read-only; choose a new batch date')
    if args.workers < 1 or args.max_reports < 0 or (args.end_date and args.end_date < args.start_date):
        parser.error('Invalid workers, max-reports, or date range')
    if args.embedding_batch_size < 1 or not 0 < args.embedding_gpu_memory_utilization <= 1:
        parser.error('Invalid embedding batch size or GPU memory utilization')
    args.api_base = f'http://127.0.0.1:{args.port}/v1'
    if args.command == 'check':
        discover(args)
        return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
