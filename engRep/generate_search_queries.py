#!/usr/bin/env python3
"""Generate independent search queries from published target images; no GPU locally.

results.jsonl is the durable, per-visual source of truth. queries.jsonl is rebuilt
from it on resume, so a crash between the two writes cannot lose or duplicate queries.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import time

import httpx

HERE = Path(__file__).resolve().parent
STOP = False
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "reason", "queries"],
    "properties": {
        "status": {"type": "string", "enum": ["ok", "review", "skip"]},
        "reason": {"type": "string"},
        "queries": {"type": "array", "maxItems": 1, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "text", "evidence"],
            "properties": {
                "type": {"type": "string", "enum": ["core"]},
                "text": {"type": "string"},
                "evidence": {"type": "array", "minItems": 1, "maxItems": 5, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["source", "text"], "properties": {
                        "source": {"type": "string", "enum": ["image", "caption", "report_heading", "table_labels"]},
                        "text": {"type": "string"}}}}}}}}
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def dump_line(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def report_records(path):
    run_info = json.loads((path.parent / "run_result.json").read_text())
    report_id = path.parent.name
    original = Path(run_info["report_dir"]) / f"{report_id}.json"
    source_meta = json.loads(original.read_text())
    source_data = source_meta.get("data") or {}
    tables = source_data.get("tables") or {}
    result = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("publish_ready") is not True:
                continue
            stat = Path(row["image_file"]).stat()
            # Only labels from this exact source table; never neighboring context.
            own_table = tables.get(row['id'].split(':')[-1], {}) if isinstance(tables, dict) else {}
            labels = []
            for cell in re.split(r"[|\n]", own_table.get("markdown", "")):
                label = html.unescape(cell.strip())
                if re.search(r"[A-Za-z]", label) and label not in labels:
                    labels.append(label[:160])
            row.update(_source_file=str(path.resolve()), _image_size=stat.st_size,
                       _image_mtime_ns=stat.st_mtime_ns,
                       _report_security_code=source_meta.get("security_code") or "",
                       _target_table_labels=labels[:80])
            result.append(row)
    return result


def prepare(args):
    if args.manifest.exists():
        raise FileExistsError(f"Frozen manifest already exists: {args.manifest}")
    records = {}
    sources = args.sources or [HERE / "visual_2026_07", HERE / "visual_2026_newest_first"]
    for source in sources:
        files = sorted((source / "reports").glob("*/published_visuals.jsonl"))
        if not files:
            raise ValueError(f"No published records in {source}")
        with ThreadPoolExecutor(max_workers=16) as pool:
            for n, batch in enumerate(pool.map(report_records, files), 1):
                for row in batch:
                    if row['id'] in records:
                        old = {k: v for k, v in records[row['id']].items() if not k.startswith('_')}
                        new = {k: v for k, v in row.items() if not k.startswith('_')}
                        if old != new:
                            raise ValueError(f"Conflicting published ID: {row['id']}")
                        continue
                    records[row['id']] = row
                if n % 500 == 0:
                    print(f"Prepared {n}/{len(files)} reports in {source.name}", flush=True)
    ordered = sorted(records.values(), key=lambda r: (r["publish_time"], r["id"]), reverse=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.manifest.with_suffix(".tmp")
    with tmp.open("w") as out:
        for row in ordered:
            out.write(dump_line(row))
    tmp.replace(args.manifest)
    print(json.dumps({"manifest": str(args.manifest), "visuals": len(ordered),
                      "reports": len({r['id'].split(':')[0] for r in ordered}),
                      "types": dict(Counter(r['type'] for r in ordered))}))


def load_results(path):
    """Only a torn FINAL line can be removed; never ignore interior corruption."""
    result = {}
    if not path.exists():
        return result
    with path.open("rb+") as f:
        end = f.seek(0, 2)
        f.seek(0)
        while f.tell() < end:
            start = f.tell()
            line = f.readline()
            try:
                row = json.loads(line)
                result[row["visual_id"]] = row
                if not line.endswith(b"\n"):
                    f.seek(0, 2)
                    f.write(b"\n")
                    break
            except (ValueError, KeyError):
                if f.tell() != end or line.endswith(b"\n"):
                    raise ValueError(f"Invalid result record at byte {start}: {path}")
                f.seek(start)
                f.truncate()
                break
    return result


def payload(row):
    # Intentionally exclude generated title/image_content and target_companies.
    return {
        "original_caption": row.get("original_caption", ""),
        "original_or_source_footnote": row.get("source_footnote", ""),
        "report_heading_background_only": row.get("report_title", ""),
        "report_security_code_background_only": row.get("_report_security_code", ""),
        "target_table_labels_partial_transcription": row.get("_target_table_labels", []),
        "instruction": "Generate independent short search queries for this target. The report heading is background, not proof of every target's subject or metrics.",
    }


def validate(obj):
    if set(obj) != {"status", "reason", "queries"}:
        raise ValueError("Incorrect response fields")
    status, queries = obj["status"], obj["queries"]
    if status not in {"ok", "review", "skip"} or not isinstance(queries, list):
        raise ValueError("Invalid status/queries")
    if status != "ok":
        if queries or not obj["reason"].strip():
            raise ValueError("review/skip requires reason and no queries")
        return obj
    if len(queries) != 1 or queries[0]["type"] != "core":
        raise ValueError("ok requires exactly one core query")
    types, texts = set(), set()
    for q in queries:
        if set(q) != {"type", "text", "evidence"}:
            raise ValueError("Incorrect query fields")
        q["text"] = " ".join(q["text"].split())
        normalized = re.sub(r"[^a-z0-9]", "", q["text"].lower())
        if q["type"] != "core" or q["type"] in types or normalized in texts:
            raise ValueError("Invalid/duplicate query type or text")
        if not 2 <= len(q["text"].split()) <= 18 or "?" in q["text"] or re.search(r"[\u4e00-\u9fff]", q["text"]):
            raise ValueError("Query must be a short English search phrase")
        if not 1 <= len(q["evidence"]) <= 5:
            raise ValueError("Missing/excessive evidence")
        for e in q["evidence"]:
            if set(e) != {"source", "text"} or e["source"] not in {"image", "caption", "report_heading", "table_labels"} or not e["text"].strip():
                raise ValueError("Invalid evidence")
        types.add(q["type"])
        texts.add(normalized)
    return obj


def flattened(result):
    if result["status"] != "ok":
        return []
    return [{"query_id": f"{result['visual_id']}:{q['type']}",
             "visual_id": result["visual_id"], "report_id": result["report_id"],
             "image_file": result["image_file"], "type": q["type"], "query": q["text"],
             "evidence": q["evidence"], "prompt_sha256": result["prompt_sha256"],
             "label_status": "synthetic_unverified"} for q in result["queries"]]


def fragment_reason(row):
    """A narrow, auditable guard for numeric New/Old/EPS-impact panels."""
    labels = [" ".join(x.lower().split()) for x in row.get('_target_table_labels', [])]
    if row.get('type') != 'table' or not {'new', 'old', 'eps impact'}.issubset(labels):
        return None
    headings = {'new', 'old', 'eps impact', 'eps', 'impact', 'chng', 'chg', 'change'}
    numeric_or_period = re.compile(r"(?:[1-4]q\d{2,4}[ae]?|(?:fy|cy)?\s*\d{2,4}[ae]?|[-+()\d.,%$\s]+(?:bp|bps|x)?)$")
    if all(label in headings or numeric_or_period.fullmatch(label) for label in labels):
        return 'Numeric New/Old/EPS Impact panel lacks metric row labels in the source table; query generation requires review.'
    return None


def analyze(client, row, args, prompt, prompt_hash):
    started = time.monotonic()
    base = {"visual_id": row["id"], "report_id": row["id"].split(":")[0],
            "image_file": row["image_file"], "visual_type": row["type"],
            "prompt_sha256": prompt_hash, "model": args.model,
            "created_at": utcnow(), "input_evidence": payload(row)}
    raw, usage, attempts = "", {}, 0
    try:
        precheck = fragment_reason(row)
        if precheck:
            return {**base, 'status': 'review', 'reason': precheck, 'queries': [],
                    'raw_response': '', 'usage': {}, 'api_attempts': 0,
                    'review_stage': 'source_fragment_guard', 'seconds': 0.0}
        path = Path(row["image_file"])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["_image_size"], row["_image_mtime_ns"]):
            raise ValueError("Source image changed since manifest creation")
        data = path.read_bytes()
        base["image_sha256"] = digest(data)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        url = f"data:{mime};base64," + base64.b64encode(data).decode()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": json.dumps(payload(row), ensure_ascii=False)}]}]
        for attempt in range(args.retries + 1):
            attempts += 1
            try:
                response = client.post(args.api_base.rstrip("/") + "/chat/completions", json={
                    "model": args.model, "messages": messages, "temperature": 0.1,
                    "max_tokens": 1000, "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "search_queries", "strict": True, "schema": SCHEMA}}})
                response.raise_for_status()
                response = response.json()
                choice = response["choices"][0]
                raw = choice["message"]["content"] or ""
                usage = response.get("usage", {})
                if choice.get("finish_reason") == "length":
                    raise ValueError("Truncated response")
                parsed = validate(json.loads(raw))
                return {**base, **parsed, "raw_response": raw, "usage": usage,
                        "api_attempts": attempts, "seconds": round(time.monotonic() - started, 3)}
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                if attempt == args.retries:
                    raise
                time.sleep(min(2 ** attempt, 8))
    except Exception as exc:
        return {**base, "status": "error", "reason": str(exc)[:1600], "queries": [],
                "raw_response": raw, "usage": usage, "api_attempts": attempts,
                "seconds": round(time.monotonic() - started, 3)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "active.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This output directory already has an active worker")
        run_locked(args)


def run_locked(args):
    global STOP
    if args.workers < 1 or args.retries < 0:
        raise ValueError("Invalid workers/retries")
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.manifest.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate visual IDs in manifest")
    prompt = args.prompt.read_text()
    prompt_hash = digest(prompt.encode())
    identity = {"manifest_sha256": digest(data), "prompt_sha256": prompt_hash,
                "model": args.model, "temperature": 0.1, "max_tokens": 1000,
                "format_version": 4}
    manifest_file = args.output / "run_manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text())["identity"] != identity:
            raise ValueError("Input/prompt/model changed; use a new output directory")
    else:
        atomic_json(manifest_file, {"identity": identity, "created_at": utcnow(),
                    "input_manifest": str(args.manifest.resolve()), "visuals": len(rows),
                    "api_base": args.api_base, "script": str(Path(__file__).resolve())})
        (args.output / "system_prompt.txt").write_text(prompt)
    results_file = args.output / "results.jsonl"
    latest = load_results(results_file)
    ids = {r['id'] for r in rows}
    if set(latest) - ids:
        raise ValueError("Unexpected IDs in results")
    counts = Counter(r['status'] for r in latest.values())
    query_count = sum(len(r['queries']) for r in latest.values() if r['status'] == 'ok')
    flat = args.output / "queries.jsonl"
    tmp = flat.with_suffix(".tmp")
    with tmp.open("w") as f:
        for result in latest.values():
            for q in flattened(result):
                f.write(dump_line(q))
    tmp.replace(flat)
    pending_rows = [r for r in rows if latest.get(r['id'], {}).get('status') not in {'ok', 'review', 'skip'}]
    started = time.monotonic()
    completed_now = 0
    stop_reason = None
    stop_file = args.output / "STOP"
    def signal_stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    def progress(status, inflight=0):
        done = counts['ok'] + counts['review'] + counts['skip']
        atomic_json(args.output / "progress.json", {
            "status": status, "updated_at": utcnow(), "pid": os.getpid(),
            "total_visuals": len(rows), "completed_visuals": done,
            "ok": counts['ok'], "review": counts['review'], "skip": counts['skip'],
            "errors": counts['error'], "remaining_visuals": len(rows) - done,
            "queries": query_count, "inflight": inflight,
            "session_completed": completed_now, "session_seconds": round(time.monotonic() - started, 1),
            "stop_reason": stop_reason, "workers": args.workers})
    progress("starting")
    iterator = iter(pending_rows)
    exhausted = False
    consecutive_errors = 0
    # Disable environment proxy variables for the explicitly supplied cluster endpoint.
    with httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0), trust_env=False,
                      limits=httpx.Limits(max_connections=args.workers + 1)) as client, \
         ThreadPoolExecutor(max_workers=args.workers) as pool, \
         results_file.open("a") as out, flat.open("a") as queries_out:
        active = {}
        while active or not exhausted:
            if STOP or stop_file.exists():
                stop_reason = stop_reason or "stop_requested"
            while len(active) < args.workers and not exhausted and not stop_reason:
                row = next(iterator, None)
                if row is None:
                    exhausted = True
                    break
                active[pool.submit(analyze, client, row, args, prompt, prompt_hash)] = row['id']
            if not active:
                break
            done, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
            for future in done:
                key = active.pop(future)
                result = future.result()
                out.write(dump_line(result))
                out.flush()
                os.fsync(out.fileno())
                for q in flattened(result):
                    queries_out.write(dump_line(q))
                queries_out.flush()
                if key in latest:
                    counts[latest[key]['status']] -= 1
                latest[key] = result
                counts[result['status']] += 1
                query_count += len(result['queries'])
                completed_now += 1
                consecutive_errors = consecutive_errors + 1 if result['status'] == 'error' else 0
                if consecutive_errors >= max(12, args.workers * 2):
                    stop_reason = "consecutive_errors"
                print(f"[{counts['ok'] + counts['review'] + counts['skip']}/{len(rows)}] {key} {result['status']} queries={len(result['queries'])} {result['seconds']}s" + (f" error={result['reason']}" if result['status'] == 'error' else ''), flush=True)
            progress("stopping" if stop_reason else "running", len(active))
    progress("stopped" if stop_reason else ("complete_with_errors" if counts['error'] else "complete"))
    print(json.dumps({"status": "stopped" if stop_reason else "finished", "counts": dict(counts), "queries": query_count}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sources", nargs="+", type=Path)
    prep.add_argument("--manifest", required=True, type=Path)
    runner = sub.add_parser("run")
    runner.add_argument("--manifest", required=True, type=Path)
    runner.add_argument("--output", required=True, type=Path)
    runner.add_argument("--prompt", type=Path, default=HERE / "search_query_prompt.txt")
    runner.add_argument("--api-base", default="http://interactive-qrfpqqwmogik:8088/v1")
    runner.add_argument("--model", default="qwen3.8-27b")
    runner.add_argument("--workers", type=int, default=8)
    runner.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args)


if __name__ == "__main__":
    main()
