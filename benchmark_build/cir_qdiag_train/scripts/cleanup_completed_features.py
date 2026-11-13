"""Conservative cleanup: only completed evaluation feature arrays, never teachers."""
import argparse
import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'outputs'
NAMES = {'gallery_features.npy', 'query_features.npy', 'corpus_embeddings.npy', 'query_embeddings.npy'}


def candidates():
    for p in ROOT.rglob('*.npy'):
        if p.name not in NAMES or p.is_symlink() or p.resolve() != p:
            continue
        rel = p.relative_to(ROOT)
        lower = str(rel).lower()
        # Shared training caches, mining, and all teacher/original-teacher outputs stay intact.
        if any(x in lower for x in ('on_policy/', 'bge', 'teacher', 'embedding-8b', 'embedding_8b', 'fixed_gallery')):
            continue
        if 'cache' in rel.parts:
            cache = next(a for a in p.parents if a.name == 'cache')
            base = cache.parent
            if not ((base / 'official_metrics.csv').is_file() or (base / 'summary.csv').is_file()):
                continue
        else:
            base = p.parent
            if not ((base / 'metrics.json').is_file() or (base / 'val_metrics.json').is_file()):
                continue
        # Pending Instruct runs may be active on another server.
        if 'qwen_instruct_shared_e4' in rel.parts and any(x in rel.parts for x in (
                'qwen2_instruct_shared_cmse_s2', 'qwen2_instruct_shared_cmse_qwen8')):
            continue
        yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true')
    args = parser.parse_args()
    files = list(candidates())
    print(json.dumps({'files': len(files), 'GiB': sum(p.stat().st_size for p in files) / 2**30}))
    locks = []
    blocked = set()
    try:
        for p in files:
            for a in p.parents:
                if a == ROOT:
                    break
                lock = a / '.run.lock'
                if lock.is_file() and lock not in blocked and not any(x[0] == lock for x in locks):
                    handle = lock.open('r')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locks.append((lock, handle))
                    except BlockingIOError:
                        blocked.add(lock)
                        handle.close()
        count = size = 0
        for p in files:
            if any(lock.parent in p.parents for lock in blocked):
                continue
            if args.delete:
                size += p.stat().st_size
                p.unlink()
                count += 1
            print(('DELETED ' if args.delete else 'PLAN ') + str(p))
        print(json.dumps({'deleted_files': count, 'deleted_GiB': size / 2**30, 'locked_runs_skipped': len(blocked)}))
    finally:
        for _, handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
