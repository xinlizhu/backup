"""Use the released, provenance-checked S2 builder with an on-policy config."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT.parent / 'release_code/OPRD'
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(RELEASE))
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import load_image_map, load_cirr_positive_records
from cir_qdiag_train.on_policy_mining import _load_teacher_cache
from oprd.teacher_cache import build_teacher_cache
from oprd.data import load_cirr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--teacher-model', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    c = load_on_policy_config(args.config)
    fields = {'image_root': c.data.image_root,
              'train_images': c.data.train_image_split_path,
              'train_annotations': c.data.train_annotation_path,
              'teacher_cache': c.data.teacher_cache_dir}
    released = {'data': {k: os.path.relpath(v, RELEASE) for k, v in fields.items()}}
    ids, paths = load_image_map(c.data.train_image_split_path, c.data.image_root)
    records = load_cirr_positive_records(c.data.train_annotation_path, ids, paths)
    rid, rp, rows = load_cirr(released)
    assert rid == ids and rp == paths
    assert [(r.pair_id, r.reference_id, r.target_id, r.text) for r in records] == [
        (int(r['pairid']), r['reference'], r['target'], r['caption']) for r in rows]
    result = build_teacher_cache(
        released, teacher_model=os.path.relpath(args.teacher_model.resolve(), RELEASE),
        dtype='fp16', image_batch_size=args.batch_size, query_batch_size=args.batch_size,
        num_workers=8, prefetch_factor=2,
    )
    _load_teacher_cache(config=c, records=records, image_ids=ids)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
