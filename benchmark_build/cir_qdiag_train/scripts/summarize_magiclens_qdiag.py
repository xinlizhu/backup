"""Reuse the validated QDiag statistics on native MagicLens score exports."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from evaluate_external_qdiag import load_v7_dataset, DEFAULT_INPUT, write_metrics, write_json_atomic, sha256_file

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
root=args.output
identity=json.loads((root/'identity.json').read_text())
assert identity['input_sha256']==sha256_file(DEFAULT_INPUT)
groups,queries,coverage=load_v7_dataset(DEFAULT_INPUT,limit_groups=None)
keys=json.loads((root/'cache/query_keys.json').read_text())
assert keys==[q.query_key for q in queries]
targets=json.loads((root/'cache/target_ids.json').read_text())
matrix=np.load(root/'cache/query_target_scores.npy',mmap_mode='r')
assert matrix.shape==(len(queries),len(targets))
write_metrics(groups,queries,matrix,targets,root,SimpleNamespace(bootstrap_replicates=2000,seed=42))
metadata=json.loads((root/'run_metadata.json').read_text())
metadata.update(complete=True,bootstrap_replicates=2000,seed=42,random_samples=10,
                statistics_environment='/mnt/data/datasets/search_dataset/zxl/multi_modal_py3.12')
write_json_atomic(root/'run_metadata.json',metadata)
print('[done metrics]',root,flush=True)
