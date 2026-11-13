"""Collect only completed, coverage-checked external QDiag evaluations."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/external_qdiag_20261109')
args=p.parse_args()
rows=[]
completed=[]
input_hashes=set()
for directory in sorted(args.root.iterdir()):
    metadata_file=directory/'run_metadata.json'
    if not directory.is_dir() or not metadata_file.is_file():continue
    metadata=json.loads(metadata_file.read_text())
    if not metadata.get('complete'):continue
    identity=json.loads((directory/'identity.json').read_text())
    input_hashes.add(identity['input_sha256'])
    with (directory/'paper_metrics.csv').open() as f:
        result=list(csv.DictReader(f))
    assert len(result)==11,(directory,len(result))
    overall=next(r for r in result if r['subset']=='all')
    assert tuple(int(overall[k]) for k in ['groups','instructions','negatives'])==(499,1862,5576)
    with (directory/'per_negative_random10.csv').open() as f:
        negatives=list(csv.DictReader(f))
    assert len(negatives)==5576
    assert len({r['negative_id'] for r in negatives})==5576
    for row in result:rows.append({'model':directory.name,**row})
    completed.append(directory.name)
assert rows and len(input_hashes)==1
for filename,selection in [('comparison.csv',rows),('main_table.csv',[r for r in rows if r['subset']=='all'])]:
    target=args.root/filename
    temp=target.with_suffix('.tmp.csv')
    with temp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(selection)
    temp.replace(target)
status_file=args.root/'experiments.json'
status=json.loads(status_file.read_text()) if status_file.exists() else {}
status['completed']=completed
status['input_sha256']=next(iter(input_hashes))
status['validation_per_model']={name:json.loads((args.root/name/'smoke.json').read_text()) for name in completed}
status_file.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print('Completed:',', '.join(completed))
for row in rows:
    if row['subset']=='all':print(row)
