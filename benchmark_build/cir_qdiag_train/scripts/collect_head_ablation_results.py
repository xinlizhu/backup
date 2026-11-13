"""Collect trained-head QDiag subsets and full retrieval metrics without rounding."""
import argparse
import json
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'outputs/head_ablation_oprd_20261109'
HEADS = ['mmret_add', 'clip4cir_combiner', 'magiclens']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    rows, subsets, complete = [], [], []
    for head in HEADS:
        folder = ROOT / head
        for stage in ['initial', 'final']:
            metrics = folder / f'qdiag_{stage}/paper_metrics.csv'
            if metrics.exists():
                frame = pd.read_csv(metrics)
                frame.insert(0, 'stage', stage); frame.insert(0, 'head', head)
                subsets.append(frame)
                row = frame[frame.subset == 'all'].iloc[0].to_dict()
                if stage == 'final':
                    ret = folder / 'retrieval'
                    files = [ret / f for f in ['cirr_val/val_metrics.json', 'fashioniq_metrics.json', 'pinpoint/metrics.json']]
                    if all(f.exists() for f in files):
                        cirr, fiq, pp = [json.loads(f.read_text()) for f in files]
                        assert (cirr['num_queries'], fiq['num_queries'], int(pp['total_queries'])) == (4181, 6016, 7635)
                        row.update({f'CIRR_{k}': v for k, v in cirr.items() if 'recall_at_' in k})
                        row.update({f'FIQ_macro_{k}': v for k, v in fiq['macro_average'].items()})
                        row.update({f'PinPoint_{k}': v for k, v in pp.items() if k not in ('model', 'total_queries')})
                        complete.append(head)
                rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(ROOT / 'comparison.csv', index=False)
        pd.concat(subsets).to_csv(ROOT / 'qdiag_all_subsets.csv', index=False)
    if args.require_complete:
        assert complete == HEADS, complete
    print(json.dumps({'completed': complete, 'comparison': str(ROOT / 'comparison.csv')}), flush=True)


if __name__ == '__main__':
    main()
