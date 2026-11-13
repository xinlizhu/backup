import argparse
import _bootstrap
from review_audit import load_config,prepare,probe,probe_from_saved,redact,run,summarize


def main():
    p=argparse.ArgumentParser(description='Stratified cross-model review of accepted QDiag V7 edits')
    p.add_argument('--config',default='configs/accepted_450_seed42.yaml')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--probe-only',action='store_true')
    mode.add_argument('--probe-cache-only',action='store_true',help='Revalidate saved real probe responses without any API call')
    mode.add_argument('--summarize-only',action='store_true')
    p.add_argument('--models',nargs='+')
    p.add_argument('--max-workers',type=int)
    p.add_argument('--max-groups',type=int,help='Limit new group requests per selected model for a smoke run')
    p.add_argument('--output-dir')
    a=p.parse_args();c=load_config(a.config,a.output_dir)
    if a.max_workers is not None:
        if a.max_workers<1:p.error('--max-workers must be positive')
        c['runtime']['max_workers']=a.max_workers
    if a.max_groups is not None and a.max_groups<1:p.error('--max-groups must be positive')
    if a.prepare_only:
        meta=prepare(c);summarize(c)
        print({k:meta[k] for k in ['sampled_edits','sampled_groups','per_cell','planned_total_group_calls']})
    elif a.probe_only:
        results=probe(c,a.models)
        if not all(r['ready_for_audit'] for r in results):raise SystemExit(2)
    elif a.probe_cache_only:
        results=probe_from_saved(c,a.models)
        if not all(r['ready_for_audit'] for r in results):raise SystemExit(2)
    elif a.summarize_only:prepare(c);summarize(c)
    else:run(c,a.models,a.max_groups)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[stopped] '+redact(exc));raise SystemExit(1)
