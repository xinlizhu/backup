import argparse
import _bootstrap
from single_target_audit import load_config,prepare,redact,run,summarize


def main():
    p=argparse.ArgumentParser(description='Simple single-target review of the frozen 450 QDiag edits')
    p.add_argument('--config',default='configs/accepted_450_five_models.yaml')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--summarize-only',action='store_true')
    p.add_argument('--models',nargs='+')
    p.add_argument('--max-cases',type=int,help='Limit NEW requests per model; subsequent runs resume')
    p.add_argument('--max-workers',type=int)
    p.add_argument('--output-dir')
    a=p.parse_args();c=load_config(a.config,a.output_dir)
    if a.max_workers is not None:
        if a.max_workers<1:p.error('--max-workers must be positive')
        c['runtime']['max_workers']=a.max_workers
    if a.max_cases is not None and a.max_cases<1:p.error('--max-cases must be positive')
    if a.prepare_only:
        meta=prepare(c);summarize(c)
        print({k:meta[k] for k in ['sampled_edits','per_cell','reference_count_by_edit','models','planned_calls']})
    elif a.summarize_only:prepare(c);summarize(c)
    else:run(c,a.models,a.max_cases)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[stopped] '+redact(exc));raise SystemExit(1)
