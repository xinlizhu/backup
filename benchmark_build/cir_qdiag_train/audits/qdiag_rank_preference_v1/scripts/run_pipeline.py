import argparse
import logging
from pathlib import Path
import sys
import _bootstrap
from rank_preference.common import load_config,redact
from rank_preference.workflow import run


class Redact(logging.Filter):
    def filter(self,r):r.msg=redact(r.getMessage());r.args=();return True


def main():
    p=argparse.ArgumentParser(description='GC epochs 1/4, ranks 1/2/3, one random A/B order; identical images skip GPT')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--resume',action='store_true',help='Safe resume is always enabled')
    p.add_argument('--max-cases',type=int)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--mock',action='store_true')
    a=p.parse_args()
    if a.max_cases is not None and a.max_cases<1:p.error('--max-cases must be positive')
    if a.mock and a.output_dir is None:p.error('--mock requires a separate --output-dir')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    for h in logging.getLogger().handlers:h.addFilter(Redact())
    run(load_config(a.config,a.output_dir),a.prepare_only,a.summarize_only,a.max_cases,a.mock)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[stopped] '+redact(exc),file=sys.stderr,flush=True);raise SystemExit(1)
