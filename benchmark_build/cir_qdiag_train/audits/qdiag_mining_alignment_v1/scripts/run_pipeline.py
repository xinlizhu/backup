"""Prepare recorded GC candidates, propose/review witnesses, and export audit results."""
import argparse
import logging
from pathlib import Path
import sys
import _bootstrap
from mining_audit.common import load_config,redact
from mining_audit.workflow import run


class Redact(logging.Filter):
    def filter(self,record):
        record.msg=redact(record.getMessage());record.args=()
        return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true',help='Validate caches, prepare images/IDs and HTML, without API calls')
    p.add_argument('--summarize-only',action='store_true',help='Rebuild tables and HTML from saved results')
    p.add_argument('--resume',action='store_true',help='Accepted for compatibility; safe resume is always enabled')
    p.add_argument('--max-pairs',type=int,help='Process at most this many pending pairs; keep the full sampling denominator')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--mock',action='store_true',help='Offline pipeline test; requires a separate output directory; never scientific data')
    a=p.parse_args()
    if a.max_pairs is not None and a.max_pairs<1:p.error('--max-pairs must be positive')
    if a.mock and a.output_dir is None:p.error('--mock requires --output-dir pointing to a separate test folder')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    for h in logging.getLogger().handlers:h.addFilter(Redact())
    c=load_config(a.config,a.output_dir)
    run(c,prepare_only=a.prepare_only,summarize_only=a.summarize_only,max_pairs=a.max_pairs,mock=a.mock)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[stopped] '+redact(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
