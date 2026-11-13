import argparse
import _bootstrap
from rank_preference.common import load_config
from rank_preference.prepare import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    a=p.parse_args();prepare(load_config(a.config))
