"""Resume the frozen four-model experiment, serializing GPU-heavy models."""
import argparse,fcntl,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import _bootstrap
from alignment_data import ROOT,load_config,prepare,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full_four_models.yaml');p.add_argument('--models',nargs='+');a=p.parse_args()
 os.chdir(ROOT);c=load_config(a.config);prepare(c);out=Path(c['runtime']['output_dir'])
 models=a.models or list(c['models']);unknown=set(models)-set(c['models'])
 if unknown:raise ValueError(f'Unknown models: {unknown}')
 lock=(out/'pipeline.lock').open('a+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status={'pid':os.getpid(),'models':models,'started_at_utc':datetime.now(timezone.utc).isoformat(),'complete':False}
 for model in models:
  path=out/model;path.mkdir(exist_ok=True)
  for stage,script,marker in [('encoding','encode_model.py','encoding_complete.json'),('ranking','score_ranks.py','ranks/complete.json'),('verification','verify_saved_ranks.py','ranks/verification.json'),('analysis','analyze_model.py','analysis_complete.json')]:
   if (path/marker).exists():print('[complete]',model,stage,flush=True);continue
   status.update(current_model=model,current_stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat());write_json(out/'pipeline_status.json',status)
   print('[start]',model,stage,status['updated_at_utc'],flush=True)
   with (path/(stage+'.pipeline.log')).open('a',buffering=1) as log:
    result=subprocess.run([sys.executable,'scripts/'+script,'--config',a.config,'--model',model],stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    status.update(error={'stage':stage,'model':model,'exit_code':result.returncode});write_json(out/'pipeline_status.json',status);raise SystemExit(result.returncode)
  subprocess.run([sys.executable,'scripts/collect_results.py'],check=True)
 status.update(complete=True,current_model=None,current_stage=None,finished_at_utc=datetime.now(timezone.utc).isoformat());write_json(out/'pipeline_status.json',status)
 print('[pipeline complete]',flush=True)
if __name__=='__main__':main()
