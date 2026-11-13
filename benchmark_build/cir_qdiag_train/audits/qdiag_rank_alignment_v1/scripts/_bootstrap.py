import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[3]
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('HF_MODULES_CACHE',str(WORKSPACE/'models/hf_modules'))
sys.path.insert(0,str(WORKSPACE/'models/oprd_teacher_compat'))
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT.parents[2]/'release_code/OPRD'))
