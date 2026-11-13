from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT.parents[1]/'src'))
sys.path.insert(0,str(ROOT.parents[2]/'cir_qdiag_openai_cirr/src'))
