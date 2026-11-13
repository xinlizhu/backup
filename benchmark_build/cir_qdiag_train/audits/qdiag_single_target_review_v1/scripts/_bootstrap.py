from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parents[2]/'cir_qdiag_openai'
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(ROOT/'src'))
