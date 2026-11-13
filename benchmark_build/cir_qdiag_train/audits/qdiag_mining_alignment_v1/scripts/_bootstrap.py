from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
# Reuse the already-tested HTTP client, retry handling and response cache.
sys.path.insert(0, str(ROOT.parents[2] / 'cir_qdiag_openai_cirr/src'))
