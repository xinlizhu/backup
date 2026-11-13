from __future__ import annotations

import sys

from cir_qdiag.cli import main


if __name__ == "__main__":
    main(["screen", *sys.argv[1:]])
