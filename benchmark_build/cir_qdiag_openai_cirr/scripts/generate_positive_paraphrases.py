from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from cir_qdiag_openai_cirr.cli import main


if __name__ == "__main__":
    main(["paraphrase", *sys.argv[1:]])
