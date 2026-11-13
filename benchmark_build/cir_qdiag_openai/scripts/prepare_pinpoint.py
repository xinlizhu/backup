from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from cir_qdiag.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
