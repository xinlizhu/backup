from __future__ import annotations

import argparse
from pathlib import Path

from cir_qdiag_train.oap_config import load_oap_config
from cir_qdiag_train.oap_trainer import train_oap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen-global Object/Attribute/Predicate residual adapters "
            "or semantic attention tokens."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    checkpoint = train_oap(
        load_oap_config(args.config), resume_from=args.resume_from
    )
    print(checkpoint)


if __name__ == "__main__":
    main()
