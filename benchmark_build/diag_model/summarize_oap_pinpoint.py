from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASELINE_BY_MODEL = {
    "CLIP-L-OAP-epoch2-beta0.1": "CLIP-L-full-baseline",
    "CLIP-L-OAP-epoch5-beta0.1": "CLIP-L-full-baseline",
    "BGE-VL-large-OAP-epoch5-beta0.025": "BGE-VL-large-full-hard-w0.1",
    "CLIP-L-OAP-Attention-epoch2-beta0.1": "CLIP-L-full-baseline",
    "CLIP-L-OAP-Attention-epoch5-beta0.1": "CLIP-L-full-baseline",
    "BGE-VL-large-OAP-Attention-epoch2-beta0.1": (
        "BGE-VL-large-full-hard-w0.1"
    ),
    "BGE-VL-large-OAP-Attention-epoch5-beta0.1": (
        "BGE-VL-large-full-hard-w0.1"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oap-summary", type=Path, required=True)
    parser.add_argument("--clip-baseline", type=Path, required=True)
    parser.add_argument("--bge-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    oap = pd.read_csv(args.oap_summary)
    baselines = pd.concat(
        [pd.read_csv(args.clip_baseline), pd.read_csv(args.bge_baseline)],
        ignore_index=True,
    ).set_index("model")
    rows = []
    for _, row in oap.iterrows():
        model_name = str(row["model"])
        baseline_name = BASELINE_BY_MODEL[model_name]
        baseline = baselines.loc[baseline_name]
        rows.append(
            {
                "model": model_name,
                "baseline": baseline_name,
                "mAP@10": row["mAP@10"],
                "baseline_mAP@10": baseline["mAP@10"],
                "delta_mAP@10": row["mAP@10"] - baseline["mAP@10"],
                "NegRecall@10": row["NegRecall@10"],
                "baseline_NegRecall@10": baseline["NegRecall@10"],
                "delta_NegRecall@10": (
                    row["NegRecall@10"] - baseline["NegRecall@10"]
                ),
                "total_queries": row["total_queries"],
            }
        )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")
    print(output)


if __name__ == "__main__":
    main()
