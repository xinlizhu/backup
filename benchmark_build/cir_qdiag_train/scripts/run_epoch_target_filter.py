"""Four epoch-Top32 controls; reuse the completed Reverse/Top10/inject baseline."""

import argparse
import csv
import fcntl
import importlib.util
from pathlib import Path

from cir_qdiag_train.on_policy_config import load_on_policy_config

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "outputs/epoch_target_filter_e4"
RUNS = ("kl_all_inject", "kl_top10_natural", "kl_all_natural", "cmse_top10_inject")
spec = importlib.util.spec_from_file_location(
    "epoch_control_runner", PROJECT / "scripts/run_reverse_rank_r128.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.ROOT = ROOT
runner.LABEL_PREFIX = "CLIP-L-EPOCH"
runner.CONFIG_PATHS = {name: PROJECT / f"configs/epoch_control_{name}.yaml" for name in RUNS}


def summarize():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = PROJECT / "outputs/opd_k_tau_kl_ablation"
        baseline = (
            next(
                (
                    r
                    for r in csv.DictReader((old / "all_metrics.csv").open())
                    if r["Model"] == "K32 Reverse T=0.05"
                ),
                {},
            )
            if (old / "all_metrics.csv").exists()
            else {}
        )
        rows = []
        for name in ("existing_kl_top10_inject", *RUNS):
            existing = name.startswith("existing_")
            run = old / "k32_reverse_t005" if existing else ROOT / name
            label = f"{runner.LABEL_PREFIX}-{name}"
            pp = runner.read_csv(run / "pinpoint_official" / label / "official_metrics.csv")
            qd = runner.read_csv(run / "qdiag" / label / "summary.csv", "strict")
            for epoch in range(1, 5):
                cirr = runner.read_json(run / f"val_epoch_{epoch:03d}.json")
                fiq = runner.read_json(
                    run / "fashioniq" / f"epoch_{epoch:03d}" / "metrics.json"
                ).get("macro_average", {})
                final = epoch == 4
                rows.append(
                    {
                        "Run": name,
                        "Epoch": epoch,
                        "Checkpoint": (run / f"checkpoint_epoch_{epoch:03d}.pt").is_file(),
                        "CIRR R@1": cirr.get("recall_at_1"),
                        "CIRR R@10": cirr.get("recall_at_10"),
                        "FashionIQ R@10": baseline.get("FashionIQ R@10")
                        if existing and final
                        else fiq.get("recall_at_10"),
                        "PinPoint mAP@10": (
                            baseline.get("PinPoint mAP@10") if existing else pp.get("mAP@10")
                        )
                        if final
                        else None,
                        "QDiag Delta": (
                            baseline.get("QDiag Delta") if existing else qd.get("mean_target_delta")
                        )
                        if final
                        else None,
                        "QDiag Stable Win": (
                            baseline.get("QDiag Stable Win")
                            if existing
                            else qd.get("target_stable_win_rate")
                        )
                        if final
                        else None,
                    }
                )
        fields = list(rows[0])
        tmp = ROOT / "all_metrics.csv.tmp"
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(ROOT / "all_metrics.csv")
        lines = [
            "# Epoch Top-32: target insertion / eligibility / Centered MSE",
            "",
            "CLIP-L ← S2, full FT, seed 42, four epochs. Reverse KL T=0.05; CMSE uses raw centered scores (no temperature scaling).",
            "Natural candidates never receive an injected target. Eligibility alone uses the virtual target + Top-31 non-target set.",
            "Existing baseline is read only; no baseline retraining. QDiag wins are fractions, not percentages. Missing results are —.",
            "",
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join(["---"] * len(fields)) + " |",
        ]
        for row in rows:
            lines.append(
                "| " + " | ".join("—" if row[f] is None else str(row[f]) for f in fields) + " |"
            )
        tmp = ROOT / "RESULTS.md.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(ROOT / "RESULTS.md")
    print(ROOT / "RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-indices", type=int, nargs="+", default=list(range(4)))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-extended-eval", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize()
        return
    if len(set(args.run_indices)) != len(args.run_indices) or any(
        i not in range(4) for i in args.run_indices
    ):
        parser.error("run indices must be unique integers from 0 to 3")
    for index in args.run_indices:
        name = RUNS[index]
        config = load_on_policy_config(runner.CONFIG_PATHS[name])
        print(
            f"[{index}] {name}: loss={config.objective.loss_type}, filter={config.objective.teacher_target_max_rank}, inject={config.mining.force_target_insertion}, output={config.experiment.output_dir}",
            flush=True,
        )
        if not args.dry_run and not runner.read_json(
            config.data.teacher_cache_dir / "metadata.json"
        ).get("complete"):
            raise RuntimeError("missing/incomplete shared S2 Teacher cache")
    if args.dry_run:
        return
    for index in args.run_indices:
        runner.run_one(RUNS[index], extended=not args.skip_extended_eval)
    if not args.skip_summary:
        summarize()


if __name__ == "__main__":
    main()
