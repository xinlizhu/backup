"""Run Instruct CMSE-Qwen8 independently of the main suite's global lock."""
import argparse
import run_qwen_instruct_shared_e4 as instruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = instruct.suite
    name = "qwen2_instruct_shared_cmse_qwen8"
    suite.ROOT = suite.runner.ROOT = instruct.ROOT
    suite.runner.LABEL_PREFIX = "INSTRUCT-SHARED"
    suite.runner.CONFIG_PATHS[name] = instruct.PROJECT / f"configs/{name}.yaml"
    config = suite.load(name)
    instruct.validate(config)
    assert config.optimizer.world_size == 2
    assert config.optimizer.gradient_accumulation_steps == 1
    print(f"{name}: 2 GPUs x {config.optimizer.batch_size_per_gpu}; "
          f"epochs={config.optimizer.epochs}; output={config.experiment.output_dir}", flush=True)
    if args.dry_run:
        return
    suite.preflight(config)
    # run_qwen holds the shared per-experiment lock through training and evaluation.
    # The original suite waits here if it reaches this experiment concurrently.
    suite.run_qwen(name, extended=True)


if __name__ == "__main__":
    main()
