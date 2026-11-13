"""Qwen3.5 paired runs with CLIP-L training batches: 32 x 1, image chunks of 256."""

from run_qwen35_e4 import main


if __name__ == "__main__":
    main(clip_batch=True)
