"""Shared all-linear LoRA with CLIP-style direct backward (no gradient cache)."""

from run_qwen_shared_lora_e4 import main


if __name__ == "__main__":
    main(direct=True)
