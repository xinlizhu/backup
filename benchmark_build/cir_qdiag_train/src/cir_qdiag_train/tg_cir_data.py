"""CPU-only, spawn-safe TG-CIR preprocessing and CIRR training loader."""

from __future__ import annotations

import html
from pathlib import Path

import torch
from ftfy import fix_text
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTokenizer

from cir_qdiag_train.inference_data import _initialize_inference_worker


class TGCIRProcessor:
    """OpenAI CLIP eval transform and fixed 77-token, zero-padded tokenization.

    The same deterministic transform is used for train/val in the official code.
    Long evaluation captions are explicitly truncated at 77, preserving EOS.
    """

    def __init__(self, model_path):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, local_files_only=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # OpenAI CLIP basic_clean also unescapes HTML twice; HF alone does not.
        texts = [html.unescape(html.unescape(fix_text(str(text)))).strip() for text in texts]
        tokens = self.tokenizer(
            list(texts),
            padding=False,
            truncation=True,
            max_length=77,
            return_attention_mask=False,
        )["input_ids"]
        result = torch.zeros((len(tokens), 77), dtype=torch.long)
        for index, row in enumerate(tokens):
            result[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        return result

    def __call__(self, *, images=None, text=None, **kwargs):
        result = {}
        if images is not None:
            result["pixel_values"] = torch.stack(
                [self.transform(image.convert("RGB")) for image in images]
            )
        if text is not None:
            result["input_ids"] = self.tokenize(text)
        return result


class TripletDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return str(row.reference_path), row.text, str(row.target_path)


class TripletCollator:
    def __init__(self, processor):
        self.processor = processor

    def _images(self, paths):
        tensors = []
        for path in paths:
            with Image.open(Path(path)) as image:
                tensors.append(self.processor.transform(image.convert("RGB")))
        return torch.stack(tensors)

    def __call__(self, rows):
        return {
            "reference_pixels": self._images([row[0] for row in rows]),
            "input_ids": self.processor.tokenize([row[1] for row in rows]),
            "target_pixels": self._images([row[2] for row in rows]),
        }


def epoch_indices(count, batch_size, seed, epoch, start_step=0):
    if batch_size < 2 or count < batch_size:
        raise ValueError("TG-CIR requires at least one full batch of >= 2 examples")
    steps = count // batch_size
    if not 0 <= start_step <= steps:
        raise ValueError("Invalid resume batch offset")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
    # Explicit drop_last, identical negative-set size on every optimizer step.
    return order[start_step * batch_size : steps * batch_size].tolist()


def build_train_loader(records, processor, config, epoch, start_step=0, pin_memory=False):
    train = config["training"]
    indices = epoch_indices(
        len(records),
        train["batch_size"],
        config["seed"],
        epoch,
        start_step,
    )
    kwargs = {
        "dataset": Subset(TripletDataset(records), indices),
        "batch_size": train["batch_size"],
        "shuffle": False,
        "drop_last": True,
        "num_workers": train["num_workers"],
        "pin_memory": pin_memory,
        "collate_fn": TripletCollator(processor),
        "generator": torch.Generator().manual_seed(config["seed"] + epoch),
    }
    if train["num_workers"]:
        kwargs.update(
            multiprocessing_context="spawn",
            prefetch_factor=train["prefetch_factor"],
            worker_init_fn=_initialize_inference_worker,
        )
    return DataLoader(**kwargs)
