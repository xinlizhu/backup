"""Ordered CPU-only query preprocessing; workers never receive the GPU model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader


class QueryCollator:
    def __init__(self, processor, config):
        self.processor = processor
        self.config = config

    def __call__(self, rows):
        from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent, _composed_conversation

        if self.config.llava:
            from cir_qdiag_train.llava_student import preprocess_llava
            return [r[0] for r in rows], preprocess_llava(
                self.processor, self.config, [[r[1]] for r in rows], [r[2] for r in rows], query=True
            )

        conversations = [
            _composed_conversation(
                [path],
                text,
                instruction=QwenQueryLoRAStudent.QUERY_INSTRUCTION,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )
            for _, path, text in rows
        ]
        if self.config.qwen35:
            from cir_qdiag_train.qwen35_student import preprocess_qwen35
            return [row[0] for row in rows], preprocess_qwen35(
                self.processor, conversations, self.config.max_length
            )
        if self.config.variant == "qwen3_vl_instruct_query_lora":
            from cir_qdiag_train.qwen_instruct_retrieval import preprocess_query_only
            return [row[0] for row in rows], preprocess_query_only(
                self.processor, conversations, self.config.max_length
            )
        if self.config.shared_lora:
            from cir_qdiag_train.qwen_retrieval import preprocess
            return [row[0] for row in rows], preprocess(
                self.processor, conversations, self.config.max_length
            )
        root = str(Path(self.config.pretrained_model_name_or_path).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

        # Invoke the exact existing processor implementation without constructing an embedder/model.
        state = SimpleNamespace(processor=self.processor, max_length=self.config.max_length)
        prepared = Qwen3VLEmbedder._preprocess_inputs(state, conversations)
        return [index for index, _, _ in rows], dict(prepared)


def query_batches(*, student, records, order, batch_size, workers, prefetch_factor, seed):
    if workers == 0:
        for begin in range(0, len(order), batch_size):
            indices = order[begin : begin + batch_size]
            yield (
                indices,
                student.prepare_query_batch(
                    [records[i].reference_path for i in indices],
                    [records[i].text for i in indices],
                ),
            )
        return
    rows = [(i, records[i].reference_path, records[i].text) for i in order]
    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context="spawn",
        collate_fn=QueryCollator(student.embedder.processor, student.config),
        pin_memory=student.device.type == "cuda",
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )
    yield from loader


class ImageCollator:
    def __init__(self, processor, config):
        self.processor, self.config = processor, config

    def __call__(self, rows):
        from cir_qdiag_train.qwen_retrieval import preprocess
        from cir_qdiag_train.qwen_student import QwenQueryLoRAStudent, _composed_conversation

        if self.config.llava:
            from cir_qdiag_train.llava_student import preprocess_llava
            return [r[0] for r in rows], preprocess_llava(
                self.processor, self.config, [[r[1]] for r in rows], query=False
            )

        if self.config.qwen35:
            from cir_qdiag_train.qwen35_student import preprocess_qwen35 as preprocess

        conversations = [
            _composed_conversation(
                [path], "", instruction=QwenQueryLoRAStudent.TARGET_INSTRUCTION,
                min_pixels=self.config.min_pixels, max_pixels=self.config.max_pixels,
            ) for _, path in rows
        ]
        return [i for i, _ in rows], preprocess(
            self.processor, conversations, self.config.max_length
        )


class SharedTrainingCollator:
    def __init__(self, processor, config, image_batch_size):
        self.queries = QueryCollator(processor, config)
        self.images = ImageCollator(processor, config)
        self.image_batch_size = image_batch_size

    def __call__(self, rows):
        indices, queries = self.queries([row[:3] for row in rows])
        # Deduplicate candidates across queries; gather-backward sums their gradients.
        paths, positions = [], {}
        inverse = []
        for row in rows:
            local = []
            for path in row[3]:
                if path not in positions:
                    positions[path] = len(paths)
                    paths.append(path)
                local.append(positions[path])
            inverse.append(local)
        images = [
            self.images(list(enumerate(paths))[start:start + self.image_batch_size])[1]
            for start in range(0, len(paths), self.image_batch_size)
        ]
        return indices, {"queries": queries, "images": images,
                         "candidate_positions": torch.tensor(inverse, dtype=torch.long)}


def _cpu_batches(rows, collator, *, student, batch_size, workers, prefetch_factor, seed):
    if workers == 0:
        for start in range(0, len(rows), batch_size):
            yield collator(rows[start:start + batch_size])
        return
    yield from DataLoader(
        rows, batch_size=batch_size, shuffle=False, num_workers=workers,
        prefetch_factor=prefetch_factor, multiprocessing_context="spawn",
        collate_fn=collator, pin_memory=student.device.type == "cuda",
        persistent_workers=False, generator=torch.Generator().manual_seed(seed),
    )


def image_batches(*, student, image_paths, batch_size, workers, prefetch_factor, seed):
    yield from _cpu_batches(
        list(enumerate(image_paths)), ImageCollator(student.embedder.processor, student.config),
        student=student, batch_size=batch_size, workers=workers,
        prefetch_factor=prefetch_factor, seed=seed,
    )


def shared_training_batches(*, student, records, order, candidates, image_paths,
                            batch_size, image_batch_size, workers, prefetch_factor, seed):
    rows = [(i, records[i].reference_path, records[i].text,
             [image_paths[int(j)] for j in candidates[i]]) for i in order]
    yield from _cpu_batches(
        rows, SharedTrainingCollator(student.embedder.processor, student.config, image_batch_size),
        student=student, batch_size=batch_size, workers=workers,
        prefetch_factor=prefetch_factor, seed=seed,
    )
