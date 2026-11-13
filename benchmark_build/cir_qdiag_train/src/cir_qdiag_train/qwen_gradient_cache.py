"""Exact first-order embedding gradient replay with dropout RNG preservation.

The loss is evaluated once on detached embedding leaves. Its (possibly AMP-scaled)
gradients are replayed through one encoder chunk at a time. No optimizer update
may occur between the cache pass and replay. Inputs stay on CPU between passes.
"""

from contextlib import contextmanager

import torch


@contextmanager
def replay_rng(cpu_state, cuda_state, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        yield


class CachedEmbedding:
    def __init__(self, encode, prepared, device):
        self.encode, self.prepared, self.device = encode, prepared, device
        self.cpu_state = torch.get_rng_state()
        self.cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        with torch.no_grad():
            self.embedding = encode(prepared, requires_grad=True).detach()
        self.embedding.requires_grad_(True)

    def backward(self):
        if self.embedding.grad is None:
            raise RuntimeError("Missing cached embedding gradient")
        with replay_rng(self.cpu_state, self.cuda_state, self.device):
            live = self.encode(self.prepared, requires_grad=True)
            torch.autograd.backward(live, self.embedding.grad)


def shared_student_scores(student, prepared, *, gradient_cache):
    if not gradient_cache:
        from cir_qdiag_train.qwen_live_training import compute_live_qwen_scores
        return compute_live_qwen_scores(student, prepared), lambda: None
    chunks = []

    def encode(fn, inputs):
        cached = CachedEmbedding(fn, inputs, student.device)
        chunks.append(cached)
        return cached.embedding

    queries = encode(student.encode_prepared_queries, prepared["queries"])
    images = torch.cat([
        encode(student.encode_prepared_images, chunk) for chunk in prepared["images"]
    ])
    positions = prepared["candidate_positions"].to(student.device)
    scores = torch.einsum("bd,bkd->bk", queries, images[positions])

    def replay():
        for chunk in chunks:
            chunk.backward()

    return scores, replay
