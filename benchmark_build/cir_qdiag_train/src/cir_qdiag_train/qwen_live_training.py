"""CLIP-style live scoring: differentiable encoders, one loss, one backward.

Image chunks retain their autograd graphs until the shared loss is backpropagated.
There is no detached embedding cache or encoder-gradient replay in this path.
"""

import torch


def compute_live_qwen_scores(student, prepared):
    queries = student.encode_prepared_queries(prepared["queries"], requires_grad=True)
    images = torch.cat([
        student.encode_prepared_images(chunk, requires_grad=True)
        for chunk in prepared["images"]
    ])
    positions = prepared["candidate_positions"].to(student.device, non_blocking=True)
    return torch.einsum("bd,bkd->bk", queries, images[positions])
