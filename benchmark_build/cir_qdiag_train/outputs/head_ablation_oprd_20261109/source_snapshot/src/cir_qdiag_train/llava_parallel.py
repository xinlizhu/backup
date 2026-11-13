"""Synchronous data parallelism for multi-forward, shared-LoRA retrieval.

Each rank retains the native PEFT model. Reduce adapter gradients once, after
all query/candidate backwards (including optional replay), before clipping and
stepping. This avoids DDP reducer hooks around repeated encoder forwards and
natively unused final vision-block adapters. Only LoRA gradients communicate.
"""
import os
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def setup_parallel(expected_world_size, *, allow_cpu=False):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != expected_world_size or world < 2:
        raise ValueError(f"Requires {expected_world_size} ranks via torchrun; got WORLD_SIZE={world}")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise ValueError(f"LOCAL_RANK={local_rank} has no visible GPU")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("LLaVA multi-GPU training requires visible CUDA GPUs")
    # Other ranks legitimately wait while rank 0 refreshes the full gallery.
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                            timeout=timedelta(hours=12))
    return rank, world, device


def on_rank_zero(action):
    """Broadcast completion/errors; never race cache/checkpoint writers."""
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = (True, action())
        except Exception:
            result[0] = (False, traceback.format_exc())
    dist.broadcast_object_list(result, src=0)
    success, payload = result[0]
    if not success:
        raise RuntimeError(f"Rank-zero operation failed: {payload}")
    return payload


def shard_order(order, batch_size, rank, world_size):
    """Shard each global batch without padding/duplicating or dropping queries."""
    if batch_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid data-parallel shard")
    return [i for start in range(0, len(order), batch_size * world_size)
            for i in order[start + rank * batch_size:min(start + (rank + 1) * batch_size, len(order))]]


@torch.no_grad()
def broadcast_parameters(parameters):
    for parameter in parameters:
        dist.broadcast(parameter, src=0)


@torch.no_grad()
def sum_gradients(parameters, bucket_bytes=32 * 1024 * 1024):
    """SUM pre-weighted gradients; preserve globally unused parameters as None.

Each local mean loss is weighted by local_queries / actual_global_queries.
Do not divide by world size a second time. Empty final shards still participate.
"""
    parameters = list(parameters)
    present = torch.tensor([p.grad is not None for p in parameters],
                           dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    active = [p for p, used in zip(parameters, present.tolist(), strict=True) if used]
    bucket, nbytes = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in bucket:
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()

    for p in active:
        size = p.numel() * p.element_size()
        if bucket and (nbytes + size > bucket_bytes or bucket[0].dtype != p.dtype):
            flush()
            bucket, nbytes = [], 0
        bucket.append(p)
        nbytes += size
    flush()


class OverlappedGradientSum:
    """One backward per update: launch ordered async buckets as gradients arrive.

    Separate bucket buffers are retained until wait(), never reducing a gradient
    still being accumulated. Missing local gradients contribute zero; globally
    unused parameters retain grad=None. No world-size division: loss is weighted
    by local/global query count by the caller.
    """

    def __init__(self, parameters, bucket_bytes=8 * 1024 * 1024):
        self.parameters = list(parameters)
        self.buckets = []
        bucket, size = [], 0
        # Decoder layers typically finish backwards in reverse parameter order.
        for i in reversed(range(len(self.parameters))):
            p = self.parameters[i]
            if bucket and (size + p.numel() * p.element_size() > bucket_bytes
                           or self.parameters[bucket[0]].dtype != p.dtype):
                self.buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(i)
            size += p.numel() * p.element_size()
        if bucket:
            self.buckets.append(bucket)
        self.enabled = False
        self.handles = [p.register_post_accumulate_grad_hook(self._hook(i))
                        for i, p in enumerate(self.parameters)]

    def begin(self):
        if self.enabled:
            raise RuntimeError("Previous gradient reduction has not finished")
        self.ready, self.pending, self.next_bucket = set(), [], 0
        self.enabled = True

    def _hook(self, index):
        def ready(parameter):
            if not self.enabled:
                return
            if index in self.ready:
                raise RuntimeError("Overlap requires exactly one backward per update; no replay/accumulation")
            self.ready.add(index)
            self._launch_ready()
        return ready

    @torch.no_grad()
    def _launch_ready(self, force=False):
        while self.next_bucket < len(self.buckets):
            indices = self.buckets[self.next_bucket]
            if not force and any(i not in self.ready for i in indices):
                break
            flat = torch.cat([(self.parameters[i].grad if self.parameters[i].grad is not None
                               else torch.zeros_like(self.parameters[i])).reshape(-1) for i in indices])
            work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
            self.pending.append((indices, flat, work))
            self.next_bucket += 1

    @torch.no_grad()
    def finish(self):
        if not self.enabled:
            raise RuntimeError("Call begin before backward")
        self.enabled = False
        self._launch_ready(force=True)
        present = torch.tensor([p.grad is not None for p in self.parameters],
                               dtype=torch.int32, device=self.parameters[0].device)
        presence_work = dist.all_reduce(present, op=dist.ReduceOp.SUM, async_op=True)
        for _, _, work in self.pending:
            work.wait()
        presence_work.wait()
        used = present.tolist()
        for indices, flat, _ in self.pending:
            offset = 0
            for i in indices:
                p = self.parameters[i]
                if used[i]:
                    if p.grad is None:
                        p.grad = torch.empty_like(p)
                    p.grad.copy_(flat[offset:offset + p.numel()].view_as(p))
                else:
                    p.grad = None
                offset += p.numel()
        self.pending.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def average_metrics(metrics, count, device):
    """Query-weighted log metrics, including a rank with an empty final shard."""
    keys_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(keys_by_rank, list(metrics))
    keys = sorted(set().union(*keys_by_rank))
    values = torch.tensor([float(metrics.get(k, 0.0)) * count for k in keys] + [count],
                          device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values.cpu().tolist()
    return {key: value / max(values[-1], 1) for key, value in zip(keys, values[:-1], strict=True)}
