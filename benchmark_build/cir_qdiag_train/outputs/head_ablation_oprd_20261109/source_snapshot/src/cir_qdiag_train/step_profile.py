"""Sampled host and CUDA-stream timings; no synchronization between phases."""
import time
import torch


class StepProfile:
    def __init__(self, device):
        self.device = device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.started = time.perf_counter()
        self.host = {}
        self.events = []

    def compute_start(self):
        now = time.perf_counter()
        self.host["data_wait_s"] = now - self.started
        self.last_host = now
        self._event("start")

    def _event(self, name):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.events.append((name, event))

    def mark(self, name):
        self._event(name)
        now = time.perf_counter()
        self.host[name + "_host_s"] = now - self.last_host
        self.last_host = now

    def finish(self):
        gpu = {}
        if self.events:
            self.events[-1][1].synchronize()
            for (_, start), (name, end) in zip(self.events, self.events[1:]):
                gpu[name + "_cuda_stream_s"] = start.elapsed_time(end) / 1000
        return {**self.host, **gpu, "step_wall_s": time.perf_counter() - self.started}


def prepared_stats(prepared):
    """Read shapes only; never move GPU data to CPU to compute diagnostics."""
    result = {"candidate_images": 0, "candidate_chunks": 0}
    if prepared is None:
        return result
    query = prepared.get("queries", {})
    if isinstance(query, dict) and "input_ids" in query:
        result["query_padded_tokens"] = query["input_ids"].shape[-1]
    images = prepared.get("images", [])
    result["candidate_chunks"] = len(images)
    lengths = []
    for batch in images:
        if isinstance(batch, dict) and "input_ids" in batch:
            result["candidate_images"] += batch["input_ids"].shape[0]
            lengths.append(batch["input_ids"].shape[-1])
    result["candidate_max_padded_tokens"] = max(lengths, default=0)
    return result
