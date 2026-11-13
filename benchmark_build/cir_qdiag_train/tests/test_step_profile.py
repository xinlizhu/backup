import torch
from cir_qdiag_train.step_profile import StepProfile, prepared_stats


def test_cpu_profile_and_shapes():
    timer = StepProfile(torch.device("cpu"))
    timer.compute_start()
    for name in ("forward_loss", "backward_replay", "gradient_finish", "optimizer", "metrics"):
        timer.mark(name)
    result = timer.finish()
    assert result["step_wall_s"] >= sum(v for k, v in result.items() if k.endswith("_host_s"))
    assert all(v >= 0 for v in result.values())
    assert not any("cuda" in k for k in result)
    stats = prepared_stats({"queries": {"input_ids": torch.zeros(2, 11)},
                            "images": [{"input_ids": torch.zeros(3, 20)}, {"input_ids": torch.zeros(1, 8)}]})
    assert stats == {"candidate_images": 4, "candidate_chunks": 2,
                     "query_padded_tokens": 11, "candidate_max_padded_tokens": 20}
    assert prepared_stats(None)["candidate_images"] == 0


def test_cuda_events_not_synchronized_between_phases(monkeypatch):
    syncs = []
    class FakeEvent:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            pass
        def synchronize(self):
            syncs.append("end")
        def elapsed_time(self, other):
            return 12.0
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: syncs.append("start"))
    timer = StepProfile(torch.device("cuda"))
    timer.compute_start()
    timer.mark("forward_loss")
    timer.mark("backward_replay")
    assert syncs == ["start"]
    result = timer.finish()
    assert syncs == ["start", "end"]
    assert result["forward_loss_cuda_stream_s"] == .012
    assert result["backward_replay_cuda_stream_s"] == .012
