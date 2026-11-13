"""KL on natural candidates, with independent target eligibility metadata."""

import torch
import torch.nn.functional as F


def natural_candidate_rank_loss(
    *,
    student_scores,
    teacher_scores,
    target_positions,
    teacher_filter_target_ranks,
    temperature,
    direction,
    max_teacher_rank,
    rank_weight=1.0,
):
    if student_scores.ndim != 2 or student_scores.shape != teacher_scores.shape:
        raise ValueError("invalid natural candidate score shapes")
    b, k = student_scores.shape
    if k < 2 or temperature <= 0 or direction not in {"forward", "reverse"}:
        raise ValueError("invalid natural candidate KL settings")
    if target_positions.shape != (b,) or teacher_filter_target_ranks.shape != (b,):
        raise ValueError("invalid target metadata shapes")
    if bool(((target_positions < -1) | (target_positions >= k)).any()):
        raise ValueError("invalid natural target position")
    valid = torch.ones(b, dtype=torch.bool, device=student_scores.device)
    if max_teacher_rank is not None:
        valid &= teacher_filter_target_ranks <= max_teacher_rank
    s_log = F.log_softmax(student_scores / temperature, dim=-1)
    t_log = F.log_softmax(teacher_scores / temperature, dim=-1)
    if bool(valid.any()):
        if direction == "reverse":
            loss = F.kl_div(t_log[valid], s_log[valid].exp(), reduction="batchmean")
        else:
            loss = F.kl_div(s_log[valid], t_log[valid].exp(), reduction="batchmean")
    else:
        loss = student_scores.sum() * 0.0
    s_top, t_top = student_scores.argmax(-1), teacher_scores.argmax(-1)
    # Missing target is -1, never silently treated as candidate zero.
    metrics = {
        "loss": (rank_weight * loss).detach(),
        "rank_loss": loss.detach(),
        "valid_teacher_rate": valid.float().mean(),
        "raw_mining_target_recall_at_k": (target_positions >= 0).float().mean(),
        "student_target_top1_rate": (s_top == target_positions).float().mean(),
        "teacher_target_top1_rate": (t_top == target_positions).float().mean(),
        "student_teacher_top1_agreement": (s_top == t_top).float().mean(),
        "teacher_filter_target_top10_rate": (teacher_filter_target_ranks <= 10).float().mean(),
        "teacher_filter_target_mean_rank": teacher_filter_target_ranks.float().mean(),
        "rank_kl_is_reverse": student_scores.new_tensor(float(direction == "reverse")),
        "teacher_rank_entropy": -(t_log.exp() * t_log).sum(-1).mean().detach(),
        "student_rank_entropy": -(s_log.exp() * s_log).sum(-1).mean().detach(),
    }
    return rank_weight * loss, metrics
