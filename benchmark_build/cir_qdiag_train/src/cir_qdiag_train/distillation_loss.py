from __future__ import annotations

import torch
import torch.nn.functional as F


def semantic_ability_distillation_loss(
    *,
    positive_queries: torch.Tensor,
    negative_queries: torch.Tensor | None,
    negative_owner: torch.Tensor,
    candidate_features: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_indices: torch.Tensor,
    teacher_positive_scores: torch.Tensor,
    teacher_negative_scores: torch.Tensor,
    teacher_target_top1: torch.Tensor,
    positive_rank_weight: float,
    semantic_response_weight: float,
    positive_rank_temperature: float,
    semantic_response_beta: float,
    require_teacher_target_top1: bool,
    semantic_min_teacher_target_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill Teacher group ranking and its q+ minus q- response matrix.

    The candidate dimension contains the target plus the four non-reference CIRR
    group members. Negative queries remain flattened, so a batch supports a
    variable number N of semantic perturbations per sample without duplication.
    """

    batch_size, candidate_count, embedding_dim = candidate_features.shape
    if positive_queries.shape != (batch_size, embedding_dim):
        raise ValueError("positive query and candidate feature shapes do not match")
    if candidate_mask.shape != (batch_size, candidate_count):
        raise ValueError("candidate mask shape does not match candidate features")
    if teacher_positive_scores.shape != candidate_mask.shape:
        raise ValueError("Teacher positive score shape does not match candidates")
    if target_indices.shape != (batch_size,):
        raise ValueError("target index shape does not match batch")
    if not bool(candidate_mask.gather(1, target_indices[:, None]).all()):
        raise ValueError("a target index points to a padded candidate")

    zero = positive_queries.new_zeros(())
    student_positive_scores = torch.einsum(
        "bd,bmd->bm", positive_queries, candidate_features
    )

    rank_loss = zero
    rank_valid = candidate_mask.sum(dim=1) >= 2
    if require_teacher_target_top1:
        rank_valid = rank_valid & teacher_target_top1
    if positive_rank_weight > 0 and bool(rank_valid.any()):
        teacher_logits = (teacher_positive_scores / positive_rank_temperature).masked_fill(
            ~candidate_mask, -torch.inf
        )
        student_logits = (student_positive_scores / positive_rank_temperature).masked_fill(
            ~candidate_mask, -torch.inf
        )
        teacher_probabilities = F.softmax(teacher_logits, dim=-1)
        student_log_probabilities = F.log_softmax(student_logits, dim=-1)
        element_kl = F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="none",
        ).masked_fill(~candidate_mask, 0.0)
        per_sample_kl = element_kl.sum(dim=-1)
        rank_loss = per_sample_kl[rank_valid].mean()

    masked_student_positive = student_positive_scores.masked_fill(
        ~candidate_mask, -torch.inf
    )
    student_positive_top1 = masked_student_positive.argmax(dim=1)
    student_target_top1 = student_positive_top1 == target_indices
    teacher_masked_positive = teacher_positive_scores.masked_fill(
        ~candidate_mask, -torch.inf
    )
    teacher_positive_top1_index = teacher_masked_positive.argmax(dim=1)

    response_loss = zero
    response_relation_mask = torch.empty(
        0, dtype=torch.bool, device=positive_queries.device
    )
    student_target_delta = positive_queries.new_empty(0)
    teacher_target_delta = positive_queries.new_empty(0)
    response_absolute_error = zero
    response_element_count = 0
    if semantic_response_weight > 0:
        if negative_queries is None or negative_queries.shape[0] == 0:
            raise ValueError("semantic response KD received no negative queries")
        if negative_owner.shape != (negative_queries.shape[0],):
            raise ValueError("negative owners do not match negative queries")
        if teacher_negative_scores.shape != (
            negative_queries.shape[0],
            candidate_count,
        ):
            raise ValueError("Teacher negative scores do not match negative queries")
        owned_candidates = candidate_features[negative_owner]
        owned_mask = candidate_mask[negative_owner]
        student_negative_scores = torch.einsum(
            "kd,kmd->km", negative_queries, owned_candidates
        )
        student_response = (
            student_positive_scores[negative_owner] - student_negative_scores
        )
        teacher_response = (
            teacher_positive_scores[negative_owner] - teacher_negative_scores
        )
        owned_target_indices = target_indices[negative_owner]
        student_target_delta = student_response.gather(
            1, owned_target_indices[:, None]
        ).squeeze(1)
        teacher_target_delta = teacher_response.gather(
            1, owned_target_indices[:, None]
        ).squeeze(1)
        response_relation_mask = (
            teacher_target_delta > semantic_min_teacher_target_delta
        )
        if bool(response_relation_mask.any()):
            safe_student = torch.where(owned_mask, student_response, torch.zeros_like(student_response))
            safe_teacher = torch.where(owned_mask, teacher_response, torch.zeros_like(teacher_response))
            element_loss = F.smooth_l1_loss(
                safe_student,
                safe_teacher,
                reduction="none",
                beta=semantic_response_beta,
            )
            per_relation = element_loss.sum(dim=1) / owned_mask.sum(dim=1).clamp_min(1)
            response_loss = per_relation[response_relation_mask].mean()
            absolute_error = (safe_student - safe_teacher).abs()
            selected_elements = owned_mask & response_relation_mask[:, None]
            response_absolute_error = absolute_error[selected_elements].mean()
            response_element_count = int(selected_elements.sum().item())

    total = positive_rank_weight * rank_loss + semantic_response_weight * response_loss
    metrics = {
        "loss": total.detach(),
        "positive_rank_loss": rank_loss.detach(),
        "semantic_response_loss": response_loss.detach(),
        "teacher_target_top1_rate": teacher_target_top1.float().mean().detach(),
        "student_target_top1_rate": student_target_top1.float().mean().detach(),
        "student_teacher_top1_agreement": (
            student_positive_top1 == teacher_positive_top1_index
        ).float().mean().detach(),
        "rank_sample_count": positive_queries.new_tensor(float(rank_valid.sum().item())),
        "semantic_relation_count": positive_queries.new_tensor(
            float(response_relation_mask.sum().item())
        ),
        "semantic_response_element_count": positive_queries.new_tensor(
            float(response_element_count)
        ),
        "semantic_response_mae": response_absolute_error.detach(),
        "teacher_target_delta": (
            teacher_target_delta[response_relation_mask].mean().detach()
            if bool(response_relation_mask.any())
            else zero.detach()
        ),
        "student_target_delta": (
            student_target_delta[response_relation_mask].mean().detach()
            if bool(response_relation_mask.any())
            else zero.detach()
        ),
        "student_semantic_raw_win_rate": (
            (student_target_delta[response_relation_mask] > 0).float().mean().detach()
            if bool(response_relation_mask.any())
            else zero.detach()
        ),
    }
    return total, metrics
