from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalPolicyRollouts:
    """A fixed group of variable-length target-stopped ranking trajectories."""

    rankings: torch.Tensor
    valid_step_mask: torch.Tensor
    old_step_log_probabilities: torch.Tensor
    rewards: torch.Tensor
    target_ranks: torch.Tensor
    advantages: torch.Tensor


def causal_prefix_anchor_kl(
    *,
    student_scores: torch.Tensor,
    anchor_scores: torch.Tensor,
    rollouts: CausalPolicyRollouts,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average anchor-to-current KL over each sampled causal prefix.

    Both policies are categorical Plackett--Luce policies over the same live
    candidate IDs. Previously selected actions are masked before each step, so
    this regularizes the actual sequential policy rather than only its initial
    categorical distribution.
    """

    if student_scores.ndim != 2 or anchor_scores.shape != student_scores.shape:
        raise ValueError("anchor and current candidate scores must have shape [B, K]")
    if temperature <= 0:
        raise ValueError("anchor KL temperature must be positive")
    batch_size, candidate_count = student_scores.shape
    rankings = rollouts.rankings
    valid_step_mask = rollouts.valid_step_mask
    if rankings.ndim != 3 or rankings.shape[:1] != (batch_size,):
        raise ValueError("anchor KL rollouts have an invalid batch shape")
    if valid_step_mask.shape != rankings.shape:
        raise ValueError("anchor KL rollout mask has an invalid shape")

    rollout_count, max_steps = rankings.shape[1:]
    current_logits = (
        student_scores.float() / temperature
    )[:, None, :].expand(batch_size, rollout_count, candidate_count)
    fixed_anchor_logits = (
        anchor_scores.detach().float() / temperature
    )[:, None, :].expand(batch_size, rollout_count, candidate_count)
    available = torch.ones(
        (batch_size, rollout_count, candidate_count),
        dtype=torch.bool,
        device=student_scores.device,
    )
    kl_sum = student_scores.new_zeros((batch_size, rollout_count), dtype=torch.float32)
    step_count = valid_step_mask.sum(dim=-1).clamp_min(1).to(torch.float32)
    step_values: list[torch.Tensor] = []
    for step in range(max_steps):
        masked_current = current_logits.masked_fill(~available, -torch.inf)
        masked_anchor = fixed_anchor_logits.masked_fill(~available, -torch.inf)
        current_log_probabilities = F.log_softmax(masked_current, dim=-1)
        anchor_log_probabilities = F.log_softmax(masked_anchor, dim=-1)
        anchor_probabilities = anchor_log_probabilities.exp()
        log_probability_delta = torch.where(
            available,
            anchor_log_probabilities - current_log_probabilities,
            torch.zeros_like(anchor_log_probabilities),
        )
        step_kl = (
            anchor_probabilities
            * log_probability_delta
        ).sum(dim=-1)
        valid = valid_step_mask[:, :, step]
        step_values.append(step_kl[valid])
        kl_sum = kl_sum + step_kl * valid.to(step_kl.dtype)

        selected = rankings[:, :, step]
        safe_selected = selected.clamp_min(0)
        selected_mask = F.one_hot(
            safe_selected, num_classes=candidate_count
        ).to(torch.bool)
        available = available & ~(selected_mask & valid.unsqueeze(-1))

    per_rollout_kl = kl_sum / step_count
    mean_kl = per_rollout_kl.mean()
    valid_step_values = [values for values in step_values if values.numel()]
    flattened = (
        torch.cat(valid_step_values)
        if valid_step_values
        else mean_kl.new_zeros((1,))
    )
    return mean_kl, {
        "anchor_policy_kl": mean_kl.detach(),
        "anchor_policy_kl_max_step": flattened.max().detach(),
        "anchor_policy_kl_mean_step": flattened.mean().detach(),
    }


def merge_mixed_candidate_features(
    *,
    image_candidate_features: torch.Tensor,
    negative_query_features: torch.Tensor,
    negative_query_owners: torch.Tensor,
    negative_query_positions: torch.Tensor,
    candidate_is_negative_query: torch.Tensor,
) -> torch.Tensor:
    """Merge image and composed q- embeddings into one fixed-width candidate set."""

    if image_candidate_features.ndim != 3:
        raise ValueError("image candidate features must have shape [B, K, D]")
    batch_size, candidate_count, embedding_dim = image_candidate_features.shape
    if candidate_is_negative_query.shape != (batch_size, candidate_count):
        raise ValueError("mixed candidate mask has an invalid shape")
    negative_count = int(candidate_is_negative_query.sum().item())
    if negative_query_features.shape != (negative_count, embedding_dim):
        raise ValueError("q- candidate features have an invalid shape")
    if negative_query_owners.shape != (negative_count,) or negative_query_positions.shape != (
        negative_count,
    ):
        raise ValueError("q- candidate coordinates have an invalid shape")
    if negative_count == 0:
        return image_candidate_features
    if bool(
        ((negative_query_owners < 0) | (negative_query_owners >= batch_size)).any()
    ) or bool(
        (
            (negative_query_positions < 0)
            | (negative_query_positions >= candidate_count)
        ).any()
    ):
        raise ValueError("q- candidate coordinate is out of range")
    flattened = negative_query_owners * candidate_count + negative_query_positions
    if flattened.unique().numel() != negative_count:
        raise ValueError("q- candidate coordinates contain duplicates")
    if not bool(
        candidate_is_negative_query[
            negative_query_owners, negative_query_positions
        ].all()
    ):
        raise ValueError("q- candidate coordinates disagree with the candidate mask")
    query_candidates = torch.zeros_like(image_candidate_features)
    query_candidates[
        negative_query_owners, negative_query_positions
    ] = negative_query_features
    return torch.where(
        candidate_is_negative_query.unsqueeze(-1),
        query_candidates,
        image_candidate_features,
    )


def candidate_infonce_loss(
    *,
    student_scores: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    temperature: float,
    teacher_scores: torch.Tensor | None = None,
    require_teacher_target_top1: bool = False,
    teacher_target_max_rank: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast the forced target against the selected non-target candidates.

    Candidate zero is always the ground-truth target.  Teacher scores may
    optionally define the same target-rank validity mask used by Rank KD, but
    they never enter the one-hot cross-entropy target.
    """

    if student_scores.ndim != 2 or student_scores.shape[1] < 2:
        raise ValueError("candidate InfoNCE requires [batch, candidates>=2] scores")
    if raw_target_in_topk.shape != (student_scores.shape[0],):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if temperature <= 0:
        raise ValueError("candidate InfoNCE temperature must be positive")
    if teacher_target_max_rank is not None and teacher_target_max_rank < 1:
        raise ValueError("teacher_target_max_rank must be at least one")
    if require_teacher_target_top1 and teacher_target_max_rank not in (None, 1):
        raise ValueError(
            "require_teacher_target_top1 conflicts with teacher_target_max_rank > 1"
        )
    uses_teacher_filter = (
        require_teacher_target_top1 or teacher_target_max_rank is not None
    )
    if uses_teacher_filter and teacher_scores is None:
        raise ValueError("Teacher-filtered candidate InfoNCE requires Teacher scores")
    if teacher_scores is not None and teacher_scores.shape != student_scores.shape:
        raise ValueError("Teacher and Student scores must have the same shape")

    valid = torch.ones(
        student_scores.shape[0], dtype=torch.bool, device=student_scores.device
    )
    teacher_target_rank: torch.Tensor | None = None
    if teacher_scores is not None:
        teacher_target_rank = 1 + (
            teacher_scores[:, 1:] > teacher_scores[:, :1]
        ).sum(dim=1)
        if require_teacher_target_top1:
            valid &= teacher_target_rank <= 1
        elif teacher_target_max_rank is not None:
            valid &= teacher_target_rank <= teacher_target_max_rank

    labels = torch.zeros(
        int(valid.sum().item()), dtype=torch.long, device=student_scores.device
    )
    if bool(valid.any()):
        loss = F.cross_entropy(student_scores[valid] / temperature, labels)
    else:
        loss = student_scores.sum() * 0.0
    student_top1 = student_scores.argmax(dim=1)
    student_target_margin = student_scores[:, 0] - student_scores[:, 1:].amax(dim=1)
    metrics = {
        "loss": loss.detach(),
        "candidate_infonce_loss": loss.detach(),
        "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean().detach(),
        "student_target_top1_rate": (student_top1 == 0).float().mean().detach(),
        "student_target_margin": student_target_margin.mean().detach(),
    }
    if teacher_scores is not None and teacher_target_rank is not None:
        teacher_top1 = teacher_scores.argmax(dim=1)
        teacher_target_margin = (
            teacher_scores[:, 0] - teacher_scores[:, 1:].amax(dim=1)
        )
        metrics.update(
            {
                "valid_teacher_rate": valid.float().mean().detach(),
                "teacher_target_top1_rate": (teacher_top1 == 0).float().mean().detach(),
                "teacher_target_top10_rate": (
                    teacher_target_rank <= min(10, student_scores.shape[1])
                ).float().mean().detach(),
                "teacher_target_mean_rank": teacher_target_rank.float().mean().detach(),
                "student_teacher_top1_agreement": (
                    student_top1 == teacher_top1
                ).float().mean().detach(),
                "teacher_target_margin": teacher_target_margin.mean().detach(),
            }
        )
    return loss, metrics


def centered_similarity_mse_loss(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    require_teacher_target_top1: bool = False,
    teacher_target_max_rank: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regress centered raw Student-Teacher similarity residuals.

    For each query this computes ``mean_k(((s - t) - mean_k(s - t)) ** 2)``.
    No softmax, temperature scaling, or probability normalization is applied.
    """

    if student_scores.shape != teacher_scores.shape or student_scores.ndim != 2:
        raise ValueError("Teacher and Student scores must have the same 2-D shape")
    if student_scores.shape[1] < 2:
        raise ValueError("centered similarity MSE requires at least two candidates")
    batch_size = student_scores.shape[0]
    if raw_target_in_topk.shape != (batch_size,):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if teacher_target_max_rank is not None and teacher_target_max_rank < 1:
        raise ValueError("teacher_target_max_rank must be at least one")
    if require_teacher_target_top1 and teacher_target_max_rank not in (None, 1):
        raise ValueError(
            "require_teacher_target_top1 conflicts with teacher_target_max_rank > 1"
        )

    teacher_target_rank = 1 + (
        teacher_scores[:, 1:] > teacher_scores[:, :1]
    ).sum(dim=1)
    valid = torch.ones(batch_size, dtype=torch.bool, device=student_scores.device)
    if require_teacher_target_top1:
        valid &= teacher_target_rank <= 1
    elif teacher_target_max_rank is not None:
        valid &= teacher_target_rank <= teacher_target_max_rank

    residual = student_scores - teacher_scores
    centered_residual = residual - residual.mean(dim=1, keepdim=True)
    if bool(valid.any()):
        # torch.mean over [valid_batch, K] is mean_b 1/K ||centered residual||_2^2.
        loss = centered_residual[valid].square().mean()
        centered_residual_mae = centered_residual[valid].abs().mean().detach()
    else:
        loss = student_scores.sum() * 0.0
        centered_residual_mae = loss.detach()

    student_top1 = student_scores.argmax(dim=1)
    teacher_top1 = teacher_scores.argmax(dim=1)
    student_target_margin = student_scores[:, 0] - student_scores[:, 1:].amax(dim=1)
    teacher_target_margin = teacher_scores[:, 0] - teacher_scores[:, 1:].amax(dim=1)
    metrics = {
        "loss": loss.detach(),
        "centered_mse_loss": loss.detach(),
        "centered_residual_mae": centered_residual_mae,
        "valid_teacher_rate": valid.float().mean().detach(),
        "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean().detach(),
        "student_target_top1_rate": (student_top1 == 0).float().mean().detach(),
        "teacher_target_top1_rate": (teacher_top1 == 0).float().mean().detach(),
        "teacher_target_top10_rate": (
            teacher_target_rank <= min(10, student_scores.shape[1])
        ).float().mean().detach(),
        "teacher_target_mean_rank": teacher_target_rank.float().mean().detach(),
        "student_teacher_top1_agreement": (
            student_top1 == teacher_top1
        ).float().mean().detach(),
        "student_target_margin": student_target_margin.mean().detach(),
        "teacher_target_margin": teacher_target_margin.mean().detach(),
    }
    return loss, metrics


def centered_scores(scores: torch.Tensor) -> torch.Tensor:
    """Remove each model/query's scalar similarity offset over its candidates."""

    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("centered scores require [batch, candidates>=2]")
    return scores - scores.mean(dim=1, keepdim=True)


def sample_plackett_luce_rankings(
    logits: torch.Tensor,
    *,
    rollout_count: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample independent ordered Top-K lists without replacement.

    Images cannot repeat within one rollout. Different rollouts remain independent
    Monte Carlo samples and may therefore be identical; forcing cross-rollout
    uniqueness would change the policy distribution.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Plackett-Luce sampling requires [batch, candidates>=2]")
    if rollout_count < 2:
        raise ValueError("group-relative policy loss requires at least two rollouts")
    if top_k < 1 or top_k > logits.shape[1]:
        raise ValueError("sampled Top-K must be within the candidate count")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Plackett-Luce logits must be finite")

    batch_size, candidate_count = logits.shape
    expanded = logits[:, None, :].expand(batch_size, rollout_count, candidate_count)
    available = torch.ones_like(expanded, dtype=torch.bool)
    sampled_steps: list[torch.Tensor] = []
    log_probability_steps: list[torch.Tensor] = []
    for _ in range(top_k):
        masked_logits = expanded.masked_fill(~available, -torch.inf)
        step_log_probabilities = F.log_softmax(masked_logits, dim=-1)
        sampling_probabilities = F.softmax(masked_logits.detach(), dim=-1)
        sampled = torch.multinomial(
            sampling_probabilities.reshape(-1, candidate_count),
            num_samples=1,
        ).reshape(batch_size, rollout_count)
        sampled_steps.append(sampled)
        log_probability_steps.append(
            step_log_probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        )
        available = available.scatter(-1, sampled.unsqueeze(-1), False)

    rankings = torch.stack(sampled_steps, dim=-1)
    sequence_log_probabilities = torch.stack(
        log_probability_steps, dim=-1
    ).sum(dim=-1)
    return rankings, sequence_log_probabilities


def sample_causal_plackett_luce_rollouts(
    logits: torch.Tensor,
    *,
    rollout_count: int,
    max_steps: int,
    target_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample target-stopped Plackett-Luce prefixes without replacement.

    A trajectory terminates as soon as it selects the target. A miss contains
    exactly ``max_steps`` valid actions. Suffix slots use ``-1`` and have zero
    log probability, which makes accidental suffix credit easy to detect.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("causal Plackett-Luce sampling requires [batch, candidates>=2]")
    if rollout_count < 2:
        raise ValueError("causal group-relative policy loss requires at least two rollouts")
    if max_steps < 1 or max_steps > logits.shape[1]:
        raise ValueError("causal rollout length must be within the candidate count")
    if target_index < 0 or target_index >= logits.shape[1]:
        raise ValueError("target_index is outside the candidate set")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("causal Plackett-Luce logits must be finite")

    batch_size, candidate_count = logits.shape
    expanded = logits[:, None, :].expand(batch_size, rollout_count, candidate_count)
    available = torch.ones_like(expanded, dtype=torch.bool)
    active = torch.ones(
        batch_size,
        rollout_count,
        dtype=torch.bool,
        device=logits.device,
    )
    sampled_steps: list[torch.Tensor] = []
    log_probability_steps: list[torch.Tensor] = []
    valid_steps: list[torch.Tensor] = []
    for _ in range(max_steps):
        step_valid = active
        # Finished rows need one harmless finite category for torch.multinomial.
        # Their sampled value is replaced by -1 and never enters the loss.
        sampling_available = available.clone()
        sampling_available[~step_valid] = False
        sampling_available[..., target_index] |= ~step_valid
        masked_logits = expanded.masked_fill(~sampling_available, -torch.inf)
        step_log_probabilities = F.log_softmax(masked_logits, dim=-1)
        sampling_probabilities = F.softmax(masked_logits.detach(), dim=-1)
        sampled = torch.multinomial(
            sampling_probabilities.reshape(-1, candidate_count),
            num_samples=1,
        ).reshape(batch_size, rollout_count)
        selected_log_probability = step_log_probabilities.gather(
            -1, sampled.unsqueeze(-1)
        ).squeeze(-1)
        sampled_steps.append(torch.where(step_valid, sampled, -torch.ones_like(sampled)))
        log_probability_steps.append(
            torch.where(
                step_valid,
                selected_log_probability,
                torch.zeros_like(selected_log_probability),
            )
        )
        valid_steps.append(step_valid)
        available = available.scatter(-1, sampled.unsqueeze(-1), False)
        active = step_valid & (sampled != target_index)

    return (
        torch.stack(sampled_steps, dim=-1),
        torch.stack(log_probability_steps, dim=-1),
        torch.stack(valid_steps, dim=-1),
    )


def causal_plackett_luce_step_log_probabilities(
    logits: torch.Tensor,
    *,
    rankings: torch.Tensor,
    valid_step_mask: torch.Tensor,
) -> torch.Tensor:
    """Score fixed target-stopped prefixes under the current policy."""

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("causal Plackett-Luce scoring requires [batch, candidates>=2]")
    if rankings.ndim != 3 or rankings.shape[0] != logits.shape[0]:
        raise ValueError("causal rankings require [batch, rollouts, steps]")
    if valid_step_mask.shape != rankings.shape or valid_step_mask.dtype != torch.bool:
        raise ValueError("causal valid-step mask must be boolean and match rankings")
    if bool((valid_step_mask[..., 1:] & ~valid_step_mask[..., :-1]).any()):
        raise ValueError("causal valid-step masks must be contiguous prefixes")
    valid_actions = rankings[valid_step_mask]
    if valid_actions.numel() and bool(
        ((valid_actions < 0) | (valid_actions >= logits.shape[1])).any()
    ):
        raise ValueError("a valid causal action is outside the candidate set")
    if bool((rankings[~valid_step_mask] != -1).any()):
        raise ValueError("causal suffix slots must use the -1 sentinel")

    batch_size, candidate_count = logits.shape
    rollout_count, max_steps = rankings.shape[1:]
    expanded = logits[:, None, :].expand(batch_size, rollout_count, candidate_count)
    available = torch.ones_like(expanded, dtype=torch.bool)
    log_probability_steps: list[torch.Tensor] = []
    for step in range(max_steps):
        step_valid = valid_step_mask[..., step]
        safe_action = rankings[..., step].clamp_min(0)
        if bool(
            (
                step_valid
                & ~available.gather(-1, safe_action.unsqueeze(-1)).squeeze(-1)
            ).any()
        ):
            raise ValueError("a causal rollout repeats a candidate")
        scoring_available = available.clone()
        scoring_available[~step_valid] = False
        scoring_available[..., 0] |= ~step_valid
        masked_logits = expanded.masked_fill(~scoring_available, -torch.inf)
        step_log_probabilities = F.log_softmax(masked_logits, dim=-1)
        selected = step_log_probabilities.gather(
            -1, safe_action.unsqueeze(-1)
        ).squeeze(-1)
        log_probability_steps.append(
            torch.where(step_valid, selected, torch.zeros_like(selected))
        )
        # Invalid suffix actions are harmless: no later step may become valid.
        available = available.scatter(-1, safe_action.unsqueeze(-1), False)
    return torch.stack(log_probability_steps, dim=-1)


def target_ndcg_rewards(
    rankings: torch.Tensor,
    *,
    target_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return single-positive NDCG and target rank for sampled lists."""

    if rankings.ndim != 3 or rankings.shape[-1] < 1:
        raise ValueError("sampled rankings require [batch, rollouts, top_k]")
    top_k = rankings.shape[-1]
    positions = torch.arange(
        1,
        top_k + 1,
        device=rankings.device,
        dtype=torch.long,
    )
    matches = rankings == target_index
    missing_rank = torch.full_like(rankings, top_k + 1)
    ranks = torch.where(matches, positions, missing_rank).amin(dim=-1)
    found = ranks <= top_k
    rewards = torch.where(
        found,
        torch.log2(ranks.to(torch.float32) + 1.0).reciprocal(),
        torch.zeros_like(ranks, dtype=torch.float32),
    )
    return rewards, ranks


def causal_group_policy_loss(
    *,
    student_scores: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    temperature: float,
    rollout_count: int,
    rollout_top_k: int,
    algorithm: Literal["reinforce", "grpo", "gspo"],
    clip_epsilon: float | None,
    rollouts: CausalPolicyRollouts | None = None,
    teacher_scores: torch.Tensor | None = None,
    reward_tilt_beta: float | None = None,
) -> tuple[torch.Tensor, CausalPolicyRollouts, dict[str, torch.Tensor]]:
    """Optimize target NDCG using causal-prefix group-relative policy updates.

    ``reinforce`` uses the exact stopped-trajectory score function. ``grpo``
    clips per-step importance ratios. ``gspo`` clips one length-normalized
    sequence ratio. Passing a previously collected rollout group activates a
    genuine old-policy update without retaining an old model copy.
    """

    if student_scores.ndim != 2 or student_scores.shape[1] < 2:
        raise ValueError("causal policy loss requires [batch, candidates>=2] scores")
    if raw_target_in_topk.shape != (student_scores.shape[0],):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if temperature <= 0:
        raise ValueError("causal policy temperature must be positive")
    if algorithm not in {"reinforce", "grpo", "gspo"}:
        raise ValueError(f"unsupported causal policy algorithm: {algorithm}")
    if algorithm == "grpo" and clip_epsilon is None:
        raise ValueError("causal GRPO requires a clipping epsilon")
    if clip_epsilon is not None and clip_epsilon <= 0:
        raise ValueError("policy clipping epsilon must be positive")
    reward_tilted = teacher_scores is not None or reward_tilt_beta is not None
    if reward_tilted:
        if teacher_scores is None or teacher_scores.shape != student_scores.shape:
            raise ValueError(
                "reward-tilted policy loss requires matching Teacher scores"
            )
        if reward_tilt_beta is None or reward_tilt_beta <= 0:
            raise ValueError("reward_tilt_beta must be positive")
        if algorithm != "gspo":
            raise ValueError("reward-tilted policy loss currently requires GSPO")

    logits = student_scores.float() / temperature
    if rollouts is None:
        rankings, current_step_log_probabilities, valid_step_mask = (
            sample_causal_plackett_luce_rollouts(
                logits,
                rollout_count=rollout_count,
                max_steps=rollout_top_k,
            )
        )
        rewards, target_ranks = target_ndcg_rewards(rankings)
        if reward_tilted:
            assert teacher_scores is not None and reward_tilt_beta is not None
            teacher_step_log_probabilities = (
                causal_plackett_luce_step_log_probabilities(
                    teacher_scores.detach().float() / temperature,
                    rankings=rankings,
                    valid_step_mask=valid_step_mask,
                )
            )
            teacher_log_ratio = (
                (current_step_log_probabilities.detach() - teacher_step_log_probabilities)
                * valid_step_mask.to(current_step_log_probabilities.dtype)
            ).sum(dim=-1)
            objective_rewards = rewards - reward_tilt_beta * teacher_log_ratio
        else:
            objective_rewards = rewards
        group_reward_mean = objective_rewards.mean(dim=1, keepdim=True)
        raw_advantages = objective_rewards - group_reward_mean
        objective_reward_std_per_query = objective_rewards.std(
            dim=1, unbiased=False
        )
        effective_query = objective_reward_std_per_query > 1e-8
        advantages = torch.where(
            effective_query[:, None],
            raw_advantages
            / objective_reward_std_per_query.clamp_min(1e-8)[:, None],
            torch.zeros_like(raw_advantages),
        )
        rollouts = CausalPolicyRollouts(
            rankings=rankings.detach(),
            valid_step_mask=valid_step_mask.detach(),
            old_step_log_probabilities=current_step_log_probabilities.detach(),
            rewards=rewards.detach(),
            target_ranks=target_ranks.detach(),
            advantages=advantages.detach(),
        )
    else:
        expected_shape = (
            student_scores.shape[0],
            rollout_count,
            rollout_top_k,
        )
        if rollouts.rankings.shape != expected_shape:
            raise ValueError("reused causal rollouts do not match the current batch")
        current_step_log_probabilities = causal_plackett_luce_step_log_probabilities(
            logits,
            rankings=rollouts.rankings,
            valid_step_mask=rollouts.valid_step_mask,
        )
        if reward_tilted:
            assert teacher_scores is not None and reward_tilt_beta is not None
            teacher_step_log_probabilities = (
                causal_plackett_luce_step_log_probabilities(
                    teacher_scores.detach().float() / temperature,
                    rankings=rollouts.rankings,
                    valid_step_mask=rollouts.valid_step_mask,
                )
            )
            teacher_log_ratio = (
                (
                    rollouts.old_step_log_probabilities
                    - teacher_step_log_probabilities
                )
                * rollouts.valid_step_mask.to(
                    rollouts.old_step_log_probabilities.dtype
                )
            ).sum(dim=-1)
            objective_rewards = (
                rollouts.rewards - reward_tilt_beta * teacher_log_ratio
            )
        else:
            objective_rewards = rollouts.rewards
        objective_reward_std_per_query = objective_rewards.std(
            dim=1, unbiased=False
        )
        effective_query = objective_reward_std_per_query > 1e-8

    step_mask = rollouts.valid_step_mask
    step_mask_float = step_mask.to(current_step_log_probabilities.dtype)
    prefix_lengths = step_mask.sum(dim=-1).clamp_min(1)
    advantages = rollouts.advantages
    log_ratios = current_step_log_probabilities - rollouts.old_step_log_probabilities
    if algorithm == "reinforce":
        sequence_log_probabilities = (
            current_step_log_probabilities * step_mask_float
        ).sum(dim=-1)
        per_rollout_objective = advantages * sequence_log_probabilities
        importance_ratios = torch.ones_like(per_rollout_objective)
        clipped = torch.zeros_like(per_rollout_objective, dtype=torch.bool)
    elif algorithm == "grpo":
        assert clip_epsilon is not None
        importance_ratios = log_ratios.exp()
        unclipped = importance_ratios * advantages.unsqueeze(-1)
        clipped_ratios = importance_ratios.clamp(
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        )
        clipped_objective = clipped_ratios * advantages.unsqueeze(-1)
        per_step_objective = torch.minimum(unclipped, clipped_objective)
        per_rollout_objective = (
            per_step_objective * step_mask_float
        ).sum(dim=-1) / prefix_lengths
        clipped = (
            (importance_ratios < 1.0 - clip_epsilon)
            | (importance_ratios > 1.0 + clip_epsilon)
        ) & step_mask
    else:
        mean_sequence_log_ratio = (
            log_ratios * step_mask_float
        ).sum(dim=-1) / prefix_lengths
        importance_ratios = mean_sequence_log_ratio.exp()
        unclipped = importance_ratios * advantages
        if clip_epsilon is None:
            per_rollout_objective = unclipped
            clipped = torch.zeros_like(unclipped, dtype=torch.bool)
        else:
            clipped_ratios = importance_ratios.clamp(
                1.0 - clip_epsilon,
                1.0 + clip_epsilon,
            )
            per_rollout_objective = torch.minimum(
                unclipped,
                clipped_ratios * advantages,
            )
            clipped = (
                (importance_ratios < 1.0 - clip_epsilon)
                | (importance_ratios > 1.0 + clip_epsilon)
            )

    loss = -per_rollout_objective.mean()
    with torch.no_grad():
        initial_probabilities = F.softmax(logits, dim=-1)
        initial_log_probabilities = F.log_softmax(logits, dim=-1)
        policy_entropy = -(
            initial_probabilities * initial_log_probabilities
        ).sum(dim=-1).mean()
        unique_prefix_counts = torch.stack(
            [
                torch.tensor(
                    torch.unique(row, dim=0).shape[0],
                    device=student_scores.device,
                    dtype=torch.float32,
                )
                for row in rollouts.rankings
            ]
        )
        unique_reward_counts = torch.stack(
            [
                torch.tensor(
                    torch.unique(row).numel(),
                    device=student_scores.device,
                    dtype=torch.float32,
                )
                for row in rollouts.rewards
            ]
        )
        unique_target_rank_counts = torch.stack(
            [
                torch.tensor(
                    torch.unique(row).numel(),
                    device=student_scores.device,
                    dtype=torch.float32,
                )
                for row in rollouts.target_ranks
            ]
        )
        student_top1 = student_scores.argmax(dim=1)
        student_target_margin = (
            student_scores[:, 0] - student_scores[:, 1:].amax(dim=1)
        )
        valid_log_ratios = log_ratios[step_mask]
        approximate_kl = (
            -valid_log_ratios.mean()
            if valid_log_ratios.numel()
            else student_scores.new_zeros(())
        )
        if algorithm == "grpo":
            valid_ratios = importance_ratios[step_mask]
            clip_fraction = clipped.float().sum() / step_mask_float.sum().clamp_min(1)
        else:
            valid_ratios = importance_ratios.reshape(-1)
            clip_fraction = clipped.float().mean()
        metrics = {
            "loss": loss.detach(),
            "causal_policy_loss": loss.detach(),
            "rollout_reward_mean": rollouts.rewards.mean(),
            "rollout_reward_std": rollouts.rewards.std(
                dim=1, unbiased=False
            ).mean(),
            "objective_reward_mean": objective_rewards.mean(),
            "objective_reward_std": objective_reward_std_per_query.mean(),
            "effective_rl_query_rate": effective_query.float().mean(),
            "causal_prefix_mean_length": prefix_lengths.float().mean(),
            "causal_suffix_masked_steps": (
                rollout_top_k - prefix_lengths.float()
            ).mean(),
            "unique_causal_prefix_count": unique_prefix_counts.mean(),
            "unique_causal_prefix_rate": (
                unique_prefix_counts / float(rollout_count)
            ).mean(),
            "unique_reward_count": unique_reward_counts.mean(),
            "unique_target_rank_count": unique_target_rank_counts.mean(),
            "sampled_target_recall_at_k": (
                rollouts.target_ranks <= rollout_top_k
            ).float().mean(),
            "sampled_target_rank_with_miss": rollouts.target_ranks.float().mean(),
            "policy_entropy": policy_entropy,
            "policy_target_probability": initial_probabilities[:, 0].mean(),
            "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean(),
            "student_target_top1_rate": (student_top1 == 0).float().mean(),
            "student_target_margin": student_target_margin.mean(),
            "importance_ratio_mean": valid_ratios.mean(),
            "importance_ratio_min": valid_ratios.min(),
            "importance_ratio_max": valid_ratios.max(),
            "clip_fraction": clip_fraction,
            "approx_policy_kl": approximate_kl,
        }
        if reward_tilted:
            assert teacher_scores is not None and reward_tilt_beta is not None
            teacher_target_rank = 1 + (
                teacher_scores[:, 1:] > teacher_scores[:, :1]
            ).sum(dim=1)
            metrics.update(
                {
                    "reward_tilt_beta": student_scores.new_tensor(
                        reward_tilt_beta
                    ),
                    "teacher_reverse_kl_sample": teacher_log_ratio.mean(),
                    "teacher_rollout_log_probability": (
                        teacher_step_log_probabilities
                        * step_mask_float
                    ).sum(dim=-1).mean(),
                    "old_rollout_log_probability": (
                        rollouts.old_step_log_probabilities
                        * step_mask_float
                    ).sum(dim=-1).mean(),
                    "teacher_target_top1_rate": (
                        teacher_target_rank == 1
                    ).float().mean(),
                    "teacher_target_top10_rate": (
                        teacher_target_rank
                        <= min(10, teacher_scores.shape[1])
                    ).float().mean(),
                    "teacher_target_mean_rank": (
                        teacher_target_rank.float().mean()
                    ),
                }
            )
        for rank in range(1, rollout_top_k + 1):
            metrics[f"sampled_target_rank_{rank}_rate"] = (
                rollouts.target_ranks == rank
            ).float().mean()
        metrics["sampled_target_miss_rate"] = (
            rollouts.target_ranks > rollout_top_k
        ).float().mean()
    return loss, rollouts, metrics


def reward_tilted_reverse_opd_loss(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    temperature: float,
    rollout_count: int,
    rollout_top_k: int,
    reward_tilt_beta: float,
    require_teacher_target_top1: bool = False,
    teacher_target_max_rank: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reverse-KL OPD toward an NDCG-tilted Teacher rollout distribution.

    Student rollouts are self-normalized importance samples.  The proposal
    correction removes their old-policy sampling frequency from both the
    Teacher+reward target and the differentiable current-Student group policy.
    This is a direct reverse KL, not a policy-gradient surrogate.
    """

    if student_scores.ndim != 2 or student_scores.shape[1] < 2:
        raise ValueError("reward-tilted Reverse-OPD requires [B, K>=2] scores")
    if teacher_scores.shape != student_scores.shape:
        raise ValueError("Teacher and Student scores must have the same shape")
    if raw_target_in_topk.shape != (student_scores.shape[0],):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if temperature <= 0 or reward_tilt_beta <= 0:
        raise ValueError("temperature and reward_tilt_beta must be positive")
    if require_teacher_target_top1 and teacher_target_max_rank not in (None, 1):
        raise ValueError(
            "require_teacher_target_top1 conflicts with teacher_target_max_rank > 1"
        )

    student_logits = student_scores.float() / temperature
    teacher_logits = teacher_scores.detach().float() / temperature
    rankings, old_step_log_probabilities, valid_step_mask = (
        sample_causal_plackett_luce_rollouts(
            student_logits,
            rollout_count=rollout_count,
            max_steps=rollout_top_k,
        )
    )
    current_step_log_probabilities = causal_plackett_luce_step_log_probabilities(
        student_logits,
        rankings=rankings,
        valid_step_mask=valid_step_mask,
    )
    teacher_step_log_probabilities = causal_plackett_luce_step_log_probabilities(
        teacher_logits,
        rankings=rankings,
        valid_step_mask=valid_step_mask,
    )
    rewards, target_ranks = target_ndcg_rewards(rankings)
    step_mask_float = valid_step_mask.to(current_step_log_probabilities.dtype)
    old_sequence_log_probabilities = (
        old_step_log_probabilities.detach() * step_mask_float
    ).sum(dim=-1)
    current_sequence_log_probabilities = (
        current_step_log_probabilities * step_mask_float
    ).sum(dim=-1)
    teacher_sequence_log_probabilities = (
        teacher_step_log_probabilities * step_mask_float
    ).sum(dim=-1)

    target_group_logits = (
        teacher_sequence_log_probabilities
        - old_sequence_log_probabilities
        + rewards / reward_tilt_beta
    ).detach()
    student_group_logits = (
        current_sequence_log_probabilities - old_sequence_log_probabilities
    )
    target_group_log_probabilities = F.log_softmax(target_group_logits, dim=1)
    target_group_probabilities = target_group_log_probabilities.exp()
    student_group_log_probabilities = F.log_softmax(student_group_logits, dim=1)
    student_group_probabilities = student_group_log_probabilities.exp()
    per_query_reverse_kl = (
        student_group_probabilities
        * (student_group_log_probabilities - target_group_log_probabilities)
    ).sum(dim=1)

    teacher_target_rank = 1 + (
        teacher_scores[:, 1:] > teacher_scores[:, :1]
    ).sum(dim=1)
    valid = torch.ones(
        student_scores.shape[0], dtype=torch.bool, device=student_scores.device
    )
    if require_teacher_target_top1:
        valid &= teacher_target_rank <= 1
    elif teacher_target_max_rank is not None:
        valid &= teacher_target_rank <= teacher_target_max_rank
    loss = (
        per_query_reverse_kl[valid].mean()
        if bool(valid.any())
        else student_scores.sum() * 0.0
    )

    with torch.no_grad():
        target_entropy = -(
            target_group_probabilities * target_group_log_probabilities
        ).sum(dim=1)
        target_ess = target_group_probabilities.square().sum(dim=1).reciprocal()
        student_initial_probabilities = F.softmax(student_logits, dim=1)
        student_initial_log_probabilities = F.log_softmax(student_logits, dim=1)
        metrics = {
            "loss": loss.detach(),
            "reward_tilted_reverse_opd_loss": loss.detach(),
            "reward_tilt_beta": student_scores.new_tensor(reward_tilt_beta),
            "valid_teacher_rate": valid.float().mean(),
            "rollout_reward_mean": rewards.mean(),
            "rollout_reward_std": rewards.std(dim=1, unbiased=False).mean(),
            "sampled_target_recall_at_k": (
                target_ranks <= rollout_top_k
            ).float().mean(),
            "sampled_target_rank_with_miss": target_ranks.float().mean(),
            "tilted_target_entropy": target_entropy.mean(),
            "tilted_target_effective_sample_size": target_ess.mean(),
            "tilted_target_max_probability": (
                target_group_probabilities.max(dim=1).values.mean()
            ),
            "student_group_entropy": -(
                student_group_probabilities * student_group_log_probabilities
            ).sum(dim=1).mean(),
            "teacher_rollout_log_probability": (
                teacher_sequence_log_probabilities.mean()
            ),
            "old_rollout_log_probability": old_sequence_log_probabilities.mean(),
            "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean(),
            "student_target_top1_rate": (
                student_scores.argmax(dim=1) == 0
            ).float().mean(),
            "teacher_target_top1_rate": (
                teacher_target_rank == 1
            ).float().mean(),
            "teacher_target_top10_rate": (
                teacher_target_rank <= min(10, teacher_scores.shape[1])
            ).float().mean(),
            "teacher_target_mean_rank": teacher_target_rank.float().mean(),
            "policy_target_probability": student_initial_probabilities[:, 0].mean(),
            "policy_entropy": -(
                student_initial_probabilities
                * student_initial_log_probabilities
            ).sum(dim=1).mean(),
        }
    return loss, metrics


def listwise_group_relative_loss(
    *,
    student_scores: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    temperature: float,
    rollout_count: int,
    rollout_top_k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize sampled target NDCG with group-normalized REINFORCE."""

    if student_scores.ndim != 2 or student_scores.shape[1] < 2:
        raise ValueError(
            "listwise group-relative policy loss requires [batch, candidates>=2] scores"
        )
    if raw_target_in_topk.shape != (student_scores.shape[0],):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if temperature <= 0:
        raise ValueError("policy sampling temperature must be positive")

    logits = student_scores.float() / temperature
    rankings, sequence_log_probabilities = sample_plackett_luce_rankings(
        logits,
        rollout_count=rollout_count,
        top_k=rollout_top_k,
    )
    rewards, target_ranks = target_ndcg_rewards(rankings)
    group_reward_mean = rewards.mean(dim=1, keepdim=True)
    raw_advantages = rewards - group_reward_mean
    reward_std_per_query = rewards.std(dim=1, unbiased=False)
    effective_query = reward_std_per_query > 1e-8
    advantages = torch.where(
        effective_query[:, None],
        raw_advantages / reward_std_per_query.clamp_min(1e-8)[:, None],
        torch.zeros_like(raw_advantages),
    )
    loss = -(
        advantages.detach() * sequence_log_probabilities
    ).mean()

    with torch.no_grad():
        initial_probabilities = F.softmax(logits, dim=-1)
        initial_log_probabilities = F.log_softmax(logits, dim=-1)
        policy_entropy = -(
            initial_probabilities * initial_log_probabilities
        ).sum(dim=-1).mean()
        unique_counts = torch.stack(
            [
                torch.tensor(
                    torch.unique(row, dim=0).shape[0],
                    device=student_scores.device,
                    dtype=torch.float32,
                )
                for row in rankings.detach()
            ]
        )
        student_top1 = student_scores.argmax(dim=1)
        student_target_margin = (
            student_scores[:, 0] - student_scores[:, 1:].amax(dim=1)
        )
        metrics = {
            "loss": loss.detach(),
            "listwise_group_relative_loss": loss.detach(),
            "rollout_reward_mean": rewards.mean(),
            "rollout_reward_std": reward_std_per_query.mean(),
            "effective_rl_query_rate": effective_query.float().mean(),
            "unique_rollout_count": unique_counts.mean(),
            "unique_rollout_rate": (unique_counts / float(rollout_count)).mean(),
            "sampled_target_recall_at_k": (
                target_ranks <= rollout_top_k
            ).float().mean(),
            "sampled_target_rank_with_miss": target_ranks.float().mean(),
            "policy_entropy": policy_entropy,
            "policy_target_probability": initial_probabilities[:, 0].mean(),
            "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean(),
            "student_target_top1_rate": (student_top1 == 0).float().mean(),
            "student_target_margin": student_target_margin.mean(),
        }
        for rank in range(1, rollout_top_k + 1):
            metrics[f"sampled_target_rank_{rank}_rate"] = (
                target_ranks == rank
            ).float().mean()
        metrics["sampled_target_miss_rate"] = (
            target_ranks > rollout_top_k
        ).float().mean()
    return loss, metrics


def on_policy_centered_distillation_loss(
    *,
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    teacher_target_top1: torch.Tensor,
    raw_target_in_topk: torch.Tensor,
    rank_weight: float,
    centered_response_weight: float,
    rank_temperature: float,
    centered_response_beta: float,
    require_teacher_target_top1: bool,
    teacher_target_max_rank: int | None = None,
    rank_kl_direction: str = "forward",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill Teacher ordering and centered score response on current Top-K.

    Candidate zero is always the ground-truth target.  The remaining candidates
    are the current Student's highest-scoring non-target retrievals.
    """

    if student_scores.shape != teacher_scores.shape or student_scores.ndim != 2:
        raise ValueError("Student and Teacher scores must have the same 2-D shape")
    if student_scores.shape[1] < 2:
        raise ValueError("on-policy KD requires at least two candidates")
    batch_size = student_scores.shape[0]
    if teacher_target_top1.shape != (batch_size,):
        raise ValueError("teacher_target_top1 shape does not match scores")
    if raw_target_in_topk.shape != (batch_size,):
        raise ValueError("raw_target_in_topk shape does not match scores")
    if rank_weight < 0 or centered_response_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if rank_weight == 0 and centered_response_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    if rank_kl_direction not in {"forward", "reverse"}:
        raise ValueError("rank_kl_direction must be 'forward' or 'reverse'")
    if teacher_target_max_rank is not None and teacher_target_max_rank < 1:
        raise ValueError("teacher_target_max_rank must be at least one")
    if require_teacher_target_top1 and teacher_target_max_rank not in (None, 1):
        raise ValueError(
            "require_teacher_target_top1 conflicts with teacher_target_max_rank > 1"
        )

    valid = torch.ones_like(teacher_target_top1, dtype=torch.bool)
    teacher_target_rank = 1 + (
        teacher_scores[:, 1:] > teacher_scores[:, :1]
    ).sum(dim=1)
    if require_teacher_target_top1:
        valid &= teacher_target_rank <= 1
    elif teacher_target_max_rank is not None:
        valid &= teacher_target_rank <= teacher_target_max_rank
    differentiable_zero = student_scores.sum() * 0.0

    rank_loss = differentiable_zero
    teacher_rank_entropy = differentiable_zero.detach()
    student_rank_entropy = differentiable_zero.detach()
    if rank_weight > 0 and bool(valid.any()):
        teacher_logits = teacher_scores[valid] / rank_temperature
        student_logits = student_scores[valid] / rank_temperature
        teacher_probabilities = F.softmax(teacher_logits, dim=1)
        teacher_log_probabilities = F.log_softmax(teacher_logits, dim=1)
        student_probabilities = F.softmax(student_logits, dim=1)
        student_log_probabilities = F.log_softmax(
            student_logits,
            dim=1,
        )
        teacher_rank_entropy = (
            -(teacher_probabilities * teacher_log_probabilities)
            .sum(dim=1)
            .mean()
            .detach()
        )
        student_rank_entropy = (
            -(student_probabilities * student_log_probabilities)
            .sum(dim=1)
            .mean()
            .detach()
        )
        if rank_kl_direction == "forward":
            # D_KL(p_teacher || p_student), the objective used by all existing
            # Rank-OPD checkpoints in this project.
            rank_loss = F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="batchmean",
            )
        else:
            # D_KL(p_student || p_teacher), exposed explicitly for the
            # MiniLLM-style reverse-KL control.
            rank_loss = F.kl_div(
                teacher_log_probabilities,
                student_probabilities,
                reduction="batchmean",
            )

    student_centered = centered_scores(student_scores)
    teacher_centered = centered_scores(teacher_scores)
    centered_loss = differentiable_zero
    centered_mae = differentiable_zero.detach()
    if centered_response_weight > 0 and bool(valid.any()):
        centered_loss = F.smooth_l1_loss(
            student_centered[valid],
            teacher_centered[valid],
            beta=centered_response_beta,
        )
        centered_mae = (
            student_centered[valid] - teacher_centered[valid]
        ).abs().mean().detach()

    total = rank_weight * rank_loss + centered_response_weight * centered_loss
    student_top1 = student_scores.argmax(dim=1)
    teacher_top1 = teacher_scores.argmax(dim=1)
    student_target_margin = student_scores[:, 0] - student_scores[:, 1:].amax(dim=1)
    teacher_target_margin = teacher_scores[:, 0] - teacher_scores[:, 1:].amax(dim=1)
    metrics = {
        "loss": total.detach(),
        "rank_loss": rank_loss.detach(),
        "teacher_rank_entropy": teacher_rank_entropy,
        "student_rank_entropy": student_rank_entropy,
        "rank_kl_is_reverse": torch.tensor(
            float(rank_kl_direction == "reverse"), device=student_scores.device
        ),
        "centered_response_loss": centered_loss.detach(),
        "centered_response_mae": centered_mae,
        "valid_teacher_rate": valid.float().mean().detach(),
        "raw_mining_target_recall_at_k": raw_target_in_topk.float().mean().detach(),
        "student_target_top1_rate": (student_top1 == 0).float().mean().detach(),
        "teacher_target_top1_rate": (teacher_top1 == 0).float().mean().detach(),
        "teacher_target_top10_rate": (
            teacher_target_rank <= min(10, student_scores.shape[1])
        ).float().mean().detach(),
        "teacher_target_mean_rank": teacher_target_rank.float().mean().detach(),
        "student_teacher_top1_agreement": (
            student_top1 == teacher_top1
        ).float().mean().detach(),
        "student_target_margin": student_target_margin.mean().detach(),
        "teacher_target_margin": teacher_target_margin.mean().detach(),
    }
    return total, metrics
