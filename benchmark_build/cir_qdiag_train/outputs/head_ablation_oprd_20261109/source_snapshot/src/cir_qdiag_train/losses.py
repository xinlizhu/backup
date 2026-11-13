from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def composed_infonce_loss(
    *,
    positive_queries: torch.Tensor,
    target_features: torch.Tensor,
    negative_queries: torch.Tensor | None,
    negative_owner: torch.Tensor,
    scale: torch.Tensor,
    hard_negative_weight: float,
    hard_negative_margin: float = 0.0,
    loss_mode: str = "batch_plus_hard",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute batch InfoNCE plus hard loss, or optimize the hard loss alone."""

    if loss_mode not in {"batch_plus_hard", "hard_only"}:
        raise ValueError(f"unsupported loss_mode: {loss_mode}")

    if loss_mode == "hard_only":
        # Keep batch metrics for diagnosis, but detach this branch so ordinary
        # in-batch negatives cannot affect optimization.
        with torch.no_grad():
            batch_loss, batch_accuracy = _batch_infonce_loss(
                positive_queries=positive_queries.detach(),
                target_features=target_features.detach(),
                scale=scale.detach(),
            )
    else:
        batch_loss, batch_accuracy = _batch_infonce_loss(
            positive_queries=positive_queries,
            target_features=target_features,
            scale=scale,
        )

    hard_loss = positive_queries.new_zeros(())
    hard_accuracy = positive_queries.new_ones(())
    hard_margin_accuracy = positive_queries.new_ones(())
    hard_mean_gap = positive_queries.new_zeros(())
    if negative_queries is not None and negative_queries.shape[0] > 0:
        (
            hard_loss,
            hard_accuracy,
            hard_margin_accuracy,
            hard_mean_gap,
        ) = _hard_negative_loss(
            positive_queries=positive_queries,
            target_features=target_features,
            negative_queries=negative_queries,
            negative_owner=negative_owner,
            scale=scale,
            margin=hard_negative_margin,
        )
    elif loss_mode == "hard_only":
        raise ValueError("hard_only loss received a batch with no hard negatives")

    total = (
        hard_loss
        if loss_mode == "hard_only"
        else batch_loss + hard_negative_weight * hard_loss
    )
    return total, {
        "loss": total.detach(),
        "batch_loss": batch_loss.detach(),
        "hard_loss": hard_loss.detach(),
        "batch_accuracy": batch_accuracy.detach(),
        "hard_accuracy": hard_accuracy.detach(),
        "hard_margin_accuracy": hard_margin_accuracy.detach(),
        "hard_mean_gap": hard_mean_gap.detach(),
    }


def typed_candidate_counterfactual_loss(
    *,
    positive_queries: torch.Tensor,
    target_features: torch.Tensor,
    negative_queries: torch.Tensor | None,
    negative_owner: torch.Tensor,
    negative_semantic_class: torch.Tensor,
    candidate_features: torch.Tensor | None,
    candidate_owner: torch.Tensor,
    scale: torch.Tensor,
    candidate_negative_weight: float,
    counterfactual_square_weight: float,
    candidate_margin: float,
    counterfactual_margin: float,
    mining_min_advantage: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mine typed image candidates and optimize forward and 2x2 matching orders.

    The ordinary candidate term ranks each positive target above the hardest image-set
    distractor. The square term chooses, per owner and O/A/P class, the (negative query,
    candidate image) pair with the largest counterfactual affinity
    ``sim(q-, z-) - sim(q+, z-)`` and enforces the correct diagonal of the 2x2 score
    matrix to beat the crossed diagonal.
    """

    zero = positive_queries.new_zeros(())
    candidate_loss = zero
    square_loss = zero
    candidate_accuracy = positive_queries.new_tensor(1.0)
    candidate_margin_accuracy = positive_queries.new_tensor(1.0)
    square_accuracy = positive_queries.new_tensor(1.0)
    square_margin_accuracy = positive_queries.new_tensor(1.0)
    mining_coverage = positive_queries.new_tensor(0.0)
    mined_advantage_mean = zero
    candidate_relation_count = 0
    typed_relation_count = 0
    typed_attempt_count = 0
    class_attempts = [0, 0, 0]
    class_selected = [0, 0, 0]

    if candidate_features is not None and candidate_features.shape[0] > 0:
        if candidate_owner.shape != (candidate_features.shape[0],):
            raise ValueError("candidate owners do not match candidate features")
        if candidate_owner.min() < 0 or candidate_owner.max() >= positive_queries.shape[0]:
            raise ValueError("candidate owner is out of range")

        positive_scores = scale * (positive_queries * target_features).sum(dim=-1)
        hardest_candidate_scores: list[torch.Tensor] = []
        for owner in range(positive_queries.shape[0]):
            owned_candidates = torch.nonzero(
                candidate_owner == owner, as_tuple=False
            ).flatten()
            if not owned_candidates.numel():
                continue
            scores = scale * (
                candidate_features[owned_candidates] * positive_queries[owner]
            ).sum(dim=-1)
            hardest_candidate_scores.append(scores.max())
        if hardest_candidate_scores:
            owners_with_candidates = torch.tensor(
                [
                    owner
                    for owner in range(positive_queries.shape[0])
                    if bool((candidate_owner == owner).any())
                ],
                dtype=torch.long,
                device=positive_queries.device,
            )
            hardest = torch.stack(hardest_candidate_scores)
            clean = positive_scores[owners_with_candidates]
            candidate_relations = hardest - clean + scale * candidate_margin
            candidate_loss = F.softplus(candidate_relations).mean()
            candidate_accuracy = (clean > hardest).float().mean()
            candidate_margin_accuracy = (
                clean >= hardest + scale * candidate_margin
            ).float().mean()
            candidate_relation_count = int(hardest.numel())

        if negative_queries is not None and negative_queries.shape[0] > 0:
            if negative_owner.shape != (negative_queries.shape[0],):
                raise ValueError("negative owners do not match negative queries")
            if negative_semantic_class.shape != negative_owner.shape:
                raise ValueError("negative semantic classes do not match owners")
            if negative_semantic_class.min() < 0 or negative_semantic_class.max() > 2:
                raise ValueError("negative semantic class is out of range")

            square_gaps: list[torch.Tensor] = []
            selected_advantages: list[torch.Tensor] = []
            for owner in range(positive_queries.shape[0]):
                owned_candidates = torch.nonzero(
                    candidate_owner == owner, as_tuple=False
                ).flatten()
                if not owned_candidates.numel():
                    continue
                for semantic_class in range(3):
                    owned_negatives = torch.nonzero(
                        (negative_owner == owner)
                        & (negative_semantic_class == semantic_class),
                        as_tuple=False,
                    ).flatten()
                    if not owned_negatives.numel():
                        continue
                    typed_attempt_count += 1
                    class_attempts[semantic_class] += 1
                    candidates = candidate_features[owned_candidates]
                    negatives = negative_queries[owned_negatives]
                    positive_candidate_scores = positive_queries[owner] @ candidates.t()
                    negative_candidate_scores = negatives @ candidates.t()
                    affinity = negative_candidate_scores - positive_candidate_scores
                    flat_index = int(affinity.detach().argmax().item())
                    negative_index = flat_index // candidates.shape[0]
                    candidate_index = flat_index % candidates.shape[0]
                    advantage = affinity[negative_index, candidate_index]
                    if float(advantage.detach()) < mining_min_advantage:
                        continue
                    typed_relation_count += 1
                    class_selected[semantic_class] += 1
                    negative_query = negatives[negative_index]
                    candidate = candidates[candidate_index]
                    diagonal = (
                        (positive_queries[owner] * target_features[owner]).sum()
                        + (negative_query * candidate).sum()
                    )
                    crossed = (
                        (positive_queries[owner] * candidate).sum()
                        + (negative_query * target_features[owner]).sum()
                    )
                    square_gaps.append(diagonal - crossed)
                    selected_advantages.append(advantage)

            if square_gaps:
                gaps = torch.stack(square_gaps)
                square_loss = F.softplus(
                    scale * (counterfactual_margin - gaps)
                ).mean()
                square_accuracy = (gaps > 0).float().mean()
                square_margin_accuracy = (gaps >= counterfactual_margin).float().mean()
                mined_advantage_mean = torch.stack(selected_advantages).mean().detach()
            if typed_attempt_count:
                mining_coverage = positive_queries.new_tensor(
                    typed_relation_count / typed_attempt_count
                )

    total = (
        candidate_negative_weight * candidate_loss
        + counterfactual_square_weight * square_loss
    )
    metrics = {
        "candidate_loss": candidate_loss.detach(),
        "candidate_accuracy": candidate_accuracy.detach(),
        "candidate_margin_accuracy": candidate_margin_accuracy.detach(),
        "candidate_relation_count": positive_queries.new_tensor(
            float(candidate_relation_count)
        ),
        "counterfactual_square_loss": square_loss.detach(),
        "counterfactual_square_accuracy": square_accuracy.detach(),
        "counterfactual_square_margin_accuracy": square_margin_accuracy.detach(),
        "typed_candidate_attempt_count": positive_queries.new_tensor(
            float(typed_attempt_count)
        ),
        "typed_candidate_relation_count": positive_queries.new_tensor(
            float(typed_relation_count)
        ),
        "typed_candidate_mining_coverage": mining_coverage.detach(),
        "typed_candidate_advantage": mined_advantage_mean,
    }
    for semantic_class, label in enumerate(("object", "attribute", "predicate")):
        attempts = class_attempts[semantic_class]
        selected = class_selected[semantic_class]
        metrics[f"{label}_candidate_mining_coverage"] = positive_queries.new_tensor(
            selected / attempts if attempts else 0.0
        )
    return total, metrics


def _batch_infonce_loss(
    *,
    positive_queries: torch.Tensor,
    target_features: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_positive_queries = _all_gather_with_grad(positive_queries)
    all_targets = _all_gather_with_grad(target_features)
    batch_size = positive_queries.shape[0]
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    labels = rank * batch_size + torch.arange(
        batch_size,
        device=positive_queries.device,
    )
    query_to_target_logits = scale * positive_queries @ all_targets.t()
    target_to_query_logits = scale * target_features @ all_positive_queries.t()
    query_to_target = F.cross_entropy(query_to_target_logits, labels)
    target_to_query = F.cross_entropy(target_to_query_logits, labels)
    loss = 0.5 * (query_to_target + target_to_query)
    with torch.no_grad():
        accuracy = (query_to_target_logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def _hard_negative_loss(
    *,
    positive_queries: torch.Tensor,
    target_features: torch.Tensor,
    negative_queries: torch.Tensor,
    negative_owner: torch.Tensor,
    scale: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = positive_queries.shape[0]
    positive_scores = scale * (positive_queries * target_features).sum(dim=-1)
    negative_scores = scale * (
        negative_queries * target_features[negative_owner]
    ).sum(dim=-1)
    counts = torch.bincount(negative_owner, minlength=batch_size)
    max_negatives = int(counts.max().item())
    logits = positive_scores.new_full(
        (batch_size, max_negatives + 1),
        -torch.inf,
    )
    logits[:, 0] = positive_scores
    next_slot = torch.zeros(batch_size, dtype=torch.long, device=positive_queries.device)
    for negative_index, owner in enumerate(negative_owner.tolist()):
        slot = int(next_slot[owner].item()) + 1
        logits[owner, slot] = negative_scores[negative_index]
        next_slot[owner] += 1
    valid = counts > 0
    valid_logits = logits[valid]
    margin_logits = valid_logits.clone()
    # An additive cosine margin makes the target-conditioned rejection explicit:
    # every q+ score must exceed every owned q- score by at least `margin`.
    margin_logits[:, 0] -= scale * margin
    labels = torch.zeros(valid_logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(margin_logits, labels)
    accuracy = (valid_logits.argmax(dim=1) == 0).float().mean()
    hardest_negative = valid_logits[:, 1:].max(dim=1).values
    scaled_gaps = valid_logits[:, 0] - hardest_negative
    margin_accuracy = (scaled_gaps >= scale * margin).float().mean()
    mean_gap = (scaled_gaps / scale).mean()
    return loss, accuracy, margin_accuracy, mean_gap


def _all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor), dim=0)
