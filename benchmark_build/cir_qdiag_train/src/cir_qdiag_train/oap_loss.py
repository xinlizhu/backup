from __future__ import annotations

import torch
import torch.distributed as dist

from cir_qdiag_train.oap_model import NUM_SEMANTIC_COMPONENTS, OAPSemanticModel


def oap_ordering_loss(
    *,
    model: OAPSemanticModel,
    positive_global: torch.Tensor,
    positive_semantic: torch.Tensor,
    target_global: torch.Tensor,
    target_semantic: torch.Tensor,
    negative_semantic: torch.Tensor | None,
    negative_owner: torch.Tensor,
    negative_semantic_class: torch.Tensor,
    temperature: float,
    retrieval_margin: float,
    patch_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One log-sum-exp over retrieval and typed semantic-patching orderings."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = positive_global.shape[0]
    all_target_global = _all_gather_without_grad(target_global)
    all_target_semantic = _all_gather_with_grad(target_semantic)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    labels = rank * batch_size + torch.arange(
        batch_size, device=positive_global.device
    )

    final_gallery = model.pairwise_score(
        positive_global,
        positive_semantic,
        all_target_global,
        all_target_semantic,
    ).float()
    global_gallery = (positive_global @ all_target_global.t()).float()
    positive_own = final_gallery.gather(1, labels[:, None]).squeeze(1)
    gallery_mask = torch.ones_like(final_gallery, dtype=torch.bool)
    gallery_mask.scatter_(1, labels[:, None], False)

    patched_scores = positive_own.new_empty((0,))
    if negative_semantic is not None and negative_semantic.shape[0] > 0:
        if negative_owner.numel() != negative_semantic.shape[0]:
            raise ValueError("negative owner count does not match negative embeddings")
        if negative_semantic_class.shape != negative_owner.shape:
            raise ValueError("negative semantic labels do not match owners")
        if (
            negative_semantic_class.min() < 0
            or negative_semantic_class.max() >= NUM_SEMANTIC_COMPONENTS
        ):
            raise ValueError("invalid O/A/P semantic class")
        indices = torch.arange(negative_semantic.shape[0], device=positive_global.device)
        classes = negative_semantic_class
        owned_target_semantic = target_semantic[negative_owner]
        clean_components = (
            positive_semantic[negative_owner] * owned_target_semantic
        ).sum(dim=-1)
        corrupted_components = (
            negative_semantic[indices, classes]
            * owned_target_semantic[indices, classes]
        ).sum(dim=-1)
        clean_selected = clean_components.gather(1, classes[:, None]).squeeze(1)
        patched_scores = positive_own[negative_owner] + (
            model.beta / NUM_SEMANTIC_COMPONENTS
        ) * (corrupted_components - clean_selected)

    sample_losses: list[torch.Tensor] = []
    relation_values: list[torch.Tensor] = []
    zero = positive_own.new_zeros(1)
    for sample_index in range(batch_size):
        relations = [
            final_gallery[sample_index, gallery_mask[sample_index]]
            - positive_own[sample_index]
            + retrieval_margin
        ]
        if patched_scores.numel():
            owned = torch.nonzero(
                negative_owner == sample_index, as_tuple=False
            ).flatten()
            if owned.numel():
                relations.append(
                    patched_scores[owned]
                    - positive_own[sample_index]
                    + patch_margin
                )
        values = torch.cat([relation.reshape(-1) for relation in relations])
        sample_losses.append(
            torch.logsumexp(torch.cat([zero, values / temperature]), dim=0)
        )
        relation_values.append(values.detach())

    loss = torch.stack(sample_losses).mean()
    all_relations = torch.cat(relation_values)
    with torch.no_grad():
        global_correct = global_gallery.argmax(dim=1) == labels
        final_correct = final_gallery.argmax(dim=1) == labels
        patch_correct = positive_own.new_tensor(1.0)
        patch_margin_correct = positive_own.new_tensor(1.0)
        if patched_scores.numel():
            clean_owned = positive_own[negative_owner]
            patch_correct = (clean_owned > patched_scores).float().mean()
            patch_margin_correct = (
                clean_owned >= patched_scores + patch_margin
            ).float().mean()

    metrics: dict[str, torch.Tensor] = {
        "loss": loss.detach(),
        "relation_count": positive_own.new_tensor(float(all_relations.numel())),
        "margin_violation_rate": (all_relations > 0).float().mean(),
        "global_batch_accuracy": global_correct.float().mean(),
        "final_batch_accuracy": final_correct.float().mean(),
        "patch_accuracy": patch_correct,
        "patch_margin_accuracy": patch_margin_correct,
        "top1_correction_rate": (
            ((~global_correct) & final_correct).float().sum()
            / (~global_correct).float().sum().clamp_min(1.0)
        ),
        "top1_regression_rate": (
            (global_correct & (~final_correct)).float().sum()
            / global_correct.float().sum().clamp_min(1.0)
        ),
    }
    if patched_scores.numel():
        clean_owned = positive_own[negative_owner]
        for class_id, label in enumerate(("object", "attribute", "predicate")):
            mask = negative_semantic_class == class_id
            metrics[f"{label}_patch_count"] = mask.float().sum()
            metrics[f"{label}_patch_accuracy"] = (
                (clean_owned[mask] > patched_scores[mask]).float().mean()
                if mask.any()
                else positive_own.new_zeros(())
            )
    return loss, metrics


def _all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(tensor), dim=0)


def _all_gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)
