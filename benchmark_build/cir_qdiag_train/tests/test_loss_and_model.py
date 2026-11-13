import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cir_qdiag_train.losses import (
    composed_infonce_loss,
    typed_candidate_counterfactual_loss,
)
from cir_qdiag_train.distillation_loss import semantic_ability_distillation_loss
from cir_qdiag_train.on_policy_loss import (
    candidate_infonce_loss,
    causal_group_policy_loss,
    causal_prefix_anchor_kl,
    causal_plackett_luce_step_log_probabilities,
    centered_similarity_mse_loss,
    centered_scores,
    listwise_group_relative_loss,
    on_policy_centered_distillation_loss,
    reward_tilted_reverse_opd_loss,
    sample_causal_plackett_luce_rollouts,
    sample_plackett_luce_rankings,
    target_ndcg_rewards,
)
from cir_qdiag_train.on_policy_mining import (
    _mine_teacher_topk,
    _sample_random_candidates,
    force_target_into_topk,
)
from cir_qdiag_train.on_policy_trainer import (
    _compute_live_student_scores,
    _materialize_selected_candidate_batch,
    _select_live_topk_from_reservoir,
)
from cir_qdiag_train.pixel_cache import build_gallery_pixel_cache
from cir_qdiag_train.model import (
    ComposedRetrievalModel,
    LoRALinear,
    _validate_clip_backbone,
    load_checkpoint,
    save_checkpoint,
)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_projection = nn.Linear(4, 4, bias=False)
        self.text_embedding = nn.Embedding(20, 4)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_projection(pixel_values)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.text_embedding(input_ids).mean(dim=1)


class FakeTrainableBackbone(FakeBackbone):
    # Match CLIP's encoder discovery without loading any model weights.
    @property
    def vision_model(self):
        return self.image_projection

    @property
    def text_model(self):
        return self.text_embedding


class FakeImageProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = []
        for image in images:
            red, green, blue = image.getpixel((0, 0))
            values.append([red / 255.0, green / 255.0, blue / 255.0, 1.0])
        return {"pixel_values": torch.tensor(values, dtype=torch.float32)}


def test_clip_b_backbone_validation_defaults_to_no_custom_code() -> None:
    backbone = SimpleNamespace(
        config=SimpleNamespace(
            model_type="clip",
            auto_map=None,
            projection_dim=512,
            vision_config=SimpleNamespace(
                hidden_size=768,
                num_hidden_layers=12,
                patch_size=16,
            ),
        )
    )
    _validate_clip_backbone(backbone, "clip_vit_b16")


def test_centered_scores_remove_per_query_similarity_offsets() -> None:
    scores = torch.tensor([[0.1, 0.4, -0.2], [1.0, 2.0, 4.0]])
    offsets = torch.tensor([[7.0], [-3.5]])
    assert torch.allclose(centered_scores(scores), centered_scores(scores + offsets))
    assert torch.allclose(
        centered_scores(scores).mean(dim=1), torch.zeros(2), atol=1e-7
    )


def test_on_policy_centered_loss_uses_forced_target_and_backpropagates() -> None:
    student = torch.tensor(
        [[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], requires_grad=True
    )
    teacher = torch.tensor([[0.6, 0.1, 0.0], [0.5, 0.3, 0.1]])
    loss, metrics = on_policy_centered_distillation_loss(
        student_scores=student,
        teacher_scores=teacher,
        teacher_target_top1=torch.tensor([True, True]),
        raw_target_in_topk=torch.tensor([True, False]),
        rank_weight=1.0,
        centered_response_weight=1.0,
        rank_temperature=0.05,
        centered_response_beta=0.01,
        require_teacher_target_top1=True,
    )
    assert torch.isfinite(loss)
    assert metrics["raw_mining_target_recall_at_k"].item() == 0.5
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad != 0).any())


def test_on_policy_centered_loss_keeps_teacher_target_up_to_configured_rank() -> None:
    student = torch.tensor(
        [[0.2, 0.3, 0.1], [0.2, 0.3, 0.4]], requires_grad=True
    )
    # Candidate zero is the target: its Teacher ranks are respectively 2 and 3.
    teacher = torch.tensor([[0.5, 0.6, 0.1], [0.1, 0.3, 0.2]])
    loss, metrics = on_policy_centered_distillation_loss(
        student_scores=student,
        teacher_scores=teacher,
        teacher_target_top1=torch.tensor([False, False]),
        raw_target_in_topk=torch.tensor([True, True]),
        rank_weight=1.0,
        centered_response_weight=1.0,
        rank_temperature=0.05,
        centered_response_beta=0.01,
        require_teacher_target_top1=False,
        teacher_target_max_rank=2,
    )
    assert torch.isfinite(loss)
    assert metrics["valid_teacher_rate"].item() == 0.5
    assert metrics["teacher_target_mean_rank"].item() == 2.5
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad[0] != 0).any())
    assert bool((student.grad[1] == 0).all())


def test_on_policy_rank_reverse_kl_matches_student_to_teacher_direction() -> None:
    student = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
    teacher = torch.tensor([[0.1, 0.4, -0.1]])
    temperature = 0.2
    loss, metrics = on_policy_centered_distillation_loss(
        student_scores=student,
        teacher_scores=teacher,
        teacher_target_top1=torch.tensor([False]),
        raw_target_in_topk=torch.tensor([True]),
        rank_weight=1.0,
        centered_response_weight=0.0,
        rank_temperature=temperature,
        centered_response_beta=0.01,
        require_teacher_target_top1=False,
        rank_kl_direction="reverse",
    )
    student_log = F.log_softmax(student / temperature, dim=1)
    teacher_log = F.log_softmax(teacher / temperature, dim=1)
    expected = (
        student_log.exp() * (student_log - teacher_log)
    ).sum(dim=1).mean()
    assert torch.allclose(loss, expected)
    assert metrics["rank_kl_is_reverse"].item() == 1.0
    loss.backward()
    assert student.grad is not None


def test_plackett_luce_rollouts_do_not_repeat_candidates_within_a_list() -> None:
    torch.manual_seed(7)
    logits = torch.zeros(2, 5, requires_grad=True)
    rankings, log_probabilities = sample_plackett_luce_rankings(
        logits,
        rollout_count=8,
        top_k=3,
    )
    assert rankings.shape == (2, 8, 3)
    assert log_probabilities.shape == (2, 8)
    assert all(
        torch.unique(rollout).numel() == 3
        for query in rankings
        for rollout in query
    )
    expected_log_probability = -sum(torch.log(torch.tensor([5.0, 4.0, 3.0])))
    assert torch.allclose(
        log_probabilities,
        torch.full_like(log_probabilities, expected_log_probability),
    )


def test_causal_rollout_stops_at_target_and_uses_negative_one_suffix() -> None:
    logits = torch.tensor([[1000.0, -1000.0, -1000.0, -1000.0]])
    rankings, step_log_probabilities, valid_step_mask = (
        sample_causal_plackett_luce_rollouts(
            logits,
            rollout_count=8,
            max_steps=3,
        )
    )
    assert torch.equal(rankings[..., 0], torch.zeros(1, 8, dtype=torch.long))
    assert bool((rankings[..., 1:] == -1).all())
    assert bool(valid_step_mask[..., 0].all())
    assert not bool(valid_step_mask[..., 1:].any())
    assert torch.equal(
        step_log_probabilities[..., 1:],
        torch.zeros_like(step_log_probabilities[..., 1:]),
    )


def test_causal_rollout_rescoring_matches_collection_policy() -> None:
    torch.manual_seed(19)
    logits = torch.randn(2, 6, requires_grad=True)
    rankings, collected_log_probabilities, valid_step_mask = (
        sample_causal_plackett_luce_rollouts(
            logits,
            rollout_count=8,
            max_steps=4,
        )
    )
    rescored = causal_plackett_luce_step_log_probabilities(
        logits,
        rankings=rankings,
        valid_step_mask=valid_step_mask,
    )
    assert torch.allclose(rescored, collected_log_probabilities)
    assert torch.isfinite(rescored).all()


def test_causal_reinforce_has_variable_prefix_gradient_and_diagnostics() -> None:
    torch.manual_seed(23)
    student = torch.zeros(4, 8, requires_grad=True)
    loss, rollouts, metrics = causal_group_policy_loss(
        student_scores=student,
        raw_target_in_topk=torch.tensor([True, False, True, False]),
        temperature=1.0,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="reinforce",
        clip_epsilon=None,
    )
    assert torch.isfinite(loss)
    assert bool((rollouts.rankings[~rollouts.valid_step_mask] == -1).all())
    assert metrics["causal_suffix_masked_steps"].item() > 0
    assert metrics["causal_prefix_mean_length"].item() < 5
    assert metrics["effective_rl_query_rate"].item() > 0
    assert metrics["importance_ratio_mean"].item() == 1.0
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad != 0).any())


def test_causal_grpo_and_gspo_reuse_old_policy_with_expected_ratio_granularity() -> None:
    torch.manual_seed(29)
    old_scores = torch.zeros(3, 7, requires_grad=True)
    _, rollouts, first_metrics = causal_group_policy_loss(
        student_scores=old_scores,
        raw_target_in_topk=torch.ones(3, dtype=torch.bool),
        temperature=1.0,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="gspo",
        clip_epsilon=0.2,
    )
    assert first_metrics["importance_ratio_mean"].item() == 1.0
    assert first_metrics["clip_fraction"].item() == 0.0

    changed_scores_grpo = torch.tensor(
        [[2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0]] * 3,
        requires_grad=True,
    )
    grpo_loss, _, grpo_metrics = causal_group_policy_loss(
        student_scores=changed_scores_grpo,
        raw_target_in_topk=torch.ones(3, dtype=torch.bool),
        temperature=1.0,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="grpo",
        clip_epsilon=0.2,
        rollouts=rollouts,
    )
    assert grpo_metrics["clip_fraction"].item() > 0
    assert grpo_metrics["importance_ratio_min"].item() < 1
    assert grpo_metrics["importance_ratio_max"].item() > 1
    grpo_loss.backward()
    assert changed_scores_grpo.grad is not None

    changed_scores_gspo = changed_scores_grpo.detach().clone().requires_grad_()
    gspo_loss, _, gspo_metrics = causal_group_policy_loss(
        student_scores=changed_scores_gspo,
        raw_target_in_topk=torch.ones(3, dtype=torch.bool),
        temperature=1.0,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="gspo",
        clip_epsilon=None,
        rollouts=rollouts,
    )
    assert gspo_metrics["clip_fraction"].item() == 0.0
    assert gspo_metrics["importance_ratio_min"].item() < 1
    assert gspo_metrics["importance_ratio_max"].item() > 1
    gspo_loss.backward()
    assert changed_scores_gspo.grad is not None


def test_reward_tilted_reverse_opd_is_direct_reverse_kl_with_gradient() -> None:
    torch.manual_seed(30)
    student = torch.zeros(4, 8, requires_grad=True)
    teacher = torch.tensor(
        [[1.5, 0.8, 0.5, 0.2, 0.0, -0.2, -0.5, -0.8]] * 4
    )
    loss, metrics = reward_tilted_reverse_opd_loss(
        student_scores=student,
        teacher_scores=teacher,
        raw_target_in_topk=torch.ones(4, dtype=torch.bool),
        temperature=0.5,
        rollout_count=32,
        rollout_top_k=5,
        reward_tilt_beta=0.1,
        teacher_target_max_rank=5,
    )
    assert torch.isfinite(loss)
    assert metrics["valid_teacher_rate"].item() == 1.0
    assert 1.0 <= metrics["tilted_target_effective_sample_size"].item() <= 32.0
    assert 0.0 < metrics["tilted_target_max_probability"].item() <= 1.0
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad != 0).any())


def test_reward_tilted_reverse_kl_gspo_uses_teacher_shaped_reward() -> None:
    torch.manual_seed(32)
    student = torch.zeros(4, 8, requires_grad=True)
    teacher = torch.tensor(
        [[1.2, 0.9, 0.6, 0.3, 0.0, -0.3, -0.6, -0.9]] * 4
    )
    loss, rollouts, metrics = causal_group_policy_loss(
        student_scores=student,
        teacher_scores=teacher,
        raw_target_in_topk=torch.ones(4, dtype=torch.bool),
        temperature=0.5,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="gspo",
        clip_epsilon=0.2,
        reward_tilt_beta=0.1,
    )
    assert torch.isfinite(loss)
    assert torch.isclose(
        metrics["reward_tilt_beta"], torch.tensor(0.1)
    )
    assert torch.isfinite(metrics["teacher_reverse_kl_sample"])
    assert metrics["effective_rl_query_rate"].item() > 0
    loss.backward()
    assert student.grad is not None

    changed = student.detach().clone()
    changed[:, 0] += 1.0
    changed.requires_grad_()
    reused_loss, _, reused_metrics = causal_group_policy_loss(
        student_scores=changed,
        teacher_scores=teacher,
        raw_target_in_topk=torch.ones(4, dtype=torch.bool),
        temperature=0.5,
        rollout_count=32,
        rollout_top_k=5,
        algorithm="gspo",
        clip_epsilon=0.2,
        rollouts=rollouts,
        reward_tilt_beta=0.1,
    )
    assert reused_metrics["clip_fraction"].item() > 0
    reused_loss.backward()
    assert changed.grad is not None


def test_causal_prefix_anchor_kl_is_zero_at_anchor_and_penalizes_drift() -> None:
    torch.manual_seed(31)
    anchor_scores = torch.randn(3, 8)
    _, rollouts, _ = causal_group_policy_loss(
        student_scores=anchor_scores.clone().requires_grad_(),
        raw_target_in_topk=torch.ones(3, dtype=torch.bool),
        temperature=0.5,
        rollout_count=16,
        rollout_top_k=5,
        algorithm="gspo",
        clip_epsilon=0.2,
    )
    equal_scores = anchor_scores.clone().requires_grad_()
    equal_kl, _ = causal_prefix_anchor_kl(
        student_scores=equal_scores,
        anchor_scores=anchor_scores,
        rollouts=rollouts,
        temperature=0.5,
    )
    assert torch.allclose(equal_kl, torch.zeros_like(equal_kl), atol=1e-6)

    changed_scores = anchor_scores.clone()
    changed_scores[:, 0] += 2.0
    changed_scores.requires_grad_()
    changed_kl, metrics = causal_prefix_anchor_kl(
        student_scores=changed_scores,
        anchor_scores=anchor_scores,
        rollouts=rollouts,
        temperature=0.5,
    )
    assert changed_kl.item() > 0
    assert metrics["anchor_policy_kl_max_step"].item() >= changed_kl.item()
    changed_kl.backward()
    assert changed_scores.grad is not None
    assert bool((changed_scores.grad != 0).any())


def test_live_reservoir_screening_reforwards_only_selected_images(tmp_path) -> None:
    torch.manual_seed(37)
    image_paths = []
    for index in range(7):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", (2, 2), color=(20 * index, 10 * index, 5 * index)).save(path)
        image_paths.append(path)
    model = ComposedRetrievalModel(
        FakeTrainableBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        temperature=0.07,
        learnable_temperature=False,
    )
    processor = FakeImageProcessor()
    pixel_cache = build_gallery_pixel_cache(
        image_paths=image_paths,
        processor=processor,
        dtype=torch.float32,
        batch_size=3,
        workers=2,
        prefetch_factor=2,
    )
    assert len(pixel_cache) == len(image_paths)
    for path in image_paths:
        path.unlink()
    source_batch = {
        "pair_ids": [1, 2],
        "reference_indices": torch.tensor([5, 6]),
        "target_indices": torch.tensor([1, 2]),
        "reservoir_candidate_indices": torch.tensor(
            [[1, 0, 2, 3, 4], [2, 0, 1, 3, 4]]
        ),
        "raw_target_in_reservoir": torch.tensor([True, True]),
        # Scores encode image IDs, so any candidate/score misalignment is visible.
        "reservoir_teacher_scores": torch.tensor(
            [[0.1, 0.0, 0.2, 0.3, 0.4], [0.2, 0.0, 0.1, 0.3, 0.4]]
        ),
        "input_ids": torch.tensor([[1, 2], [2, 3]]),
        "attention_mask": torch.ones(2, 2, dtype=torch.long),
    }
    selected = _select_live_topk_from_reservoir(
        model=model,
        processor=processor,
        batch=source_batch,
        image_paths=image_paths,
        candidate_count=3,
        image_batch_size=4,
        mixed_precision="no",
        image_loader_workers=0,
        device=torch.device("cpu"),
        pixel_cache=pixel_cache,
        transfer_prefetch_batches=2,
        pin_memory=False,
    )
    assert selected["selected_candidate_indices"].shape == (2, 3)
    assert torch.allclose(
        selected["selected_teacher_scores"],
        selected["selected_candidate_indices"].float() / 10,
    )
    assert torch.equal(
        selected["selected_candidate_indices"][:, 0],
        source_batch["target_indices"],
    )
    training_batch = _materialize_selected_candidate_batch(
        processor=processor,
        source_batch=source_batch,
        selected_candidate_indices=selected["selected_candidate_indices"],
        selected_anchor_scores=None,
        selected_teacher_scores=selected["selected_teacher_scores"],
        raw_target_in_topk=selected["raw_target_in_topk"],
        image_paths=image_paths,
        preprocess_batch_size=4,
        image_loader_workers=0,
        device=torch.device("cpu"),
        pixel_cache=pixel_cache,
        transfer_prefetch_batches=2,
        pin_memory=False,
    )
    scores = _compute_live_student_scores(
        model=model,
        batch=training_batch,
        live_image_batch_size=4,
    )
    assert scores.shape == (2, 3)
    loss, _ = on_policy_centered_distillation_loss(
        student_scores=scores,
        teacher_scores=training_batch["teacher_scores"],
        teacher_target_top1=training_batch["teacher_target_top1"],
        raw_target_in_topk=training_batch["raw_target_in_topk"],
        rank_weight=1.0, centered_response_weight=0.0,
        rank_temperature=0.05, centered_response_beta=0.01,
        rank_kl_direction="reverse", require_teacher_target_top1=False,
    )
    loss.backward()
    assert model.fusion[-1].weight.grad is not None
    assert model.backbone.image_projection.weight.grad.abs().sum() > 0
    assert model.backbone.text_embedding.weight.grad.abs().sum() > 0


def test_target_ndcg_reward_uses_rank_and_zero_for_a_miss() -> None:
    rankings = torch.tensor([[[0, 1, 2], [1, 0, 2], [1, 2, 3]]])
    rewards, ranks = target_ndcg_rewards(rankings)
    assert torch.equal(ranks, torch.tensor([[1, 2, 4]]))
    assert torch.allclose(
        rewards,
        torch.tensor([[1.0, 1.0 / torch.log2(torch.tensor(3.0)), 0.0]]),
    )


def test_listwise_group_relative_loss_has_gradient_and_diagnostics() -> None:
    torch.manual_seed(11)
    student = torch.zeros(4, 6, requires_grad=True)
    loss, metrics = listwise_group_relative_loss(
        student_scores=student,
        raw_target_in_topk=torch.tensor([True, False, True, False]),
        temperature=1.0,
        rollout_count=16,
        rollout_top_k=3,
    )
    assert torch.isfinite(loss)
    assert metrics["effective_rl_query_rate"].item() > 0
    assert 0 < metrics["sampled_target_recall_at_k"].item() < 1
    assert 0 < metrics["unique_rollout_rate"].item() <= 1
    assert torch.allclose(
        metrics["sampled_target_miss_rate"],
        1.0 - metrics["sampled_target_recall_at_k"],
    )
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad != 0).any())


def test_candidate_infonce_uses_forced_target_without_teacher() -> None:
    student = torch.tensor(
        [[0.1, 0.3, 0.2], [0.5, 0.2, 0.1]], requires_grad=True
    )
    loss, metrics = candidate_infonce_loss(
        student_scores=student,
        raw_target_in_topk=torch.tensor([False, True]),
        temperature=0.05,
    )
    expected = F.cross_entropy(student / 0.05, torch.zeros(2, dtype=torch.long))
    assert torch.allclose(loss, expected)
    assert metrics["raw_mining_target_recall_at_k"].item() == 0.5
    assert metrics["student_target_top1_rate"].item() == 0.5
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad != 0).any())


def test_candidate_infonce_can_use_teacher_target_rank_filter() -> None:
    student = torch.tensor(
        [[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], requires_grad=True
    )
    # Candidate zero is the target. Its Teacher ranks are respectively 1 and 3.
    teacher = torch.tensor([[0.6, 0.1, 0.0], [0.1, 0.3, 0.2]])
    loss, metrics = candidate_infonce_loss(
        student_scores=student,
        teacher_scores=teacher,
        raw_target_in_topk=torch.tensor([True, False]),
        temperature=0.05,
        teacher_target_max_rank=2,
    )
    expected = F.cross_entropy(
        student[:1] / 0.05, torch.zeros(1, dtype=torch.long)
    )
    assert torch.allclose(loss, expected)
    assert metrics["valid_teacher_rate"].item() == 0.5
    assert metrics["teacher_target_mean_rank"].item() == 2.0
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad[0] != 0).any())
    assert bool((student.grad[1] == 0).all())


def test_centered_similarity_mse_matches_centered_residual_formula() -> None:
    student = torch.tensor(
        [[0.2, 0.4, 0.1], [0.3, 0.1, 0.5]], requires_grad=True
    )
    # The target at candidate zero has Teacher ranks 2 and 3. Only row zero
    # survives a target-rank <= 2 filter.
    teacher = torch.tensor([[0.5, 0.7, 0.1], [0.1, 0.2, 0.4]])
    loss, metrics = centered_similarity_mse_loss(
        student_scores=student,
        teacher_scores=teacher,
        raw_target_in_topk=torch.tensor([True, False]),
        teacher_target_max_rank=2,
    )
    residual = student[:1] - teacher[:1]
    expected = (residual - residual.mean(dim=1, keepdim=True)).square().mean()
    assert torch.allclose(loss, expected)
    assert torch.allclose(loss, torch.tensor(0.02))
    assert metrics["valid_teacher_rate"].item() == 0.5
    loss.backward()
    assert student.grad is not None
    assert bool((student.grad[0] != 0).any())
    assert bool((student.grad[1] == 0).all())


def test_force_target_into_topk_keeps_exact_k_and_target_at_zero() -> None:
    already_present, hit = force_target_into_topk(
        torch.tensor([8, 3, 5, 2]).numpy(), target_index=5, top_k=4
    )
    assert hit is True
    assert already_present.tolist() == [5, 8, 3, 2]
    inserted, hit = force_target_into_topk(
        torch.tensor([8, 3, 5, 2]).numpy(), target_index=9, top_k=4
    )
    assert hit is False
    assert inserted.tolist() == [9, 8, 3, 5]
    assert len(set(inserted.tolist())) == 4


def test_random_candidates_are_unique_and_exclude_reference_and_target() -> None:
    references = torch.tensor([1, 8, 4]).numpy()
    targets = torch.tensor([3, 2, 7]).numpy()
    first, hits = _sample_random_candidates(
        sample_count=3,
        gallery_count=12,
        reference_indices=references,
        target_indices=targets,
        top_k=5,
        seed=43,
    )
    repeated, _ = _sample_random_candidates(
        sample_count=3,
        gallery_count=12,
        reference_indices=references,
        target_indices=targets,
        top_k=5,
        seed=43,
    )
    assert torch.equal(torch.from_numpy(first), torch.from_numpy(repeated))
    assert not bool(hits.any())
    for row, reference, target in zip(first, references, targets, strict=True):
        assert row[0] == target
        assert len(set(row.tolist())) == 5
        assert reference not in row
        assert target not in row[1:]


def test_teacher_topk_uses_teacher_policy_and_forces_target(tmp_path) -> None:
    image_ids = ["i0", "i1", "i2", "i3", "i4"]
    pair_ids = [10, 11]
    gallery = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.0, 1.0]]
    ).numpy()
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).numpy()
    (tmp_path / "metadata.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    (tmp_path / "image_ids.json").write_text(
        json.dumps(image_ids), encoding="utf-8"
    )
    (tmp_path / "pair_ids.json").write_text(
        json.dumps(pair_ids), encoding="utf-8"
    )
    np.save(tmp_path / "gallery_features.npy", gallery)
    np.save(tmp_path / "query_features.npy", queries)
    config = SimpleNamespace(
        data=SimpleNamespace(teacher_cache_dir=tmp_path),
        mining=SimpleNamespace(top_k=3, query_batch_size=2),
    )
    records = [SimpleNamespace(pair_id=value) for value in pair_ids]
    candidates, hits = _mine_teacher_topk(
        config=config,
        records=records,
        image_ids=image_ids,
        reference_indices=np.asarray([0, 4]),
        target_indices=np.asarray([4, 0]),
        epoch=1,
        device=torch.device("cpu"),
    )
    assert candidates[:, 0].tolist() == [4, 0]
    assert candidates[0, 1:].tolist() == [1, 2]
    assert candidates[1, 1:].tolist() == [3, 2]
    assert hits.tolist() == [False, False]


class FakeEncoder(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeTransformerLayer() for _ in range(count)])


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)


class FakeTransformerLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = FakeAttention()


class FakeTower(nn.Module):
    def __init__(self, count: int, *, vision: bool) -> None:
        super().__init__()
        self.encoder = FakeEncoder(count)
        if vision:
            self.post_layernorm = nn.LayerNorm(4)
        else:
            self.final_layer_norm = nn.LayerNorm(4)


class FakeStandardCLIP(FakeBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = FakeTower(6, vision=True)
        self.text_model = FakeTower(5, vision=False)
        self.visual_projection = nn.Linear(4, 4, bias=False)
        self.text_projection = nn.Linear(4, 4, bias=False)


def test_model_composes_positive_and_owned_negative_queries() -> None:
    model = ComposedRetrievalModel(
        FakeBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=True,
        temperature=0.07,
        learnable_temperature=True,
    )
    outputs = model(
        reference_pixel_values=torch.randn(2, 4),
        target_pixel_values=torch.randn(2, 4),
        positive_input_ids=torch.tensor([[1, 2], [3, 4]]),
        positive_attention_mask=torch.ones(2, 2, dtype=torch.long),
        negative_input_ids=torch.tensor([[5, 6], [7, 8], [9, 10]]),
        negative_attention_mask=torch.ones(3, 2, dtype=torch.long),
        negative_owner=torch.tensor([0, 0, 1]),
        candidate_pixel_values=torch.randn(3, 4),
    )
    assert outputs["positive_queries"].shape == (2, 4)
    assert outputs["target_features"].shape == (2, 4)
    assert outputs["negative_queries"].shape == (3, 4)
    assert outputs["candidate_features"].shape == (3, 4)
    assert not any(
        key.startswith("backbone.") for key in model.trainable_state_dict()
    )


def test_initial_composition_matches_standard_clip_sum_baseline() -> None:
    model = ComposedRetrievalModel(
        FakeBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=True,
        temperature=0.07,
        learnable_temperature=True,
    )
    reference = F.normalize(torch.randn(3, 4), dim=-1)
    text = F.normalize(torch.randn(3, 4), dim=-1)
    expected = F.normalize(reference + text, dim=-1)
    assert torch.allclose(model.compose(reference, text), expected)


def test_infonce_loss_uses_same_target_negative_queries() -> None:
    targets = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    positives = targets.clone().requires_grad_(True)
    negatives = F.normalize(
        torch.tensor([[-1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]),
        dim=-1,
    ).requires_grad_(True)
    loss, metrics = composed_infonce_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=negatives,
        negative_owner=torch.tensor([0, 1, 0]),
        scale=torch.tensor(10.0),
        hard_negative_weight=1.0,
    )
    assert torch.isfinite(loss)
    assert metrics["hard_accuracy"].item() == 1.0
    loss.backward()
    assert positives.grad is not None
    assert negatives.grad is not None


def test_target_conditioned_hard_margin_requires_an_explicit_cosine_gap() -> None:
    targets = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0]])
    negatives = F.normalize(torch.tensor([[0.98, 0.2]]), dim=-1)
    no_margin_loss, no_margin_metrics = composed_infonce_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=negatives,
        negative_owner=torch.tensor([0]),
        scale=torch.tensor(10.0),
        hard_negative_weight=1.0,
        hard_negative_margin=0.0,
    )
    margin_loss, margin_metrics = composed_infonce_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=negatives,
        negative_owner=torch.tensor([0]),
        scale=torch.tensor(10.0),
        hard_negative_weight=1.0,
        hard_negative_margin=0.05,
    )
    assert margin_loss > no_margin_loss
    assert no_margin_metrics["hard_accuracy"].item() == 1.0
    assert margin_metrics["hard_accuracy"].item() == 1.0
    assert margin_metrics["hard_margin_accuracy"].item() == 0.0
    assert 0.0 < margin_metrics["hard_mean_gap"].item() < 0.05


def test_normal_training_is_symmetric_batch_infonce_only() -> None:
    targets = F.normalize(torch.randn(4, 8), dim=-1)
    positives = F.normalize(torch.randn(4, 8), dim=-1).requires_grad_(True)
    loss, metrics = composed_infonce_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=None,
        negative_owner=torch.empty(0, dtype=torch.long),
        scale=torch.tensor(10.0),
        hard_negative_weight=0.0,
    )
    assert torch.allclose(loss, metrics["batch_loss"])
    assert metrics["hard_loss"].item() == 0.0
    loss.backward()
    assert positives.grad is not None


def test_hard_only_reports_but_does_not_backpropagate_batch_infonce() -> None:
    targets = F.normalize(torch.randn(2, 8), dim=-1).requires_grad_(True)
    positives = F.normalize(torch.randn(2, 8), dim=-1).requires_grad_(True)
    negatives = F.normalize(torch.randn(3, 8), dim=-1).requires_grad_(True)
    loss, metrics = composed_infonce_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=negatives,
        negative_owner=torch.tensor([0, 0, 1]),
        scale=torch.tensor(10.0),
        hard_negative_weight=1.0,
        loss_mode="hard_only",
    )
    assert torch.allclose(loss.detach(), metrics["hard_loss"])
    assert torch.isfinite(metrics["batch_loss"])
    loss.backward()
    assert positives.grad is not None
    assert negatives.grad is not None
    assert targets.grad is not None


def test_typed_candidate_loss_mines_each_available_semantic_class() -> None:
    positives = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1
    ).requires_grad_(True)
    targets = positives.detach().clone().requires_grad_(True)
    negatives = F.normalize(
        torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]), dim=-1
    ).requires_grad_(True)
    candidates = F.normalize(
        torch.tensor([[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]]), dim=-1
    ).requires_grad_(True)
    loss, metrics = typed_candidate_counterfactual_loss(
        positive_queries=positives,
        target_features=targets,
        negative_queries=negatives,
        negative_owner=torch.tensor([0, 0, 1]),
        negative_semantic_class=torch.tensor([0, 1, 2]),
        candidate_features=candidates,
        candidate_owner=torch.tensor([0, 0, 1]),
        scale=torch.tensor(10.0),
        candidate_negative_weight=0.1,
        counterfactual_square_weight=0.1,
        candidate_margin=0.0,
        counterfactual_margin=0.0,
        mining_min_advantage=0.0,
    )
    assert torch.isfinite(loss)
    assert metrics["typed_candidate_attempt_count"].item() == 3
    assert metrics["typed_candidate_relation_count"].item() == 3
    assert metrics["typed_candidate_mining_coverage"].item() == 1.0
    assert metrics["counterfactual_square_accuracy"].item() == 1.0
    loss.backward()
    assert positives.grad is not None
    assert targets.grad is not None
    assert negatives.grad is not None
    assert candidates.grad is not None


def test_last_n_finetuning_selects_tail_blocks_and_projections() -> None:
    model = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        temperature=0.07,
        learnable_temperature=True,
        unfreeze_last_n_layers=2,
    )
    assert not any(
        parameter.requires_grad
        for layer in model.backbone.vision_model.encoder.layers[:-2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.backbone.vision_model.encoder.layers[-2:]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.backbone.text_model.encoder.layers[-2:]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.backbone.visual_projection.parameters()
    )
    state = model.trainable_state_dict()
    assert not any(
        key.startswith("backbone.vision_model.encoder.layers.0") for key in state
    )
    assert any(
        key.startswith("backbone.vision_model.encoder.layers.5") for key in state
    )
    summary = model.parameter_summary()
    assert summary["backbone_finetune_mode"] == "last_2"
    assert 0 < summary["trainable_backbone_parameter_count"] < summary[
        "backbone_parameter_count"
    ]


def test_lora_only_unfreezes_attention_adapters() -> None:
    model = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=True,
        temperature=0.07,
        learnable_temperature=True,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    wrapped = model.backbone.vision_model.encoder.layers[0].self_attn.q_proj
    assert isinstance(wrapped, LoRALinear)
    assert wrapped.lora_A.requires_grad
    assert wrapped.lora_B.requires_grad
    assert not wrapped.base.weight.requires_grad
    backbone_trainable = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    ]
    assert backbone_trainable
    assert all(name.endswith(("lora_A", "lora_B")) for name in backbone_trainable)
    assert model.parameter_summary()["backbone_finetune_mode"] == "lora_r2"


def test_lora_linear_starts_as_exact_base_and_receives_gradients() -> None:
    base = nn.Linear(4, 3)
    lora = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    inputs = torch.randn(5, 4)
    expected = base(inputs).detach()
    actual = lora(inputs)
    assert torch.allclose(actual, expected)
    actual.square().mean().backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None


def _distillation_fixture():
    candidates = F.normalize(
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]],
            ]
        ),
        dim=-1,
    )
    candidate_mask = torch.tensor([[True, True, False], [True, True, True]])
    positives = F.normalize(
        torch.tensor([[1.0, 0.2], [0.1, 1.0]]), dim=-1
    ).requires_grad_(True)
    negatives = F.normalize(
        torch.tensor([[0.5, 1.0], [-0.5, 1.0], [1.0, 0.3]]), dim=-1
    ).requires_grad_(True)
    owners = torch.tensor([0, 0, 1])
    student_positive_scores = torch.einsum("bd,bmd->bm", positives, candidates)
    student_negative_scores = torch.einsum(
        "kd,kmd->km", negatives, candidates[owners]
    )
    teacher_positive = student_positive_scores.detach().clone()
    teacher_negative = student_negative_scores.detach().clone()
    teacher_positive[0, 2] = torch.nan
    teacher_negative[:2, 2] = torch.nan
    return {
        "positive_queries": positives,
        "negative_queries": negatives,
        "negative_owner": owners,
        "candidate_features": candidates,
        "candidate_mask": candidate_mask,
        "target_indices": torch.tensor([0, 0]),
        "teacher_positive_scores": teacher_positive,
        "teacher_negative_scores": teacher_negative,
        "teacher_target_top1": torch.tensor([True, True]),
    }


def test_distillation_losses_support_rank_response_and_joint_with_padding() -> None:
    fixture = _distillation_fixture()
    common = {
        **fixture,
        "positive_rank_temperature": 0.05,
        "semantic_response_beta": 0.01,
        "require_teacher_target_top1": True,
        "semantic_min_teacher_target_delta": -2.0,
    }
    rank, rank_metrics = semantic_ability_distillation_loss(
        **common,
        positive_rank_weight=1.0,
        semantic_response_weight=0.0,
    )
    response, response_metrics = semantic_ability_distillation_loss(
        **common,
        positive_rank_weight=0.0,
        semantic_response_weight=1.0,
    )
    joint, _ = semantic_ability_distillation_loss(
        **common,
        positive_rank_weight=1.0,
        semantic_response_weight=1.0,
    )
    assert torch.isfinite(rank)
    assert torch.isfinite(response)
    assert torch.isfinite(joint)
    assert torch.allclose(rank, torch.zeros_like(rank), atol=1e-6)
    assert torch.allclose(response, torch.zeros_like(response), atol=1e-6)
    assert torch.allclose(joint, rank + response, atol=1e-6)
    assert rank_metrics["rank_sample_count"].item() == 2
    assert response_metrics["semantic_relation_count"].item() == 3
    assert response_metrics["semantic_response_element_count"].item() == 7


def test_semantic_response_kd_supervises_off_target_matrix_cells() -> None:
    fixture = _distillation_fixture()
    teacher_negative = fixture["teacher_negative_scores"].clone()
    # Keep the own-target response unchanged, but alter a non-target candidate.
    teacher_negative[0, 1] += 0.2
    loss, metrics = semantic_ability_distillation_loss(
        **{**fixture, "teacher_negative_scores": teacher_negative},
        positive_rank_weight=0.0,
        semantic_response_weight=1.0,
        positive_rank_temperature=0.05,
        semantic_response_beta=0.01,
        require_teacher_target_top1=True,
        semantic_min_teacher_target_delta=-2.0,
    )
    assert loss > 0
    assert metrics["semantic_response_mae"] > 0
    loss.backward()
    assert fixture["positive_queries"].grad is not None
    assert fixture["negative_queries"].grad is not None


def test_text_only_mode_freezes_clip_vision_modules() -> None:
    model = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        freeze_vision_encoder=True,
        temperature=0.07,
        learnable_temperature=False,
    )
    assert not any(
        parameter.requires_grad for parameter in model.backbone.vision_model.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.backbone.visual_projection.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.backbone.text_model.parameters()
    )
    assert model.parameter_summary()["backbone_finetune_mode"] == "text_and_fusion"


def test_incremental_checkpoint_inherits_frozen_vision_state(tmp_path) -> None:
    base = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        temperature=0.07,
        learnable_temperature=True,
    )
    with torch.no_grad():
        base.backbone.visual_projection.weight.fill_(0.25)
    base_path = tmp_path / "base.pt"
    save_checkpoint(
        base_path,
        model=base,
        optimizer=None,
        scheduler=None,
        epoch=0,
        global_step=0,
        config={},
    )

    student = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        freeze_vision_encoder=True,
        temperature=0.07,
        learnable_temperature=False,
    )
    load_checkpoint(base_path, model=student)
    with torch.no_grad():
        student.fusion[-1].bias.fill_(0.75)
    delta_path = tmp_path / "delta.pt"
    save_checkpoint(
        delta_path,
        model=student,
        optimizer=None,
        scheduler=None,
        epoch=3,
        global_step=10,
        config={},
        base_checkpoint=base_path,
    )

    restored = ComposedRetrievalModel(
        FakeStandardCLIP(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=False,
        freeze_vision_encoder=True,
        temperature=0.07,
        learnable_temperature=False,
    )
    load_checkpoint(delta_path, model=restored)
    assert torch.allclose(
        restored.backbone.visual_projection.weight,
        torch.full_like(restored.backbone.visual_projection.weight, 0.25),
    )
    assert torch.allclose(
        restored.fusion[-1].bias,
        torch.full_like(restored.fusion[-1].bias, 0.75),
    )
