from pathlib import Path

import pytest
from pydantic import ValidationError

from cir_qdiag_train.config import load_config
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.qwen_on_policy_config import load_qwen_on_policy_config


def test_reverse_r128_suite_has_matched_no_filter_control() -> None:
    root = Path(__file__).parents[1]
    names = ("static", "teacher", "random", "no_filter", "opd")
    configs = [load_on_policy_config(
        root / f"configs/clip_l_reverse_rank_{name}_r128_e4.yaml"
    ) for name in names]
    assert len({c.experiment.output_dir for c in configs}) == 5
    for c in configs:
        assert c.experiment.initial_checkpoint is None
        assert c.objective.rank_kl_direction == "reverse"
        assert c.objective.loss_type == "rank_kd"
        assert not c.objective.is_policy_loss
        assert c.objective.needs_teacher_scores
        assert c.objective.centered_response_weight == 0
        assert not c.objective.anchor_kl_enabled
        assert c.optimizer == configs[0].optimizer
        assert c.model == configs[0].model
        assert c.data == configs[0].data
    no_filter, opd = configs[3:]
    assert no_filter.mining == opd.mining
    assert no_filter.objective.model_copy(update={"teacher_target_max_rank": 10}) == opd.objective
    assert opd.mining.reservoir_top_k == 128
    assert opd.mining.top_k == 32
    assert not opd.mining.persist_candidate_cache
    assert configs[0].mining.refresh_mode == "once"
    assert configs[1].mining.candidate_source == "teacher"
    assert configs[2].mining.candidate_source == "random"
    payload = opd.model_dump()
    payload["optimizer"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ValidationError, match="one optimizer update per batch"):
        type(opd).model_validate(payload)


def test_clip_b_three_way_comparison_is_controlled() -> None:
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "clip_b_baseline.yaml")
    without_delete = load_config(root / "configs" / "clip_b_add_substitute.yaml")
    with_delete = load_config(root / "configs" / "clip_b_add_substitute_delete.yaml")
    assert baseline.experiment.negative_mode == "none"
    assert without_delete.experiment.negative_mode == "add_substitute"
    assert with_delete.experiment.negative_mode == "add_substitute_delete"
    assert with_delete.data == without_delete.data
    assert with_delete.model == without_delete.model
    assert with_delete.optimizer == without_delete.optimizer
    assert with_delete.evaluation == without_delete.evaluation
    assert with_delete.experiment.seed == without_delete.experiment.seed
    assert baseline.data == without_delete.data
    assert baseline.model == without_delete.model
    assert baseline.optimizer.model_copy(update={"hard_negative_weight": 1.0}) == (
        without_delete.optimizer
    )
    assert baseline.experiment.seed == without_delete.experiment.seed


def test_standard_clip_variants_are_explicit() -> None:
    root = Path(__file__).parents[1]
    clip_b = load_config(root / "configs" / "clip_b_baseline.yaml")
    clip_l = load_config(root / "configs" / "clip_l_baseline.yaml")
    assert clip_b.model.variant == "clip_vit_b16"
    assert Path(clip_b.model.pretrained_model_name_or_path).name == "clip-vit-base-patch16"
    assert clip_b.model.local_files_only is True
    assert clip_l.model.variant == "clip_vit_l14"
    assert Path(clip_l.model.pretrained_model_name_or_path).name == "clip-vit-large-patch14"
    assert clip_l.model.local_files_only is True


def test_opd_student_teacher_matrix_has_matched_clip_protocols() -> None:
    root = Path(__file__).parents[1]
    names = (
        "opd_clip_b_s2_rank.yaml",
        "opd_clip_b_qwen8_rank.yaml",
        "opd_clip_l_qwen8_rank.yaml",
    )
    configs = [load_on_policy_config(root / "configs" / name) for name in names]
    for config in configs:
        assert config.objective.rank_kl_direction == "forward"
        assert config.objective.teacher_target_max_rank == 10
        assert config.mining.candidate_source == "student"
        assert config.mining.top_k == 32
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
        assert config.model.freeze_backbone is False
    assert configs[0].model.variant == configs[1].model.variant == "clip_vit_b16"
    assert configs[2].model.variant == "clip_vit_l14"
    assert configs[0].data.teacher_cache_dir != configs[1].data.teacher_cache_dir
    assert configs[1].data.teacher_cache_dir == configs[2].data.teacher_cache_dir


def test_reranker_teacher_opd_uses_the_existing_rank_protocol() -> None:
    root = Path(__file__).parents[1]
    config = load_on_policy_config(
        root / "configs/opd_clip_l_qwen3_vl_reranker_2b_rank.yaml"
    )
    assert config.model.variant == "clip_vit_l14"
    assert config.teacher.backend == "http_reranker"
    assert config.teacher.model_name == "qwen3-vl-reranker-2b"
    assert config.teacher.api_format == "score"
    assert config.teacher.instruction is None
    assert config.teacher.image_transport == "data_url"
    assert config.data.teacher_cache_dir is None
    assert config.mining.candidate_source == "student"
    assert config.mining.top_k == 32
    assert config.mining.cache_dtype == "float32"
    assert config.objective.rank_temperature == 0.05
    assert config.objective.rank_kl_direction == "forward"
    assert config.objective.teacher_target_max_rank == 10
    assert config.optimizer.epochs == 4
    assert config.optimizer.save_every_epochs == 1


def test_s2_reverse_kl_is_a_direction_only_clip_l_control() -> None:
    root = Path(__file__).parents[1]
    forward = load_on_policy_config(
        root / "configs/rank_ablation_dynamic_student_rank_e4.yaml"
    )
    reverse = load_on_policy_config(
        root / "configs/opd_clip_l_s2_reverse_rank.yaml"
    )
    assert forward.objective.rank_kl_direction == "forward"
    assert reverse.objective.rank_kl_direction == "reverse"
    assert reverse.teacher.backend == "embedding_cache"
    assert reverse.data == forward.data
    assert reverse.model == forward.model
    assert reverse.mining == forward.mining
    assert reverse.optimizer == forward.optimizer
    assert reverse.evaluation == forward.evaluation
    assert reverse.objective.model_copy(update={"rank_kl_direction": "forward"}) == (
        forward.objective
    )


def test_s2_qminus_candidate_opd_changes_only_the_candidate_layout() -> None:
    root = Path(__file__).parents[1]
    control = load_on_policy_config(
        root / "configs/rank_ablation_dynamic_student_rank_e4.yaml"
    )
    mixed = load_on_policy_config(
        root / "configs/opd_clip_l_s2_qminus_candidates_rank.yaml"
    )
    assert mixed.mining.include_negative_queries_as_candidates is True
    assert mixed.data.negative_query_manifest is not None
    assert mixed.teacher.backend == "embedding_cache"
    assert mixed.objective.rank_kl_direction == "forward"
    assert mixed.model == control.model
    assert mixed.objective == control.objective
    assert mixed.optimizer == control.optimizer
    assert mixed.evaluation == control.evaluation
    assert mixed.data.model_copy(update={"negative_query_manifest": None}) == control.data
    assert mixed.mining.model_copy(
        update={"include_negative_queries_as_candidates": False}
    ) == control.mining


def test_s2_qminus_candidate_k_sweep_is_schedule_matched() -> None:
    root = Path(__file__).parents[1]
    configs = [
        load_on_policy_config(
            root / f"configs/opd_clip_l_s2_qminus_candidates_rank{suffix}.yaml"
        )
        for suffix in ("_k16", "", "_k64")
    ]
    assert [config.mining.top_k for config in configs] == [16, 32, 64]
    assert [
        config.mining.max_negative_query_candidates_per_sample
        for config in configs
    ] == [14, None, None]
    control = configs[1]
    for config in configs:
        assert config.data == control.data
        assert config.model == control.model
        assert config.objective == control.objective
        assert config.teacher == control.teacher
        assert config.optimizer == control.optimizer
        assert config.evaluation == control.evaluation
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
        assert config.mining.include_negative_queries_as_candidates is True
        assert config.mining.model_copy(
            update={
                "top_k": 32,
                "max_negative_query_candidates_per_sample": None,
            }
        ) == control.mining


def test_s2_opd_k_temperature_kl_grid_is_controlled() -> None:
    root = Path(__file__).parents[1]
    control = load_on_policy_config(
        root / "configs/rank_ablation_dynamic_student_rank_e4.yaml"
    )
    specifications = (
        ("opd_clip_l_s2_k16_forward_t005.yaml", 16, "forward", 0.05),
        ("opd_clip_l_s2_k8_forward_t005.yaml", 8, "forward", 0.05),
        ("opd_clip_l_s2_k32_reverse_t005.yaml", 32, "reverse", 0.05),
        ("opd_clip_l_s2_k32_forward_t1.yaml", 32, "forward", 1.0),
        ("opd_clip_l_s2_k32_reverse_t1.yaml", 32, "reverse", 1.0),
    )
    for name, top_k, direction, temperature in specifications:
        config = load_on_policy_config(root / "configs" / name)
        assert config.mining.top_k == top_k
        assert config.objective.rank_kl_direction == direction
        assert config.objective.rank_temperature == temperature
        assert config.data == control.data
        assert config.model == control.model
        assert config.teacher == control.teacher
        assert config.optimizer == control.optimizer
        assert config.evaluation == control.evaluation
        assert config.mining.model_copy(update={"top_k": 32}) == control.mining
        assert config.objective.model_copy(
            update={
                "rank_kl_direction": "forward",
                "rank_temperature": 0.05,
            }
        ) == control.objective


def test_opd_post_training_group_relative_and_target_ce_are_matched() -> None:
    root = Path(__file__).parents[1]
    group_relative = load_on_policy_config(
        root / "configs/opd_clip_l_e4_group_relative_e4.yaml"
    )
    target_ce = load_on_policy_config(
        root / "configs/opd_clip_l_e4_target_ce_e4.yaml"
    )
    assert (
        group_relative.experiment.initial_checkpoint
        == target_ce.experiment.initial_checkpoint
    )
    assert group_relative.experiment.initial_checkpoint is not None
    assert group_relative.experiment.initial_checkpoint.name == "checkpoint_epoch_004.pt"
    assert group_relative.objective.loss_type == "listwise_group_relative"
    assert group_relative.objective.rollout_count == 16
    assert group_relative.objective.rollout_top_k == 10
    assert group_relative.objective.rank_temperature == 0.05
    assert group_relative.objective.uses_teacher_filter is False
    assert group_relative.objective.needs_teacher_scores is False
    assert target_ce.objective.loss_type == "candidate_infonce"
    assert target_ce.objective.uses_teacher_filter is False
    assert target_ce.objective.needs_teacher_scores is False
    assert group_relative.objective.model_copy(
        update={"loss_type": "candidate_infonce"}
    ) == target_ce.objective
    assert group_relative.data == target_ce.data
    assert group_relative.data.teacher_cache_dir is None
    assert group_relative.model == target_ce.model
    assert group_relative.mining == target_ce.mining
    assert group_relative.optimizer == target_ce.optimizer
    assert group_relative.optimizer.epochs == 4
    assert group_relative.optimizer.save_every_epochs == 1
    assert group_relative.evaluation == target_ce.evaluation


def test_raw_clip_policy_control_differs_only_in_initialization_and_output() -> None:
    root = Path(__file__).resolve().parents[1]
    opd = load_on_policy_config(
        root / "configs/opd_clip_l_e4_group_relative_e4.yaml"
    )
    raw = load_on_policy_config(root / "configs/raw_clip_l_group_relative_e4.yaml")
    assert opd.experiment.initial_checkpoint is not None
    assert raw.experiment.initial_checkpoint is None
    assert opd.experiment.seed == raw.experiment.seed
    assert opd.data == raw.data
    assert opd.model == raw.model
    assert opd.objective == raw.objective
    assert opd.teacher == raw.teacher
    assert opd.mining == raw.mining
    assert opd.optimizer == raw.optimizer
    assert opd.evaluation == raw.evaluation


def test_causal_policy_ablation_is_controlled_and_full_finetuned() -> None:
    root = Path(__file__).resolve().parents[1]
    names = (
        "opd_clip_l_cp_reinforce_e4.yaml",
        "opd_clip_l_cp_grpo_e4.yaml",
        "opd_clip_l_cp_gspo_noclip_e4.yaml",
        "opd_clip_l_cp_gspo_e4.yaml",
    )
    configs = [load_on_policy_config(root / "configs" / name) for name in names]
    control = configs[0]
    for config in configs:
        assert config.experiment.initial_checkpoint == control.experiment.initial_checkpoint
        assert config.experiment.seed == control.experiment.seed == 42
        assert config.data == control.data
        assert config.model == control.model
        assert config.teacher == control.teacher
        assert config.mining == control.mining
        assert config.optimizer == control.optimizer
        assert config.evaluation == control.evaluation
        assert config.objective.is_causal_policy_loss
        assert config.objective.rollout_count == 16
        assert config.objective.rollout_top_k == 10
        assert config.objective.rank_temperature == 0.05
        assert config.objective.deterministic_policy_dropout is True
        assert config.objective.uses_teacher_filter is False
        assert config.objective.needs_teacher_scores is False
        assert config.model.freeze_backbone is False
        assert config.model.freeze_vision_encoder is False
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
    assert control.objective.loss_type == "causal_reinforce"
    assert control.objective.policy_update_epochs == 1
    assert control.objective.policy_clip_epsilon is None
    assert configs[1].objective.loss_type == "causal_grpo"
    assert configs[1].objective.policy_update_epochs == 2
    assert configs[1].objective.policy_clip_epsilon == 0.2
    assert configs[2].objective.loss_type == "causal_gspo"
    assert configs[2].objective.policy_update_epochs == 2
    assert configs[2].objective.policy_clip_epsilon is None
    assert configs[3].objective.loss_type == "causal_gspo"
    assert configs[3].objective.policy_update_epochs == 2
    assert configs[3].objective.policy_clip_epsilon == 0.2


def test_live_reservoir_anchor_kl_ablation_is_controlled() -> None:
    root = Path(__file__).resolve().parents[1]
    names = (
        "opd_clip_l_live_r512_grpo_e4.yaml",
        "opd_clip_l_live_r512_grpo_anchor_kl_e4.yaml",
        "opd_clip_l_live_r512_gspo_e4.yaml",
        "opd_clip_l_live_r512_gspo_anchor_kl_e4.yaml",
    )
    configs = [load_on_policy_config(root / "configs" / name) for name in names]
    for config in configs:
        assert config.mining.top_k == 32
        assert config.mining.reservoir_top_k == 512
        assert config.mining.live_reservoir_selection is True
        assert config.mining.pixel_cache_enabled is True
        assert config.mining.pixel_cache_dtype == "float16"
        assert config.mining.pixel_cache_build_batch_size == 128
        assert config.mining.pixel_cache_workers == 12
        assert config.mining.pixel_cache_build_prefetch_factor == 3
        assert config.mining.pixel_cache_transfer_prefetch_batches == 2
        assert config.mining.pixel_cache_pin_memory is True
        assert config.mining.persist_candidate_cache is False
        assert config.mining.live_selection_image_batch_size == 1024
        assert config.objective.policy_update_epochs == 2
        assert config.objective.policy_clip_epsilon == 0.2
        assert config.objective.needs_teacher_scores is False
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
    assert configs[0].objective.loss_type == "causal_grpo"
    assert configs[0].objective.anchor_kl_enabled is False
    assert configs[1].objective.loss_type == "causal_grpo"
    assert configs[1].objective.anchor_kl_enabled is True
    assert configs[2].objective.loss_type == "causal_gspo"
    assert configs[2].objective.anchor_kl_enabled is False
    assert configs[3].objective.loss_type == "causal_gspo"
    assert configs[3].objective.anchor_kl_enabled is True
    for config in (configs[1], configs[3]):
        assert config.objective.anchor_kl_target == 0.02
        assert config.objective.anchor_kl_beta_initial == 0.1


def test_reward_tilted_reverse_experiments_start_from_original_clip_l() -> None:
    root = Path(__file__).resolve().parents[1]
    reverse_opd = load_on_policy_config(
        root / "configs/clip_l_reward_tilted_reverse_opd_e4.yaml"
    )
    reverse_gspo = load_on_policy_config(
        root / "configs/clip_l_reward_tilted_reverse_gspo_e4.yaml"
    )
    for config in (reverse_opd, reverse_gspo):
        assert config.experiment.initial_checkpoint is None
        assert config.model.variant == "clip_vit_l14"
        assert config.model.freeze_backbone is False
        assert config.model.freeze_vision_encoder is False
        assert config.objective.rank_kl_direction == "reverse"
        assert config.objective.reward_tilt_beta == 0.1
        assert config.objective.needs_teacher_scores is True
        assert config.objective.anchor_kl_enabled is False
        assert config.objective.rollout_count == 16
        assert config.objective.rollout_top_k == 10
        assert config.mining.top_k == 32
        assert config.mining.candidate_source == "student"
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
    assert reverse_opd.objective.loss_type == "reward_tilted_reverse_opd"
    assert reverse_opd.objective.policy_update_epochs == 1
    assert reverse_opd.objective.policy_clip_epsilon is None
    assert reverse_opd.objective.teacher_target_max_rank == 10
    assert reverse_gspo.objective.loss_type == "reward_tilted_reverse_gspo"
    assert reverse_gspo.objective.policy_update_epochs == 2
    assert reverse_gspo.objective.policy_clip_epsilon == 0.2
    assert reverse_gspo.objective.uses_teacher_filter is False


def test_reranker_reverse_kl_is_a_direction_only_control() -> None:
    root = Path(__file__).parents[1]
    forward = load_on_policy_config(
        root / "configs/opd_clip_l_qwen3_vl_reranker_2b_rank.yaml"
    )
    reverse = load_on_policy_config(
        root / "configs/opd_clip_l_qwen3_vl_reranker_2b_reverse_rank.yaml"
    )
    assert forward.objective.rank_kl_direction == "forward"
    assert reverse.objective.rank_kl_direction == "reverse"
    assert reverse.teacher == forward.teacher
    assert reverse.data == forward.data
    assert reverse.model == forward.model
    assert reverse.mining == forward.mining
    assert reverse.optimizer == forward.optimizer
    assert reverse.evaluation == forward.evaluation
    assert reverse.objective.model_copy(update={"rank_kl_direction": "forward"}) == (
        forward.objective
    )


def test_qwen2_qwen8_opd_is_dynamic_query_lora_with_fixed_gallery() -> None:
    root = Path(__file__).parents[1]
    config = load_qwen_on_policy_config(
        root / "configs/opd_qwen2_qwen8_rank.yaml"
    )
    assert config.model.variant == "qwen3_vl_embedding_query_lora"
    assert config.model.lora_rank == 16
    assert config.mining.candidate_source == "student"
    assert config.mining.refresh_mode == "every_epoch"
    assert config.mining.top_k == 32
    assert config.objective.teacher_target_max_rank == 10
    assert config.objective.rank_kl_direction == "forward"
    assert config.optimizer.epochs == 4
    assert config.optimizer.save_every_epochs == 1
    assert config.optimizer.batch_size_per_gpu * (
        config.optimizer.gradient_accumulation_steps
    ) == 32


def test_on_policy_centered_ablation_is_controlled_and_target_forced_top32() -> None:
    root = Path(__file__).parents[1]
    rank = load_on_policy_config(root / "configs/on_policy_clip_l_rank.yaml")
    response = load_on_policy_config(
        root / "configs/on_policy_clip_l_centered_response.yaml"
    )
    joint = load_on_policy_config(root / "configs/on_policy_clip_l_joint.yaml")
    assert rank.objective.rank_weight == 1.0
    assert rank.objective.centered_response_weight == 0.0
    assert response.objective.rank_weight == 0.0
    assert response.objective.centered_response_weight == 1.0
    assert joint.objective.rank_weight == 1.0
    assert joint.objective.centered_response_weight == 1.0
    for config in (rank, response, joint):
        assert config.optimizer.epochs == 8
        assert config.optimizer.batch_size_per_gpu == 32
        assert config.mining.top_k == 32
        assert config.mining.refresh_every_epochs == 1
        assert config.model.freeze_backbone is False
        assert config.model.freeze_vision_encoder is False
    assert rank.data == response.data == joint.data
    assert rank.model == response.model == joint.model
    assert rank.mining == response.mining == joint.mining
    assert rank.optimizer.save_every_epochs == 4
    assert response.optimizer.save_every_epochs == 2
    assert joint.optimizer.save_every_epochs == 2
    assert rank.optimizer.model_copy(update={"save_every_epochs": 2}) == (
        response.optimizer
    ) == joint.optimizer


def test_rank_candidate_policy_ablation_is_schedule_matched() -> None:
    root = Path(__file__).parents[1]
    names = (
        "dynamic_student_rank",
        "teacher_topk_rank",
        "random_rank",
        "static_student_rank",
        "dynamic_student_infonce",
        "dynamic_student_infonce_teacher_top10",
    )
    configs = {
        name: load_on_policy_config(
            root / "configs" / f"rank_ablation_{name}_e4.yaml"
        )
        for name in names
    }
    control = configs["dynamic_student_rank"]
    for config in configs.values():
        assert config.data == control.data
        assert config.model == control.model
        assert config.optimizer == control.optimizer
        assert config.evaluation == control.evaluation
        assert config.optimizer.epochs == 4
        assert config.optimizer.save_every_epochs == 1
        assert config.mining.top_k == 32
        assert config.model.freeze_backbone is False
        assert config.model.freeze_vision_encoder is False

    assert control.mining.candidate_source == "student"
    assert control.mining.refresh_mode == "every_epoch"
    assert configs["teacher_topk_rank"].mining.candidate_source == "teacher"
    assert configs["teacher_topk_rank"].mining.refresh_mode == "once"
    assert configs["random_rank"].mining.candidate_source == "random"
    assert configs["random_rank"].mining.refresh_mode == "every_epoch"
    assert configs["static_student_rank"].mining.candidate_source == "student"
    assert configs["static_student_rank"].mining.refresh_mode == "once"
    assert configs["dynamic_student_infonce"].mining.candidate_source == "student"
    assert configs["dynamic_student_infonce"].mining.refresh_mode == "every_epoch"
    assert (
        configs["dynamic_student_infonce_teacher_top10"].mining.candidate_source
        == "student"
    )
    assert (
        configs["dynamic_student_infonce_teacher_top10"].mining.refresh_mode
        == "every_epoch"
    )

    for name in names[:4]:
        assert configs[name].objective == control.objective
        assert configs[name].objective.loss_type == "rank_kd"
        assert configs[name].objective.rank_weight == 1.0
        assert configs[name].objective.centered_response_weight == 0.0
        assert configs[name].objective.teacher_target_max_rank == 10
    infonce = configs["dynamic_student_infonce"].objective
    assert infonce.loss_type == "candidate_infonce"
    assert infonce.rank_weight == 0.0
    assert infonce.centered_response_weight == 0.0
    assert infonce.teacher_target_max_rank is None
    assert infonce.needs_teacher_scores is False
    filtered_infonce = configs[
        "dynamic_student_infonce_teacher_top10"
    ].objective
    assert filtered_infonce.loss_type == "candidate_infonce"
    assert filtered_infonce.rank_weight == 0.0
    assert filtered_infonce.centered_response_weight == 0.0
    assert filtered_infonce.teacher_target_max_rank == 10
    assert filtered_infonce.uses_teacher_filter is True
    assert filtered_infonce.needs_teacher_scores is True


def test_centered_mse_baseline_uses_unfiltered_random_candidates() -> None:
    root = Path(__file__).parents[1]
    rank = load_on_policy_config(
        root / "configs" / "rank_ablation_dynamic_student_rank_e4.yaml"
    )
    centered_mse = load_on_policy_config(
        root / "configs" / "opd_clip_l_s2_centered_mse_e4.yaml"
    )
    assert centered_mse.data == rank.data
    assert centered_mse.model == rank.model
    assert centered_mse.optimizer == rank.optimizer
    assert centered_mse.evaluation == rank.evaluation
    assert centered_mse.mining == rank.mining.model_copy(
        update={"candidate_source": "random"}
    )
    assert centered_mse.objective.loss_type == "centered_mse"
    assert centered_mse.objective.rank_weight == 0.0
    assert centered_mse.objective.centered_response_weight == 0.0
    assert centered_mse.objective.require_teacher_target_top1 is False
    assert centered_mse.objective.teacher_target_max_rank is None
    assert centered_mse.objective.uses_teacher_filter is False
    assert centered_mse.objective.needs_teacher_scores is True


def test_batch64_weight_study_changes_only_hard_weight() -> None:
    root = Path(__file__).parents[1]
    for variant in ("clip_b", "clip_l"):
        baseline = load_config(root / "configs" / f"{variant}_b64_baseline.yaml")
        half = load_config(root / "configs" / f"{variant}_b64_hard_w0_5.yaml")
        full = load_config(root / "configs" / f"{variant}_b64_hard_w1_0.yaml")
        assert baseline.optimizer.batch_size_per_gpu == 64
        assert half.optimizer.batch_size_per_gpu == 64
        assert full.optimizer.batch_size_per_gpu == 64
        assert baseline.experiment.negative_mode == "none"
        assert half.experiment.negative_mode == "add_substitute"
        assert full.experiment.negative_mode == "add_substitute"
        assert baseline.optimizer.hard_negative_weight == 0.0
        assert half.optimizer.hard_negative_weight == 0.5
        assert full.optimizer.hard_negative_weight == 1.0
        assert half.data == full.data
        assert half.model == full.model
        assert half.evaluation == full.evaluation
        assert half.optimizer.model_copy(update={"hard_negative_weight": 1.0}) == (
            full.optimizer
        )


def test_clip_l_last4_pair_is_controlled() -> None:
    root = Path(__file__).parents[1]
    baseline = load_config(root / "configs" / "clip_l_last4_baseline.yaml")
    hard = load_config(root / "configs" / "clip_l_last4_hard_w0_5.yaml")
    assert baseline.model.freeze_backbone is False
    assert baseline.model.unfreeze_last_n_layers == 4
    assert baseline.model.gradient_checkpointing is True
    assert hard.model == baseline.model
    assert hard.data == baseline.data
    assert hard.evaluation == baseline.evaluation
    assert baseline.optimizer.batch_size_per_gpu == 64
    assert baseline.optimizer.gradient_accumulation_steps == 1
    assert baseline.optimizer.hard_negative_weight == 0.0
    assert hard.optimizer.model_copy(update={"hard_negative_weight": 0.0}) == (
        baseline.optimizer
    )
    assert baseline.experiment.negative_mode == "none"
    assert hard.experiment.negative_mode == "add_substitute"


def test_clip_l_full_and_lora_pairs_are_controlled() -> None:
    root = Path(__file__).parents[1]
    for method in ("full", "lora_r16"):
        baseline = load_config(root / "configs" / f"clip_l_{method}_baseline.yaml")
        hard = load_config(root / "configs" / f"clip_l_{method}_hard_w0_5.yaml")
        assert baseline.model == hard.model
        assert baseline.data == hard.data
        assert baseline.evaluation == hard.evaluation
        assert hard.optimizer.model_copy(update={"hard_negative_weight": 0.0}) == (
            baseline.optimizer
        )
        assert baseline.optimizer.batch_size_per_gpu == 64
        assert baseline.optimizer.save_every_epochs == 2
    full = load_config(root / "configs" / "clip_l_full_baseline.yaml")
    assert full.model.freeze_backbone is False
    assert full.model.unfreeze_last_n_layers is None
    assert full.model.lora_rank is None
    lora = load_config(root / "configs" / "clip_l_lora_r16_baseline.yaml")
    assert lora.model.freeze_backbone is True
    assert lora.model.lora_rank == 16
    assert lora.model.lora_alpha == 32.0
    assert lora.optimizer.backbone_learning_rate == 1e-4


def test_hard_only_requires_negatives_and_unit_weight() -> None:
    root = Path(__file__).parents[1]
    hard = load_config(root / "configs" / "clip_l_full_hard_w0_5.yaml")
    hard_payload = hard.model_dump()
    hard_payload["optimizer"]["loss_mode"] = "hard_only"
    hard_payload["optimizer"]["hard_negative_weight"] = 1.0
    hard.__class__.model_validate(hard_payload)

    hard_payload["optimizer"]["hard_negative_weight"] = 0.5
    with pytest.raises(ValidationError, match="hard_only loss must set"):
        hard.__class__.model_validate(hard_payload)


def test_typed_candidate_config_uses_image_set_and_three_losses() -> None:
    root = Path(__file__).parents[1]
    config = load_config(
        root / "configs" / "clip_l_full_typed_candidate_w0_1.yaml"
    )
    assert config.data.candidate_image_mode == "image_set"
    assert config.data.train_image_split_path is not None
    assert config.optimizer.hard_negative_weight == 0.1
    assert config.optimizer.candidate_negative_weight == 0.1
    assert config.optimizer.counterfactual_square_weight == 0.1
    assert config.optimizer.batch_size_per_gpu == 16
    assert config.optimizer.gradient_accumulation_steps == 4


def test_target_margin_config_is_candidate_free_and_controlled() -> None:
    root = Path(__file__).parents[1]
    control = load_config(
        root / "configs" / "clip_l_full_hard_w0_1_continue_v5.yaml"
    )
    margin = load_config(
        root / "configs" / "clip_l_full_target_margin0_05_continue_v5.yaml"
    )
    assert margin.data.candidate_image_mode == "none"
    assert margin.optimizer.candidate_negative_weight == 0.0
    assert margin.optimizer.counterfactual_square_weight == 0.0
    assert control.optimizer.hard_negative_margin == 0.0
    assert margin.optimizer.hard_negative_margin == 0.05
    assert margin.optimizer.model_copy(update={"hard_negative_margin": 0.0}) == (
        control.optimizer
    )
    assert margin.data == control.data
    assert margin.model == control.model
    assert margin.evaluation == control.evaluation
