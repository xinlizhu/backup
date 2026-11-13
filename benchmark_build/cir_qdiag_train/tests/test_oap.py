from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cir_qdiag_train.manifest import HardNegativeText, TrainManifestRecord
from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.oap_config import load_oap_config
from cir_qdiag_train.oap_data import CIRROAPDataset
from cir_qdiag_train.oap_evaluation import (
    _finalize_rank_totals,
    _new_rank_totals,
    _update_rank_totals,
)
from cir_qdiag_train.oap_loss import oap_ordering_loss
from cir_qdiag_train.oap_model import (
    OAPAttentionModel,
    OAPResidualModel,
    load_oap_checkpoint,
    save_oap_checkpoint,
)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image = nn.Linear(4, 4, bias=False)
        self.text = nn.Embedding(20, 4)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image(pixel_values)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        return self.text(input_ids).mean(dim=1)


class FakeTowerOutput:
    def __init__(self, hidden: torch.Tensor, pooled: torch.Tensor) -> None:
        self.last_hidden_state = hidden
        self.pooler_output = pooled

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.last_hidden_state, self.pooler_output)[index]


class FakeVisionTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 6)

    def forward(
        self, pixel_values: torch.Tensor, return_dict: bool = True
    ) -> FakeTowerOutput:
        del return_dict
        base = self.projection(pixel_values)
        hidden = torch.stack([base + float(index) / 10 for index in range(4)], dim=1)
        return FakeTowerOutput(hidden, hidden[:, 0])


class FakeTextTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(20, 5)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> FakeTowerOutput:
        del return_dict
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return FakeTowerOutput(hidden, pooled)


class FakeTokenBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=6),
            text_config=SimpleNamespace(hidden_size=5),
        )
        self.vision_model = FakeVisionTower()
        self.text_model = FakeTextTower()
        self.visual_projection = nn.Linear(6, 4, bias=False)
        self.text_projection = nn.Linear(5, 4, bias=False)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(self.vision_model(pixel_values).pooler_output)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.text_model(input_ids, attention_mask)
        return self.text_projection(output.pooler_output)


def make_model() -> OAPResidualModel:
    base = ComposedRetrievalModel(
        FakeBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=True,
        temperature=0.07,
        learnable_temperature=True,
    )
    return OAPResidualModel(
        base,
        semantic_hidden_dim=5,
        semantic_dim=2,
        beta=0.1,
    )


def make_attention_model() -> OAPAttentionModel:
    base = ComposedRetrievalModel(
        FakeTokenBackbone(),
        embedding_dim=4,
        hidden_dim=8,
        dropout=0.0,
        freeze_backbone=True,
        temperature=0.07,
        learnable_temperature=True,
    )
    return OAPAttentionModel(
        base,
        attention_hidden_dim=8,
        attention_num_heads=2,
        attention_ffn_dim=12,
        attention_dropout=0.0,
        semantic_dim=2,
        beta=0.1,
    )


def test_oap_preserves_frozen_global_model_and_aligned_initialization() -> None:
    model = make_model()
    assert not any(parameter.requires_grad for parameter in model.base.parameters())
    inputs = torch.randn(3, 4)
    for query_adapter, target_adapter in zip(
        model.query_adapters, model.target_adapters, strict=True
    ):
        assert torch.allclose(query_adapter(inputs), target_adapter(inputs))
        assert query_adapter is not target_adapter


def test_packed_dot_product_matches_final_score() -> None:
    model = make_model()
    query_global = F.normalize(torch.randn(3, 4), dim=-1)
    target_global = F.normalize(torch.randn(3, 4), dim=-1)
    query_semantic = model.project_query_semantics(query_global)
    target_semantic = model.project_target_semantics(target_global)
    expected = model.aligned_score(
        query_global, query_semantic, target_global, target_semantic
    )
    actual = (
        model.pack_query(query_global, query_semantic)
        * model.pack_target(target_global, target_semantic)
    ).sum(dim=-1)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_reranker_rank_metrics_follow_the_supplied_order() -> None:
    totals = _new_rank_totals([1, 2], [1, 2])
    _update_rank_totals(
        totals,
        order=torch.tensor([2, 0, 3, 1]),
        target_index=0,
        group_indices=[0, 1, 3],
        cutoffs=[1, 2],
        group_cutoffs=[1, 2],
    )
    metrics = _finalize_rank_totals(totals, 1, 4, [1, 2], [1, 2])
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 100.0
    assert metrics["group_recall_at_1"] == 100.0
    assert metrics["mean_rank"] == 2.0
    assert metrics["mrr"] == 0.5


def test_oap_ordering_loss_backpropagates_through_semantic_adapters() -> None:
    model = make_model()
    global_rows = F.normalize(torch.randn(3, 4), dim=-1)
    positive_semantic = model.project_query_semantics(global_rows)
    target_semantic = model.project_target_semantics(global_rows)
    negative_global = F.normalize(torch.randn(3, 4), dim=-1)
    negative_semantic = model.project_query_semantics(negative_global)
    loss, metrics = oap_ordering_loss(
        model=model,
        positive_global=global_rows,
        positive_semantic=positive_semantic,
        target_global=global_rows,
        target_semantic=target_semantic,
        negative_semantic=negative_semantic,
        negative_owner=torch.tensor([0, 1, 2]),
        negative_semantic_class=torch.tensor([0, 1, 2]),
        temperature=0.07,
        retrieval_margin=0.0,
        patch_margin=0.01,
    )
    assert torch.isfinite(loss)
    assert metrics["relation_count"].item() == 9
    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_attention_tokens_preserve_global_geometry_and_pack_score() -> None:
    model = make_attention_model().eval()
    references = torch.randn(3, 4)
    targets = torch.randn(3, 4)
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5], [6, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])

    global_query, semantic_query = model.encode_queries(
        references, input_ids, attention_mask
    )
    global_target, semantic_target = model.encode_targets(targets)
    expected_query = model.base.compose(
        model.base.encode_reference(references),
        model.base.encode_text(input_ids, attention_mask),
    )
    expected_target = model.base.encode_image(targets)
    assert torch.allclose(global_query, expected_query, atol=1e-6)
    assert torch.allclose(global_target, expected_target, atol=1e-6)
    assert semantic_query.shape == semantic_target.shape == (3, 3, 2)
    assert torch.allclose(
        semantic_query.norm(dim=-1), torch.ones(3, 3), atol=1e-6
    )
    packed_score = (
        model.pack_query(global_query, semantic_query)
        * model.pack_target(global_target, semantic_target)
    ).sum(dim=-1)
    assert torch.allclose(
        packed_score,
        model.aligned_score(
            global_query, semantic_query, global_target, semantic_target
        ),
        atol=1e-6,
    )


def test_attention_token_loss_reaches_every_new_parameter() -> None:
    model = make_attention_model()
    outputs = model(
        reference_pixel_values=torch.randn(3, 4),
        target_pixel_values=torch.randn(3, 4),
        positive_input_ids=torch.tensor([[1, 2], [3, 4], [5, 6]]),
        positive_attention_mask=torch.ones(3, 2, dtype=torch.long),
        negative_input_ids=torch.tensor([[7, 8], [9, 10], [11, 12]]),
        negative_attention_mask=torch.ones(3, 2, dtype=torch.long),
        negative_owner=torch.tensor([0, 1, 2]),
    )
    loss, _ = oap_ordering_loss(
        model=model,
        positive_global=outputs["positive_global"],
        positive_semantic=outputs["positive_semantic"],
        target_global=outputs["target_global"],
        target_semantic=outputs["target_semantic"],
        negative_semantic=outputs["negative_semantic"],
        negative_owner=torch.tensor([0, 1, 2]),
        negative_semantic_class=torch.tensor([0, 1, 2]),
        temperature=0.07,
        retrieval_margin=0.0,
        patch_margin=0.01,
    )
    loss.backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert not any(parameter.grad is not None for parameter in model.base.parameters())


def test_attention_checkpoint_contains_only_new_modules(tmp_path: Path) -> None:
    model = make_attention_model()
    expected = {
        key: value.detach().clone()
        for key, value in model.semantic_state_dict().items()
    }
    checkpoint = tmp_path / "attention.pt"
    save_oap_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        scheduler=None,
        epoch=1,
        global_step=3,
        config={"model": {"architecture": "attention_tokens"}},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["semantic_model"]
    assert not any(key.startswith("base.") for key in payload["semantic_model"])
    with torch.no_grad():
        next(parameter for parameter in model.parameters() if parameter.requires_grad).zero_()
    load_oap_checkpoint(checkpoint, model=model)
    actual = model.semantic_state_dict()
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)


def test_oap_dataset_balances_semantics_and_excludes_delete(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    target = tmp_path / "target.png"
    Image.new("RGB", (8, 8)).save(reference)
    Image.new("RGB", (8, 8)).save(target)
    negatives = []
    for semantic in ("object", "attribute", "predicate"):
        for index, operation in enumerate(("add", "add", "substitute", "delete")):
            negatives.append(
                HardNegativeText(
                    negative_id=f"{semantic}-{operation}-{index}",
                    text=f"{semantic} {operation} {index}",
                    semantic_target=semantic,
                    edit_operation=operation,
                    construction_mode="text_only_editable",
                )
            )
    record = TrainManifestRecord(
        sample_id="sample",
        pair_id=0,
        reference_id="r",
        reference_path=str(reference),
        target_id="t",
        target_path=str(target),
        positive_text="positive",
        negatives=negatives,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    dataset = CIRROAPDataset(
        manifest, max_negatives_per_semantic=2, seed=42
    )
    selected = dataset[0]["negatives"]
    assert len(selected) == 6
    assert all(item.edit_operation != "delete" for item in selected)
    assert {
        semantic: sum(item.semantic_target == semantic for item in selected)
        for semantic in ("object", "attribute", "predicate")
    } == {"object": 2, "attribute": 2, "predicate": 2}


def test_real_oap_configs_are_controlled() -> None:
    root = Path(__file__).parents[1]
    clip = load_oap_config(root / "configs/oap_residual_clip_l_beta0_1.yaml")
    bge = load_oap_config(root / "configs/oap_residual_bge_vl_large_beta0_1.yaml")
    assert clip.data == bge.data
    assert clip.optimizer == bge.optimizer
    assert clip.evaluation == bge.evaluation
    assert clip.model.beta == bge.model.beta == 0.1
    assert clip.model.semantic_dim == bge.model.semantic_dim == 64
    assert clip.model.architecture == bge.model.architecture == "residual_adapter"
    assert clip.data.train_manifest.name == "cirr_train_all_v5_no_va.jsonl"


def test_real_attention_configs_are_controlled() -> None:
    root = Path(__file__).parents[1]
    clip = load_oap_config(root / "configs/oap_attention_clip_l_beta0_1.yaml")
    bge = load_oap_config(
        root / "configs/oap_attention_bge_vl_large_beta0_1.yaml"
    )
    assert clip.data == bge.data
    assert clip.optimizer == bge.optimizer
    assert clip.evaluation == bge.evaluation
    assert clip.model.architecture == bge.model.architecture == "attention_tokens"
    assert clip.model.beta == bge.model.beta == 0.1
    assert clip.model.semantic_dim == bge.model.semantic_dim == 64
    assert clip.model.attention_hidden_dim == bge.model.attention_hidden_dim == 256
