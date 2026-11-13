from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cir_qdiag_train.on_policy_data import (
    NegativeQueryLayout,
    OnPolicyTrainingCollator,
)
from cir_qdiag_train.on_policy_loss import merge_mixed_candidate_features
from cir_qdiag_train.on_policy_mining import (
    build_mixed_candidate_layout,
    mix_teacher_candidate_scores,
)


def _layout() -> NegativeQueryLayout:
    return NegativeQueryLayout(
        ids=["q0", "q1", "q2"],
        texts=["negative zero", "negative one", "negative two"],
        owners=np.asarray([0, 0, 1], dtype=np.int64),
        indices_by_owner=[[0, 1], [2]],
    )


def test_mixed_topk_keeps_target_all_owned_qminus_and_remaining_images() -> None:
    retrieved = np.asarray(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
        dtype=np.int64,
    )
    images, negatives = build_mixed_candidate_layout(
        retrieved,
        _layout(),
        top_k=5,
    )
    assert images.tolist() == [
        [10, -1, -1, 11, 12],
        [20, -1, 21, 22, 23],
    ]
    assert negatives.tolist() == [
        [-1, 0, 1, -1, -1],
        [-1, 2, -1, -1, -1],
    ]
    assert ((images >= 0) ^ (negatives >= 0)).all()


def test_teacher_scores_follow_the_mixed_candidate_layout() -> None:
    retrieved = np.asarray(
        [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]],
        dtype=np.int64,
    )
    images, negatives = build_mixed_candidate_layout(
        retrieved,
        _layout(),
        top_k=5,
    )
    mixed = mix_teacher_candidate_scores(
        image_scores=np.asarray(
            [[0.9, 0.8, 0.7, 0.6, 0.5], [0.4, 0.3, 0.2, 0.1, 0.0]],
            dtype=np.float32,
        ),
        candidate_image_indices=images,
        negative_query_candidate_indices=negatives,
        negative_query_scores=np.asarray([0.11, 0.22, 0.33], dtype=np.float32),
    )
    assert np.allclose(
        mixed,
        [[0.9, 0.11, 0.22, 0.8, 0.7], [0.4, 0.33, 0.3, 0.2, 0.1]],
    )


def test_k16_explicitly_caps_qminus_to_retain_one_hard_image() -> None:
    negative_count = 16
    layout = NegativeQueryLayout(
        ids=[f"q{index}" for index in range(negative_count)],
        texts=[f"negative {index}" for index in range(negative_count)],
        owners=np.zeros(negative_count, dtype=np.int64),
        indices_by_owner=[list(range(negative_count))],
    )
    retrieved = np.arange(100, 116, dtype=np.int64)[None, :]
    images, negatives = build_mixed_candidate_layout(
        retrieved,
        layout,
        top_k=16,
        max_negative_queries=14,
    )
    assert int((negatives >= 0).sum()) == 14
    assert int((images >= 0).sum()) == 2
    assert images[0, 0] == 100
    assert images[0, -1] == 101


def test_live_mixed_candidates_backpropagate_into_images_and_qminus() -> None:
    images = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    images.requires_grad_()
    qminus = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    qminus.requires_grad_()
    mask = torch.tensor(
        [[False, True, False, False], [False, False, True, False]]
    )
    mixed = merge_mixed_candidate_features(
        image_candidate_features=images,
        negative_query_features=qminus,
        negative_query_owners=torch.tensor([0, 1]),
        negative_query_positions=torch.tensor([1, 2]),
        candidate_is_negative_query=mask,
    )
    assert torch.equal(mixed[0, 1], qminus[0])
    assert torch.equal(mixed[1, 2], qminus[1])
    assert torch.equal(mixed[0, 0], images[0, 0])
    mixed.sum().backward()
    assert qminus.grad is not None and bool((qminus.grad != 0).all())
    assert images.grad is not None
    assert bool((images.grad[~mask] != 0).all())
    assert bool((images.grad[mask] == 0).all())


def test_collator_keeps_qminus_coordinates_separate_from_image_positions(
    tmp_path,
) -> None:
    image_paths = []
    for index in range(6):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), color=(index, index, index)).save(path)
        image_paths.append(path)

    class FakeProcessor:
        def __call__(self, *, images=None, text=None, **kwargs):
            del kwargs
            if images is not None:
                return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}
            assert text is not None
            return {
                "input_ids": torch.ones(len(text), 2, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 2, dtype=torch.long),
            }

    collator = OnPolicyTrainingCollator(FakeProcessor(), image_paths)
    batch = collator(
        [
            {
                "pair_id": 1,
                "reference_index": 0,
                "candidate_indices": np.asarray([1, -1, 2]),
                "negative_query_candidate_texts": [None, "wrong edit", None],
                "text": "right edit",
                "raw_target_in_topk": True,
            },
            {
                "pair_id": 2,
                "reference_index": 3,
                "candidate_indices": np.asarray([4, 5, 0]),
                "negative_query_candidate_texts": [None, None, None],
                "text": "another edit",
                "raw_target_in_topk": False,
            },
        ]
    )
    assert batch["candidate_is_negative_query"].tolist() == [
        [False, True, False],
        [False, False, False],
    ]
    assert batch["negative_query_owners"].tolist() == [0]
    assert batch["negative_query_positions"].tolist() == [1]
    assert batch["negative_query_input_ids"].shape[0] == 1
    # The q- slot uses a harmless image placeholder and is replaced before scoring.
    assert (
        batch["candidate_image_positions"][0, 1]
        == batch["reference_image_positions"][0]
    )
