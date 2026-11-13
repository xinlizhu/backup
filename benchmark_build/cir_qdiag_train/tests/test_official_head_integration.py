"""Protect query/target asymmetry and gradients in mining and live OPRD scoring."""
from pathlib import Path
import pytest
import torch
from torch import nn
from cir_qdiag_train.model import ComposedRetrievalModel
from cir_qdiag_train.on_policy_trainer import _compute_live_student_scores


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(4, 768)
        self.text_model = nn.Embedding(32, 768)

    def get_image_features(self, pixel_values):
        return self.vision_model(pixel_values)

    def get_text_features(self, input_ids, attention_mask=None):
        return self.text_model(input_ids).mean(1)


@pytest.mark.parametrize('head', ['mmret_add', 'clip4cir_combiner', 'magiclens'])
def test_live_scores_match_direct_encoding_and_target_gradients(head):
    torch.manual_seed(42)
    init = Path(__file__).resolve().parents[3] / 'models/head_architecture_sources/magiclens_large_init_seed42.npz'
    if head == 'magiclens' and not init.exists():
        pytest.skip('Generate official Flax oracle first')
    model = ComposedRetrievalModel(TinyClip(), embedding_dim=768, hidden_dim=1024,
        dropout=.1, freeze_backbone=False, temperature=.07, learnable_temperature=False,
        composition_head=head, fusion_init_path=str(init) if head == 'magiclens' else None)
    model.empty_input_ids = torch.tensor([[0]])
    model.eval()
    batch = {'unique_pixel_values': torch.randn(5, 4),
             'reference_image_positions': torch.tensor([0, 1]),
             'candidate_image_positions': torch.tensor([[2, 3, 4], [4, 2, 3]]),
             'input_ids': torch.tensor([[1, 2], [3, 4]]), 'attention_mask': None,
             'negative_query_input_ids': None, 'negative_query_attention_mask': None,
             'negative_query_owners': torch.empty(0, dtype=torch.long),
             'negative_query_positions': torch.empty(0, dtype=torch.long),
             'candidate_is_negative_query': torch.zeros(2, 3, dtype=torch.bool)}
    actual = _compute_live_student_scores(model=model, batch=batch, live_image_batch_size=3)
    pixels = batch['unique_pixel_values']
    ref = model.encode_reference(pixels[:2])
    assert torch.allclose(ref, model._encode_image_raw(pixels[:2]))
    queries = model.compose(ref, model.encode_text(batch['input_ids'], None))
    targets = model.encode_image(pixels)
    expected = torch.einsum('bd,bkd->bk', queries, targets[batch['candidate_image_positions']])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    if head == 'magiclens':
        assert not torch.allclose(targets, torch.nn.functional.normalize(model._encode_image_raw(pixels), dim=-1))
    (actual * torch.randn_like(actual)).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    if head == 'magiclens':
        assert model.backbone.text_model.weight.grad[0].abs().sum() > 0, 'Gallery empty text must receive gradients'
    else:
        assert model.backbone.text_model.weight.grad[0].abs().sum() == 0
