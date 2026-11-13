"""Actual CLIP-L, actual CIRR images: validate one OPRD backward pass per head."""
import json
from pathlib import Path
import sys
import gc
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from cir_qdiag_train.model import ComposedRetrievalModel, save_checkpoint
from cir_qdiag_train.on_policy_config import load_on_policy_config
from cir_qdiag_train.on_policy_data import load_image_map, load_cirr_positive_records
from cir_qdiag_train.on_policy_trainer import _compute_live_student_scores
from cir_qdiag_train.trainer import _audit_trainable_gradients
from cir_qdiag_train.on_policy_loss import on_policy_centered_distillation_loss


def main():
    torch.set_num_threads(8)
    output = PROJECT / 'outputs/head_ablation_oprd_20261109'
    results = {}
    for head in ['mmret_add', 'clip4cir_combiner', 'magiclens']:
        torch.manual_seed(42)
        c = load_on_policy_config(PROJECT / f'configs/head_{head}_oprd_s2_e4.yaml')
        kw = c.model.model_dump()
        pretrained = kw.pop('pretrained_model_name_or_path')
        model, processor = ComposedRetrievalModel.from_pretrained(pretrained, **kw,
            temperature=c.optimizer.temperature, learnable_temperature=c.optimizer.learnable_temperature)
        ids, paths = load_image_map(c.data.train_image_split_path, c.data.image_root)
        rows = load_cirr_positive_records(c.data.train_annotation_path, ids, paths)[:2]
        selected = [r.reference_path for r in rows] + [r.target_path for r in rows] + paths[:30]
        images = []
        for path in selected:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        pixels = processor(images=images, return_tensors='pt')['pixel_values']
        texts = processor(text=[r.text for r in rows], padding=True, truncation=True, return_tensors='pt')
        candidates = torch.stack([torch.arange(2, 34), torch.cat([torch.tensor([3, 2]), torch.arange(4, 34)])])
        batch = dict(unique_pixel_values=pixels, reference_image_positions=torch.tensor([0, 1]),
            candidate_image_positions=candidates, input_ids=texts['input_ids'], attention_mask=texts.get('attention_mask'),
            negative_query_input_ids=None, negative_query_attention_mask=None,
            negative_query_owners=torch.empty(0,dtype=torch.long), negative_query_positions=torch.empty(0,dtype=torch.long),
            candidate_is_negative_query=torch.zeros(2, 32, dtype=torch.bool))
        model.cuda().train()
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scores = _compute_live_student_scores(model=model, batch=batch, live_image_batch_size=32)
        loss, _ = on_policy_centered_distillation_loss(student_scores=scores,
            teacher_scores=torch.randn_like(scores) * .1,
            teacher_target_top1=torch.ones(2,device='cuda',dtype=torch.bool),
            raw_target_in_topk=torch.ones(2,device='cuda',dtype=torch.bool),
            rank_weight=1., centered_response_weight=0., rank_temperature=.05,
            centered_response_beta=.01, require_teacher_target_top1=False, rank_kl_direction='reverse')
        loss.backward()
        results[head] = {'loss': float(loss.detach()), **model.parameter_summary(), **_audit_trainable_gradients(model)}
        # Save the untouched initialization for fair before/after diagnostics.
        save_checkpoint(output / head / 'initial.pt', model=model, optimizer=None, scheduler=None,
                        epoch=-1, global_step=0, config=c.model_dump(mode='json'))
        del model, processor, batch, scores, loss
        gc.collect(); torch.cuda.empty_cache()
        (output / 'gpu_smoke.json').write_text(json.dumps(results, indent=2))
        print(head, results[head], flush=True)


if __name__ == '__main__':
    main()
