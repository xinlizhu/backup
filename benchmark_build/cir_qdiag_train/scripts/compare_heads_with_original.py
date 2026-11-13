"""Recover the historical original-head metrics and compare the three new heads.

The historical row is recovered from its original run log and paper report;
it is not a fresh evaluation or retraining in the head-ablation environment.
All five comparison columns use percentages, including both PinPoint metrics.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'outputs/head_ablation_oprd_20261109'
BUNDLE = PROJECT.parent / 'release_code/OPRD/checkpoints/clip_l_s2'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    recovered = OUTPUT / 'original_head_recovered'
    recovered.mkdir(exist_ok=True)
    log_path = PROJECT / 'epoch_target_filter.log'
    log = log_path.read_bytes().decode('utf-8')
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r'(?m)^\{', log):
        try:
            obj, end = decoder.raw_decode(log[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append((obj, log[:match.start()].count('\n') + 1))
    suffix = '/epoch_target_filter_e4/kl_all_inject/checkpoint_epoch_004.pt'
    cirr, cirr_line = next((obj, line) for obj, line in objects
        if str(obj.get('checkpoint', '')).endswith(suffix) and 'recall_at_1' in obj)
    fiq, fiq_line = next((obj, line) for obj, line in objects
        if str(obj.get('model_source', '')).endswith(suffix) and 'macro_average' in obj)
    match = re.search(r'\[metric\] CLIP-L-EPOCH-kl_all_inject mAP@10=([\d.]+) NegRecall@10=([\d.]+)', log)
    assert match is not None
    pinpoint = {'mAP@10': float(match[1]), 'NegRecall@10': float(match[2])}
    pinpoint_line = log[:match.start()].count('\n') + 1
    report_path = PROJECT / 'PAPER_MAIN_TABLES.md'
    report = report_path.read_text()
    section = report.split('### QDiag — all\n', 1)[1]
    line = next(s for s in section.splitlines() if s.startswith('| CLIP-L OPRD ← S2 |'))
    cells = [s.strip() for s in line.strip('|').split('|')]
    qdiag = dict(group_count=int(cells[1]), instruction_count=int(cells[2]), negative_count=int(cells[3]),
        mean_delta_reported=float(cells[4]), DRR_percent_reported=float(cells[5]),
        CRR_percent_reported=float(cells[6]), b_psi_reported=float(cells[7]),
        subset='all (ADD + SUBSTITUTE + DELETE)', source_precision='rounded values preserved in historical report')
    report_line = report[:report.index(line)].count('\n') + 1
    bundled = json.loads((BUNDLE / 'metrics.json').read_text())
    assert all(cirr[k] == v for k, v in bundled.items())
    expected = json.loads((BUNDLE / 'checksum.json').read_text())['model.safetensors']
    with (BUNDLE / 'model.safetensors').open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    assert actual == expected
    write_json(recovered / 'cirr_metrics_from_log.json', cirr)
    write_json(recovered / 'fashioniq_metrics_from_log.json', fiq)
    write_json(recovered / 'pinpoint_selected_metrics_from_log.json', pinpoint)
    write_json(recovered / 'qdiag_all_from_historical_report.json', qdiag)
    fields = ['head', 'result_origin', 'CIRR_R1_percent', 'FashionIQ_macro_R10_percent',
              'PinPoint_mAP10_percent', 'PinPoint_NegRecall10_percent', 'CRR_all_percent']
    rows = [dict(zip(fields, ['original_residual_mlp', 'historical_run_log_and_report',
        cirr['recall_at_1'], fiq['macro_average']['recall_at_10'],
        100 * pinpoint['mAP@10'], 100 * pinpoint['NegRecall@10'], qdiag['CRR_percent_reported']]))]
    provenance = {'original_residual_mlp': {
        'run': 'epoch_target_filter_e4/kl_all_inject', 'checkpoint_epoch': 4,
        'CIRR': {'file': str(log_path), 'line': cirr_line},
        'FashionIQ': {'file': str(log_path), 'line': fiq_line},
        'PinPoint': {'file': str(log_path), 'line': pinpoint_line},
        'CRR': {'file': str(report_path), 'line': report_line},
        'available_inference_weights': str(BUNDLE / 'model.safetensors'),
        'weight_sha256_verified': actual,
        'historical_run_directory_present': (PROJECT / 'outputs/epoch_target_filter_e4/kl_all_inject').exists(),
        'rerun_in_current_environment': False,
    }}
    for head in ['mmret_add', 'clip4cir_combiner', 'magiclens']:
        folder = OUTPUT / head
        paths = {'CIRR': folder / 'retrieval/cirr_val/val_metrics.json',
                 'FashionIQ': folder / 'retrieval/fashioniq_metrics.json',
                 'PinPoint': folder / 'retrieval/pinpoint/metrics.json',
                 'CRR': folder / 'qdiag_final/paper_metrics.csv'}
        c, f, p = [json.loads(paths[k].read_text()) for k in ['CIRR', 'FashionIQ', 'PinPoint']]
        with paths['CRR'].open() as handle:
            q = next(row for row in csv.DictReader(handle) if row['subset'] == 'all')
        assert (c['num_queries'], f['num_queries'], int(p['total_queries'])) == (4181, 6016, 7635)
        rows.append(dict(zip(fields, [head, 'head_ablation_20261109_epoch4', c['recall_at_1'],
            f['macro_average']['recall_at_10'], 100*p['mAP@10'], 100*p['NegRecall@10'], float(q['CRR_percent'])])))
        provenance[head] = {k: str(v) for k, v in paths.items()}
    with (OUTPUT / 'comparison_with_original_head.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(OUTPUT / 'comparison_with_original_head_sources.json', {
        'units': 'all five performance columns are percentages',
        'CRR_subset': 'all; includes ADD, SUBSTITUTE, DELETE; not strict',
        'common_recipe': 'CLIP-L/14, BGE-VL-MLLM-S2, OPRD reverse KL T=0.05, on-policy K=32 with target insertion, seed42, epoch4',
        'comparison_type': 'Historical original-head results compared with new-head runs; not a same-run retraining of all four heads',
        'sources': provenance,
    })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
