"""Recover historical paper tables and index surviving experiment artifacts."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def markdown_rows(path):
    path = ROOT / path
    rows, header, section = [], None, ''
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith('#'):
            section = line.lstrip('# ').strip()
        if not line.startswith('|'):
            header = None
            continue
        cells = [v.strip().replace(r'\|', '|').replace(' ↑', '').replace(' ↓', '')
                 for v in re.split(r'(?<!\\)\|', line.strip('|'))]
        if all(re.fullmatch(r'[-: ]+', c) for c in cells):
            continue
        if header is None:
            header = cells
            continue
        assert len(cells) == len(header), (path.name, line_number)
        rows.append(dict(zip(header, cells), _section=section,
                         _source=str(path), _line=line_number))
    return rows


PAPER = markdown_rows('PAPER_MAIN_TABLES.md')
QWEN = markdown_rows('QWEN_COMPLETED_EXPERIMENTS_RESULTS.md')
INST = markdown_rows('INSTRUCT_ORIGINAL_FINAL_RESULTS.md')


def pick(rows, required, filters=None, section=None):
    matches = [r for r in rows if set(required) <= set(r)
               and all(r.get(k) == v for k, v in (filters or {}).items())
               and (section is None or r['_section'] == section)]
    assert len(matches) == 1, (required, filters, section, len(matches))
    return matches[0]


def ref(row):
    return row['_source'] + ':' + str(row['_line'])


def interval(text):
    return [float(v) for v in text.strip('[]').split(',')]


def qdiag_record(rows, kind, filters, subset):
    if kind == 'paper':
        section = 'QDiag — ' + subset.replace('|', ' / ')
        row = pick(rows, ['mean_target_delta', 'target_stable_win_rate (%)'], filters, section)
        ci = pick(rows, ['stable_win_rate_ci_low (%)'], filters, section)
        lo, hi = float(ci['stable_win_rate_ci_low (%)']), float(ci['stable_win_rate_ci_high (%)'])
        return {'subset': subset, 'groups': int(row['group_count']),
                'instructions': int(row['instruction_count']), 'edits': int(row['negative_count']),
                'mean_delta': float(row['mean_target_delta']),
                'DRR_percent': float(row['target_raw_win_rate (%)']),
                'CRR_percent': float(row['target_stable_win_rate (%)']),
                'CRR_CI_low_percent': lo, 'CRR_CI_high_percent': hi,
                'b_psi': float(row['normal_perturbation_threshold']),
                'source_metrics': ref(row), 'source_CI': ref(ci), 'source_threshold': ref(row)}
    row = pick(rows, ['Mean Target Δ', 'Stable Win（%）'], dict(filters, Subset=subset))
    threshold = pick(rows, ['Normal Perturbation Threshold'], filters)
    lo, hi = interval(row['Stable Win CI（%）'])
    return {'subset': subset, 'groups': int(row['Groups']), 'instructions': int(row['Instructions']),
            'edits': int(row['Negatives']), 'mean_delta': float(row['Mean Target Δ']),
            'DRR_percent': float(row['Raw Win（%）']), 'CRR_percent': float(row['Stable Win（%）']),
            'CRR_CI_low_percent': lo, 'CRR_CI_high_percent': hi,
            'b_psi': float(threshold['Normal Perturbation Threshold']),
            'source_metrics': ref(row), 'source_CI': ref(row), 'source_threshold': ref(threshold)}


def retrieval_record(rows, kind, filters):
    if kind == 'paper':
        cirr = pick(rows, ['Group R@1', 'R@10'], filters, '表 1：CIRR 完整检索指标')
        fiq = pick(rows, ['R@10'], filters, '2a：FashionIQ 宏平均')
        pp = pick(rows, ['mAP@10'], filters, '2d：PinPoint mAP')
        neg = pick(rows, ['NegRecall@10'], filters, '2g：PinPoint NegRecall（越低越好）')
        map10, neg10 = float(pp['mAP@10']) * 100, float(neg['NegRecall@10']) * 100
    else:
        cirr = pick(rows, ['Group R@1', 'R@10'], filters)
        fiq = pick(rows, ['R@10'], dict(filters, 类别='Macro Average'))
        pp = neg = pick(rows, ['mAP@K', 'NegRecall@K'], dict(filters, K='10'))
        map10, neg10 = float(pp['mAP@K']) * 100, float(pp['NegRecall@K']) * 100
    return {'CIRR_R1_percent': float(cirr['R@1']), 'CIRR_R10_percent': float(cirr['R@10']),
            'CIRR_subset_R1_percent': float(cirr['Group R@1']), 'PinPoint_mAP10_percent': map10,
            'PinPoint_NegRecall10_percent': neg10, 'FashionIQ_macro_R10_percent': float(fiq['R@10']),
            'FashionIQ_macro_R50_percent': float(fiq['R@50']), 'source_CIRR': ref(cirr),
            'source_FashionIQ': ref(fiq), 'source_PinPoint_mAP': ref(pp), 'source_PinPoint_NegRecall': ref(neg)}


def recover_main_tables():
    models = [('BGE-MLLM-S2', 'Teacher', 'paper', {'模型／方法': 'S2 教师'}),
              ('Qwen3-VL-Emb-8B', 'Teacher', 'paper', {'模型／方法': 'Qwen-8B 教师'})]
    for model in ['CLIP-L', 'CLIP-B']:
        for method, suffix in [('Original', '原始'), ('InfoNCE', 'InfoNCE'),
                               ('CMSE', 'Centered MSE ← S2'), ('OPRD', 'OPRD ← S2')]:
            models.append((model, method, 'paper', {'模型／方法': model + ' ' + suffix}))
    for model, name in [('Qwen3-VL-Emb-2B', 'Embedding-2B'), ('Qwen3-VL-2B-Instruct', 'Instruct-2B')]:
        models.append((model, 'Original', 'paper' if name == 'Embedding-2B' else 'instruct',
                       {'模型／方法': 'Qwen-2B 原始'} if name == 'Embedding-2B' else {}))
        for method, name_in_report in [('InfoNCE', 'InfoNCE'), ('CMSE', 'CMSE-S2'), ('OPRD', 'OPRD-S2')]:
            models.append((model, method, 'qwen', {'模型': name, '方法': name_in_report}))
    qdiag, retrieval, subsets = [], [], []
    for model, method, kind, filters in models:
        rows = {'paper': PAPER, 'qwen': QWEN, 'instruct': INST}[kind]
        key = {'model': model, 'method': method, 'value_origin': 'preserved historical report'}
        qdiag.append(dict(key, **qdiag_record(rows, kind, filters, 'all')))
        retrieval.append(dict(key, **retrieval_record(rows, kind, filters)))
        for subset in ['all', 'strict', 'add', 'substitute', 'delete', 'object', 'attribute', 'predicate',
                       'strict|object', 'strict|attribute', 'strict|predicate']:
            subsets.append(dict(key, **qdiag_record(rows, kind, filters, subset)))
    assert len(qdiag) == len(retrieval) == 18 and len(subsets) == 198
    write_csv('paper_qdiag_main_18rows.csv', qdiag)
    write_csv('paper_retrieval_main_18rows.csv', retrieval)
    write_csv('paper_qdiag_all_subsets_198rows.csv', subsets)
    return qdiag, retrieval


def artifact(path, category, model, purpose):
    path = Path(path)
    return {'category': category, 'model': model, 'purpose': purpose, 'path': str(path),
            'exists': path.exists(), 'bytes': path.stat().st_size if path.is_file() else None}


def verify_qdiag(directory):
    df = pd.read_csv(directory / 'per_negative_random10.csv')
    assert len(df) == df.negative_id.nunique() == 5576
    values = df.groupby(['group_id', 'instruction_id'])[
        ['mean_target_delta', 'target_raw_win_rate', 'target_stable_win_rate']].mean().groupby(level=0).mean().mean()
    row = next(r for r in csv_rows(directory / 'paper_metrics.csv') if r['subset'] == 'all')
    assert np.isclose(values.mean_target_delta, float(row['mean_delta']), atol=1e-10)
    assert np.isclose(values.target_raw_win_rate * 100, float(row['DRR_percent']), atol=1e-10)
    assert np.isclose(values.target_stable_win_rate * 100, float(row['CRR_percent']), atol=1e-10)
    return {'directory': str(directory), 'edits': len(df), 'groups': int(df.group_id.nunique()),
            'hierarchical_summary_recomputed_and_matches': True}


def main():
    qdiag, historical_retrieval = recover_main_tables()
    artifacts, checks = [], []
    eq = ROOT / 'outputs/external_qdiag_20261109'
    er = ROOT / 'outputs/external_retrieval_20261109'
    hr = ROOT / 'outputs/head_ablation_oprd_20261109'
    summaries = read_json(er / 'all_metrics.json')
    external, external_subsets = [], []
    for model, result in summaries.items():
        qrows = csv_rows(eq / model / 'paper_metrics.csv')
        qr = next(r for r in qrows if r['subset'] == 'all')
        cirr, fiq, pp = result['cirr'], result['fashioniq']['macro_average'], result['pinpoint']
        external.append({'model': model, 'CIRR_R1_percent': cirr['recall_at_1'],
                         'CIRR_R10_percent': cirr['recall_at_10'], 'CIRR_subset_R1_percent': cirr['group_recall_at_1'],
                         'FashionIQ_macro_R10_percent': fiq['recall_at_10'], 'FashionIQ_macro_R50_percent': fiq['recall_at_50'],
                         'PinPoint_mAP10_percent': 100 * pp['mAP@10'], 'PinPoint_NegRecall10_percent': 100 * pp['NegRecall@10'],
                         **{k: float(qr[k]) for k in ['mean_delta', 'DRR_percent', 'CRR_percent', 'CRR_CI_low_percent', 'CRR_CI_high_percent', 'b_psi']}})
        external_subsets.extend(dict(model=model, **r) for r in qrows)
        for path, purpose in [(eq / model / 'paper_metrics.csv', 'QDiag summaries by subset'),
                              (eq / model / 'per_negative_random10.csv', '5576 per-edit scores and labels'),
                              (eq / model / 'calibration.json', 'QDiag calibration'),
                              (eq / model / 'runner.log', 'QDiag execution log'),
                              (er / model / 'runner.log', 'retrieval execution log'),
                              (er / model / 'cirr_val/val_metrics.json', 'CIRR results'),
                              (er / model / 'fashioniq_metrics.json', 'FashionIQ macro and category results'),
                              (er / model / 'pinpoint/metrics.json', 'PinPoint official results'),
                              (er / model / 'pinpoint/top50.npz', 'PinPoint Top-50 results'),
                              (er / model / f'pinpoint/{model}.json', 'PinPoint submission predictions'),
                              (Path(cirr['checkpoint']), 'evaluated official checkpoint')]:
            artifacts.append(artifact(path, 'external_model', model, purpose))
        checks.append(verify_qdiag(eq / model))
    write_csv('extra_cir_models_complete_metrics.csv', external)
    write_csv('extra_cir_models_qdiag_subsets.csv', external_subsets)
    heads = csv_rows(hr / 'comparison_with_original_head.csv')
    write_csv('heads_comparison_with_original.csv', heads)
    write_csv('heads_qdiag_initial_final_subsets.csv', csv_rows(hr / 'qdiag_all_subsets.csv'))
    historical_qdiag = next(r for r in qdiag if r['model'] == 'CLIP-L' and r['method'] == 'OPRD')
    historical_ret = next(r for r in historical_retrieval if r['model'] == 'CLIP-L' and r['method'] == 'OPRD')
    diagnostic_keys = ['mean_delta', 'DRR_percent', 'CRR_percent', 'CRR_CI_low_percent', 'CRR_CI_high_percent', 'b_psi']
    retrieval_keys = ['CIRR_R1_percent', 'CIRR_R10_percent', 'CIRR_subset_R1_percent',
                      'FashionIQ_macro_R10_percent', 'FashionIQ_macro_R50_percent',
                      'PinPoint_mAP10_percent', 'PinPoint_NegRecall10_percent']
    head_details = [{'head': 'original_residual_mlp', 'result_origin': 'historical_run_log_and_report',
                     **{k: historical_ret[k] for k in retrieval_keys},
                     **{k: historical_qdiag[k] for k in diagnostic_keys}}]
    for head in ['mmret_add', 'clip4cir_combiner', 'magiclens']:
        run = hr / head
        cirr = read_json(run / 'retrieval/cirr_val/val_metrics.json')
        fiq = read_json(run / 'retrieval/fashioniq_metrics.json')['macro_average']
        pp = read_json(run / 'retrieval/pinpoint/metrics.json')
        qr = next(r for r in csv_rows(run / 'qdiag_final/paper_metrics.csv') if r['subset'] == 'all')
        head_details.append({'head': head, 'result_origin': 'head_ablation_20261109_epoch4',
                             'CIRR_R1_percent': cirr['recall_at_1'], 'CIRR_R10_percent': cirr['recall_at_10'],
                             'CIRR_subset_R1_percent': cirr['group_recall_at_1'],
                             'FashionIQ_macro_R10_percent': fiq['recall_at_10'], 'FashionIQ_macro_R50_percent': fiq['recall_at_50'],
                             'PinPoint_mAP10_percent': 100 * pp['mAP@10'], 'PinPoint_NegRecall10_percent': 100 * pp['NegRecall@10'],
                             **{k: float(qr[k]) for k in diagnostic_keys}})
        files = [('train.log', 'training log'), ('train_metrics.jsonl', 'training metrics'),
                 ('effective_config.json', 'effective training configuration'), ('run_summary.json', 'training summary'),
                 ('qdiag_final.log', 'QDiag execution log'), ('retrieval.log', 'retrieval execution log'),
                 ('qdiag_final/per_negative_random10.csv', '5576 per-edit scores and labels'),
                 ('qdiag_final/paper_metrics.csv', 'QDiag summaries by subset'),
                 ('retrieval/cirr_val/val_metrics.json', 'CIRR results'),
                 ('retrieval/fashioniq_metrics.json', 'FashionIQ macro and category results'),
                 ('retrieval/pinpoint/metrics.json', 'PinPoint official results'),
                 ('retrieval/pinpoint/top50.npz', 'PinPoint Top-50 results')]
        files += [(f'checkpoint_epoch_{e:03d}.pt', f'epoch {e} checkpoint') for e in range(1, 5)]
        artifacts.extend(artifact(run / name, 'head_ablation', head, purpose) for name, purpose in files)
        checks.append(verify_qdiag(run / 'qdiag_final'))
    write_csv('heads_complete_metrics.csv', head_details)
    for filename in ['PAPER_MAIN_TABLES.md', 'QWEN_COMPLETED_EXPERIMENTS_RESULTS.md',
                     'EMBEDDING_OPRD_S2_FINAL_RESULTS.md', 'INSTRUCT_OPRD_S2_FINAL_RESULTS.md',
                     'INSTRUCT_ORIGINAL_FINAL_RESULTS.md', 'TARGET_INSERTION_ABLATION_REPORT.md',
                     'epoch_target_filter.log', 'main_comparison_e4.log', 'main_comparison_e4_resume.log',
                     'qwen_embedding_shared_2gpu_e1.log', 'qwen_instruct_shared_e4_2.log', 'qwen_instruct_original.log']:
        artifacts.append(artifact(ROOT / filename, 'historical_main_tables', 'multiple', 'preserved report or execution log'))
    for directory in ['outputs/epoch_target_filter_e4/kl_all_inject', 'outputs/epoch_target_filter_e4/kl_all_noinject',
                      'outputs/qwen_embedding_shared_2gpu_e1', 'outputs/qwen_instruct_shared_e4', 'outputs/main_comparison_e4']:
        artifacts.append(artifact(ROOT / directory, 'historical_run_directory', 'multiple', 'original output location'))
    artifacts.append(artifact(ROOT.parent / 'release_code/OPRD/checkpoints/clip_l_s2/model.safetensors',
                              'released_original_head', 'CLIP-L OPRD', 'preserved inference weights'))
    candidates = csv_rows(hr / 'original_head_candidate_ablation_crr.csv')
    write_csv('candidate_selection_target_insertion.csv', candidates)
    write_csv('artifact_index.csv', artifacts)
    write_json('raw_qdiag_verification.json', checks)
    write_json('inventory_notes.json', {
        'paper_table_1_and_2_rows_recovered': 18,
        'historical_values_origin': 'Recovered from preserved reports with row-specific file/line references; not newly recomputed from absent old outputs.',
        'paper_qdiag_scope': 'pilot500 / all: 499 groups with accepted edits, 1862 instructions, 5576 edits; 500 groups retained for calibration. Distinct from newer full-release 10131-edit rank-alignment audit.',
        'pinpoint_units': 'CSV exports multiply historical 0-1 PinPoint metrics by 100. CIRR and FashionIQ already percentages.',
        'extra_models': 'Four completed checkpoints: CLIP4CIR CIRR RN50x4, TEMA MCIRR, MagicLens CLIP Base and Large. QuRe CIRR was skipped because weights were unavailable.',
        'head_protocol': read_json(hr / 'experiment_protocol.json'),
        'head_architecture_checks': read_json(hr / 'architecture_equivalence.json'),
        'original_head_comparison_sources': read_json(hr / 'comparison_with_original_head_sources.json'),
        'limitations': ['Some historical run directories are absent at their original paths. Their log/report values survive, but full per-query artifacts are not claimed for every historical model.',
                        'On-policy without target insertion has strict CRR=10.1565%; its all-operation CRR is not present in the located historical records and remains blank.',
                        'Use QWEN_COMPLETED_EXPERIMENTS_RESULTS.md for final trained Qwen rows; older Qwen rows in PAPER_MAIN_TABLES.md describe different earlier runs.',
                        'Original-head versus new-head comparison uses preserved historical original-head results and later new-head runs; it is not a same-run retraining of all four heads.']})
    text = ['Experiment results inventory', '',
            'paper_qdiag_main_18rows.csv: recovered first screenshot, including CRR CI and threshold.',
            'paper_retrieval_main_18rows.csv: recovered second screenshot, all seven retrieval metrics.',
            'paper_qdiag_all_subsets_198rows.csv: all/strict/operation/semantic-category breakdowns.',
            'candidate_selection_target_insertion.csv: third screenshot plus available strict/all CRR.',
            'extra_cir_models_complete_metrics.csv: four extra model checkpoints, retrieval and QDiag.',
            'heads_comparison_with_original.csv: original head and three replacement heads.',
            'heads_complete_metrics.csv: full seven retrieval metrics and QDiag mean/DRR/CRR/CI/threshold for all four heads.',
            'artifact_index.csv: exact paths, existence and file sizes for logs, checkpoints and raw data.',
            'inventory_notes.json: protocols, caveats and historical-run provenance.',
            'raw_qdiag_verification.json: per-edit records independently re-aggregated for seven new runs.', '',
            'Missing historical original output directories are explicitly marked; they are not reconstructed raw data.',
            'No-insertion all-operation CRR remains missing; strict CRR must not be substituted for it.']
    (OUT / 'INDEX.txt').write_text('\n'.join(text) + '\n')
    print(json.dumps({'output_directory': str(OUT), 'main_table_rows': len(qdiag),
                      'extra_models': len(external), 'heads_including_original': len(heads),
                      'new_runs_with_raw_qdiag_verified': len(checks),
                      'missing_artifacts': [a['path'] for a in artifacts if not a['exists']]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
