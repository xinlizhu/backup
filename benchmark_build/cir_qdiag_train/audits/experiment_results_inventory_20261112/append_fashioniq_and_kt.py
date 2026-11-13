"""Append category-wise FashionIQ results and preserved K/temperature tables."""
from copy import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'excel_dependencies'))
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from export_inventory import ROOT, PAPER, QWEN, INST, csv_rows, pick, ref

SOURCE = HERE / 'CIR实验结果汇总_配置更新_20261112.xlsx'
OUTPUT = HERE / 'CIR实验结果汇总_含FashionIQ分类及KT实验_20261112.xlsx'
CATEGORIES = [('dress', 'Dress'), ('shirt', 'Shirt'), ('toptee', 'Toptee'), ('macro_average', 'Macro Average')]
HEADS = {'original_residual_mlp': '原始残差 MLP 头', 'mmret_add': 'MMRet 加法头',
         'clip4cir_combiner': 'CLIP4CIR Combiner', 'magiclens': 'MagicLens 融合头'}
recall_keys = ['R1', 'R5', 'R10', 'R50']


def report_spec(model, method):
    if method == 'Teacher':
        return PAPER, 'paper', {'模型／方法': 'S2 教师' if model == 'BGE-MLLM-S2' else 'Qwen-8B 教师'}
    if model in ['CLIP-L', 'CLIP-B']:
        suffix = {'Original': '原始', 'InfoNCE': 'InfoNCE', 'CMSE': 'Centered MSE ← S2', 'OPRD': 'OPRD ← S2'}[method]
        return PAPER, 'paper', {'模型／方法': model + ' ' + suffix}
    if method == 'Original':
        if model == 'Qwen3-VL-Emb-2B':
            return PAPER, 'paper', {'模型／方法': 'Qwen-2B 原始'}
        return INST, 'instruct', {}
    return QWEN, 'qwen', {'模型': 'Embedding-2B' if model == 'Qwen3-VL-Emb-2B' else 'Instruct-2B',
                          '方法': {'InfoNCE': 'InfoNCE', 'CMSE': 'CMSE-S2', 'OPRD': 'OPRD-S2'}[method]}


def from_report(model, method):
    records, kind, filters = report_spec(model, method)
    result = []
    for key, label in CATEGORIES:
        if kind == 'paper':
            section = '2a：FashionIQ 宏平均' if key == 'macro_average' else '2c：FashionIQ — ' + key
            record = pick(records, ['R@1', 'R@5', 'R@10', 'R@50'], filters, section)
            queries, gallery = record['Queries'], record.get('Gallery')
        else:
            record = pick(records, ['R@1', 'R@5', 'R@10', 'R@50'], dict(filters, 类别=label if key == 'macro_average' else key))
            queries, gallery = record['查询数'], record.get('Gallery规模')
        result.append({'model': model, 'method': method, 'category': label, 'queries': int(queries),
                       'gallery': int(gallery) if gallery and gallery != '—' else None,
                       **{out: float(record[source]) for out, source in zip(recall_keys, ['R@1', 'R@5', 'R@10', 'R@50'])},
                       'source': ref(record)})
    return result


def from_json(path, model, method):
    data = json.loads(path.read_text())
    result = []
    for key, label in CATEGORIES:
        values = data['macro_average'] if key == 'macro_average' else data['categories'][key]
        result.append({'model': model, 'method': method, 'category': label,
                       'queries': data['num_queries'] if key == 'macro_average' else values['num_queries'],
                       'gallery': values.get('gallery_size'),
                       **{out: values[source] for out, source in zip(recall_keys, ['recall_at_1', 'recall_at_5', 'recall_at_10', 'recall_at_50'])},
                       'source': str(path) + ('#macro_average' if key == 'macro_average' else '#categories.' + key)})
    return result


def check_macro(rows):
    checks = []
    keys = list(dict.fromkeys((r['model'], r['method']) for r in rows))
    for model, method in keys:
        selected = [r for r in rows if (r['model'], r['method']) == (model, method)]
        assert len(selected) == 4
        mean = next(r for r in selected if r['category'] == 'Macro Average')
        cats = [r for r in selected if r['category'] != 'Macro Average']
        assert sum(r['queries'] for r in cats) == mean['queries'] == 6016
        errors = {key: abs(sum(r[key] for r in cats) / 3 - mean[key]) for key in recall_keys}
        assert max(errors.values()) <= 0.00011, (model, method, errors)
        checks.append({'model': model, 'method': method, 'maximum_rounding_difference': max(errors.values())})
    return checks


main = []
main_summary = csv_rows(HERE / 'paper_retrieval_main_18rows.csv')
for record in main_summary:
    main.extend(from_report(record['model'], record['method']))
external = []
for record in csv_rows(HERE / 'extra_cir_models_complete_metrics.csv'):
    model = record['model']
    external.extend(from_json(ROOT / 'outputs/external_retrieval_20261109' / model / 'fashioniq_metrics.json', model, '官方权重'))
heads = []
for record in from_report('CLIP-L', 'OPRD'):
    heads.append(dict(record, model=HEADS['original_residual_mlp']))
for head in ['mmret_add', 'clip4cir_combiner', 'magiclens']:
    heads.extend(from_json(ROOT / 'outputs/head_ablation_oprd_20261109' / head / 'retrieval/fashioniq_metrics.json', HEADS[head], 'OPRD'))
assert [len(main), len(external), len(heads)] == [72, 16, 16]
macro_checks = check_macro(main) + check_macro(external) + check_macro(heads)
for row in main_summary:
    macro = next(r for r in main if r['model'] == row['model'] and r['method'] == row['method'] and r['category'] == 'Macro Average')
    assert macro['R10'] == float(row['FashionIQ_macro_R10_percent'])
    assert macro['R50'] == float(row['FashionIQ_macro_R50_percent'])

kt = []
for k, t, config_name in [(8, '0.05', 'opd_clip_l_s2_k8_forward_t005.yaml'),
                           (16, '0.05', 'opd_clip_l_s2_k16_forward_t005.yaml'),
                           (32, '0.05', 'opd_clip_l_s2_k32_reverse_t005.yaml'),
                           (32, '1.0', 'opd_clip_l_s2_k32_reverse_t1.yaml')]:
    record = pick(PAPER, ['K', 'T', 'Training hours'], {'K': str(k), 'T': t})
    config_path = ROOT / 'configs' / config_name
    config = yaml.safe_load(config_path.read_text())
    assert config['mining']['top_k'] == k and config['objective']['rank_temperature'] == float(t)
    kt.append({'K': k, 'T': float(t), 'KL': config['objective']['rank_kl_direction'].title(),
               'teacher_top_rank': config['objective']['teacher_target_max_rank'], 'epochs': config['optimizer']['epochs'],
               'seed': config['experiment']['seed'], 'training_hours': float(record['Training hours']),
               **{key: float(record[label]) for key, label in {
                   'CIRR_R1': 'CIRR R@1', 'CIRR_R5': 'CIRR R@5', 'CIRR_R10': 'CIRR R@10', 'CIRR_R50': 'CIRR R@50',
                   'CIRR_subset_R1': 'Group R@1', 'CIRR_subset_R2': 'Group R@2', 'CIRR_subset_R3': 'Group R@3',
                   'FashionIQ_R10': 'FashionIQ R@10', 'FashionIQ_R50': 'FashionIQ R@50',
                   'mean_delta_strict': 'QDiag Delta', 'DRR_strict': 'QDiag Raw Win (%)',
                   'CRR_strict': 'QDiag Stable Win (%)', 'b_psi': 'QDiag tau95', 'own_minus_random': 'QDiag Own-Random'}.items()},
               'PinPoint_mAP10': float(record['PinPoint mAP@10']) * 100,
               'PinPoint_NegRecall10': float(record['PinPoint NegRecall@10']) * 100,
               'source': ref(record), 'config_source': str(config_path)})
expected_image = [(8, .05, 43.7694, 32.2379, 1.0563), (16, .05, 43.1476, 31.7746, 1.6255),
                  (32, .05, 42.6214, 31.5293, 2.7460), (32, 1., 41.9995, 31.4182, 2.7378)]
assert [tuple(r[k] for k in ['K', 'T', 'CIRR_R1', 'FashionIQ_R10', 'training_hours']) for r in kt] == expected_image


def save_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


save_csv('fashioniq_main_categories.csv', main)
save_csv('fashioniq_extra_model_categories.csv', external)
save_csv('fashioniq_head_categories.csv', heads)
save_csv('candidate_count_temperature_complete.csv', kt)

before = load_workbook(SOURCE)
book = load_workbook(SOURCE)
new_checks, additions = [], []


def add_sheet(name, title, rows, columns, source_file):
    assert name not in book.sheetnames
    ws = book.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = '087E8B'
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    cell = ws.cell(1, 1, title)
    cell.font = Font(name='Calibri', size=17, bold=True, color='FFFFFF')
    cell.fill = PatternFill('solid', fgColor='17365D')
    cell.alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 36
    for column, (key, column_label, width, fmt) in enumerate(columns, 1):
        cell = ws.cell(4, column, column_label)
        cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='087E8B')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.column_dimensions[get_column_letter(column)].width = width
    ws.row_dimensions[4].height = 36
    for row_idx, record in enumerate(rows, 5):
        ws.row_dimensions[row_idx].height = 27
        for column, (key, column_label, width, fmt) in enumerate(columns, 1):
            value = record.get(key)
            if value is None:
                value = '—'
            cell = ws.cell(row_idx, column, value)
            cell.font = Font(name='Calibri', size=11, color='111827')
            cell.fill = PatternFill('solid', fgColor='EAF2F8' if row_idx % 2 == 0 else 'FFFFFF')
            cell.alignment = Alignment(horizontal='right' if isinstance(value, (int, float)) else 'left', vertical='center', wrap_text=True)
            if fmt and isinstance(value, (int, float)):
                cell.number_format = fmt
            if key == 'source':
                ws.row_dimensions[row_idx].height = 58
            if record.get('category') == 'Macro Average':
                cell.font = Font(name='Calibri', size=11, bold=True, color='17365D')
            new_checks.append((name, row_idx, column, value))
    last = 4 + len(rows)
    table = Table(displayName='SupplementTable' + str(len(additions) + 1), ref=f'A4:{get_column_letter(len(columns))}{last}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = 'C5'
    ws.print_title_rows = '1:4'
    ws.print_area = table.ref.replace('A4:', 'A1:')
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3 if len(columns) > 12 else ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    additions.append({'sheet': name, 'title': title, 'rows': len(rows), 'source': source_file})


all_rows = [dict(r, family='主表') for r in main] + [dict(r, family='新增模型') for r in external]
all_rows += [dict(r, family='新增头') for r in heads if r['model'] != HEADS['original_residual_mlp']]
wide = []
for family, model, method in dict.fromkeys((r['family'], r['model'], r['method']) for r in all_rows):
    row = {'family': family, 'model': model, 'method': method}
    for record in all_rows:
        if (record['family'], record['model'], record['method']) == (family, model, method):
            for metric in ['R10', 'R50']:
                row[record['category'] + '_' + metric] = record[metric]
    wide.append(row)
assert len(wide) == 25
base_cols = [('model', '模型 / 合成头', 32, None), ('method', '方法', 17, None)]
wide_cols = [('family', '实验类别', 16, None)] + base_cols
for _, label in CATEGORIES:
    for metric in ['R10', 'R50']:
        wide_cols.append((label + '_' + metric, label + ' ' + metric.replace('R', 'R@') + ' (%)', 23, '0.0000'))
add_sheet('FashionIQ_分类总览', 'FashionIQ · Dress / Shirt / Toptee / Macro Average', wide, wide_cols,
          'fashioniq_main_categories.csv; fashioniq_extra_model_categories.csv; fashioniq_head_categories.csv')
detail_cols = base_cols + [('category', '类别', 22, None), ('queries', 'Queries', 13, '#,##0'), ('gallery', 'Gallery', 13, '#,##0')]
detail_cols += [(key, key.replace('R', 'R@') + ' ↑ (%)', 18, '0.0000') for key in recall_keys]
for name, title, records, filename in [
    ('FashionIQ_主表分类', 'FashionIQ · 原主表模型分类结果', main, 'fashioniq_main_categories.csv'),
    ('FashionIQ_新增模型分类', 'FashionIQ · 新增 CIR 模型分类结果', external, 'fashioniq_extra_model_categories.csv'),
    ('FashionIQ_头消融分类', 'FashionIQ · 原始头与新增头分类结果', heads, 'fashioniq_head_categories.csv')]:
    add_sheet(name, title, records, detail_cols, filename)

simple = [('K', 'K', 12, '0'), ('T', 'T', 14, '0.0#'), ('CIRR_R1', 'CIRR R@1 (%)', 23, '0.0000'),
          ('FashionIQ_R10', 'FashionIQ R@10 (%)', 25, '0.0000'), ('training_hours', 'Training Time (h)', 25, '0.0000')]
add_sheet('Candidate_Count_Temperature', 'Candidate Count and Temperature', kt, simple, 'candidate_count_temperature_complete.csv')
full_cols = simple[:2] + [('KL', 'KL direction', 19, None), ('teacher_top_rank', 'Teacher target max rank', 24, '0'), ('epochs', 'Epochs', 12, '0'), ('seed', 'Seed', 12, '0')]
for key, title in [('CIRR_R1', 'CIRR R@1 (%)'), ('CIRR_R5', 'CIRR R@5 (%)'), ('CIRR_R10', 'CIRR R@10 (%)'),
                   ('CIRR_R50', 'CIRR R@50 (%)'), ('CIRR_subset_R1', 'CIRR Subset R@1 (%)'),
                   ('CIRR_subset_R2', 'CIRR Subset R@2 (%)'), ('CIRR_subset_R3', 'CIRR Subset R@3 (%)'),
                   ('FashionIQ_R10', 'FashionIQ R@10 (%)'), ('FashionIQ_R50', 'FashionIQ R@50 (%)'),
                   ('PinPoint_mAP10', 'PinPoint mAP@10 (%)'), ('PinPoint_NegRecall10', 'PinPoint NegRecall@10 (%)')]:
    full_cols.append((key, title, 23, '0.0000'))
full_cols += [('mean_delta_strict', 'QDiag strict Mean Δ', 23, '0.000000'), ('DRR_strict', 'DRR strict (%)', 21, '0.0000'),
              ('CRR_strict', 'CRR strict (%)', 21, '0.0000'), ('b_psi', 'bψ', 17, '0.000000'),
              ('own_minus_random', 'Own − Random score', 24, '0.000000'), simple[-1]]
add_sheet('K_T实验完整指标', 'Candidate Count and Temperature · 完整指标', kt, full_cols, 'candidate_count_temperature_complete.csv')

sources = [dict(r, family=family) for family, rows in [('主表', main), ('新增模型', external), ('头消融', heads)] for r in rows]
sources += [{'family': 'K / T', 'model': 'CLIP-L', 'method': f"K={r['K']}, T={r['T']}, {r['KL']} KL", 'category': '汇总',
             'source': r['source']} for r in kt]
sources += [{'family': 'K / T', 'model': 'CLIP-L', 'method': f"K={r['K']}, T={r['T']}, {r['KL']} KL", 'category': '训练配置',
             'source': r['config_source']} for r in kt]
add_sheet('补充数据来源', 'FashionIQ 与 K / T 实验 · 来源索引', sources,
          [('family', '实验类别', 17, None)] + base_cols + [('category', '类别', 23, None), ('source', '来源文件 / 字段', 108, None)],
          'fashioniq_*_categories.csv; candidate_count_temperature_complete.csv')
directory = book['目录与数据说明']
for entry in additions:
    row = directory.max_row + 1
    for col, key in enumerate(['sheet', 'title', 'rows', 'source'], 1):
        cell = directory.cell(row, col, entry[key])
        cell.font = Font(name='Calibri', size=11, color='111827')
        cell.alignment = Alignment(vertical='center', wrap_text=True)
    directory.row_dimensions[row].height = 46
    directory.cell(row, 1).hyperlink = f"#'{entry['sheet']}'!A1"
    directory.cell(row, 1).font = Font(name='Calibri', size=11, color='0563C1', underline='single')
next(iter(directory.tables.values())).ref = f'A4:D{directory.max_row}'
directory.print_area = f'A1:D{directory.max_row}'
book.active = book.sheetnames.index('FashionIQ_分类总览')
book.save(OUTPUT)

result = load_workbook(OUTPUT)
for original_sheet in before:
    for row in original_sheet:
        for cell in row:
            assert result[original_sheet.title][cell.coordinate].value == cell.value, (original_sheet.title, cell.coordinate)
for name, row, col, expected in new_checks:
    actual = result[name].cell(row, col).value
    if isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert actual == expected
assert result['分项_原表QDiag']['D5'].value == 930
verification = {'workbook': str(OUTPUT), 'new_sheets': additions, 'new_cells_verified': len(new_checks),
                'existing_cells_unchanged': True, 'macro_average_checks': macro_checks, 'image_table_exact_match': True,
                'KT_KL_directions': [{'K': r['K'], 'T': r['T'], 'KL': r['KL']} for r in kt],
                'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
(HERE / 'fashioniq_kt_excel_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'workbook': str(OUTPUT), 'new_sheets': additions, 'fashioniq_records': len(main) + len(external) + len(heads),
                  'new_cells_verified': len(new_checks), 'existing_cells_unchanged': True}, ensure_ascii=False, indent=2))
