"""Create a formatted, source-traceable workbook from the recovered results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'excel_dependencies'))
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

OUTPUT = ROOT / 'CIR实验结果汇总_20261112.xlsx'
NAVY, TEAL, LIGHT, GRAY = '17365D', '087E8B', 'EAF2F8', '667085'
HEADS = {'original_residual_mlp': '原始残差 MLP 头', 'mmret_add': 'MMRet 加法头',
         'clip4cir_combiner': 'CLIP4CIR Combiner', 'magiclens': 'MagicLens 融合头'}
SUBSETS = {'all': 'ALL', 'strict': 'ADD + SUBSTITUTE', 'add': 'ADD',
           'substitute': 'SUBSTITUTE', 'delete': 'DELETE', 'object': 'Object',
           'attribute': 'Attribute', 'predicate': 'Predicate',
           'strict|object': 'ADD + SUBSTITUTE / Object',
           'strict|attribute': 'ADD + SUBSTITUTE / Attribute',
           'strict|predicate': 'ADD + SUBSTITUTE / Predicate'}


def read(name):
    with (ROOT / name).open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == '':
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def confidence(row):
    return f"[{row['CRR_CI_low_percent']:.2f}, {row['CRR_CI_high_percent']:.2f}]"


def model(row):
    return row.get('model') or HEADS[row['head']]


RETRIEVAL = [('CIRR_R1_percent', 'CIRR R@1 ↑ (%)', 17, '0.00'),
             ('CIRR_R10_percent', 'CIRR R@10 ↑ (%)', 17, '0.00'),
             ('CIRR_subset_R1_percent', 'CIRR Subset R@1 ↑ (%)', 20, '0.00'),
             ('FashionIQ_macro_R10_percent', 'FashionIQ R@10 ↑ (%)', 20, '0.00'),
             ('FashionIQ_macro_R50_percent', 'FashionIQ R@50 ↑ (%)', 20, '0.00'),
             ('PinPoint_mAP10_percent', 'PinPoint mAP@10 ↑ (%)', 21, '0.00'),
             ('PinPoint_NegRecall10_percent', 'PinPoint NegRecall@10 ↓ (%)', 24, '0.00')]
QDIAG = [('mean_delta', 'Mean Δ', 16, '0.000000'), ('DRR_percent', 'DRR ↑ (%)', 15, '0.00'),
         ('CRR_percent', 'CRR ↑ (%)', 15, '0.00'),
         ('CRR_CI_low_percent', 'CRR 95% CI 下限 (%)', 21, '0.00'),
         ('CRR_CI_high_percent', 'CRR 95% CI 上限 (%)', 21, '0.00'),
         ('b_psi', 'bψ', 16, '0.000000')]
MODEL = ('_model', '模型 / Backbone', 32, None)
METHOD = ('method', '方法', 16, None)

wb = Workbook()
wb.remove(wb.active)
wb.properties.title = 'CIR 实验结果汇总'
wb.properties.subject = '新增 CIR 模型、合成头消融及三张原表'
wb.properties.creator = 'CIR experiment results'
expected = []
catalog = []


def sheet(name, title, rows, columns, subtitle='', key_columns=1, source=None):
    ws = wb.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.sheet_properties.tabColor = TEAL if name.startswith('新增') else NAVY
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    ws.cell(1, 1, title).font = Font(name='Calibri', size=17, bold=True, color='FFFFFF')
    ws.cell(1, 1).fill = PatternFill('solid', fgColor=NAVY)
    ws.cell(1, 1).alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 36
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(columns))
    ws.cell(2, 1, subtitle).font = Font(name='Calibri', size=10, color=GRAY)
    ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical='center')
    ws.row_dimensions[2].height = 31
    for j, (_, label, width, _) in enumerate(columns, 1):
        cell = ws.cell(4, j, label)
        cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor=TEAL)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.row_dimensions[4].height = 35
    for i, row in enumerate(rows, 5):
        ws.row_dimensions[i].height = 25
        for j, (key, _, _, number_format) in enumerate(columns, 1):
            value = model(row) if key == '_model' else row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            display_value = '—' if value is None else value
            cell = ws.cell(i, j, display_value)
            expected.append((name, i, j, display_value))
            cell.font = Font(name='Calibri', size=11, color='111827' if value is not None else GRAY)
            cell.fill = PatternFill('solid', fgColor='FFFFFF' if i % 2 else LIGHT)
            cell.alignment = Alignment(horizontal='right' if isinstance(value, (int, float)) else 'left',
                                       vertical='center', wrap_text=True)
            if number_format and isinstance(value, (int, float)):
                cell.number_format = number_format
            if key.startswith('source_') or key == 'path':
                cell.font = Font(name='Calibri', size=10, color=GRAY)
                ws.row_dimensions[i].height = 62
    last = 4 + len(rows)
    if rows:
        tab = Table(displayName=f'DataTable{len(wb.sheetnames)}',
                    ref=f'A4:{get_column_letter(len(columns))}{last}')
        tab.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
        ws.add_table(tab)
    ws.freeze_panes = f'{get_column_letter(key_columns + 1)}5'
    ws.print_title_rows = '1:4'
    ws.print_options.horizontalCentered = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3 if len(columns) > 10 else ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_area = f'A1:{get_column_letter(len(columns))}{max(4, last)}'
    ws.oddFooter.center.text = '第 &P 页 / 共 &N 页'
    ws.oddFooter.right.text = name
    catalog.append({'sheet': name, 'rows': len(rows), 'source': source or '由已核对结果合并', 'title': title})
    return ws


def best(ws, rows, columns, metrics, grouping=None):
    groups = {}
    for i, row in enumerate(rows, 5):
        group = grouping(row) if grouping else 'all'
        groups.setdefault(group, []).append((i, row))
    positions = {c[0]: j for j, c in enumerate(columns, 1)}
    for items in groups.values():
        for key in metrics:
            available = [(i, r[key]) for i, r in items if isinstance(r.get(key), (float, int))]
            if not available:
                continue
            optimum = (min if 'NegRecall' in key else max)(v for _, v in available)
            for i, value in available:
                if abs(value - optimum) < 1e-12:
                    ws.cell(i, positions[key]).font = Font(name='Calibri', size=11, bold=True, color=NAVY)


qmain = read('paper_qdiag_main_18rows.csv')
rmain = read('paper_retrieval_main_18rows.csv')
external = read('extra_cir_models_complete_metrics.csv')
heads = read('heads_complete_metrics.csv')
keyed = {(r['model'], r['method']): r for r in rmain}
overview = []
for q in qmain:
    overview.append(dict(keyed[q['model'], q['method']], **{k: v for k, v in q.items() if k not in {'value_origin'}}, category='原表'))
for r in external:
    overview.append(dict(r, category='新增 CIR 模型', method='官方权重'))
for r in heads:
    if r['head'] != 'original_residual_mlp':
        overview.append(dict(r, category='新增合成头', model='CLIP-L · ' + HEADS[r['head']], method='OPRD'))
assert len(overview) == 25
short_metrics = [RETRIEVAL[i] for i in [0, 3, 5, 6]] + [QDIAG[2]]
sheet('总览', 'CIR 实验结果总览', overview,
      [('category', '实验类别', 20, None), MODEL, METHOD] + short_metrics,
      '主表 18 项 + 新增 CIR 模型 4 项 + 新增合成头 3 项。指标单位：%；FashionIQ 为类别宏平均。', 3)

for row in qmain:
    row['_ci'] = confidence(row)
qcols = [MODEL, METHOD, QDIAG[0], QDIAG[1], QDIAG[2], ('_ci', 'CRR 95% bootstrap CI (%)', 27, None), QDIAG[-1]]
ws = sheet('原表1_QDiag', '原表 1 · QDiag 诊断结果', qmain, qcols,
           'DRR、CRR 和区间单位：%；Mean Δ、bψ 保留原始分数尺度。单元格保留来源精度。', 2, 'paper_qdiag_main_18rows.csv')
best(ws, qmain, qcols, ['DRR_percent', 'CRR_percent'], lambda r: 'Teacher' if r['method'] == 'Teacher' else r['model'])

rcols = [MODEL, METHOD] + [RETRIEVAL[i] for i in [0, 1, 2, 5, 6, 3, 4]]
ws = sheet('原表2_检索', '原表 2 · CIRR / PinPoint / FashionIQ', rmain, rcols,
           '全部指标为百分数；↑ 越高越好，↓ 越低越好；FashionIQ 为 dress / shirt / toptee 宏平均。', 2, 'paper_retrieval_main_18rows.csv')
best(ws, rmain, rcols, [c[0] for c in RETRIEVAL], lambda r: 'Teacher' if r['method'] == 'Teacher' else r['model'])

candidate = read('candidate_selection_target_insertion.csv')
ccols = [('candidates', 'Candidates', 26, None), ('target_insertion', 'Target insertion', 20, None),
         RETRIEVAL[0], ('FashionIQ_R10_percent', 'FashionIQ R@10 ↑ (%)', 22, '0.00'), RETRIEVAL[5], RETRIEVAL[6]]
ws = sheet('原表3_候选消融', '原表 3 · 候选选择与 Target Insertion', candidate, ccols,
           '全部检索指标为百分数。', 2, 'candidate_selection_target_insertion.csv')
best(ws, candidate, ccols, [c[0] for c in ccols[2:]])

complete_columns = [MODEL] + RETRIEVAL + QDIAG
ws = sheet('新增CIR模型', '新增 CIR 模型 · 完整检索与 QDiag 指标', external, complete_columns,
           'CLIP4CIR、TEMA、MagicLens Base / Large；包含完整七项检索指标与 QDiag 校准结果。', 1, 'extra_cir_models_complete_metrics.csv')
best(ws, external, complete_columns, [c[0] for c in RETRIEVAL] + ['DRR_percent', 'CRR_percent'])
ws = sheet('新增头消融', '合成头消融 · 原始头与三个新增头', heads, complete_columns,
           'CLIP-L + OPRD，BGE-MLLM-S2 教师，报告 E4。包含原始残差 MLP 头作为对照。', 1, 'heads_complete_metrics.csv')
best(ws, heads, complete_columns, [c[0] for c in RETRIEVAL] + ['DRR_percent', 'CRR_percent'])

subset_columns = [('subset', '子集', 32, None), ('groups', 'Groups', 12, '#,##0'),
                  ('instructions', 'Instructions', 15, '#,##0'), ('edits', 'Edits', 12, '#,##0')] + QDIAG
for filename, name, title, keys in [
    ('paper_qdiag_all_subsets_198rows.csv', '分项_原表QDiag', '主表模型 · QDiag 分项结果', [MODEL, METHOD]),
    ('extra_cir_models_qdiag_subsets.csv', '分项_新增CIR模型', '新增 CIR 模型 · QDiag 分项结果', [MODEL]),
    ('heads_qdiag_initial_final_subsets.csv', '分项_新增头', '新增合成头 · 训练前与 E4 QDiag 分项', [MODEL, ('stage', '阶段', 16, None)])]:
    rows = read(filename)
    for row in rows:
        row['subset'] = SUBSETS.get(row['subset'], row['subset'])
        if 'negatives' in row:
            row['edits'] = row['negatives']
        if 'stage' in row:
            row['stage'] = {'initial': '训练前', 'final': 'E4'}.get(row['stage'], row['stage'])
    sheet(name, title, rows, keys + subset_columns,
          '包含 ALL、ADD、SUBSTITUTE、DELETE、语义类别及相应置信区间。', len(keys), filename)

sheet('候选消融_补充CRR', '候选消融 · 补充 CRR 指标', candidate,
      ccols + [('CRR_strict_percent', 'CRR strict (%)', 19, '0.00'), ('CRR_all_percent', 'CRR all (%)', 19, '0.00')],
      '“—”表示来源记录未提供该项数值。', 2, 'candidate_selection_target_insertion.csv')

source_rows = []
for label, rows in [('原表1_QDiag', qmain), ('原表2_检索', rmain)]:
    for row in rows:
        for key, value in row.items():
            if key.startswith('source_'):
                source_rows.append({'table': label, 'model': row['model'], 'method': row['method'],
                                    'metric': key.removeprefix('source_'), 'path': value})
sheet('原表数值来源', '原表数值来源 · 文件与行号', source_rows,
      [('table', '工作表', 22, None), MODEL, METHOD, ('metric', '指标来源类别', 23, None), ('path', '来源文件 : 行号', 100, None)],
      '原表各指标均对应到保存的结果报告行。', 3)
artifact_rows = read('artifact_index.csv')
for row in artifact_rows:
    row['exists'] = '存在' if row['exists'] == 'True' else '未找到'
sheet('日志权重与明细索引', '日志、Checkpoint 与明细文件索引', artifact_rows,
      [('category', '实验类别', 28, None), ('model', '模型 / 合成头', 32, None),
       ('purpose', '文件内容', 37, None), ('exists', '当前状态', 14, None),
       ('bytes', '文件大小 (bytes)', 23, '#,##0'), ('path', '完整文件路径', 108, None)],
      '原始运行日志、配置、逐条结果、检索预测和模型权重的路径。', 2, 'artifact_index.csv')

directory = list(catalog)
ws = sheet('目录与数据说明', '工作表目录与数据说明', directory,
           [('sheet', '工作表（点击跳转）', 30, None), ('title', '内容', 53, None),
            ('rows', '数据行数', 16, '#,##0'), ('source', '来源 CSV', 65, None)],
           '所有百分比均按 0–100 存储；Mean Δ 和 bψ 保持原分数尺度。显示精度不改变单元格中的来源数值。', 1)
for i, entry in enumerate(directory, 5):
    ws.cell(i, 1).hyperlink = f"#'{entry['sheet']}'!A1"
    ws.cell(i, 1).font = Font(name='Calibri', size=11, color='0563C1', underline='single')
wb.active = 0
wb.save(OUTPUT)

# Check the saved file, including all numeric cells, against the source values.
reloaded = load_workbook(OUTPUT, data_only=True)
numeric_checks = 0
for name, i, j, expected_value in expected:
    actual = reloaded[name].cell(i, j).value
    if isinstance(expected_value, (int, float)):
        assert isinstance(actual, (int, float)), (name, i, j, actual)
        assert math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12), (name, i, j)
        numeric_checks += 1
    else:
        assert actual == expected_value, (name, i, j, actual, expected_value)
assert len(reloaded['原表1_QDiag'].tables) == len(reloaded['新增头消融'].tables) == 1
verification = {'workbook': str(OUTPUT), 'worksheets': catalog, 'cells_checked': len(expected),
                'numeric_cells_checked': numeric_checks, 'bytes': OUTPUT.stat().st_size,
                'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), 'all_checks_passed': True}
(ROOT / 'excel_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'workbook': str(OUTPUT), 'sheets': reloaded.sheetnames,
                  'numeric_cells_verified': numeric_checks, 'bytes': OUTPUT.stat().st_size}, ensure_ascii=False, indent=2))
