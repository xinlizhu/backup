"""Append verified CRR/rank results to the latest experiment workbook."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'excel_dependencies'))
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

SOURCE = ROOT / 'CIR实验结果汇总_含FashionIQ分类及KT实验_20261112.xlsx'
OUTPUT = ROOT / 'CIR实验结果汇总_含FashionIQ分类KT及排名分析_20261112.xlsx'
RESULTS = ROOT.parent / 'qdiag_rank_alignment_v1/outputs/full_four_models'
MODEL_NAMES = {'clip_original': 'CLIP-L Original', 'clip_oprd': 'CLIP-L OPRD',
               'qwen3_vl_2b': 'Qwen3-VL-Embedding-2B', 'bge_s2': 'BGE-MLLM-S2'}
records = []
for key, name in MODEL_NAMES.items():
    path = RESULTS / key / 'rank_alignment_summary.csv'
    with path.open(encoding='utf-8-sig') as stream:
        candidates = [r for r in csv.DictReader(stream) if r['scope'] == 'full' and r['subset'] == 'strict']
    assert len(candidates) == 1
    data = candidates[0]
    verification = json.loads((path.parent / 'ranks/verification.json').read_text())
    complete = json.loads((path.parent / 'analysis_complete.json').read_text())
    assert verification['complete'] and complete['complete'] and complete['bootstrap_replicates'] == 2000
    row = {'model': name, 'model_key': key, 'groups': int(data['group_count']),
           'edits': int(data['edit_count']), 'pairs': int(data['target_edit_pairs']),
           'gallery': 109598, 'bootstrap_replicates': 2000, 'source': str(path)}
    for target, source in {
        'demoted_c1_percent': 'rank_demoted_given_crr',
        'demoted_c0_percent': 'rank_demoted_given_noncrr',
        'gap_pp': 'rank_demoted_conditional_gap',
        'gap_ci_low_pp': 'rank_demoted_conditional_gap_ci_low',
        'gap_ci_high_pp': 'rank_demoted_conditional_gap_ci_high',
        'exit10_c1_percent': 'top10_exit_given_crr_and_source_top10',
        'exit10_c0_percent': 'top10_exit_given_noncrr_and_source_top10',
    }.items():
        row[target] = 100 * float(data[source])
    for target, source in {
        'rho': 'weighted_spearman_delta_logrank',
        'rho_ci_low': 'weighted_spearman_delta_logrank_ci_low',
        'rho_ci_high': 'weighted_spearman_delta_logrank_ci_high',
    }.items():
        row[target] = float(data[source])
    assert (row['groups'], row['edits'], row['pairs']) == (870, 5735, 47391)
    assert math.isclose(row['gap_pp'], row['demoted_c1_percent'] - row['demoted_c0_percent'], abs_tol=1e-12)
    row['gap_formatted'] = f"{row['gap_pp']:.2f} [{row['gap_ci_low_pp']:.2f}, {row['gap_ci_high_pp']:.2f}]"
    row['rho_formatted'] = f"{row['rho']:.4f} [{row['rho_ci_low']:.4f}, {row['rho_ci_high']:.4f}]"
    records.append(row)

csv_path = ROOT / 'crr_rank_alignment_publication_table.csv'
with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

before = load_workbook(SOURCE)
book = load_workbook(SOURCE)
checks, added = [], []


def add_sheet(name, title, rows, columns):
    assert name not in book.sheetnames
    ws = book.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = '087E8B'
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    ws.cell(1, 1, title).font = Font(name='Calibri', size=17, bold=True, color='FFFFFF')
    ws.cell(1, 1).fill = PatternFill('solid', fgColor='17365D')
    ws.cell(1, 1).alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 36
    for column, (key, label, width, fmt) in enumerate(columns, 1):
        cell = ws.cell(4, column, label)
        cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='087E8B')
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.column_dimensions[get_column_letter(column)].width = width
    ws.row_dimensions[4].height = 47
    for i, row in enumerate(rows, 5):
        ws.row_dimensions[i].height = 33 if name != 'CRR排名_指标说明' else 53
        for j, (key, label, width, fmt) in enumerate(columns, 1):
            value = row[key]
            cell = ws.cell(i, j, value)
            cell.font = Font(name='Calibri', size=11, color='111827')
            cell.fill = PatternFill('solid', fgColor='EAF2F8' if i % 2 == 0 else 'FFFFFF')
            cell.alignment = Alignment(horizontal='right' if isinstance(value, (int, float)) else 'left',
                                       vertical='center', wrap_text=True)
            if fmt and isinstance(value, (int, float)):
                cell.number_format = fmt
            checks.append((name, i, j, value))
    end = 4 + len(rows)
    table = Table(displayName='RankAlignmentTable' + str(len(added) + 1),
                  ref=f'A4:{get_column_letter(len(columns))}{end}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = 'B5'
    ws.print_title_rows = '1:4'
    ws.print_area = f'A1:{get_column_letter(len(columns))}{end}'
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3 if len(columns) > 10 else ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    added.append({'sheet': name, 'title': title, 'rows': len(rows), 'source': csv_path.name})


summary_columns = [
    ('model', '模型', 34, None),
    ('demoted_c1_percent', '排名下降 C=1 (%)', 21, '0.00'),
    ('demoted_c0_percent', '排名下降 C=0 (%)', 21, '0.00'),
    ('gap_formatted', '差值及 95% CI（百分点）', 32, None),
    ('exit10_c1_percent', 'Top-10 退出 C=1 (%)', 24, '0.00'),
    ('exit10_c0_percent', 'Top-10 退出 C=0 (%)', 24, '0.00'),
    ('rho_formatted', 'Weighted Spearman ρ 及 95% CI', 35, None),
]
add_sheet('CRR与实际排名变化', 'CRR 响应与编辑查询的实际检索排名变化', records, summary_columns)
details = [('model', '模型', 34, None)]
details += [(key, label, 18, '#,##0') for key, label in [
    ('groups', 'Groups'), ('edits', 'ADD / SUBSTITUTE 编辑数'), ('pairs', 'Target–Edit Pairs'),
    ('gallery', '图库图片数'), ('bootstrap_replicates', 'Bootstrap 次数')]]
details += [(key, label, 23, '0.000000') for key, label in [
    ('demoted_c1_percent', '排名下降 C=1 (%)'), ('demoted_c0_percent', '排名下降 C=0 (%)'),
    ('gap_pp', '差值（百分点）'), ('gap_ci_low_pp', '差值 CI 下限（百分点）'), ('gap_ci_high_pp', '差值 CI 上限（百分点）'),
    ('exit10_c1_percent', 'Top-10 退出 C=1 (%)'), ('exit10_c0_percent', 'Top-10 退出 C=0 (%)'),
    ('rho', 'Weighted Spearman ρ'), ('rho_ci_low', 'ρ CI 下限'), ('rho_ci_high', 'ρ CI 上限')]]
add_sheet('CRR排名_数值明细', 'CRR 与排名变化 · 数值明细', records, details)
definitions = [
    ('评估范围', '870 个 Group、5,735 条 ADD/SUBSTITUTE 编辑、47,391 个 Target–Edit Pair；图库为 109,598 张图片。'),
    ('分数变化', 'Δs = s(original query, target) − s(edited query, target)。正值表示目标分数下降。'),
    ('C=1 / C=0', 'C=1：单个 Target–Edit Pair 满足 Δs > bψ；C=0：Δs ≤ bψ。bψ 为各模型的 paraphrase calibration threshold。'),
    ('排名下降', 'r_edited > r_original；两种 C 条件使用相同的严格比较规则，排名不变不计为下降。'),
    ('条件比例', '对相应事件的加权量除以条件成立的加权量；C=1 和 C=0 使用相同的基础层级权重。'),
    ('Top-10 退出率', '分别在 C=1 或 C=0 且 r_original ≤ 10 的 Pair 中，计算 r_edited > 10 的加权比例。'),
    ('排名变化量', 'log-rank = ln(r_edited / r_original)。正值表示排名后退，负值表示排名提升。'),
    ('Weighted Spearman ρ', '目标分数下降量 Δs 与带符号 log-rank 的加权 Spearman 相关系数。'),
    ('层级权重', '依次平均一个编辑内的 targets、原始指令内的 edits、group 内的原始指令，以及全部 groups；每组基础总权重相等。'),
    ('95% 置信区间', '2,000 次 Group Cluster Bootstrap；每次有放回抽取 870 个 Group，组内 Pair 一起保留。使用 2.5% 和 97.5% 分位数，seed=42。'),
    ('校准阈值', '涉及 CRR 的比例在 bootstrap 中固定 bψ；区间不包含阈值估计本身的不确定性。Spearman ρ 不依赖该阈值。'),
    ('差值计算', '使用未舍入的原始条件比例计算，单位为百分点；并非对显示的两位小数直接相减。'),
]
definitions += [(row['model'] + ' · 来源', row['source']) for row in records]
add_sheet('CRR排名_指标说明', 'CRR 与排名变化 · 指标定义及统计方法',
          [{'item': key, 'definition': value} for key, value in definitions],
          [('item', '项目', 35, None), ('definition', '定义 / 来源', 135, None)])
added[-1]['source'] = 'qdiag_rank_alignment_v1/src/alignment_stats.py; rank_alignment_summary.csv'

directory = book['目录与数据说明']
for entry in added:
    row = directory.max_row + 1
    for column, key in enumerate(['sheet', 'title', 'rows', 'source'], 1):
        cell = directory.cell(row, column, entry[key])
        cell.font = Font(name='Calibri', size=11, color='111827')
        cell.alignment = Alignment(vertical='center', wrap_text=True)
    directory.row_dimensions[row].height = 46
    directory.cell(row, 1).hyperlink = f"#'{entry['sheet']}'!A1"
    directory.cell(row, 1).font = Font(name='Calibri', size=11, color='0563C1', underline='single')
next(iter(directory.tables.values())).ref = f'A4:D{directory.max_row}'
directory.print_area = f'A1:D{directory.max_row}'
book.active = book.sheetnames.index('CRR与实际排名变化')
book.save(OUTPUT)

reloaded = load_workbook(OUTPUT)
for original_sheet in before:
    for row in original_sheet:
        for cell in row:
            assert reloaded[original_sheet.title][cell.coordinate].value == cell.value, (original_sheet.title, cell.coordinate)
for name, row, column, expected in checks:
    value = reloaded[name].cell(row, column).value
    if isinstance(expected, (int, float)):
        assert isinstance(value, (int, float)) and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert value == expected
result = {'workbook': str(OUTPUT), 'new_sheets': added, 'new_cells_checked': len(checks),
          'existing_values_unchanged': True, 'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
(ROOT / 'rank_alignment_excel_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
