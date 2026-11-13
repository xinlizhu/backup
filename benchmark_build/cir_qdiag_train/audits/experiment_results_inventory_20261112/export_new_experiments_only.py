"""Create a standalone workbook containing the newly added experiments."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'excel_dependencies'))
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

SOURCE = ROOT / 'CIR实验结果汇总_含FashionIQ分类KT及排名分析_20261112.xlsx'
OUTPUT = ROOT / '本次新增实验结果_20261112.xlsx'
KEEP = [
    '新增CIR模型', 'FashionIQ_新增模型分类', '分项_新增CIR模型',
    '新增头消融', 'FashionIQ_头消融分类', '分项_新增头',
    'CRR与实际排名变化', 'CRR排名_数值明细', 'CRR排名_指标说明', '数据配置',
]
before = load_workbook(SOURCE)
book = load_workbook(SOURCE)
for name in list(book.sheetnames):
    if name not in KEEP:
        del book[name]
book._sheets = [book[name] for name in KEEP]
book.properties.title = '本次新增实验结果'
book.properties.subject = '额外 CIR 模型、合成头消融、FashionIQ 分类及 CRR 排名分析'
created = []
new_cells = []


def add_table_sheet(name, title, headers, rows, widths):
    ws = book.create_sheet(name)
    ws.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = '087E8B'
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws.cell(1, 1, title).font = Font(name='Calibri', size=17, bold=True, color='FFFFFF')
    ws.cell(1, 1).fill = PatternFill('solid', fgColor='17365D')
    ws.cell(1, 1).alignment = Alignment(vertical='center')
    ws.row_dimensions[1].height = 36
    for j, label in enumerate(headers, 1):
        cell = ws.cell(4, j, label)
        cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='087E8B')
        cell.alignment = Alignment(vertical='center', horizontal='center', wrap_text=True)
        ws.column_dimensions[get_column_letter(j)].width = widths[j - 1]
    ws.row_dimensions[4].height = 35
    for i, row in enumerate(rows, 5):
        ws.row_dimensions[i].height = 50 if name != '目录' else 31
        for j, value in enumerate(row, 1):
            cell = ws.cell(i, j, value)
            cell.font = Font(name='Calibri', size=11, color='111827')
            cell.fill = PatternFill('solid', fgColor='EAF2F8' if i % 2 == 0 else 'FFFFFF')
            cell.alignment = Alignment(vertical='center', wrap_text=True,
                                       horizontal='right' if isinstance(value, (int, float)) else 'left')
            if isinstance(value, (int, float)):
                cell.number_format = '#,##0'
            new_cells.append((name, i, j, value))
    last = 4 + len(rows)
    table = Table(displayName='NewExperimentTable' + str(len(created) + 1),
                  ref=f'A4:{get_column_letter(len(headers))}{last}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    ws.freeze_panes = 'B5'
    ws.print_title_rows = '1:4'
    ws.print_area = f'A1:{get_column_letter(len(headers))}{last}'
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    created.append(name)
    return ws


sources = before['补充数据来源']
source_rows = [[cell.value for cell in row] for row in sources.iter_rows(min_row=5)
               if row[0].value in ['新增模型', '头消融']]
for key, model in [('clip_original', 'CLIP-L Original'), ('clip_oprd', 'CLIP-L OPRD'),
                   ('qwen3_vl_2b', 'Qwen3-VL-Embedding-2B'), ('bge_s2', 'BGE-MLLM-S2')]:
    path = ROOT.parent / 'qdiag_rank_alignment_v1/outputs/full_four_models' / key / 'rank_alignment_summary.csv'
    source_rows.append(['CRR 排名分析', model, 'full / strict', '排名统计', str(path)])
add_table_sheet('新增实验数据来源', '新增实验 · 数据来源',
                [cell.value for cell in sources[4]], source_rows, [20, 34, 22, 22, 108])
artifacts = before['日志权重与明细索引']
artifact_rows = [[cell.value for cell in row] for row in artifacts.iter_rows(min_row=5)
                 if row[0].value in ['external_model', 'head_ablation', 'released_original_head']]
add_table_sheet('新增实验日志与权重', '新增实验 · 日志、权重与逐条结果',
                [cell.value for cell in artifacts[4]], artifact_rows, [27, 34, 39, 15, 24, 108])

descriptions = {
    '新增CIR模型': 'CLIP4CIR、TEMA、MagicLens Base / Large 的完整结果',
    'FashionIQ_新增模型分类': '新增模型的 Dress、Shirt、Toptee、Macro Average',
    '分项_新增CIR模型': '新增模型的 QDiag 操作与语义类别分项',
    '新增头消融': '原始头对照与三个替换头的完整结果',
    'FashionIQ_头消融分类': '头消融的 FashionIQ 分类结果',
    '分项_新增头': '三个替换头训练前与 E4 的 QDiag 分项',
    'CRR与实际排名变化': '四模型的排名下降、Top-10 退出及相关性',
    'CRR排名_数值明细': '排名分析的完整精度数值及置信区间',
    'CRR排名_指标说明': '排名分析的指标定义、统计方法与来源',
    '数据配置': 'QDiag 数据配置与分项规模',
    '新增实验数据来源': '分类结果与排名分析的数据来源',
    '新增实验日志与权重': '新增实验的运行日志、checkpoint 和明细路径',
}
contents = []
for name in book.sheetnames:
    sheet = book[name]
    row_count = sheet.max_row - 4 if name != '数据配置' else 33
    contents.append([name, descriptions[name], row_count])
directory = add_table_sheet('目录', '本次新增实验结果', ['工作表', '内容', '数据行数'], contents, [32, 76, 18])
for i, row in enumerate(contents, 5):
    directory.cell(i, 1).hyperlink = f"#'{row[0]}'!A1"
    directory.cell(i, 1).font = Font(name='Calibri', size=11, color='0563C1', underline='single')
book._sheets = [directory] + [sheet for sheet in book if sheet != directory]
book.active = 0
book.save(OUTPUT)

result = load_workbook(OUTPUT)
preserved_cells = 0
for name in KEEP:
    for row in before[name]:
        for cell in row:
            assert result[name][cell.coordinate].value == cell.value, (name, cell.coordinate)
            preserved_cells += 1
for name, row, column, value in new_cells:
    assert result[name].cell(row, column).value == value
assert not any(name.startswith('原表') or name in ['Candidate_Count_Temperature', 'K_T实验完整指标', '总览']
               for name in result.sheetnames)
assert result['新增CIR模型'].max_row - 4 == 4
assert result['新增头消融'].max_row - 4 == 4
assert result['CRR与实际排名变化'].max_row - 4 == 4
verification = {'workbook': str(OUTPUT), 'sheets': result.sheetnames,
                'preserved_cells_verified': preserved_cells, 'added_cells_verified': len(new_cells),
                'all_selected_experiment_values_unchanged': True,
                'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), 'bytes': OUTPUT.stat().st_size}
(ROOT / 'new_experiments_only_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(verification, ensure_ascii=False, indent=2))
