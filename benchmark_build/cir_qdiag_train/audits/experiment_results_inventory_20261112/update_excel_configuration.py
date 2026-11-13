"""Update dataset configuration counts, preserving all saved experiment metrics."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'excel_dependencies'))
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

SOURCE = ROOT / 'CIR实验结果汇总_20261112.xlsx'
OUTPUT = ROOT / 'CIR实验结果汇总_配置更新_20261112.xlsx'
DATA = ROOT.parents[2] / 'release_code/QDiag/data/final.jsonl'
STATS = DATA.with_name('statistics.json')
groups = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
statistics = json.loads(STATS.read_text())
data_hash = hashlib.sha256(DATA.read_bytes()).hexdigest()
assert data_hash == statistics['dataset_sha256']
edits = [dict(edit, group_id=group['group_id']) for group in groups for edit in group['negatives']]
assert len(groups) == 931 and len(edits) == 10131
assert len({e['negative_id'] for e in edits}) == len(edits)

SUBSETS = {'all': 'ALL', 'strict': 'ADD + SUBSTITUTE', 'add': 'ADD',
           'substitute': 'SUBSTITUTE', 'delete': 'DELETE', 'object': 'Object',
           'attribute': 'Attribute', 'predicate': 'Predicate',
           'strict|object': 'ADD + SUBSTITUTE / Object',
           'strict|attribute': 'ADD + SUBSTITUTE / Attribute',
           'strict|predicate': 'ADD + SUBSTITUTE / Predicate'}


def count(rows):
    return {'Groups': len({r['group_id'] for r in rows}),
            'Instructions': len({(r['group_id'], r['instruction_id']) for r in rows}),
            'Edits': len(rows)}


counts = {}
for subset, label in SUBSETS.items():
    operations = ['add', 'substitute'] if subset.startswith('strict') else ['add', 'substitute', 'delete']
    if subset in ['add', 'substitute', 'delete']:
        operations = [subset]
    axis = subset.split('|')[-1] if '|' in subset else (subset if subset in ['object', 'attribute', 'predicate'] else None)
    counts[label] = count([e for e in edits if e['edit_operation'] in operations
                           and (axis is None or e['semantic_target'] == axis)])
assert counts['ALL'] == {'Groups': 930, 'Instructions': 3308, 'Edits': 10131}
assert counts['ADD + SUBSTITUTE'] == {'Groups': 870, 'Instructions': 2762, 'Edits': 5735}

before = load_workbook(SOURCE)
book = load_workbook(SOURCE)
changed_cells = set()
changes = []
for name in ['分项_原表QDiag', '分项_新增CIR模型', '分项_新增头']:
    sheet = book[name]
    headers = {cell.value: cell.column for cell in sheet[4]}
    for row in range(5, sheet.max_row + 1):
        subset = sheet.cell(row, headers['子集']).value
        values = counts[subset]
        for key in ['Groups', 'Instructions', 'Edits']:
            cell = sheet.cell(row, headers[key])
            changes.append({'sheet': name, 'cell': cell.coordinate,
                            'before': cell.value, 'after': values[key], 'subset': subset, 'field': key})
            changed_cells.add((name, row, cell.column))
            cell.value = values[key]
assert len(changes) == (198 + 44 + 66) * 3

# Configuration values occupy a dedicated sheet. No status annotations or
# version warnings are inserted into the workbook.
ws = book.create_sheet('数据配置', 1)
ws.sheet_view.showGridLines = False
ws.sheet_properties.tabColor = '087E8B'
ws.merge_cells('A1:E1')
ws['A1'] = 'QDiag 数据配置'
ws['A1'].font = Font(name='Calibri', size=17, bold=True, color='FFFFFF')
ws['A1'].fill = PatternFill('solid', fgColor='17365D')
ws['A1'].alignment = Alignment(vertical='center')
ws.row_dimensions[1].height = 36
for col, width in {'A': 36, 'B': 31, 'C': 20, 'D': 20, 'E': 20}.items():
    ws.column_dimensions[col].width = width


def add_table(start_row, headers, records, table_name):
    for column, header in enumerate(headers, 1):
        cell = ws.cell(start_row, column, header)
        cell.font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='087E8B')
        cell.alignment = Alignment(vertical='center', horizontal='center', wrap_text=True)
    ws.row_dimensions[start_row].height = 31
    for row, values in enumerate(records, start_row + 1):
        ws.row_dimensions[row].height = 26
        for column, value in enumerate(values, 1):
            cell = ws.cell(row, column, value)
            cell.font = Font(name='Calibri', size=11, color='111827')
            cell.fill = PatternFill('solid', fgColor='EAF2F8' if row % 2 == 0 else 'FFFFFF')
            cell.alignment = Alignment(vertical='center', wrap_text=True,
                                       horizontal='right' if isinstance(value, (int, float)) else 'left')
            if isinstance(value, int):
                cell.number_format = '#,##0'
    end = start_row + len(records)
    table = Table(displayName=table_name, ref=f'A{start_row}:{get_column_letter(len(headers))}{end}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    ws.add_table(table)
    return end


config = [
    ['Group 总数', statistics['groups']],
    ['原始指令总数', statistics['instructions']],
    ['审核通过的编辑总数', statistics['accepted_edits']],
    ['含通过编辑的 Group 数', statistics['groups_with_edits']],
    ['含通过编辑的原始指令数', statistics['instructions_with_edits']],
    ['ADD 编辑数', statistics['by_operation']['add']],
    ['SUBSTITUTE 编辑数', statistics['by_operation']['substitute']],
    ['DELETE 编辑数', statistics['by_operation']['delete']],
    ['ADD + SUBSTITUTE 编辑数', statistics['strict_edits']],
    ['参考图片数（去重）', statistics['unique_reference_images']],
    ['正例图片数（去重）', statistics['unique_positive_images']],
    ['生成模型', 'gpt-5.6-terra'],
    ['审核模型', 'gpt-5.6-terra'],
]
end = add_table(4, ['配置项', '数值'], config, 'DatasetConfiguration')
subset_records = [[label, data['Groups'], data['Instructions'], data['Edits']] for label, data in counts.items()]
end = add_table(end + 3, ['子集', 'Groups', 'Instructions', 'Edits'], subset_records, 'FullDatasetSubsetCounts')
cell_records = []
for axis in ['object', 'attribute', 'predicate']:
    for operation in ['add', 'substitute', 'delete']:
        values = count([e for e in edits if e['semantic_target'] == axis and e['edit_operation'] == operation])
        assert values['Edits'] == statistics['by_target_operation'][axis + '/' + operation]
        cell_records.append([axis.title(), operation.upper(), values['Groups'], values['Instructions'], values['Edits']])
end = add_table(end + 3, ['语义类别', '操作', 'Groups', 'Instructions', 'Edits'], cell_records, 'FullDatasetNineCells')
ws.freeze_panes = 'B5'
ws.page_setup.orientation = 'landscape'
ws.page_setup.paperSize = ws.PAPERSIZE_A4
ws.page_setup.fitToWidth = 1
ws.page_setup.fitToHeight = 0
ws.print_area = f'A1:E{end}'

directory = book['目录与数据说明']
row = directory.max_row + 1
for col, value in enumerate(['数据配置', 'QDiag 数据配置与分项规模', len(config) + len(subset_records) + len(cell_records), str(DATA)], 1):
    cell = directory.cell(row, col, value)
    cell.font = Font(name='Calibri', size=11, color='111827')
    cell.alignment = Alignment(vertical='center', wrap_text=True)
directory.row_dimensions[row].height = 62
directory.cell(row, 1).hyperlink = "#'数据配置'!A1"
directory.cell(row, 1).font = Font(name='Calibri', size=11, color='0563C1', underline='single')
table = next(iter(directory.tables.values()))
table.ref = f'A4:D{row}'
directory.print_area = f'A1:D{row}'
book.active = 0
book.save(OUTPUT)

after = load_workbook(OUTPUT)
preserved_numeric_cells = 0
for sheet in before:
    for row in sheet:
        for cell in row:
            key = (sheet.title, cell.row, cell.column)
            if key in changed_cells:
                continue
            actual = after[sheet.title].cell(cell.row, cell.column).value
            assert actual == cell.value, (key, cell.value, actual)
            if isinstance(cell.value, (int, float)):
                preserved_numeric_cells += 1
for change in changes:
    assert after[change['sheet']][change['cell']].value == change['after']
for sheet in after:
    for row in sheet:
        for cell in row:
            assert cell.comment is None
            if isinstance(cell.value, str):
                assert not any(marker in cell.value for marker in ['待复测', '待重测', '保留旧值', '保留原值'])
result = {'input_workbook': str(SOURCE), 'output_workbook': str(OUTPUT),
          'dataset_path': str(DATA), 'dataset_sha256': data_hash,
          'updated_count_cells': len(changes), 'unchanged_numeric_cells_verified': preserved_numeric_cells,
          'all_original_nonconfiguration_cells_unchanged': True, 'subset_counts': counts,
          'changes': changes, 'output_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
(ROOT / 'excel_configuration_update_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ['output_workbook', 'updated_count_cells',
                 'unchanged_numeric_cells_verified', 'all_original_nonconfiguration_cells_unchanged']}, ensure_ascii=False, indent=2))
