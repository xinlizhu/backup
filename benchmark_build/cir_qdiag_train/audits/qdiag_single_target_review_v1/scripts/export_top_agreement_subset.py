"""Post-hoc upper-agreement subset; preserve original data and all labels."""
import argparse
from collections import Counter
from pathlib import Path

import _bootstrap
from export_model_agreement_table import LABELS, MODELS, metrics
from single_target_audit import (
    CELLS, Judgment, digest, read_json, read_jsonl, sha,
    write_csv, write_json, write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-cell', type=int, default=45)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source-dir', type=Path,
                        default=_bootstrap.ROOT / 'outputs/accepted_450_single_target')
    args = parser.parse_args()
    if not 1 <= args.per_cell <= 50:
        raise ValueError('per-cell must be between 1 and 50')
    source = args.source_dir.resolve()
    out = source / f'top_agreement_{args.per_cell}_per_cell_seed{args.seed}'
    out.mkdir(exist_ok=True)
    cases = read_jsonl(source / 'cases.private.jsonl')
    ids = {c['case_id'] for c in cases}
    assert len(cases) == len(ids) == 450
    ratings = {}
    hashes = {'cases.private.jsonl': sha(source / 'cases.private.jsonl')}
    for model in MODELS:
        identity = read_json(source / 'models' / model / 'identity.json')
        ratings[model] = {}
        for path in (source / 'models' / model / 'judgments').glob('*.json'):
            result = read_json(path)
            assert result['status'] == 'complete' and not result['simulated']
            assert result['model'] == model and result['signature'] == digest(identity)
            judgment = Judgment.model_validate(result['judgment'])
            assert judgment.case_id == result['case_id']
            assert judgment.case_id not in ratings[model]
            ratings[model][judgment.case_id] = judgment.decision
        assert set(ratings[model]) == ids
        hashes[model + '_decision_map'] = digest(ratings[model])
    original_details = read_json(source / 'table_b_model_agreement_details.json')
    assert hashes == original_details['source_hashes']

    def votes(case):
        return [ratings[model][case['case_id']] for model in MODELS]

    def agreeing_pairs(case):
        return sum(n * (n - 1) // 2 for n in Counter(votes(case)).values())

    selected, rankings, rows, details, rejections = [], [], [], [], []

    def report(operation, axis, available, kept):
        before, after = metrics([votes(c) for c in available]), metrics([votes(c) for c in kept])
        unanimous = sum(agreeing_pairs(c) == 6 for c in available)
        unanimous_kept = sum(agreeing_pairs(c) == 6 for c in kept)
        row = {
            'Operation': operation.upper() if operation != 'Overall' else operation,
            'SemanticCategory': axis.title(),
            'OriginalSamples': len(available), 'SelectedSamples': len(kept),
            'OriginalAveragePairwiseAgreementPercent': before['mean_pairwise_agreement_percent'],
            'SelectedAveragePairwiseAgreementPercent': after['mean_pairwise_agreement_percent'],
            'OriginalFleissKappa': before['fleiss_kappa'],
            'SelectedFleissKappa': after['fleiss_kappa'],
            'OriginalKrippendorffAlpha': before['krippendorff_alpha_nominal'],
            'SelectedKrippendorffAlpha': after['krippendorff_alpha_nominal'],
            'AvailableUnanimousSamples': unanimous,
            'SelectedUnanimousSamples': unanimous_kept,
            'SelectedAllFourAgreementPercent': 100 * unanimous_kept / len(kept),
        }
        rows.append(row)
        details.append({'operation': operation, 'semantic_category': axis,
                        'original': before, 'selected': after})
        rejections.append({'Operation': row['Operation'], 'SemanticCategory': row['SemanticCategory'],
                           **after['rejection_percent_by_model']})
        return row

    for operation, axis in CELLS:
        available = [c for c in cases if (c['edit_operation'], c['semantic_target']) == (operation, axis)]
        assert len(available) == 50
        ordered = sorted(available, key=lambda c: (
            -agreeing_pairs(c), digest({'seed': args.seed, 'case_id': c['case_id']}), c['case_id']))
        kept = ordered[:args.per_cell]
        selected.extend(kept)
        # Direct proof of maximum APA for this fixed-size stratum.
        assert sum(agreeing_pairs(c) for c in kept) == sum(
            sorted((agreeing_pairs(c) for c in available), reverse=True)[:args.per_cell])
        report(operation, axis, available, kept)
        for rank, case in enumerate(ordered, 1):
            rankings.append({
                'case_id': case['case_id'], 'negative_id': case['negative_id'],
                'group_id': case['group_id'], 'edit_operation': operation, 'semantic_target': axis,
                'rank_within_cell': rank, 'selected': rank <= args.per_cell,
                'agreeing_model_pairs': agreeing_pairs(case), 'total_model_pairs': 6,
                'item_pairwise_agreement_percent': 100 * agreeing_pairs(case) / 6,
                'all_four_agree': agreeing_pairs(case) == 6,
                'tie_break_hash': digest({'seed': args.seed, 'case_id': case['case_id']}),
                **dict(zip(MODELS, votes(case))),
            })

    assert len(selected) == len({c['case_id'] for c in selected}) == args.per_cell * 9
    overall = report('Overall', 'All', cases, selected)
    write_jsonl(out / 'selected_cases.private.jsonl', selected)
    write_json(out / 'selected_case_ids.json', [c['case_id'] for c in selected])
    write_csv(out / 'all_450_selection_ranking.csv', rankings)
    write_csv(out / 'selected_decision_matrix.csv', [r for r in rankings if r['selected']])
    write_csv(out / 'agreement_comparison.csv', rows)
    write_csv(out / 'model_rejection_rates.csv', rejections)
    metadata = {
        'analysis_type': 'Post-hoc maximum average pairwise agreement at fixed per-cell retention',
        'models': MODELS, 'labels': LABELS, 'source_hashes': hashes,
        'per_cell': args.per_cell, 'seed': args.seed, 'original_samples': 450,
        'selected_samples': len(selected), 'retention_percent': 100 * len(selected) / 450,
        'selection': 'Within each of the 9 cells, sort by agreeing model-pair count descending (out of 6), then SHA256 of JSON {seed, case_id} ascending. No preference for accept, reject, or uncertain. Same seed for every cell.',
        'optimized_metric': 'Average pairwise agreement only. Kappa and alpha are reported for the chosen subset, not separately maximized over tied subsets.',
        'interpretation': 'An optimistic, label-selected subset of these 450 cases, not an unbiased estimate of dataset-wide reviewer reliability or accuracy. Shared reject/uncertain labels also count as agreement.',
        'undefined_metrics': 'Null kappa/alpha means a degenerate single-label marginal with zero expected disagreement, not zero agreement.',
        'verification': 'All four original decision maps match their recorded hashes; every judgment signature and schema checked; exact agreement metrics checked independently inside metrics(); retained pair count equals the stratum-wise mathematical maximum.',
        'table': details,
    }
    write_json(out / 'analysis_details.json', metadata)
    write_json(out / 'artifacts.json', {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'artifacts.json'})
    for row in rows:
        print(row)
    print('OUTPUT:', out)


if __name__ == '__main__':
    main()
