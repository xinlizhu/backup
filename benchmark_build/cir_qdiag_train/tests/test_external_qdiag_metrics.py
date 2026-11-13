"""Regression against the original evaluator, independent of external weights."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from evaluate_external_qdiag import calibrate_scores, negative_frame
from evaluate_winrate_random10 import calibrate_threshold, score_negatives, summarize
from pinpoint_full_diagnostics import GroupSpec, QuerySpec


def test_native_score_matrix_preserves_original_qdiag():
    rng = np.random.default_rng(2026)
    groups, queries = [], []
    image_ids = [f'im{i}' for i in range(21)]
    for g in range(7):
        gid = f'g{g}'
        instructions = tuple(f'i{g}-{j}' for j in range(3))
        groups.append(GroupSpec(gid, (f'r{g}',), tuple(image_ids[g*3:g*3+3]), (), instructions))
        for j, iid in enumerate(instructions):
            common = dict(group_id=gid,instruction_id=iid,source_instruction_id=iid,
                          text=f'text {g} {j}',reference_ids=(f'r{g}',),query_category='',length_category='',
                          is_question=False,is_imperative=False)
            queries.append(QuerySpec(query_index=len(queries),query_key=f'p{iid}',query_kind='positive',
                                     negative_id='',semantic_target='',edit_operation='',**common))
            for n in range(j+1):
                queries.append(QuerySpec(query_index=len(queries),query_key=f'n{iid}-{n}',query_kind='negative',
                                         negative_id=f'n{iid}-{n}',semantic_target=['object','attribute','predicate'][j],
                                         edit_operation=['add','substitute','delete'][n],**common))
    q = rng.normal(size=(len(queries),12)).astype('float32')
    t = rng.normal(size=(len(image_ids),12)).astype('float32')
    q /= np.linalg.norm(q,axis=1,keepdims=True)
    t /= np.linalg.norm(t,axis=1,keepdims=True)
    scores = q @ t.T
    c1 = calibrate_threshold(groups,queries,q,t,{v:i for i,v in enumerate(image_ids)},bootstrap_replicates=200,seed=42)
    c2 = calibrate_scores(groups,queries,scores,image_ids,replicates=200,seed=42)
    np.testing.assert_allclose(c1['normal_perturbation_threshold'],c2['normal_perturbation_threshold'],atol=2e-7)
    np.testing.assert_allclose(c1['threshold_bootstrap_ci'],c2['threshold_bootstrap_ci'],atol=2e-7)
    threshold=c1['normal_perturbation_threshold']
    a=score_negatives(groups,queries,q,t,image_ids,threshold=threshold,random_samples=2,seed=42)
    b=negative_frame(groups,queries,scores,image_ids,threshold=threshold,random_samples=2,seed=42)
    pd.testing.assert_frame_equal(a,b,check_exact=False,atol=2e-7,rtol=2e-6)
    pd.testing.assert_frame_equal(summarize(a,calibration=c1,bootstrap_replicates=200,seed=42),
                                  summarize(b,calibration=c1,bootstrap_replicates=200,seed=42),
                                  check_exact=False,atol=2e-7,rtol=2e-6)
