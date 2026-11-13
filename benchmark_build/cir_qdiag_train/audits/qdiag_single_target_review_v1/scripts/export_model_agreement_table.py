"""Export four-model, three-category agreement; no API calls or human-error claims."""
import argparse
from collections import Counter
from fractions import Fraction
import itertools
from pathlib import Path

import numpy as np
import _bootstrap
from single_target_audit import CELLS,Judgment,digest,read_json,read_jsonl,sha,write_csv,write_json

MODELS=['grok-4.6','grok-4.7','kimi-k3','gpt-5.6-terra']
LABELS=['accept','reject','uncertain']


def nominal_alpha(matrix):
    """Krippendorff nominal alpha via pairable coincidences; None is missing."""
    pairable=[]
    for row in matrix:
        values=[v for v in row if v is not None]
        if len(values)>=2:pairable.append(values)
    total=sum(len(row) for row in pairable)
    if total<2:return None
    marginal=Counter(v for row in pairable for v in row)
    observed=Fraction()
    for row in pairable:
        counts=Counter(row);m=len(row)
        observed+=Fraction(m*m-sum(v*v for v in counts.values()),m-1)
    observed/=total
    expected=Fraction(total*total-sum(v*v for v in marginal.values()),total*(total-1))
    return 1-observed/expected if expected else None


def metrics(matrix):
    n=len(matrix);raters=len(matrix[0])
    assert n and raters==4 and all(len(row)==raters for row in matrix)
    counts=[Counter(row) for row in matrix]
    po=sum((Fraction(sum(v*(v-1) for v in row.values()),raters*(raters-1)) for row in counts),Fraction())/n
    marginal=Counter(v for row in matrix for v in row)
    pe=sum((Fraction(marginal[label],n*raters)**2 for label in LABELS),Fraction())
    kappa=(po-pe)/(1-pe) if pe!=1 else None
    alpha=nominal_alpha(matrix)
    # For complete data with a fixed number of raters, the finite-sample
    # correction gives an independent check of the coincidence calculation.
    if kappa is not None:assert alpha==kappa+(1-kappa)/(n*raters)
    else:assert alpha is None
    # Independent pairwise calculation and vectorized implementation of the
    # published pooled-marginal formula must agree with exact rational arithmetic.
    pair_agreements=[sum(row[a]==row[b] for row in matrix)/n for a,b in itertools.combinations(range(raters),2)]
    assert abs(float(po)-sum(pair_agreements)/len(pair_agreements))<1e-12
    table=np.asarray([[row[label] for label in LABELS] for row in counts],dtype=float)
    po_np=((np.square(table).sum(axis=1)-raters)/(raters*(raters-1))).mean()
    pe_np=np.square(table.sum(axis=0)/(n*raters)).sum()
    if kappa is not None:assert abs(float(kappa)-(po_np-pe_np)/(1-pe_np))<1e-12
    return {'samples':n,'mean_pairwise_agreement_percent':100*float(po),
        'fleiss_kappa':float(kappa) if kappa is not None else None,
        'krippendorff_alpha_nominal':float(alpha) if alpha is not None else None,
        'expected_chance_agreement_percent':100*float(pe),
        'po_exact':str(po),'pe_exact':str(pe),'kappa_exact':str(kappa) if kappa is not None else None,
        'alpha_exact':str(alpha) if alpha is not None else None,
        'pooled_label_counts':dict(marginal),
        'rejection_percent_by_model':{model:100*sum(row[j]=='reject' for row in matrix)/n for j,model in enumerate(MODELS)},
        'uncertain_percent_by_model':{model:100*sum(row[j]=='uncertain' for row in matrix)/n for j,model in enumerate(MODELS)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=_bootstrap.ROOT/'outputs/accepted_450_single_target')
    args=p.parse_args();out=args.output_dir.resolve()
    assert metrics([['accept']*4]*2)['fleiss_kappa'] is None
    assert metrics([['accept']*4,['reject']*4])['fleiss_kappa']==1
    assert metrics([['accept','accept','reject','reject']]*2)['fleiss_kappa']==-1/3
    assert metrics(list(itertools.product(['accept','reject'],repeat=4)))['fleiss_kappa']==0
    assert nominal_alpha([['accept','accept',None],['reject','reject',None],['reject',None,None]])==1
    assert nominal_alpha([['accept','accept','reject',None]])==0
    assert nominal_alpha([['accept',None],['reject',None]]) is None
    cases=read_jsonl(out/'cases.private.jsonl');ids={c['case_id'] for c in cases};ratings={}
    source_hashes={'cases.private.jsonl':sha(out/'cases.private.jsonl')}
    for model in MODELS:
        identity=read_json(out/'models'/model/'identity.json');ratings[model]={}
        for path in (out/'models'/model/'judgments').glob('*.json'):
            result=read_json(path)
            assert result['status']=='complete' and not result['simulated'] and result['model']==model
            assert result['signature']==digest(identity)
            j=Judgment.model_validate(result['judgment']);assert j.case_id==result['case_id']
            assert j.case_id not in ratings[model];ratings[model][j.case_id]=j.decision
        assert set(ratings[model])==ids
        source_hashes[model+'_decision_map']=digest(ratings[model])
    assert len(cases)==450
    subsets=[(op,axis,[c for c in cases if c['edit_operation']==op and c['semantic_target']==axis]) for op,axis in CELLS]
    subsets.append(('Overall','All',cases))
    records=[];agreement_rows=[];reject_rows=[]
    for operation,axis,subset in subsets:
        assert len(subset)==(450 if operation=='Overall' else 50)
        values=[[ratings[model][c['case_id']] for model in MODELS] for c in subset]
        stats=metrics(values);records.append({'operation':operation,'semantic_category':axis,**stats})
        agreement_rows.append({'Operation':operation.upper() if operation!='Overall' else operation,
            'SemanticCategory':axis.title(),'Samples':stats['samples'],
            'AveragePairwiseAgreementPercent':stats['mean_pairwise_agreement_percent'],'FleissKappa':stats['fleiss_kappa'],
            'KrippendorffAlphaNominal':stats['krippendorff_alpha_nominal']})
        reject_rows.append({'Operation':agreement_rows[-1]['Operation'],'SemanticCategory':axis.title(),
            **{model:stats['rejection_percent_by_model'][model] for model in MODELS}})
    write_csv(out/'table_b_model_agreement.csv',agreement_rows)
    write_csv(out/'table_b_model_rejection_rates.csv',reject_rows)
    metadata={'title':'Four-model review agreement under the single-target protocol',
        'models':MODELS,'labels':LABELS,'source_hashes':source_hashes,
        'agreement_definition':'Average agreement across all six unordered model pairs; uncertain retained as a third category.',
        'kappa_definition':'Fleiss pooled empirical category marginals; (observed agreement - chance agreement)/(1 - chance agreement). Overall recomputed on all 450 samples, not mean of cell kappas.',
        'alpha_definition':'Nominal Krippendorff alpha = 1 - observed disagreement / expected disagreement from pairable coincidence marginals, with the n*(n-1) finite-sample denominator. Missing judgments are None, not the uncertain category. The four included models have no missing judgments.',
        'rejection_definition':'Each model\'s reject decisions divided by all edits in the row; uncertain is not reject. These are not human-adjudicated error rates.',
        'formula_reference':'https://www.statsmodels.org/stable/generated/statsmodels.stats.inter_rater.fleiss_kappa.html',
        'alpha_reference':"https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf",
        'verification':'Exact fraction arithmetic agrees with independent pairwise and vectorized calculations; nominal alpha also matches the fixed-rater finite-sample correction. Constant-label, perfect-agreement, disagreement, chance-agreement and missing-value fixtures checked.',
        'table':records}
    write_json(out/'table_b_model_agreement_details.json',metadata)
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',
         r'\caption{Four-model review agreement under the single-target protocol. Average pairwise agreement uses all six model pairs over accept, reject, and uncertain labels; Krippendorff alpha uses the nominal metric. Rejection rates are model judgments, not human-adjudicated error rates.}',
         r'\label{tab:model-review-agreement}',r'\begin{tabular}{llrrrrrrrr}',r'\hline',
         r'Operation & Category & Samples & APA (\%) & Fleiss $\kappa$ & Krippendorff $\alpha$ & Grok-4.6 reject (\%) & Grok-4.7 reject (\%) & Kimi-K3 reject (\%) & GPT-5.6-terra reject (\%) \\',r'\hline']
    for row,rej in zip(agreement_rows,reject_rows):
        k='--' if row['FleissKappa'] is None else f"{row['FleissKappa']:.4f}"
        alpha='--' if row['KrippendorffAlphaNominal'] is None else f"{row['KrippendorffAlphaNominal']:.4f}"
        vals=[row['Operation'],row['SemanticCategory'],str(row['Samples']),f"{row['AveragePairwiseAgreementPercent']:.2f}",k,alpha,
              *[f'{rej[m]:.2f}' for m in MODELS]]
        tex.append(' & '.join(vals)+r' \\')
    tex += [r'\hline',r'\end{tabular}',r'\end{table*}']
    (out/'table_b_models.tex').write_text('\n'.join(tex)+'\n')
    for row,rej,detail in zip(agreement_rows,reject_rows,records):
        print({**row,'reject_percent':{m:rej[m] for m in MODELS},
               'expected_chance_agreement_percent':detail['expected_chance_agreement_percent']})


if __name__=='__main__':main()
