from pathlib import Path
import sys
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'plot_dependencies'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import _bootstrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alignment_data import load_config,read_json,write_json
from alignment_reporting import export_interpretation


def main():
    c=load_config('configs/full_four_models.yaml');out=Path(c['runtime']['output_dir'])
    names={'clip_original':'CLIP-L Original','clip_oprd':'CLIP-L RIM + OPRD','qwen3_vl_2b':'Qwen3-VL-Embedding-2B','bge_s2':'BGE-VL-MLLM-S2'}
    frames=[];finished=[]
    for model in c['models']:
        p=out/model/'rank_alignment_summary.csv'
        if p.exists() and (out/model/'analysis_complete.json').exists():
            frames.append(pd.read_csv(p));finished.append(model)
    if not frames:
        write_json(out/'collection_status.json',{'complete':False,'finished':[]});return
    combined=pd.concat(frames,ignore_index=True);combined.to_csv(out/'all_rank_alignment_results.csv',index=False)
    export_interpretation(out,combined,names)
    columns=['model','group_count','edit_count','target_edit_pairs','crr','normal_perturbation_threshold',
        'weighted_spearman_delta_logrank','weighted_spearman_delta_logrank_ci_low','weighted_spearman_delta_logrank_ci_high',
        'group_spearman_crr_rankdemotion','rank_demoted_given_crr','rank_demoted_given_noncrr','rank_demoted_conditional_gap',
        'rank_demoted_conditional_gap_ci_low','rank_demoted_conditional_gap_ci_high',
        'margin_decreased_given_crr','top10_exit_given_source_top10','top10_exit_given_crr_and_source_top10',
        'top10_exit_given_noncrr_and_source_top10','source_top10_pair_count','rank_calibrated_response',
        'paraphrase_mean_abs_log_rank_ratio','mean_abs_log_rank_ratio','edit_minus_paraphrase_abs_logrank']
    primary=combined[(combined.scope=='full')&(combined.subset=='strict')]
    primary[[k for k in columns if k in primary]].to_csv(out/'primary_strict_rank_alignment.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for ax,model in zip(axes.flat,c['models']):
        p=out/model/'analysis/full/strict_rank_demotion_by_delta_bin.csv'
        if not p.exists():ax.set_title(names[model]+' (pending)');continue
        data=pd.read_csv(p);y=data.rank_demotion_rate.to_numpy()*100
        lo=data.ci_low.to_numpy()*100;hi=data.ci_high.to_numpy()*100
        ax.errorbar(range(5),y,yerr=np.maximum(0,np.vstack([y-lo,hi-y])),marker='o',capsize=4,color='#2166ac')
        ax.axvline(2.5,color='#b2182b',linestyle='--',alpha=.7)
        ax.set_xticks(range(5),data.normalized_delta_bin);ax.set_ylim(0,100);ax.set_title(names[model])
        ax.set_xlabel('Target-score drop / paraphrase threshold');ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.set_ylabel('Target rank demotion (%)')
    fig.suptitle('Full-gallery rank response to ADD/SUBSTITUTE edits\nDashed line separates CRR failures and passes; group-bootstrap 95% CIs')
    fig.tight_layout();fig.savefig(out/'rank_demotion_by_score_bin.pdf');fig.savefig(out/'rank_demotion_by_score_bin.png',dpi=180);plt.close(fig)
    write_json(out/'collection_status.json',{'complete':len(finished)==len(c['models']),'finished':finished,
        'pending':[m for m in c['models'] if m not in finished]})


if __name__=='__main__':main()
