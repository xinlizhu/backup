import argparse
from pathlib import Path
import _bootstrap
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from alignment_data import load_config,prepare,read_json,read_jsonl,write_json
from alignment_stats import calibrate,paired_frame,summarize_subset,hierarchical_weights


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/full_four_models.yaml');p.add_argument('--model',required=True)
    p.add_argument('--bootstrap-replicates',type=int)
    args=p.parse_args();c=load_config(args.config);prepare(c);root=Path(c['runtime']['output_dir']);out=root/args.model
    _blas_limit=threadpool_limits(limits=1,user_api='blas')
    assert read_json(out/'ranks/complete.json')['complete']
    arrays={name:np.load(out/'ranks'/f'{name}.npy',mmap_mode='r') for name in ['target_scores','target_ranks','margin10','gallery_mean']}
    groups=read_json(root/'groups.json');queries=read_jsonl(root/'queries.jsonl');gallery=read_json(root/'gallery.json')
    pilot=set(read_json(root/'pilot_group_ids.json'));rows=[]
    replicates=args.bootstrap_replicates if args.bootstrap_replicates is not None else c['runtime']['bootstrap_replicates']
    for scope,selected in [('full',groups),('pilot500',[g for g in groups if g['group_id'] in pilot])]:
        target=out/'analysis'/scope;target.mkdir(parents=True,exist_ok=True)
        cal,controls=calibrate(selected,arrays);write_json(target/'calibration.json',cal);controls.to_csv(target/'paraphrase_controls_by_group.csv',index=False)
        selected_ids={g['group_id'] for g in selected}
        frame=paired_frame(queries,gallery,arrays,cal['normal_perturbation_threshold'])
        frame=frame[frame.group_id.isin(selected_ids)].copy();frame.to_parquet(target/'per_target_edit.parquet',index=False)
        for subset in ['strict','add','substitute','delete','all']:
            part=frame if subset=='all' else frame[frame.edit_operation.isin(['add','substitute'])] if subset=='strict' else frame[frame.edit_operation==subset]
            stats,by_group=summarize_subset(part,cal['normal_perturbation_threshold'],cal['rank_paraphrase_95_threshold'],replicates,c['runtime']['seed'])
            control=controls.set_index('group_id').loc[by_group.group_id]
            paired_gap=by_group.mean_abs_log_rank_ratio.to_numpy()-control.paraphrase_mean_abs_log_rank_ratio.to_numpy()
            rng=np.random.default_rng(c['runtime']['seed'])
            draws=rng.integers(0,len(paired_gap),size=(replicates,len(paired_gap)))
            stats['paraphrase_mean_abs_log_rank_ratio']=float(control.paraphrase_mean_abs_log_rank_ratio.mean())
            stats['edit_minus_paraphrase_abs_logrank']=float(paired_gap.mean())
            if replicates:
                low,high=np.quantile(paired_gap[draws].mean(1),[.025,.975])
                stats['edit_minus_paraphrase_abs_logrank_ci_low']=float(low)
                stats['edit_minus_paraphrase_abs_logrank_ci_high']=float(high)
            row={'model':args.model,'scope':scope,'subset':subset,**stats};rows.append(row)
            by_group.to_csv(target/f'{subset}_by_group.csv',index=False)
            print({k:row[k] for k in ['model','scope','subset','crr','weighted_spearman_delta_logrank','rank_demoted_given_crr','rank_demoted_given_noncrr']},flush=True)
            if subset=='strict':
                weights,gids,gi=hierarchical_weights(part)
                bins=np.digitize(part.delta_score.to_numpy()/cal['normal_perturbation_threshold'],[0,.5,1,2],right=True)
                labels=['<=0','(0,0.5]','(0.5,1]','(1,2]','>2'];binrows=[]
                for b,label in enumerate(labels):
                    mask=(bins==b);den=np.bincount(gi,weights=weights*mask,minlength=len(gids))
                    num=np.bincount(gi,weights=weights*mask*part.rank_demoted.to_numpy(),minlength=len(gids))
                    value=float(num.sum()/den.sum()) if den.sum() else float('nan')
                    item={'normalized_delta_bin':label,'pair_count':int(mask.sum()),'rank_demotion_rate':value}
                    if replicates and den.sum():
                        rng=np.random.default_rng(c['runtime']['seed']);sample=rng.integers(0,len(gids),size=(replicates,len(gids)))
                        ds=den[sample].sum(1);ns=num[sample].sum(1);valid=ds>0
                        item['ci_low'],item['ci_high']=map(float,np.quantile(ns[valid]/ds[valid],[.025,.975]))
                    binrows.append(item)
                pd.DataFrame(binrows).to_csv(target/'strict_rank_demotion_by_delta_bin.csv',index=False)
        pd.DataFrame([r for r in rows if r['scope']==scope]).to_csv(target/'summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'rank_alignment_summary.csv',index=False)
    write_json(out/'analysis_complete.json',{'complete':True,'bootstrap_replicates':replicates,'rows':len(rows),
        'interpretation':'CRR is a target-score diagnostic. Rank demotion is directionally expected for ADD/SUBSTITUTE; DELETE rows are descriptive, not correctness tests.'})


if __name__=='__main__':main()
