"""Paired full-gallery rank/score statistics with the original group hierarchy."""
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def group_quantile(values_by_group,q=.95):
    blocks=[np.asarray(v,dtype=np.float64).ravel() for v in values_by_group.values() if len(v)]
    values=np.concatenate(blocks);weights=np.concatenate([np.full(len(v),1/len(v)) for v in blocks])
    order=np.argsort(values,kind='stable');values=values[order];weights=weights[order]
    return float(values[min(np.searchsorted(np.cumsum(weights),q*weights.sum(),side='left'),len(values)-1)])


def hierarchical_weights(frame):
    target_count=frame.groupby(['group_id','instruction_id','negative_id'])['target_id'].transform('size').to_numpy()
    unique=frame[['group_id','instruction_id','negative_id']].drop_duplicates()
    edits=unique.groupby(['group_id','instruction_id']).size().to_dict()
    instructions=unique[['group_id','instruction_id']].drop_duplicates().groupby('group_id').size().to_dict()
    w=np.asarray([1/(n*edits[g,i]*instructions[g]) for n,g,i in zip(target_count,frame.group_id,frame.instruction_id)],dtype=float)
    groups,gi=np.unique(frame.group_id,return_inverse=True)
    assert np.allclose(np.bincount(gi,weights=w),1)
    return w,groups,gi


def weighted_spearman(x,y,w,prepared=None):
    if prepared is None:
        _,xi=np.unique(x,return_inverse=True);_,yi=np.unique(y,return_inverse=True)
    else:xi,yi=prepared
    total=w.sum()
    def ranks(inv):
        counts=np.bincount(inv,weights=w)
        return (np.cumsum(counts)-counts/2)[inv]/total
    rx=ranks(xi);ry=ranks(yi)
    dx=rx-np.dot(w,rx)/total;dy=ry-np.dot(w,ry)/total
    denom=np.sqrt(np.dot(w,dx*dx)*np.dot(w,dy*dy))
    return float(np.dot(w,dx*dy)/denom) if denom>0 else float('nan')


def paired_frame(queries,gallery,arrays,threshold):
    scores=arrays['target_scores'];ranks=arrays['target_ranks'];margins=arrays['margin10'];means=arrays['gallery_mean']
    rows=[]
    for q in queries:
        if q['kind']!='negative':continue
        a=q['source_query_index'];b=q['query_index'];n=len(q['target_indices'])
        assert queries[a]['target_indices']==q['target_indices']
        for j,index in enumerate(q['target_indices']):
            delta=float(scores[a,j])-float(scores[b,j]);rp=int(ranks[a,j]);rm=int(ranks[b,j])
            assert rp>=1 and rm>=1
            rows.append({'group_id':q['group_id'],'instruction_id':q['instruction_id'],'negative_id':q['negative_id'],
                'edit_operation':q['edit_operation'],'semantic_target':q['semantic_target'],'target_id':gallery[index]['id'],
                'source_query_index':a,'edited_query_index':b,'source_score':float(scores[a,j]),'edited_score':float(scores[b,j]),
                'delta_score':delta,'calibrated_response':delta>threshold,'directional_response':delta>0,
                'source_rank':rp,'edited_rank':rm,'rank_demoted':rm>rp,'rank_unchanged':rm==rp,
                'log_rank_ratio':float(np.log(rm/rp)),'delta_margin10':float(margins[a,j])-float(margins[b,j]),
                'gallery_mean_score_drop':float(means[a])-float(means[b]),
                'target_drop_minus_gallery_mean_drop':delta-(float(means[a])-float(means[b]))})
    return pd.DataFrame(rows)


def calibrate(groups,arrays):
    score_noise={};rank_noise={};controls=[]
    for g in groups:
        vals=[];rankvals=[];n=len(g['target_indices'])
        for a,b in combinations(g['positive_query_indices'],2):
            vals.extend(np.abs(arrays['target_scores'][a,:n]-arrays['target_scores'][b,:n]).tolist())
            ratio=np.abs(np.log(arrays['target_ranks'][a,:n]/arrays['target_ranks'][b,:n]))
            rankvals.extend(ratio.tolist())
        score_noise[g['group_id']]=np.asarray(vals);rank_noise[g['group_id']]=np.asarray(rankvals)
        controls.append({'group_id':g['group_id'],'paraphrase_mean_abs_log_rank_ratio':float(np.mean(rankvals)),
            'paraphrase_mean_abs_score_delta':float(np.mean(vals))})
    return {'normal_perturbation_threshold':group_quantile(score_noise),
        'rank_paraphrase_95_threshold':group_quantile(rank_noise),'calibration_groups':len(groups),
        'calibration_pairs_targets':sum(len(v) for v in score_noise.values()),
        'threshold_definition':'Group-balanced weighted empirical 95th percentile; fixed during event-rate bootstrap.'},pd.DataFrame(controls)


def summarize_subset(frame,threshold,rank_threshold,replicates=2000,seed=42):
    w,groups,gi=hierarchical_weights(frame);ng=len(groups)
    x=frame.delta_score.to_numpy();y=frame.log_rank_ratio.to_numpy()
    z=(x>threshold).astype(float);down=frame.rank_demoted.to_numpy(dtype=float)
    rplus=frame.source_rank.to_numpy();rminus=frame.edited_rank.to_numpy()
    margin=frame.delta_margin10.to_numpy();top10=(rplus<=10).astype(float);exit10=((rplus<=10)&(rminus>10)).astype(float)
    top50=(rplus<=50).astype(float);exit50=((rplus<=50)&(rminus>50)).astype(float)
    variables={'crr':z,'drr':(x>0).astype(float),'rank_demoted':down,'rank_unchanged':(rminus==rplus).astype(float),
        'mean_delta_score':x,'mean_log_rank_ratio':y,'mean_abs_log_rank_ratio':np.abs(y),'mean_delta_margin10':margin,
        'rank_calibrated_response':(y>rank_threshold).astype(float),
        'crr_and_rank_demoted':z*down,'noncrr':1-z,'noncrr_and_rank_demoted':(1-z)*down,
        'crr_and_margin_decreased':z*(margin>0),'source_top10':top10,'top10_exit':exit10,
        'source_top50':top50,'top50_exit':exit50,
        'crr_source_top10':z*top10,'crr_top10_exit':z*exit10,
        'noncrr_source_top10':(1-z)*top10,'noncrr_top10_exit':(1-z)*exit10}
    names=list(variables);group_values=np.stack([np.bincount(gi,weights=w*v,minlength=ng) for v in variables.values()],axis=1)
    point=dict(zip(names,group_values.mean(0)))
    def derived(values):
        d=dict(values)
        def ratio(a,b):return d[a]/d[b] if d[b]>0 else float('nan')
        d['rank_demoted_given_crr']=ratio('crr_and_rank_demoted','crr')
        d['rank_demoted_given_noncrr']=ratio('noncrr_and_rank_demoted','noncrr')
        d['rank_demoted_conditional_gap']=d['rank_demoted_given_crr']-d['rank_demoted_given_noncrr']
        d['margin_decreased_given_crr']=ratio('crr_and_margin_decreased','crr')
        d['top10_exit_given_source_top10']=ratio('top10_exit','source_top10')
        d['top50_exit_given_source_top50']=ratio('top50_exit','source_top50')
        d['top10_exit_given_crr_and_source_top10']=ratio('crr_top10_exit','crr_source_top10')
        d['top10_exit_given_noncrr_and_source_top10']=ratio('noncrr_top10_exit','noncrr_source_top10')
        return d
    point=derived(point)
    _,xi=np.unique(x,return_inverse=True);_,yi=np.unique(y,return_inverse=True)
    point['weighted_spearman_delta_logrank']=weighted_spearman(x,y,w,(xi,yi))
    point['weighted_spearman_delta_margin10']=weighted_spearman(x,margin,w)
    group_crr=group_values[:,names.index('crr')];group_rank=group_values[:,names.index('rank_demoted')]
    point['group_spearman_crr_rankdemotion']=float(spearmanr(group_crr,group_rank).statistic) if np.std(group_crr)>0 and np.std(group_rank)>0 else float('nan')
    rng=np.random.default_rng(seed);boot={k:[] for k in point}
    for _ in range(replicates):
        sampled=rng.integers(0,ng,size=ng);counts=np.bincount(sampled,minlength=ng)
        b=derived(dict(zip(names,group_values[sampled].mean(0))))
        b['weighted_spearman_delta_logrank']=weighted_spearman(x,y,w*counts[gi],(xi,yi))
        # Margin correlation is descriptive; avoid recomputing its ranks for each replicate.
        b['weighted_spearman_delta_margin10']=float('nan')
        a=group_crr[sampled];r=group_rank[sampled]
        b['group_spearman_crr_rankdemotion']=float(spearmanr(a,r).statistic) if np.std(a)>0 and np.std(r)>0 else float('nan')
        for k,v in b.items():boot[k].append(v)
    result={'group_count':ng,'instruction_count':len(frame[['group_id','instruction_id']].drop_duplicates()),
        'edit_count':frame.negative_id.nunique(),'target_edit_pairs':len(frame),
        'source_top10_pair_count':int(top10.sum()),'crr_pair_count':int(z.sum()),
        'normal_perturbation_threshold':threshold,'rank_paraphrase_95_threshold':rank_threshold,**point}
    for k,vals in boot.items():
        a=np.asarray(vals);a=a[np.isfinite(a)]
        if len(a):result[k+'_ci_low'],result[k+'_ci_high']=map(float,np.quantile(a,[.025,.975]))
    return result,pd.DataFrame({'group_id':groups,**{k:group_values[:,i] for i,k in enumerate(names)}})
