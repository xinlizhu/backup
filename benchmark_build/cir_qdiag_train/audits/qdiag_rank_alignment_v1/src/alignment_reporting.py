"""Readable summaries and explicitly retained score/rank counterexamples."""
import pandas as pd
from alignment_data import read_json,read_jsonl,write_json
from alignment_stats import hierarchical_weights

def export_interpretation(root,combined,model_names):
 queries=read_jsonl(root/'queries.jsonl');gallery=read_json(root/'gallery.json');idmap={g['id']:g['path'] for g in gallery}
 output=[];examples=[]
 for model in combined.model.unique():
  frame=pd.read_parquet(root/model/'analysis/full/per_target_edit.parquet')
  for subset in ['strict','add','substitute','delete','all']:
   part=frame if subset=='all' else frame[frame.edit_operation.isin(['add','substitute'])] if subset=='strict' else frame[frame.edit_operation==subset]
   w,_,_=hierarchical_weights(part);z=part.calibrated_response.to_numpy();den=w[z].sum()
   masks={'rank_demoted':part.edited_rank.to_numpy()>part.source_rank.to_numpy(),'rank_unchanged':part.edited_rank.to_numpy()==part.source_rank.to_numpy(),'rank_improved':part.edited_rank.to_numpy()<part.source_rank.to_numpy()}
   row={'model':model,'scope':'full','subset':subset}
   for name,mask in masks.items():row[name+'_given_crr']=float(w[z&mask].sum()/den) if den else float('nan')
   row['not_demoted_given_crr']=row['rank_unchanged_given_crr']+row['rank_improved_given_crr']
   row['margin_not_decreased_given_crr']=float(w[z&(part.delta_margin10.to_numpy()<=0)].sum()/den) if den else float('nan')
   output.append(row)
  # Illustrations selected by largest score drops, at most one per group;
  # prevalence is estimated using all rows above, not these illustrations.
  bad=frame[frame.edit_operation.isin(['add','substitute'])&frame.calibrated_response&(frame.edited_rank<=frame.source_rank)]
  for _,r in bad.sort_values('delta_score',ascending=False).drop_duplicates('group_id').head(10).iterrows():
   source=queries[int(r.source_query_index)];edited=queries[int(r.edited_query_index)]
   examples.append({'model':model,'group_id':r.group_id,'negative_id':r.negative_id,'operation':r.edit_operation,
    'original_text':source['text'],'edited_text':edited['text'],'reference_paths':[idmap[i] for i in source['reference_ids']],
    'target_id':r.target_id,'target_path':idmap[r.target_id],'source_score':float(r.source_score),'edited_score':float(r.edited_score),
    'delta_score':float(r.delta_score),'source_rank':int(r.source_rank),'edited_rank':int(r.edited_rank),
    'delta_margin10':float(r.delta_margin10),'gallery_mean_score_drop':float(r.gallery_mean_score_drop),
    'selection':'Largest score-drop exceptions, one per group; illustrative, not a random sample.'})
 pd.DataFrame(output).to_csv(root/'rank_exception_summary.csv',index=False);write_json(root/'counterexamples.json',examples)
 primary=combined[(combined.scope=='full')&(combined.subset=='strict')]
 lines=['CURRENT RESULTS: full-gallery, ADD/SUBSTITUTE subset','',
  f'Completed models: {len(primary)} / {len(model_names)}',
  'Gallery: 109,598 images; edits: 5,735 ADD/SUBSTITUTE (all 10,131 edits are separately retained).',
  'rho = weighted Spearman between target-score drop and log(edited rank / original rank).',
  'Weights follow target -> edit -> source instruction -> group. CIs resample groups, 2,000 times.',
  'CRR event: target-score drop > model-specific paraphrase threshold.',
  'P(down | CRR) is conditional on an individual target/edit triggering that event.',
  'Ranks are full-gallery ranks. Margin uses the tenth highest-scoring OTHER image, recomputed for each query.',
  'DELETE is descriptive; its weaker query does not imply the original target must be demoted.','']
 for _,r in primary.iterrows():
  lines.append(f"{model_names[r.model]}: rho={r.weighted_spearman_delta_logrank:.4f} [{r.weighted_spearman_delta_logrank_ci_low:.4f}, {r.weighted_spearman_delta_logrank_ci_high:.4f}]; P(down | CRR)={100*r.rank_demoted_given_crr:.2f}%; P(down | non-CRR)={100*r.rank_demoted_given_noncrr:.2f}%; P(relative margin decreases | CRR)={100*r.margin_decreased_given_crr:.2f}%; group-level rho(CRR_g, demotion rate_g)={r.group_spearman_crr_rankdemotion:.4f}.")
 lines+=['','Interpretation: empirical alignment, not a mathematical guarantee. Absolute score drops can coexist with stable or improved ranks when competitors fall further. Do not confuse score-drop/rank correlation with group-level CRR/rank-rate correlation or correlation across models.','These are results for the full released dataset; they are not the historical 500-group main-table CRRs. The pilot500 rows are supplied separately.']
 (root/'results.txt').write_text('\n'.join(lines)+'\n')
