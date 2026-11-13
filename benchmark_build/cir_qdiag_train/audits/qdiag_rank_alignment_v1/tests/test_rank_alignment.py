from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import _bootstrap
from score_ranks import rank_batch
from alignment_stats import group_quantile,hierarchical_weights,weighted_spearman


def test_score_drop_can_coexist_with_rank_improvement():
    scores=torch.full((2,60),-.5)
    scores[0,0]=.6;scores[0,1]=.65
    scores[1,0]=.4;scores[1,1]=.3
    result=rank_batch(scores,np.array([[0],[0]]),np.array([1,1]))
    assert result['target_scores'][0,0]-result['target_scores'][1,0]>.1
    assert tuple(result['target_ranks'][:,0])==(2,1)


def test_uniform_score_shift_leaves_rank_and_relative_margin_unchanged():
    base=torch.linspace(1,-1,60)
    scores=torch.stack([base,base-.2])
    result=rank_batch(scores,np.array([[4,12],[4,12]]),np.array([2,2]))
    assert np.allclose(result['target_scores'][0]-result['target_scores'][1],.2)
    assert np.array_equal(result['target_ranks'][0],result['target_ranks'][1])
    assert np.allclose(result['margin10'][0],result['margin10'][1],atol=1e-6)


def test_target_exclusion_top10_boundary_and_stable_ties():
    scores=torch.arange(60,0,-1,dtype=torch.float32)[None]
    result=rank_batch(scores,np.array([[0,15,0]]),np.array([2]))
    assert result['target_ranks'].tolist()==[[1,16,0]]
    assert result['margin10'][0,0]==10 # exclude target: competitor at global rank 11
    assert result['margin10'][0,1]==-6 # target outside top10: use global rank 10
    assert np.isnan(result['target_scores'][0,2])
    tied=rank_batch(torch.zeros(1,60),np.array([[55,2]]),np.array([2]))
    assert tied['target_ranks'].tolist()==[[56,3]]


def test_target_edit_instruction_group_hierarchy():
    frame=pd.DataFrame([
        ('a','i1','e1','t1',1),('a','i1','e1','t2',0),('a','i1','e2','t1',1),('a','i2','e3','t1',1),
        ('b','i1','e4','t1',0),('b','i1','e4','t2',0),('b','i1','e4','t3',0)],
        columns=['group_id','instruction_id','negative_id','target_id','event'])
    w,groups,gi=hierarchical_weights(frame)
    assert np.allclose(np.bincount(gi,weights=w),1)
    assert np.dot(w,frame.event)/len(groups)==pytest.approx(.4375)


def test_group_balanced_calibration_and_weighted_rank_correlation():
    assert group_quantile({'small':np.array([0.,2.]),'large':np.ones(100)},.95)==2
    x=np.array([1.,2.,2.,3.]);y=np.array([5.,2.,7.,8.])
    assert weighted_spearman(x,y,np.ones(4))==pytest.approx(spearmanr(x,y).statistic)
    w=np.array([1,3,2,4])
    assert weighted_spearman(x,y,w)==pytest.approx(spearmanr(np.repeat(x,w),np.repeat(y,w)).statistic)
    assert np.isnan(weighted_spearman(np.ones(4),y,np.ones(4)))
