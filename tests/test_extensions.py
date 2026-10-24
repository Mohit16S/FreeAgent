import numpy as np
import pandas as pd
import pytest
from freeagent.valuation.alternatives import alternatives
from freeagent.valuation.sell_side import sell_or_retain
from freeagent.valuation.budget import optimise


def test_similarity_excludes_self_and_role():
    r=pd.DataFrame({'player_id':['a','b','c'],'position':['F','F','D'],'eligible':[True]*3,'impact':[1,2,1],'minutes':[900,1000,900],'lower':[0,1,0],'upper':[2,3,2]})
    assert alternatives(r,'a').player_id.tolist()==['b']


def test_sell_side_accounting():
    r=sell_or_retain(np.array([-10.,10.]),20,5,2)
    assert r['mean_incremental_value']==17
    assert r['probability_prefer_sale']==1


def test_discrete_budget():
    c=pd.DataFrame({'player_id':['a','b','c'],'role':['F','F','D'],'cost':[6.,4.,4.], 'value':[8.,7.,6.],'replacement_id':['x','x','y']})
    r=optimise(c,8,{'F':1,'D':1})
    assert set(r['selected'].player_id)=={'b','c'}
    assert r['total_cost']<=8 and r['total_value']==13
    assert optimise(c,0,{'F':1,'D':1})['selected'].empty
    with pytest.raises(ValueError): optimise(c,-1,{'F':1,'D':1})
