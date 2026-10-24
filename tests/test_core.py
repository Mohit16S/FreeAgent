import json
import numpy as np
import pandas as pd
import pytest
from freeagent.data.validation import validate,require_valid,validate_transfer_date
from freeagent.data.features import design,earlier_team_form
from freeagent.impact.ridge_model import ImpactModel
from freeagent.match.dixon_coles import DixonColes,tau,score_matrix,outcomes
from freeagent.simulation.monte_carlo import simulate
from freeagent.valuation.displacement import Displacement
from freeagent.valuation.net_value import Costs,net_value
from freeagent.services.analysis_service import train,evaluate,save,load,standings
from freeagent.evaluation.backtesting import walk_forward,historical_transfer_runs
from freeagent.evaluation.metrics import probability_metrics


def test_valid_contract(data):
    m,a = data
    assert not validate(m,a)['errors']

@pytest.mark.parametrize('field,value',[('home_goals',-1),('away_goals',1.5),('date','bad'),('season','1990')])
def test_bad_match(data,field,value):
    m,a = data; m[field] = m[field].astype(object); m.loc[0,field]=value
    assert validate(m,a)['errors']


def test_duplicates_minutes_mapping(data):
    m,a=data
    assert validate(pd.concat([m,m.iloc[:1]]),a)['errors']
    assert validate(m,pd.concat([a,a.iloc[:1]]))['errors']
    a.loc[0,'minutes']=91
    assert validate(m,a)['errors']
    a.loc[0,'minutes']=90; a.loc[0,'team_id']='unknown'
    assert validate(m,a)['errors']


def test_cutoff(data):
    m,a=data
    with pytest.raises(ValueError): require_valid(m,a,'2023-02-01')
    with pytest.raises(ValueError): validate_transfer_date('2024-01-01','2023-01-01')


def test_features_alignment(data):
    m,a=data; players=sorted(a.player_id.unique()); teams=sorted(a.team_id.unique())
    x=design(m,a,players,teams)
    assert np.allclose(x.sum(axis=1),0)
    assert np.allclose(x,design(m,a.sample(frac=1,random_state=1),players,teams))
    f=earlier_team_form(m); changed=m.copy(); changed.loc[59,'home_goals']=99
    pd.testing.assert_frame_equal(f,earlier_team_form(changed))


def test_ridge(data):
    m,a=data; impact=ImpactModel().fit(m,a,'2023-04-01',bootstrap=4)
    assert len(impact.report())==44
    assert np.isfinite(impact.samples).all()
    assert np.isfinite(impact.predict(m.iloc[:2],a)).all()
    assert impact.alpha in (1,10,100)
    assert impact.report().eligible.all()
    assert impact.samples.shape==(4,44)

@pytest.mark.parametrize('x,y,expected',[(0,0,.7),(0,1,1.2),(1,0,1.15),(1,1,.9),(2,2,1)])
def test_tau(x,y,expected):
    assert tau(x,y,2,1.5,.1)==pytest.approx(expected)


def test_matrix():
    mat=score_matrix(1.7,1.1,-.08)
    assert mat.min()>=0 and mat.sum()==pytest.approx(1)
    assert outcomes(mat).sum()==pytest.approx(1)
    assert outcomes(mat)[0]>outcomes(mat)[2]
    assert np.allclose(outcomes(mat),outcomes(score_matrix(1.7,1.1,-.08,16)),atol=1e-6)
    with pytest.raises(ValueError): score_matrix(10,10,0,3)
    with pytest.raises(ValueError): score_matrix(2,2,1)


def test_fit(data):
    m,_=data; model=DixonColes().fit(m,'2023-04-01')
    assert model.fit_info['converged']
    assert abs(model.attack.sum())<1e-10 and abs(model.defence.sum())<1e-10
    assert np.isfinite(model.predict('A','B')['probabilities']).all()
    with pytest.raises(ValueError): model.matrix('X','A')
    with pytest.raises(ValueError): DixonColes().fit(m.iloc[:4],'2023-04-01')


def test_fit_failure(data,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('freeagent.match.dixon_coles.minimize',lambda *a,**k:SimpleNamespace(success=False,fun=0,message='test failure'))
    with pytest.raises(RuntimeError): DixonColes().fit(data[0],'2023-04-01')

class Fixed:
    teams=['A','B']
    def matrix(self,*args):
        m=np.zeros((3,3)); m[2,1]=1; return m


def test_simulation_points():
    f=pd.DataFrame([dict(match_id='1',date='2024-01-01',home_id='A',away_id='B')])
    a=simulate(Fixed(),f,10); b=simulate(Fixed(),f,10)
    assert (a.points[:,0]==3).all() and (a.points[:,1]==0).all()
    assert (a.goal_difference[:,0]==1).all() and (a.positions[:,0]==1).all()
    assert np.array_equal(a.points,b.points)
    assert a.summary(1,1).title.sum()==1
    with pytest.raises(ValueError): simulate(Fixed(),pd.concat([f,f]),10)
    with pytest.raises(ValueError): simulate(Fixed(),f,10,uniforms=np.full((1,10),1.))


def test_displacement():
    audit=Displacement('A','B',900,1800,20).audit()
    assert audit['total_before']==audit['total_after'] and audit['incumbent_after']==900
    with pytest.raises(ValueError): Displacement('A','B',2000,1800,20).audit()


def test_finance():
    c=Costs(20e6,3e6,4)
    assert c.breakdown()['total']==8e6
    assert c.breakdown()['total_contract_commitment']==32e6
    r=net_value(np.array([0,1,2]),c,8e6,tolerance=1e6)
    assert r['mean']==0
    assert sum(r['category_probabilities'].values())==pytest.approx(1)
    assert r['verdict']=='Fair'
    assert net_value(np.array([0,0,100]),Costs(0,0,1),1e6)['verdict']=='Fair'
    with pytest.raises(ValueError): Costs(-1,0,1).breakdown()
    with pytest.raises(ValueError): net_value(np.array([np.nan]),c,1)


def test_end_to_end(data,tmp_path):
    m,a=data; bundle=train(m,a,'2023-02-15',bootstrap=2,min_minutes=90)
    save(bundle,tmp_path/'model.joblib'); bundle=load(tmp_path/'model.joblib')
    future=m[m.date>='2023-02-15'][['match_id','date','home_id','away_id']]
    count=int(((future.home_id=='A')|(future.away_id=='A')).sum())
    d=Displacement('B0','A0',count*45,count*90,count)
    r=evaluate(bundle,future,'A',d,Costs(100,10,2),10,n=30,qualification=1,relegation=1)
    assert np.isfinite(r['epa_mean'])
    assert r['displacement']['total_before']==r['displacement']['total_after']
    zero=Displacement('B0','A0',0,count*90,count)
    r0=evaluate(bundle,future,'A',zero,Costs(0,0,1),10,n=30,qualification=1,relegation=1)
    assert np.all(r0['epa_samples']==0)
    assert np.array_equal(r0['before_points'],r0['after_points'])
    assert bundle.metadata['training_max_date']<'2023-02-15'


def test_backtest(data,tmp_path):
    m,a=data
    metrics=walk_forward(m,a,['2023-02-01','2023-02-15','2023-03-05'],tmp_path)
    p=pd.read_csv(tmp_path/'predictions.csv')
    assert (p.training_max_date<p.cutoff).all()
    assert (p.date>=p.cutoff).all()
    assert metrics['rows']==29
    assert metrics['dixon_coles']['log_loss']>0
    assert probability_metrics(np.array([[1.,0,0]]),[0])['rps']==0


def test_transfer_scaffold(tmp_path):
    calls=[]
    def runner(assumptions,cutoff):
        assert 'realised_points' not in assumptions
        calls.append(cutoff); return {'epa':0}
    rows=pd.DataFrame([{'prediction_date':'2023-01-01','transfer_date':'2022-12-01','realised_points':10},
                       {'prediction_date':'2023-01-01','transfer_date':'2024-01-01','realised_points':20}])
    historical_transfer_runs(rows,runner,tmp_path/'x.json')
    assert len(calls)==1
    assert json.loads((tmp_path/'x.json').read_text())[1]['status']=='failed'


def test_artifact_changed_data_is_rejected(data,tmp_path):
    from freeagent.services.analysis_service import verify_dataset
    m,a=data; bundle=train(m,a,'2023-02-15',bootstrap=2,min_minutes=90)
    verify_dataset(bundle,m,a)
    changed=m.copy(); changed.loc[0,'home_goals']+=1
    with pytest.raises(ValueError,match='changed'): verify_dataset(bundle,changed,a)
    save(bundle,tmp_path/'m.joblib')
    meta=json.loads((tmp_path/'m.json').read_text()); meta['version']='wrong'
    (tmp_path/'m.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='version'): load(tmp_path/'m.joblib')


def test_initial_standings_and_rank_ties():
    class Draw:
        teams=['A','B']
        def matrix(self,*args):
            result=np.zeros((3,3)); result[0,0]=1.; return result
    fixtures=pd.DataFrame([dict(match_id='1',date='2024-01-01',home_id='A',away_id='B')])
    initial=pd.DataFrame({'team_id':['A','B'],'points':[10,10],'goal_difference':[2,2],'goals_for':[3,4]})
    result=simulate(Draw(),fixtures,n=10,initial=initial)
    assert (result.points==11).all()
    assert (result.positions[:,1]==1).all()
    assert (result.goal_difference==2).all()
