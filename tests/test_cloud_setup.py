"""Cold-start and source-failure checks. Provider payloads in mocks are test-only."""
from pathlib import Path
import json
import requests
import pytest
from streamlit.testing.v1 import AppTest
from freeagent.data.ingestion import fetch, DataSourceError, load_data, _lineup_rows
from freeagent.paths import RuntimePaths
from freeagent.services.setup_service import setup, diagnostics
from freeagent.config import load_config
from freeagent.services.analysis_service import train


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr('freeagent.data.ingestion.time.sleep',lambda _:None)


def response(content=b'[]',status=200):
    result=requests.Response(); result.status_code=status; result._content=content
    return result


def test_fetch_retries_and_caches(tmp_path,monkeypatch,no_sleep):
    calls=[]
    def get(*args,**kwargs):
        calls.append(args)
        if len(calls)<3: raise requests.Timeout('test')
        return response(b'[{"id":1}]')
    monkeypatch.setattr('freeagent.data.ingestion._SESSION.get',get)
    assert fetch('data/test.json',tmp_path)==[{'id':1}]
    assert fetch('data/test.json',tmp_path)==[{'id':1}]
    assert len(calls)==3
    manifest=json.loads((tmp_path/'raw/manifest.jsonl').read_text())
    assert manifest['sha256'] and manifest['retrieved_at']


@pytest.mark.parametrize('content,status,message',[(b'<html>bad</html>',200,'Invalid JSON'),(b'{}',200,'JSON array'),(b'',404,'not found')])
def test_invalid_source_not_cached(tmp_path,monkeypatch,no_sleep,content,status,message):
    monkeypatch.setattr('freeagent.data.ingestion._SESSION.get',lambda *a,**k:response(content,status))
    with pytest.raises(DataSourceError,match=message): fetch('data/test.json',tmp_path)
    assert not (tmp_path/'raw/data/test.json').exists()


def test_network_failure_is_actionable(tmp_path,monkeypatch,no_sleep):
    def fail(*a,**k): raise requests.ConnectionError('test; must not expose response content')
    monkeypatch.setattr('freeagent.data.ingestion._SESSION.get',fail)
    with pytest.raises(DataSourceError,match='3 attempts'): fetch('data/test.json',tmp_path)


def test_corrupt_cache_can_recover(tmp_path,monkeypatch,no_sleep):
    path=tmp_path/'raw/data/test.json'; path.parent.mkdir(parents=True); path.write_text('bad')
    monkeypatch.setattr('freeagent.data.ingestion._SESSION.get',lambda *a,**k:response(b'[1]'))
    assert fetch('data/test.json',tmp_path)==[1]
    assert path.read_text()=='[1]'


def test_bad_lineup_schema_is_explained():
    with pytest.raises(DataSourceError,match='lineup/interval'):
        _lineup_rows([{'team_id':1,'lineup':[{'positions':[{'from':'bad'}]}]}],'123')


def test_cloud_without_mac_files(tmp_path,monkeypatch):
    monkeypatch.setenv('FREEAGENT_HOME',str(tmp_path))
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    assert not app.exception
    assert any(button.label=='Download real data and train models' for button in app.button)
    assert not any('.venv/bin' in code.value for code in app.code)
    assert diagnostics(RuntimePaths(tmp_path))['model_status']=='not trained'
    assert 'result' not in app.session_state


def test_match_only_data_blocks_player_training(data,tmp_path,monkeypatch):
    m,a=data; out=tmp_path/'data/processed'; out.mkdir(parents=True)
    m.to_csv(out/'matches.csv',index=False)
    matches,appearances=load_data(tmp_path/'data')
    assert appearances.empty
    with pytest.raises(ValueError,match='real player appearances'): train(matches,appearances,'2023-04-01',bootstrap=2)
    monkeypatch.setenv('FREEAGENT_HOME',str(tmp_path))
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    assert not app.exception
    assert any('player-level' in item.value for item in app.warning)
    assert 'result' not in app.session_state


def test_setup_reuses_models_and_reports_failure(data,tmp_path,monkeypatch):
    m,a=data
    def ingest(root,*args,**kwargs):
        out=root/'processed'; out.mkdir(parents=True,exist_ok=True)
        m.to_csv(out/'matches.csv',index=False); a.to_csv(out/'appearances.csv',index=False)
        return {'rows':len(m),'appearance_rows':len(a),'errors':[]}
    monkeypatch.setattr('freeagent.services.setup_service.ingest_statsbomb',ingest)
    cfg=load_config(); cfg['bootstrap_draws']=2
    first=setup(RuntimePaths(tmp_path),cfg,cutoff='2023-04-01')
    assert not first['models_reused']
    second=setup(RuntimePaths(tmp_path),cfg,cutoff='2023-04-01')
    assert second['models_reused'] and second['diagnostics']['model_status']=='trained'
    with pytest.raises(ValueError,match='at least 20'):
        setup(RuntimePaths(tmp_path),cfg,cutoff='2023-01-02')
    status=json.loads((tmp_path/'data/setup-status.json').read_text())
    assert status['state']=='failed' and status['stage']=='validation'


def test_setup_button_calls_real_orchestrator(tmp_path,monkeypatch):
    monkeypatch.setenv('FREEAGENT_HOME',str(tmp_path))
    calls=[]
    def fail(paths,config,**kwargs):
        calls.append(paths)
        raise DataSourceError('Network unavailable: test-only failure')
    monkeypatch.setattr('freeagent.services.setup_service.setup',fail)
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    app.button[0].click().run(timeout=30)
    assert len(calls)==1 and app.error and not app.exception
    assert 'result' not in app.session_state


def test_corrupt_model_keeps_setup_accessible(data,tmp_path,monkeypatch):
    m,a=data; out=tmp_path/'data/processed'; out.mkdir(parents=True)
    m.to_csv(out/'matches.csv',index=False); a.to_csv(out/'appearances.csv',index=False)
    artifact=tmp_path/'models/artifacts/core.joblib'; artifact.parent.mkdir(parents=True)
    artifact.write_text('not a model')
    monkeypatch.setenv('FREEAGENT_HOME',str(tmp_path))
    state=diagnostics(RuntimePaths(tmp_path))
    assert state['data_status']=='validated'
    assert state['model_status']=='unavailable or stale'
    app=AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    app.sidebar.radio[0].set_value('Data & models').run(timeout=30)
    assert not app.exception
    assert any(button.label=='Download real data and train models' for button in app.button)


def test_configuration_seeds_reproduce_training(data):
    import numpy as np
    matches,appearances=data
    first=train(matches,appearances,'2023-04-01',bootstrap=2,seed=42)
    second=train(matches,appearances,'2023-04-01',bootstrap=2,seed=42)
    assert np.array_equal(first.impact.samples,second.impact.samples)
    assert np.allclose(first.match_draws[0].attack,second.match_draws[0].attack)


def test_partial_evaluation_does_not_hide_valid_data(data,tmp_path):
    m,a=data; out=tmp_path/'data/processed'; out.mkdir(parents=True)
    m.to_csv(out/'matches.csv',index=False); a.to_csv(out/'appearances.csv',index=False)
    reports=tmp_path/'reports/backtest'; reports.mkdir(parents=True)
    (reports/'metrics.json').write_text('{}')
    state=diagnostics(RuntimePaths(tmp_path))
    assert state['data_status']=='validated' and state['evaluation_status']=='not calculated'
    assert any('evaluation is incomplete' in item for item in state['warnings'])


def test_invalid_pickled_artifact_is_actionable(tmp_path):
    from freeagent import __version__
    from freeagent.services.analysis_service import load
    import sklearn
    artifact=tmp_path/'model.joblib'; artifact.write_bytes(b'not-pickle-data')
    artifact.with_suffix('.json').write_text(json.dumps({'version':__version__,'sklearn':sklearn.__version__}))
    with pytest.raises(ValueError,match='corrupt or incompatible'): load(artifact)
