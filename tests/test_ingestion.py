"""Synthetic provider-shaped payloads test the connector without network requests."""
import pandas as pd
from freeagent.data.ingestion import ingest_statsbomb


def test_lineup_quarantine_and_null_final_whistle(tmp_path,monkeypatch):
    raw=[]
    for i in [1,2]:
        raw.append({'match_id':i,'match_date':f'2016-01-0{i}','competition':{'competition_name':'TEST'},
                    'season':{'season_name':'2015/2016'},'home_team':{'home_team_id':1,'home_team_name':'A'},
                    'away_team':{'away_team_id':2,'away_team_name':'B'},'home_score':1,'away_score':0})
    def fetch(relative,root):
        if 'matches/' in relative: return raw
        if not 'lineups/' in relative: return None
        teams=[]
        for t in [1,2]:
            players=[]
            count=12 if relative.endswith('/2.json') and t==1 else 11
            for p in range(count):
                players.append({'player_id':t*100+p,'player_name':f'Test {t}-{p}',
                    'positions':[{'position':'TEST','from':'00:00','to':None,'from_period':1,'to_period':None}]})
            teams.append({'team_id':t,'lineup':players})
        return teams
    monkeypatch.setattr('freeagent.data.ingestion.fetch',fetch)
    report=ingest_statsbomb(tmp_path)
    assert not report['errors']
    assert report['rows']==2 and report['appearance_rows']==22
    assert len(report['excluded'])==1
    assert (tmp_path/'processed/quarantined_appearances.csv').exists()
    a=pd.read_csv(tmp_path/'processed/appearances.csv')
    assert (a.minutes==90).all()


def test_missing_lineups_retain_real_match_records(tmp_path,monkeypatch):
    from freeagent.data.ingestion import DataSourceError, load_data
    raw=[{'match_id':1,'match_date':'2016-01-01','competition':{'competition_name':'TEST'},
          'season':{'season_name':'2015/2016'},'home_team':{'home_team_id':1,'home_team_name':'A'},
          'away_team':{'away_team_id':2,'away_team_name':'B'},'home_score':1,'away_score':0}]
    def fetch(relative,root):
        if 'matches/' in relative: return raw
        if 'lineups/' in relative: raise DataSourceError('Lineup unavailable: test')
        return None
    monkeypatch.setattr('freeagent.data.ingestion.fetch',fetch)
    report=ingest_statsbomb(tmp_path)
    matches,appearances=load_data(tmp_path)
    assert len(matches)==1 and appearances.empty
    assert not report['errors'] and report['warnings'] and report['excluded']


def test_empty_season_and_bad_schema(tmp_path,monkeypatch):
    import pytest
    from freeagent.data.ingestion import DataSourceError
    monkeypatch.setattr('freeagent.data.ingestion.fetch',lambda *args:[])
    with pytest.raises(DataSourceError,match='no matches'): ingest_statsbomb(tmp_path)
    monkeypatch.setattr('freeagent.data.ingestion.fetch',lambda *args:[{'wrong':'schema'}])
    with pytest.raises(DataSourceError,match='schema'): ingest_statsbomb(tmp_path)
