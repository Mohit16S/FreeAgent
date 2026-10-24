"""Cached StatsBomb Open Data ingestion with provenance and explicit IDs."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import logging
import json
import time
import pandas as pd
import requests
from .validation import validate

BASE = 'https://raw.githubusercontent.com/statsbomb/open-data/master/'


def fetch(relative: str, root: Path) -> object:
    path = root / 'raw' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(BASE + relative, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
        record = {'url': BASE + relative, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                  'sha256': hashlib.sha256(response.content).hexdigest(),
                  'terms': BASE + 'LICENSE.pdf', 'source': 'StatsBomb Open Data'}
        with (root / 'raw' / 'manifest.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        time.sleep(0.12)
    return json.loads(path.read_text()) if relative.endswith('.json') else None


def ingest_statsbomb(root: Path, competition: int = 2, season: int = 27, limit: int | None = None) -> dict:
    """Import regulation matches and lineup position intervals, including substitutions.

    Minutes use regulation-clock clipping: period 1 to 45, period 2 to 90.
    Stoppage time is excluded because lineups have null final-whistle timestamps.
    Extra-time matches are explicitly excluded and recorded.
    """
    fetch('LICENSE.pdf', root)
    fetch('README.md', root)
    raw = fetch(f'data/matches/{competition}/{season}.json', root)
    raw = sorted(raw, key=lambda x: (x['match_date'], x['match_id']))
    if limit: raw = raw[:limit]
    matches, appearances, excluded = [], [], []
    for number,r in enumerate(raw,1):
        if number==1 or number%50==0: logging.info('Processing lineup %s/%s',number,len(raw))
        mid = r['match_id']
        lineups = fetch(f'data/lineups/{mid}.json', root)
        # Explicit regulation-clock approximation, without inventing final-whistle times.
        def clock(value):
            if value is None: return None
            pieces = [float(p) for p in value.split(':')]
            return pieces[0] + pieces[1] / 60
        intervals = [p for t in lineups for pl in t['lineup'] for p in pl.get('positions', [])]
        if not intervals or any((p.get('to_period') or 2) > 2 or (p.get('from_period') or 1) > 2 for p in intervals):
            excluded.append({'match_id': mid, 'reason': 'Extra time or missing lineup intervals'})
            continue
        duration = 90.
        matches.append({'match_id': str(mid), 'date': r['match_date'],
                        'competition': r['competition']['competition_name'], 'season': r['season']['season_name'],
                        'home_id': str(r['home_team']['home_team_id']), 'away_id': str(r['away_team']['away_team_id']),
                        'home_name': r['home_team']['home_team_name'], 'away_name': r['away_team']['away_team_name'],
                        'home_goals': r['home_score'], 'away_goals': r['away_score'], 'source': 'StatsBomb Open Data'})
        for t in lineups:
            for pl in t['lineup']:
                positions = pl.get('positions', [])
                if not positions: continue
                spans = [(min(clock(p['from']),45. if p['from_period']==1 else 90.),
                          min(clock(p['to']),45. if p.get('to_period')==1 else 90.) if p.get('to') else duration)
                         for p in positions]
                minutes = sum(end - start for start, end in spans) * 90 / duration
                appearances.append({'match_id': str(mid), 'player_id': str(pl['player_id']),
                                    'player_name': pl['player_name'], 'team_id': str(t['team_id']),
                                    'minutes': minutes, 'position': positions[0]['position'],
                                    'started': spans[0][0] == 0})
    m, a = pd.DataFrame(matches), pd.DataFrame(appearances)
    out = root / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    # Quarantine whole appearance sets when source intervals are inconsistent.
    totals = a.groupby(['match_id','team_id']).minutes.sum()
    invalid_ids = set(totals[(totals>990.01)|(totals<720)].index.get_level_values('match_id'))
    invalid_ids |= set(a.loc[(a.minutes<0)|(a.minutes>90),'match_id'])
    if invalid_ids:
        a[a.match_id.isin(invalid_ids)].to_csv(out / 'quarantined_appearances.csv',index=False)
        excluded.extend({'match_id':mid,'reason':'Inconsistent lineup minutes; appearances quarantined, match retained'} for mid in sorted(invalid_ids))
        a = a[~a.match_id.isin(invalid_ids)].copy()
    report = validate(m, a)
    report['excluded'] = excluded
    report['source'] = BASE
    manifest = root / 'raw' / 'manifest.jsonl'
    if manifest.exists():
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        source_url = BASE + f'data/matches/{competition}/{season}.json'
        record = next((r for r in records if r['url']==source_url),{})
        report['match_file_retrieved_at'] = record.get('retrieved_at')
        report['match_file_sha256'] = record.get('sha256')
    (out / 'quality.json').write_text(json.dumps(report, indent=2, default=str))
    if report['errors']: raise ValueError(f'Data validation failed; see {out / "quality.json"}')
    m.to_csv(out / 'matches.csv', index=False)
    a.to_csv(out / 'appearances.csv', index=False)
    teams = pd.concat([m[['home_id','home_name']].rename(columns={'home_id':'team_id','home_name':'team_name'}),
                       m[['away_id','away_name']].rename(columns={'away_id':'team_id','away_name':'team_name'})]).drop_duplicates()
    teams.to_csv(out / 'teams.csv', index=False)
    a[['player_id','player_name']].drop_duplicates().to_csv(out / 'players.csv', index=False)
    return report


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(root / 'processed/matches.csv', dtype={'match_id':str,'home_id':str,'away_id':str,'season':str})
    a = pd.read_csv(root / 'processed/appearances.csv', dtype={'match_id':str,'player_id':str,'team_id':str})
    return m, a
