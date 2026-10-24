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
from .validation import validate, MATCH_FIELDS, APPEARANCE_FIELDS

BASE = 'https://raw.githubusercontent.com/statsbomb/open-data/master/'


class DataSourceError(ValueError):
    """A source failure that can be shown safely without response bodies or secrets."""


def _decode(content: bytes, relative: str):
    if not content:
        raise DataSourceError(f'Empty source file: {relative}')
    if relative.endswith('.json'):
        try:
            value = json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DataSourceError(f'Invalid JSON in {relative}; no data were imported') from exc
        if not isinstance(value, list):
            raise DataSourceError(f'Expected a JSON array in {relative}')
        return value
    if relative.endswith('.pdf') and not content.startswith(b'%PDF-'):
        raise DataSourceError(f'Invalid PDF in {relative}')
    return None


_SESSION = requests.Session()


def fetch(relative: str, root: Path) -> object:
    """Validate before atomic caching; retry transient failures, never retry missing seasons.

    Previously downloaded valid files remain usable offline. Corrupt cached files are
    replaced only after a valid response arrives. Source bodies are never logged.
    """
    path = root / 'raw' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            return _decode(path.read_bytes(), relative)
        except DataSourceError:
            logging.warning('Invalid cached source file; fetching a replacement: %s', relative)
    for attempt in range(3):
        try:
            response = _SESSION.get(BASE + relative, timeout=(10, 45))
            if response.status_code == 404:
                raise DataSourceError(f'Source not found: {relative}. Check competition/season IDs in the StatsBomb catalogue.')
            response.raise_for_status()
            value = _decode(response.content, relative)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_bytes(response.content)
            temporary.replace(path)
            record = {'url': BASE + relative, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                      'sha256': hashlib.sha256(response.content).hexdigest(),
                      'terms': BASE + 'LICENSE.pdf', 'source': 'StatsBomb Open Data'}
            with (root / 'raw' / 'manifest.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            time.sleep(0.12)
            return value
        except requests.RequestException as exc:
            if attempt == 2:
                raise DataSourceError(f'Download failed after 3 attempts: {relative}. Check network access to raw.githubusercontent.com and retry; validated downloads are cached.') from exc
            time.sleep(2 ** attempt)
    raise AssertionError('Unreachable download retry state')


def _lineup_rows(lineups: list, match_id: str) -> tuple[list[dict], bool]:
    """Parse an entire match atomically so malformed intervals cannot partially leak in."""
    rows=[]
    def clock(value):
        parts=value.split(':')
        if len(parts)!=2: raise ValueError('Expected MM:SS')
        minutes,seconds=map(float,parts)
        if not (0<=minutes<=150 and 0<=seconds<60): raise ValueError('Invalid clock')
        return minutes+seconds/60
    try:
        intervals=[p for t in lineups for pl in t['lineup'] for p in pl.get('positions',[])]
        if any((p.get('to_period') or 2)>2 or (p.get('from_period') or 1)>2 for p in intervals):
            return [],True
        for team in lineups:
            for player in team['lineup']:
                positions=player.get('positions',[])
                if not positions: continue
                spans=[]
                for p in positions:
                    start=min(clock(p['from']),45. if p['from_period']==1 else 90.)
                    end=min(clock(p['to']),45. if p.get('to_period')==1 else 90.) if p.get('to') else 90.
                    if start>end: raise ValueError('Interval ends before it starts')
                    spans.append((start,end))
                rows.append({'match_id':str(match_id),'player_id':str(player['player_id']),
                             'player_name':player['player_name'],'team_id':str(team['team_id']),
                             'minutes':sum(end-start for start,end in spans),'position':positions[0]['position'],
                             'started':spans[0][0]==0})
        return rows,False
    except (KeyError,TypeError,ValueError,AttributeError) as exc:
        raise DataSourceError(f'Invalid player lineup/interval schema for match {match_id}; player rows excluded') from exc


def ingest_statsbomb(root: Path, competition: int = 2, season: int = 27, limit: int | None = None, progress=None) -> dict:
    """Import regulation matches and lineup position intervals, including substitutions.

    Minutes use regulation-clock clipping: period 1 to 45, period 2 to 90.
    Stoppage time is excluded because lineups have null final-whistle timestamps.
    Extra-time matches are explicitly excluded and recorded.
    """
    if competition <= 0 or season <= 0 or (limit is not None and limit <= 0):
        raise ValueError('Competition, season and optional match limit must be positive')
    fetch('LICENSE.pdf', root)
    fetch('README.md', root)
    raw = fetch(f'data/matches/{competition}/{season}.json', root)
    if not raw:
        raise DataSourceError('The selected season contains no matches')
    try:
        raw = sorted(raw, key=lambda x: (x['match_date'], x['match_id']))
        for r in raw:
            for key in ['competition','season','home_team','away_team','home_score','away_score']:
                if key not in r: raise KeyError(key)
            for field,key in [('competition','competition_name'),('season','season_name'),('home_team','home_team_id'),('home_team','home_team_name'),('away_team','away_team_id'),('away_team','away_team_name')]:
                if key not in r[field]: raise KeyError(key)
    except (KeyError, TypeError) as exc:
        raise DataSourceError('Match source schema is invalid; select a supported StatsBomb season') from exc
    if limit: raw = raw[:limit]
    matches, appearances, excluded = [], [], []
    for number,r in enumerate(raw,1):
        if number==1 or number%50==0: logging.info('Processing lineup %s/%s',number,len(raw))
        mid = r['match_id']
        if progress: progress(number / len(raw), f'Processing match {number}/{len(raw)}')
        try:
            lineups = fetch(f'data/lineups/{mid}.json', root)
            if not all(isinstance(t,dict) and 'team_id' in t and 'lineup' in t for t in lineups):
                raise DataSourceError(f'Invalid lineup schema for match {mid}')
        except DataSourceError as exc:
            # Match results remain useful; never invent missing player involvement.
            lineups = []
            excluded.append({'match_id':str(mid),'reason':str(exc)})
        try:
            player_rows,extra_time=_lineup_rows(lineups,str(mid))
        except DataSourceError as exc:
            player_rows,extra_time=[],False
            excluded.append({'match_id':str(mid),'reason':str(exc)})
        if extra_time:
            excluded.append({'match_id':str(mid),'reason':'Extra time is unsupported; match excluded'})
            continue
        if not player_rows and not any(str(item['match_id'])==str(mid) for item in excluded):
            excluded.append({'match_id':str(mid),'reason':'No player intervals available; match result retained'})
        matches.append({'match_id': str(mid), 'date': r['match_date'],
                        'competition': r['competition']['competition_name'], 'season': r['season']['season_name'],
                        'home_id': str(r['home_team']['home_team_id']), 'away_id': str(r['away_team']['away_team_id']),
                        'home_name': r['home_team']['home_team_name'], 'away_name': r['away_team']['away_team_name'],
                        'home_goals': r['home_score'], 'away_goals': r['away_score'], 'source': 'StatsBomb Open Data'})
        appearances.extend(player_rows)
    m = pd.DataFrame(matches, columns=MATCH_FIELDS + ['home_name','away_name','source'])
    a = pd.DataFrame(appearances, columns=APPEARANCE_FIELDS)
    if m.empty:
        raise DataSourceError('No supported regulation matches were available in this season')
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
    report['competition_id'] = competition
    report['season_id'] = season
    report['player_matches'] = int(a.match_id.nunique())
    report['player_count'] = int(a.player_id.nunique())
    report['source_match_count'] = len(raw)
    report['partial_download'] = limit is not None
    manifest = root / 'raw' / 'manifest.jsonl'
    if manifest.exists():
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        source_url = BASE + f'data/matches/{competition}/{season}.json'
        record = next((r for r in records if r['url']==source_url),{})
        report['match_file_retrieved_at'] = record.get('retrieved_at')
        report['match_file_sha256'] = record.get('sha256')
    (out / 'quality.json').write_text(json.dumps(report, indent=2, default=str))
    if report['errors']: raise ValueError(f'Data validation failed; see {out / "quality.json"}')
    for name, frame in [('matches',m),('appearances',a)]:
        temporary = out / f'{name}.csv.tmp'
        frame.to_csv(temporary,index=False)
        temporary.replace(out / f'{name}.csv')
    teams = pd.concat([m[['home_id','home_name']].rename(columns={'home_id':'team_id','home_name':'team_name'}),
                       m[['away_id','away_name']].rename(columns={'away_id':'team_id','away_name':'team_name'})]).drop_duplicates()
    teams.to_csv(out / 'teams.csv', index=False)
    a[['player_id','player_name']].drop_duplicates().to_csv(out / 'players.csv', index=False)
    return report


def load_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(root / 'processed/matches.csv', dtype={'match_id':str,'home_id':str,'away_id':str,'season':str})
    appearance_path = root / 'processed/appearances.csv'
    a = pd.read_csv(appearance_path, dtype={'match_id':str,'player_id':str,'team_id':str}) if appearance_path.exists() else pd.DataFrame(columns=APPEARANCE_FIELDS)
    return m, a
