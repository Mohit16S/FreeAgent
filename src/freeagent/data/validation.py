"""Strict CSV contracts; reject suspicious records rather than silently removing them."""
from __future__ import annotations
import numpy as np
import pandas as pd

MATCH_FIELDS = ['match_id', 'date', 'competition', 'season', 'home_id', 'away_id', 'home_goals', 'away_goals']
APPEARANCE_FIELDS = ['match_id', 'player_id', 'player_name', 'team_id', 'minutes', 'position', 'started']


def validate(matches: pd.DataFrame, appearances: pd.DataFrame | None = None,
             cutoff: str | None = None) -> dict:
    errors = []
    warnings = []
    missing = sorted(set(MATCH_FIELDS) - set(matches.columns))
    if missing:
        return {'errors': [f'Missing match fields: {missing}'], 'rows': len(matches)}
    m = matches
    dates = pd.to_datetime(m.date, errors='coerce', utc=True, format='ISO8601')
    if dates.isna().any(): errors.append('Invalid/missing match dates')
    if m[MATCH_FIELDS].isna().any().any(): errors.append('Missing required match values')
    if m.empty: errors.append('Empty match table')
    if m.competition.nunique()>1: errors.append('Fit one competition at a time')
    for field in ['match_id','home_id','away_id']:
        if m[field].astype(str).str.strip().eq('').any(): errors.append(f'Empty {field}')
    if m.match_id.duplicated().any(): errors.append('Duplicate match IDs')
    if m.duplicated(['date', 'home_id', 'away_id']).any(): errors.append('Duplicate fixtures')
    if (m.home_id == m.away_id).any(): errors.append('Home team equals away team')
    for field in ['home_goals', 'away_goals']:
        v = pd.to_numeric(m[field], errors='coerce')
        if (~np.isfinite(v) | (v < 0) | (v % 1 != 0)).any(): errors.append(f'Invalid {field}')
    if cutoff and (dates >= pd.Timestamp(cutoff, tz='UTC')).any(): errors.append('Matches at/after prediction cutoff')
    for row in m.itertuples():
        # Supported season labels: YYYY or YYYY/YYYY. Other conventions must be mapped on import.
        try:
            years = [int(x) for x in str(row.season).split('/')]
            year = pd.Timestamp(row.date).year
            if year not in years or len(years) not in (1, 2): errors.append('Inconsistent season assignment')
        except (ValueError, TypeError): errors.append('Invalid season label')
    if appearances is not None:
        a = appearances
        absent = sorted(set(APPEARANCE_FIELDS) - set(a.columns))
        if absent:
            errors.append(f'Missing appearance fields: {absent}')
        else:
            if a[APPEARANCE_FIELDS].isna().any().any(): errors.append('Missing required appearance values')
            for field in ['player_id','team_id','match_id']:
                if a[field].astype(str).str.strip().eq('').any(): errors.append(f'Empty appearance {field}')
            if not a.started.isin([True,False,0,1]).all(): errors.append('started must be boolean')
            if a.duplicated(['match_id', 'player_id']).any(): errors.append('Duplicate player-match records')
            mins = pd.to_numeric(a.minutes, errors='coerce')
            if (~np.isfinite(mins) | (mins < 0) | (mins > 90)).any(): errors.append('Minutes must be regulation-equivalent 0..90')
            joined = a.merge(m[['match_id', 'home_id', 'away_id']], on='match_id', how='left', validate='many_to_one') if not m.match_id.duplicated().any() else pd.DataFrame()
            if len(joined) and (~((joined.team_id == joined.home_id) | (joined.team_id == joined.away_id))).any():
                errors.append('Unmapped match/team or incorrect home/away appearance')
            if not set(a.match_id).issubset(set(m.match_id)): errors.append('Unmapped appearance match')
            totals = a.groupby(['match_id', 'team_id']).minutes.sum()
            if (totals > 990.01).any(): errors.append('Squad minutes exceed 11 players × 90')
            covered = m[m.match_id.isin(a.match_id)]
            expected = set(zip(covered.match_id, covered.home_id)) | set(zip(covered.match_id, covered.away_id))
            if set(totals.index) != expected: errors.append('Missing team appearance coverage')
            if len(covered)<len(m): warnings.append(f'{len(m)-len(covered)} matches lack appearances and cannot train player impact')
            if (totals < 720).any(): errors.append('Incomplete team minutes: below 720')
    return {'rows': len(m), 'appearance_rows': 0 if appearances is None else len(appearances),
            'missingness': m.isna().sum().to_dict(), 'duplicates': int(m.match_id.duplicated().sum()),
            'date_min': str(dates.min()), 'date_max': str(dates.max()),
            'seasons': sorted(m.season.astype(str).unique().tolist()), 'errors': sorted(set(errors)), 'warnings': warnings}


def require_valid(matches: pd.DataFrame, appearances: pd.DataFrame | None = None, cutoff: str | None = None) -> dict:
    report = validate(matches, appearances, cutoff)
    if report['errors']: raise ValueError('; '.join(report['errors']))
    return report


def validate_transfer_date(transfer_date: str, prediction_date: str) -> None:
    if pd.Timestamp(transfer_date) > pd.Timestamp(prediction_date):
        raise ValueError('Transfer observation is unavailable at prediction date')
