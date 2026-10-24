"""Reproducible player involvement and team controls, using only training identities."""
from __future__ import annotations
import numpy as np
import pandas as pd


def design(matches: pd.DataFrame, appearances: pd.DataFrame, players: list[str], teams: list[str]) -> np.ndarray:
    """Signed player minutes/90; signed team effects control opponent/team quality.

    Collinearity is intentional and regularised. Team/player attribution is weakly identified.
    """
    pidx, tidx = {p:i for i,p in enumerate(players)}, {t:i for i,t in enumerate(teams)}
    x = np.zeros((len(matches), len(players) + len(teams)))
    by_match = {k: v for k,v in appearances.groupby('match_id')}
    for i,r in enumerate(matches.itertuples()):
        for t, sign in [(r.home_id,1),(r.away_id,-1)]:
            if t in tidx: x[i,len(players)+tidx[t]] = sign
        for a in by_match.get(r.match_id, pd.DataFrame()).itertuples():
            if a.player_id in pidx:
                x[i,pidx[a.player_id]] = (1 if a.team_id == r.home_id else -1) * a.minutes / 90
    return x


def earlier_team_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Pre-date average goal difference, with simultaneous dates updated together."""
    totals, counts, rows = {}, {}, []
    for date, group in matches.sort_values('date').groupby('date', sort=True):
        for r in group.itertuples():
            rows.append({'match_id':r.match_id,'home_form':totals.get(r.home_id,0)/max(1,counts.get(r.home_id,0)),
                         'away_form':totals.get(r.away_id,0)/max(1,counts.get(r.away_id,0))})
        for r in group.itertuples():
            for t,g in [(r.home_id,r.home_goals-r.away_goals),(r.away_id,r.away_goals-r.home_goals)]:
                totals[t] = totals.get(t,0)+g
                counts[t] = counts.get(t,0)+1
    return pd.DataFrame(rows)
