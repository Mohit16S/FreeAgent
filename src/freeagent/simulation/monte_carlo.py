"""Vectorised season draws with paired uniforms and auditable tie breaking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class SeasonResult:
    teams: list[str]
    points: np.ndarray
    goal_difference: np.ndarray
    goals_for: np.ndarray
    positions: np.ndarray

    def summary(self, qualification: int = 4, relegation: int = 3) -> pd.DataFrame:
        n = len(self.teams)
        if not 0<=qualification<=n or not 0<=relegation<n: raise ValueError('Invalid objective places')
        return pd.DataFrame({'team_id':self.teams,'expected_points':self.points.mean(axis=0),
                             'points_p05':np.quantile(self.points,.05,axis=0),'points_p95':np.quantile(self.points,.95,axis=0),
                             'title':(self.positions==1).mean(axis=0),
                             'qualification':(self.positions<=qualification).mean(axis=0),
                             'relegation':(self.positions>n-relegation).mean(axis=0)})


def simulate(model, fixtures: pd.DataFrame, n: int = 1000, seed: int = 42,
             shifts: dict | None = None, uniforms: np.ndarray | None = None,
             initial: pd.DataFrame | None = None, progress=None) -> SeasonResult:
    """Rank points, GD, goals scored, then shared seeded lots for exact ties.

    This is an explicitly approximate tie rule where a league requires head-to-head/playoffs.
    Initial standings require team_id, points, goal_difference, goals_for.
    """
    if n<1 or fixtures.empty: raise ValueError('Positive simulation count and nonempty fixtures required')
    if fixtures.match_id.duplicated().any(): raise ValueError('Duplicate fixture ID')
    if fixtures.duplicated(['date','home_id','away_id']).any(): raise ValueError('Duplicate fixture pairing/date')
    if (fixtures.home_id==fixtures.away_id).any(): raise ValueError('Self fixture')
    teams = model.teams
    index = {t:i for i,t in enumerate(teams)}
    rng = np.random.default_rng(seed)
    u = rng.random((len(fixtures),n)) if uniforms is None else np.asarray(uniforms)
    if u.shape != (len(fixtures),n) or not np.isfinite(u).all() or np.any((u<0)|(u>=1)): raise ValueError('Invalid paired uniforms')
    points,gd,gf = [np.zeros((n,len(teams)),dtype=int) for _ in range(3)]
    if initial is not None:
        if initial.team_id.duplicated().any() or set(initial.team_id)!=set(teams): raise ValueError('Initial standings must cover every team once')
        for r in initial.itertuples():
            i = index[r.team_id]
            if r.points<0 or r.goals_for<0: raise ValueError('Invalid initial standings')
            points[:,i],gd[:,i],gf[:,i] = r.points,r.goal_difference,r.goals_for
    for k,r in enumerate(fixtures.itertuples()):
        if r.home_id not in index or r.away_id not in index: raise ValueError('Fixture contains unseen team')
        matrix = model.matrix(r.home_id,r.away_id,shifts)
        if not np.isfinite(matrix).all() or (matrix<0).any() or not np.isclose(matrix.sum(),1): raise ValueError('Invalid score distribution')
        cdf = matrix.ravel().cumsum(); cdf[-1] = 1
        draw = np.searchsorted(cdf,u[k],side='right')
        h,a = np.divmod(draw,matrix.shape[1])
        hi,ai = index[r.home_id],index[r.away_id]
        points[:,hi] += 3*(h>a)+(h==a); points[:,ai] += 3*(a>h)+(h==a)
        gd[:,hi] += h-a; gd[:,ai] += a-h
        gf[:,hi] += h; gf[:,ai] += a
        if progress: progress((k+1)/len(fixtures))
    lots = np.random.default_rng(seed+1).random(points.shape)
    order = np.lexsort((lots,-gf,-gd,-points),axis=1)
    positions = np.empty_like(order)
    np.put_along_axis(positions,order,np.broadcast_to(np.arange(1,len(teams)+1),order.shape),axis=1)
    return SeasonResult(teams,points,gd,gf,positions)
