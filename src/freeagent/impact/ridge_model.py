"""Minutes-share adjusted plus-minus with chronological alpha selection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from freeagent.data.features import design
from freeagent.data.validation import require_valid

@dataclass
class ImpactModel:
    alphas: tuple = (1.,10.,100.)
    min_minutes: float = 270.

    def fit(self, matches: pd.DataFrame, appearances: pd.DataFrame, cutoff: str,
            bootstrap: int = 20, seed: int = 42) -> 'ImpactModel':
        require_valid(matches, appearances, cutoff)
        m = matches[matches.match_id.isin(appearances.match_id)].sort_values(['date','match_id']).reset_index(drop=True)
        if len(m)<20: raise ValueError('At least 20 matches required for impact fitting')
        dates = sorted(m.date.unique())
        if len(dates)<2: raise ValueError('Impact fitting requires matches on at least two dates for chronological validation')
        split = dates[max(1,int(len(dates)*.75))]
        train, valid = m[m.date < split], m[m.date >= split]
        a0 = appearances[appearances.match_id.isin(train.match_id)]
        p0 = sorted(a0.player_id.unique())
        t0 = sorted(set(train.home_id)|set(train.away_id))
        xt, xv = design(train,a0,p0,t0), design(valid,appearances,p0,t0)
        y0 = (train.home_goals-train.away_goals).to_numpy()
        yv = (valid.home_goals-valid.away_goals).to_numpy()
        scores = {alpha:float(np.mean((Ridge(alpha=alpha).fit(xt,y0).predict(xv)-yv)**2)) for alpha in self.alphas}
        self.alpha = min(scores,key=scores.get)
        self.validation_mse = scores
        self.players = sorted(appearances.player_id.unique())
        self.teams = sorted(set(m.home_id)|set(m.away_id))
        x = design(m,appearances,self.players,self.teams)
        y = (m.home_goals-m.away_goals).to_numpy()
        self.model = Ridge(alpha=self.alpha).fit(x,y)
        rng = np.random.default_rng(seed)
        # Match-level resampling preserves both teams' player involvements together.
        samples = []
        for _ in range(bootstrap):
            ids = rng.integers(0,len(m),len(m))
            samples.append(Ridge(alpha=self.alpha).fit(x[ids],y[ids]).coef_[:len(self.players)])
        self.samples = np.array(samples) if samples else self.model.coef_[None,:len(self.players)]
        self.cutoff = cutoff
        self.minutes = appearances.groupby('player_id').minutes.sum().to_dict()
        self.names = appearances.drop_duplicates('player_id').set_index('player_id').player_name.to_dict()
        self.positions = appearances.groupby('player_id').position.agg(lambda s:s.mode().iloc[0]).to_dict()
        self.diagnostics = {'matches':len(m),'features':x.shape[1], 'rank':int(np.linalg.matrix_rank(x)),
                            'warning':'Coefficients are regularised associations; correlated teammates are weakly identified.'}
        return self

    def predict(self, matches: pd.DataFrame, appearances: pd.DataFrame) -> np.ndarray:
        return self.model.predict(design(matches,appearances,self.players,self.teams))

    def report(self) -> pd.DataFrame:
        lo, hi = np.quantile(self.samples,[.05,.95],axis=0)
        return pd.DataFrame({'player_id':self.players,'player':[self.names[p] for p in self.players],
                             'position':[self.positions[p] for p in self.players],
                             'impact':self.model.coef_[:len(self.players)],'lower':lo,'upper':hi,
                             'minutes':[self.minutes[p] for p in self.players],
                             'eligible':[self.minutes[p]>=self.min_minutes for p in self.players]})

    def difference_samples(self, incoming: str, incumbent: str) -> np.ndarray:
        for p in (incoming,incumbent):
            if p not in self.players or self.minutes[p]<self.min_minutes:
                raise ValueError(f'Player {p} lacks minimum historical minutes')
        return self.samples[:,self.players.index(incoming)]-self.samples[:,self.players.index(incumbent)]
