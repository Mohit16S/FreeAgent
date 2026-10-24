"""Identifiable Poisson attack/defence model with fitted Dixon–Coles correction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson
from freeagent.data.validation import require_valid


def tau(x, y, lh, la, rho):
    return np.where((x==0)&(y==0),1-lh*la*rho,
           np.where((x==0)&(y==1),1+lh*rho,
           np.where((x==1)&(y==0),1+la*rho,
           np.where((x==1)&(y==1),1-rho,1.))))


def score_matrix(lh: float, la: float, rho: float = 0., max_score: int = 12) -> np.ndarray:
    if not np.isfinite([lh,la,rho]).all() or min(lh,la)<=0 or max_score<2:
        raise ValueError('Finite positive goal rates and maximum score >=2 required')
    g = np.arange(max_score+1)
    mat = np.outer(poisson.pmf(g,lh),poisson.pmf(g,la))*tau(g[:,None],g[None,:],lh,la,rho)
    if np.any(mat<0): raise ValueError('Dixon–Coles correction produced negative probability')
    mass = mat.sum()
    if mass < .999: raise ValueError('Score truncation loses >0.1% probability; increase max_score')
    return mat/mass


def outcomes(matrix: np.ndarray) -> np.ndarray:
    """Ordered home win, draw, away win; home goals are rows."""
    return np.array([np.tril(matrix,-1).sum(),np.trace(matrix),np.triu(matrix,1).sum()])

@dataclass
class DixonColes:
    decay_days: float = 365.
    correction: bool = True
    penalty: float = .1

    def fit(self, matches: pd.DataFrame, cutoff: str) -> 'DixonColes':
        require_valid(matches, cutoff=cutoff)
        if len(matches)<20: raise ValueError('At least 20 matches required')
        self.teams = sorted(set(matches.home_id)|set(matches.away_id))
        self.index = {t:i for i,t in enumerate(self.teams)}
        n = len(self.teams)
        if n<2: raise ValueError('At least two teams required')
        h = matches.home_id.map(self.index).to_numpy()
        a = matches.away_id.map(self.index).to_numpy()
        gh, ga = matches.home_goals.to_numpy(), matches.away_goals.to_numpy()
        age = (pd.Timestamp(cutoff)-pd.to_datetime(matches.date)).dt.days.to_numpy()
        weights = np.exp(-np.log(2)*age/self.decay_days) if self.decay_days>0 else np.ones(len(matches))
        def unpack(v):
            attack = np.r_[v[:n-1], -sum(v[:n-1])]
            defence = np.r_[v[n-1:2*n-2], -sum(v[n-1:2*n-2])]
            return attack,defence,v[-3],v[-2],v[-1]
        def objective(v):
            attack,defence,intercept,home,rho = unpack(v)
            logh = intercept+home+attack[h]+defence[a]
            loga = intercept+attack[a]+defence[h]
            lh,la = np.exp(logh),np.exp(loga)
            correction = tau(gh,ga,lh,la,rho)
            # Enforce positivity for all potential team pairings, not just observed scores.
            allh = np.exp(intercept+home+attack[:,None]+defence[None,:])
            alla = np.exp(intercept+attack[None,:]+defence[:,None])
            margins = np.r_[(1-allh*alla*rho).ravel(),(1+allh*rho).ravel(),(1+alla*rho).ravel(),1-rho]
            if margins.min()<=0: return 1e12+float(np.minimum(margins,0)@np.minimum(margins,0))*1e8
            logp = gh*logh-lh-gammaln(gh+1)+ga*loga-la-gammaln(ga+1)+np.log(correction)
            return -float(weights@logp)+self.penalty*(attack@attack+defence@defence)
        initial = np.zeros(2*n+1)
        initial[-3] = np.log(max(.1,float(matches.away_goals.mean())))
        initial[-2] = np.log(max(.1,float(matches.home_goals.mean()))/max(.1,float(matches.away_goals.mean())))
        bounds = [(-1.5,1.5)]*(2*n-2)+[(-1,1.5),(-.8,.8),(-.2,.2) if self.correction else (0,0)]
        result = minimize(objective,initial,method='L-BFGS-B',bounds=bounds,options={'maxiter':1200,'ftol':1e-9})
        if not result.success or not np.isfinite(result.fun): raise RuntimeError(f'Match fitting failed: {result.message}')
        self.attack,self.defence,self.intercept,self.home,self.rho = unpack(result.x)
        self.cutoff = cutoff
        self.fit_info = {'converged':bool(result.success),'objective':float(result.fun),'iterations':result.nit,'matches':len(matches)}
        return self

    def rates(self, home: str, away: str, shifts: dict[str,float] | None = None) -> tuple[float,float]:
        if home not in self.index or away not in self.index: raise ValueError('Unseen team: retraining or an explicit prior is required')
        h,a = self.index[home],self.index[away]
        lh = np.exp(self.intercept+self.home+self.attack[h]+self.defence[a])
        la = np.exp(self.intercept+self.attack[a]+self.defence[h])
        # Net GD delta is split once: half added to own scoring, half subtracted from conceding.
        # An additive GD mapping in goals/match; positivity floor is audited by the service.
        shifts = shifts or {}
        delta = shifts.get(home,0)-shifts.get(away,0)
        return max(.05,float(lh+delta/2)),max(.05,float(la-delta/2))

    def matrix(self, home: str, away: str, shifts: dict[str,float] | None = None) -> np.ndarray:
        lh,la = self.rates(home,away,shifts)
        return score_matrix(lh,la,self.rho)

    def predict(self, home: str, away: str) -> dict:
        lh,la = self.rates(home,away)
        matrix = self.matrix(home,away)
        p = outcomes(matrix)
        return {'home_xg':lh,'away_xg':la,'probabilities':p,'score_matrix':matrix,
                'home_expected_points':3*p[0]+p[1],'away_expected_points':3*p[2]+p[1],
                'expected_goal_difference':lh-la}
