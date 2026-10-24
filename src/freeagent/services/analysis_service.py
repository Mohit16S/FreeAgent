"""Model orchestration, versioned artifacts and uncertainty-aware transfer analysis."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform
import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
import sklearn
from freeagent import __version__
from freeagent.data.validation import require_valid
from freeagent.impact.ridge_model import ImpactModel
from freeagent.match.dixon_coles import DixonColes, outcomes
from freeagent.simulation.monte_carlo import simulate
from freeagent.valuation.displacement import Displacement
from freeagent.valuation.net_value import Costs, net_value

@dataclass
class Bundle:
    impact: ImpactModel
    match: DixonColes
    match_draws: list
    metadata: dict


def dataset_hash(matches: pd.DataFrame, appearances: pd.DataFrame) -> str:
    return hashlib.sha256((matches.to_csv(index=False)+appearances.to_csv(index=False)).encode()).hexdigest()


def train(matches: pd.DataFrame, appearances: pd.DataFrame, cutoff: str,
          bootstrap: int = 20, seed: int = 42, min_minutes: float = 270.,
          decay_days: float = 365., alphas: tuple = (1.,10.,100.)) -> Bundle:
    require_valid(matches,appearances)
    m = matches[pd.to_datetime(matches.date)<pd.Timestamp(cutoff)].copy()
    a = appearances[appearances.match_id.isin(m.match_id)].copy()
    require_valid(m,a,cutoff)
    covered=m[m.match_id.isin(a.match_id)]
    if len(covered)<20:
        raise ValueError(f'Player-impact training requires at least 20 pre-cutoff matches with real player appearances; found {len(covered)}. Match results alone cannot produce EPA or a transfer verdict.')
    if bootstrap<2: raise ValueError('At least two bootstrap draws required')
    impact = ImpactModel(alphas=alphas,min_minutes=min_minutes).fit(m,a,cutoff,bootstrap,seed)
    match = DixonColes(decay_days=decay_days).fit(m,cutoff)
    rng = np.random.default_rng(seed)
    draws = []
    # Conditional parametric bootstrap: draw fitted joint scores on the training schedule.
    # Separately resampled Ridge and score models approximate marginal parameter uncertainty.
    for _ in range(bootstrap):
        sample = m.copy()
        hg,ag = [],[]
        for r in m.itertuples():
            matrix = match.matrix(r.home_id,r.away_id)
            code = rng.choice(matrix.size,p=matrix.ravel())
            h,g = divmod(code,matrix.shape[1]); hg.append(h); ag.append(g)
        sample['home_goals'],sample['away_goals'] = hg,ag
        draws.append(DixonColes(decay_days=decay_days).fit(sample,cutoff))
    recent = a.merge(m[['match_id','date']],on='match_id').sort_values(['date','match_id']).drop_duplicates('player_id',keep='last')
    roster = recent.set_index('player_id').team_id.to_dict()
    meta = {'trained_at':datetime.now(timezone.utc).isoformat(),'roster_inferred_last_appearance':roster,'version':__version__,'python':platform.python_version(),'sklearn':sklearn.__version__,
            'cutoff':cutoff,'training_max_date':str(m.date.max()),'training_rows':len(m),
            'dataset_sha256':dataset_hash(m,a),'seed':seed,'bootstrap_draws':bootstrap,
            'source':sorted(m.source.unique().tolist()) if 'source' in m else ['User CSV: provenance must be supplied'],
            'uncertainty':'Match bootstrap + parametric score bootstrap; independent model resampling approximation; uncalibrated.'}
    return Bundle(impact,match,draws,meta)


def save(bundle: Bundle, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.joblib.tmp')
    joblib.dump(bundle,temporary)
    temporary.replace(path)
    metadata=path.with_suffix('.json.tmp')
    metadata.write_text(json.dumps(bundle.metadata,indent=2))
    metadata.replace(path.with_suffix('.json'))


def load(path: Path) -> Bundle:
    """Load only trusted local artifacts; joblib is not safe for untrusted uploads."""
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['version']!=__version__ or meta['sklearn']!=sklearn.__version__:
        raise ValueError('Artifact version mismatch; retrain with current environment')
    try:
        bundle=joblib.load(path)
        if not isinstance(bundle,Bundle) or not bundle.match_draws or not hasattr(bundle.impact,'model'):
            raise ValueError('Artifact does not contain a trained FreeAgent model bundle; run setup')
        return bundle
    except (pickle.UnpicklingError,EOFError,AttributeError,IndexError,KeyError,TypeError) as exc:
        raise ValueError('Saved model artifact is corrupt or incompatible; run setup to retrain') from exc


def evaluate(bundle: Bundle, fixtures: pd.DataFrame, club: str, displacement: Displacement,
             costs: Costs, price_per_point: float, n: int = 1000, seed: int = 42,
             qualification: int = 4, relegation: int = 3, initial: pd.DataFrame | None = None,
             availability: tuple[float,float] = (1.,1.), tolerance: float = 1e6,
             threshold: float = .75) -> dict:
    displacement.validate()
    if club not in bundle.match.teams: raise ValueError('Club absent from model')
    if (pd.to_datetime(fixtures.date)<pd.Timestamp(bundle.metadata['cutoff'])).any():
        raise ValueError('Counterfactual fixtures precede model cutoff')
    roster = bundle.metadata.get('roster_inferred_last_appearance',{})
    if roster and roster.get(displacement.incumbent)!=club: raise ValueError('Incumbent was not last observed at destination club')
    if roster and roster.get(displacement.incoming)==club: raise ValueError('Incoming player is already in the inferred destination squad')
    club_fixtures = int(((fixtures.home_id==club)|(fixtures.away_id==club)).sum())
    if displacement.fixtures!=club_fixtures: raise ValueError('Minutes horizon must match club fixtures')
    if not 0<=availability[0]<=availability[1]<=1: raise ValueError('Availability range must lie in [0,1]')
    if n<20: raise ValueError('At least 20 simulations per parameter draw required')
    if costs.horizon_years<=0: raise ValueError('Financial horizon required')
    diff = bundle.impact.difference_samples(displacement.incoming,displacement.incumbent)
    audit = displacement.audit()
    rng = np.random.default_rng(seed)
    expected_epa, realised_epa, before_points,after_points = [],[],[],[]
    objective_before,objective_after = [],[]
    shifts,minutes_samples, floor_count = [],[],0
    simulation_epa = []
    for k,d in enumerate(diff):
        model = bundle.match_draws[k%len(bundle.match_draws)]
        minutes = displacement.expected_minutes*rng.uniform(*availability)
        shift = float(d*minutes/(90*club_fixtures))
        for r in fixtures.itertuples():
            if club not in (r.home_id,r.away_id): continue
            h,a = model.rates(r.home_id,r.away_id)
            delta = shift if r.home_id==club else -shift
            floor_count += int(min(h+delta/2,a-delta/2)<.05)
        u = rng.random((len(fixtures),n))
        before = simulate(model,fixtures,n,seed,uniforms=u,initial=initial)
        after = simulate(model,fixtures,n,seed,shifts={club:shift},uniforms=u,initial=initial)
        ix = before.teams.index(club)
        delta = after.points[:,ix]-before.points[:,ix]
        # Integrate expected points exactly to remove inner Monte Carlo noise from EPA.
        analytic = 0.
        for fixture in fixtures.itertuples():
            if club not in (fixture.home_id,fixture.away_id): continue
            before_p = outcomes(model.matrix(fixture.home_id,fixture.away_id))
            after_p = outcomes(model.matrix(fixture.home_id,fixture.away_id,{club:shift}))
            win = 0 if fixture.home_id==club else 2
            analytic += 3*(after_p[win]-before_p[win])+(after_p[1]-before_p[1])
        expected_epa.append(analytic); simulation_epa.append(float(delta.mean())); realised_epa.extend(delta)
        before_points.extend(before.points[:,ix]); after_points.extend(after.points[:,ix])
        objective_before.append(before.summary(qualification,relegation).set_index('team_id').loc[club].to_dict())
        objective_after.append(after.summary(qualification,relegation).set_index('team_id').loc[club].to_dict())
        shifts.append(shift); minutes_samples.append(minutes)
    b,a = pd.DataFrame(objective_before).mean(),pd.DataFrame(objective_after).mean()
    finance = net_value(np.asarray(expected_epa),costs,price_per_point,tolerance,threshold)
    predictive_finance = net_value(np.asarray(realised_epa),costs,price_per_point,tolerance,threshold)
    return {'epa_samples':np.asarray(expected_epa),'realised_points_delta_samples':np.asarray(realised_epa),
            'simulation_epa_mean':float(np.mean(simulation_epa)), 'simulation_epa_by_draw':simulation_epa,
            'epa_mean':float(np.mean(expected_epa)),'epa_interval':np.quantile(expected_epa,[.05,.95]).tolist(),
            'before_points':np.asarray(before_points),'after_points':np.asarray(after_points),
            'objectives':pd.DataFrame({'without':b,'with':a,'change':a-b}),
            'finance':finance,'predictive_finance':predictive_finance,'displacement':audit,
            'goal_difference_shifts':shifts,'minutes_samples':minutes_samples,
            'rate_floor_count':floor_count,'metadata':bundle.metadata,
            'warnings':['Counterfactual impact-to-rate mapping is an unvalidated assumption.',
                        'Intervals are bootstrap/scenario ranges, not calibrated confidence intervals.',
                        'Paired realised differences depend on the common-random-number coupling.',
                        'Rank ties use points/GD/goals then seeded lots; league-specific head-to-head rules are unsupported.']}


def standings(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    rows = {t:{'team_id':t,'points':0,'goal_difference':0,'goals_for':0} for t in teams}
    for r in matches.itertuples():
        for t,g,o in [(r.home_id,r.home_goals,r.away_goals),(r.away_id,r.away_goals,r.home_goals)]:
            if t not in rows: raise ValueError('Standings contain unseen team')
            rows[t]['points'] += 3 if g>o else 1 if g==o else 0
            rows[t]['goal_difference'] += g-o; rows[t]['goals_for'] += g
    return pd.DataFrame(rows.values())


def verify_dataset(bundle: Bundle, matches: pd.DataFrame, appearances: pd.DataFrame) -> None:
    """Reject stale artifacts if the training inputs changed after fitting."""
    m=matches[pd.to_datetime(matches.date)<pd.Timestamp(bundle.metadata['cutoff'])]
    a=appearances[appearances.match_id.isin(m.match_id)]
    if dataset_hash(m,a)!=bundle.metadata['dataset_sha256']:
        raise ValueError('Historical training data changed; retrain the saved model before analysis')
