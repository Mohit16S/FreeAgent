"""Expanding chronological forecasts; no realised transfer effects used as causal labels."""
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib
import plotly.express as px
import numpy as np
import pandas as pd
from freeagent import __version__
from freeagent.data.validation import require_valid, validate_transfer_date
from freeagent.impact.ridge_model import ImpactModel
from freeagent.match.dixon_coles import DixonColes, outcomes
from freeagent.match.poisson import PoissonModel
from .metrics import probability_metrics, calibration


def walk_forward(matches: pd.DataFrame, appearances: pd.DataFrame, cutoffs: list[str], out: Path) -> dict:
    require_valid(matches,appearances)
    if cutoffs != sorted(set(cutoffs)) or len(cutoffs)<2: raise ValueError('Two or more increasing fold boundaries required')
    rows,folds,failures = [],[],[]
    for start,end in zip(cutoffs[:-1],cutoffs[1:]):
        train = matches[pd.to_datetime(matches.date)<pd.Timestamp(start)]
        test = matches[(pd.to_datetime(matches.date)>=pd.Timestamp(start))&(pd.to_datetime(matches.date)<pd.Timestamp(end))]
        a = appearances[appearances.match_id.isin(train.match_id)]
        try:
            if test.empty: raise ValueError('Empty evaluation fold')
            dc = DixonColes().fit(train,start)
            poisson = PoissonModel().fit(train,start)
            ridge = ImpactModel().fit(train,a,start,bootstrap=10)
            known = test.home_id.isin(dc.teams)&test.away_id.isin(dc.teams)
            for r in test[~known].itertuples(): failures.append({'cutoff':start,'match_id':r.match_id,'reason':'Unseen team'})
            test = test[known]
            if test.empty: raise ValueError('No evaluable fixtures')
            ta = appearances[appearances.match_id.isin(test.match_id)]
            impact_predictions = ridge.predict(test,ta)
            labels_train = np.where(train.home_goals>train.away_goals,0,np.where(train.home_goals==train.away_goals,1,2))
            frequency = (np.bincount(labels_train,minlength=3)+1)/(len(train)+3)
            for i,r in enumerate(test.itertuples()):
                p = outcomes(dc.matrix(r.home_id,r.away_id)); baseline = outcomes(poisson.matrix(r.home_id,r.away_id))
                label = 0 if r.home_goals>r.away_goals else 1 if r.home_goals==r.away_goals else 2
                has_minutes = r.match_id in set(ta.match_id)
                rows.append({'conditional_minutes_complete':has_minutes,'match_id':r.match_id,'date':r.date,'cutoff':start,'training_max_date':str(train.date.max()),
                             'version':__version__,'label':label,'home':p[0],'draw':p[1],'away':p[2],
                             'poisson_home':baseline[0],'poisson_draw':baseline[1],'poisson_away':baseline[2],
                             'frequency_home':frequency[0],'frequency_draw':frequency[1],'frequency_away':frequency[2],
                             'predicted_gd_conditional_minutes':float(impact_predictions[i]) if has_minutes else np.nan,'observed_gd':r.home_goals-r.away_goals,
                             'team_only_gd':sum(dc.rates(r.home_id,r.away_id)[j]*(1 if j==0 else -1) for j in range(2))})
            fold = pd.DataFrame([r for r in rows if r['cutoff']==start])
            folds.append({'cutoff':start,'end':end,'train_n':len(train),'test_n':len(fold),
                          **probability_metrics(fold[['home','draw','away']],fold.label)})
        except (ValueError,RuntimeError) as exc: failures.append({'cutoff':start,'reason':str(exc)})
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(folds).to_csv(out/'folds.csv',index=False)
    (out/'failures.json').write_text(json.dumps(failures,indent=2))
    if not rows: raise ValueError('No successful folds; inspect failures.json')
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out/'predictions.csv',index=False)
    metrics = {}
    for name,cols in [('dixon_coles',['home','draw','away']),('poisson',['poisson_home','poisson_draw','poisson_away']),('frequency',['frequency_home','frequency_draw','frequency_away'])]:
        metrics[name] = probability_metrics(predictions[cols],predictions.label)
    errors = predictions.predicted_gd_conditional_minutes-predictions.observed_gd
    metrics['impact_conditional_on_observed_minutes'] = {'matches_with_valid_appearances':int(errors.notna().sum()),'mae':float(abs(errors).mean()),'rmse':float(np.sqrt((errors**2).mean())),
        'note':'Held-out outcome reconstruction given realised player minutes; NOT a pre-match forecast or causal test.'}
    metrics['team_only_gd'] = {'mae':float(abs(predictions.team_only_gd-predictions.observed_gd).mean())}
    metrics['transfer_validation'] = 'Unavailable: no defensible causal EPA or financial truth labels supplied.'
    metrics['rows'] = len(predictions)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    reliability=calibration(predictions[['home','draw','away']],predictions.label)
    reliability.to_csv(out/'calibration.csv',index=False)
    fig=px.scatter(reliability,x='predicted',y='observed',color='outcome',size='count',range_x=[0,1],range_y=[0,1],title='Real historical out-of-sample match calibration')
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash'))
    fig.write_html(out/'calibration.html',include_plotlyjs=True)
    px.scatter(predictions,x='predicted_gd_conditional_minutes',y='observed_gd',color='cutoff',title='Held-out goal difference conditional on observed minutes').write_html(out/'prediction_vs_outcome.html',include_plotlyjs=True)
    probabilities=predictions[['home','draw','away']].to_numpy()
    predictions['match_log_loss']=-np.log(np.maximum(probabilities[np.arange(len(predictions)),predictions.label],1e-15))
    predictions.sort_values('match_log_loss',ascending=False).to_csv(out/'error_analysis.csv',index=False)
    metadata={'run_at':datetime.now(timezone.utc).isoformat(),'version':__version__,'cutoffs':cutoffs,'data_sha256':hashlib.sha256(matches.to_csv(index=False).encode()).hexdigest(),'source':sorted(matches.source.unique()) if 'source' in matches else ['User CSV'],'failed_runs':len(failures)}
    (out/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2))
    return metrics


def historical_transfer_runs(transfers: pd.DataFrame, runner, output: Path) -> None:
    """Run a cutoff-aware callback per historical transfer and log failures.

    Required fields prediction_date and transfer_date refer to information available then.
    The runner must train with the passed cutoff; future outcomes are never passed to it.
    """
    rows = []
    for row in transfers.sort_values('prediction_date').to_dict('records'):
        try:
            validate_transfer_date(row['transfer_date'],row['prediction_date'])
            assumptions = {k:v for k,v in row.items() if not k.startswith('realised_')}
            prediction = runner(assumptions,cutoff=row['prediction_date'])
            rows.append({'status':'ok','cutoff':row['prediction_date'],'prediction':prediction})
        except (ValueError,RuntimeError) as exc: rows.append({'status':'failed','cutoff':row['prediction_date'],'reason':str(exc)})
    output.write_text(json.dumps(rows,indent=2,default=str))
