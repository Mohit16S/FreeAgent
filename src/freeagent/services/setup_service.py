"""The same resumable runtime bootstrap for a fresh cloud instance and the CLI."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import pandas as pd
import numpy as np
from freeagent.paths import RuntimePaths
from freeagent.data.ingestion import ingest_statsbomb, load_data
from freeagent.data.validation import validate, require_valid
from freeagent.services.analysis_service import train, save, load, verify_dataset
from freeagent.evaluation.backtesting import walk_forward

_SETUP_LOCK = threading.Lock()


def _write_status(paths: RuntimePaths, state: str, stage: str, message: str) -> None:
    paths.status.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.status.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'state':state,'stage':stage,'message':message,
                                     'updated_at':datetime.now(timezone.utc).isoformat()},indent=2))
    temporary.replace(paths.status)


def diagnostics(paths: RuntimePaths) -> dict:
    """Inspect files and validate readiness, never fabricate metrics or load a verdict."""
    result = {'data_status':'missing','model_status':'not trained','evaluation_status':'not calculated',
              'match_rows':0,'appearance_rows':0,'player_count':0,'errors':[], 'warnings':[]}
    if paths.status.exists():
        try: result['last_setup'] = json.loads(paths.status.read_text())
        except (ValueError,OSError): result['warnings'].append('Setup status file is unreadable')
    if not (paths.data / 'processed/matches.csv').exists():
        result['warnings'].append('This instance has no downloaded historical matches. Run setup below.')
        return result
    try:
        matches, appearances = load_data(paths.data)
        quality = validate(matches,appearances)
        result.update(match_rows=len(matches), appearance_rows=len(appearances),
                      player_count=int(appearances.player_id.nunique()) if 'player_id' in appearances else 0,
                      quality=quality, data_status='invalid' if quality['errors'] else 'validated')
        result['errors'].extend(quality['errors'])
        result['warnings'].extend(quality.get('warnings',[]))
        if appearances.empty:
            result['warnings'].append('Real player-level appearances are unavailable. Player impact, EPA and transfer verdicts are disabled; match scores alone are insufficient.')
        if paths.artifact.exists() and not quality['errors']:
            try:
                bundle=load(paths.artifact)
                verify_dataset(bundle,matches,appearances)
                result['model_status']='trained'
                result['model_metadata']={k:v for k,v in bundle.metadata.items() if k!='roster_inferred_last_appearance'}
            except (ValueError, OSError, KeyError, EOFError, ImportError):
                result['model_status']='unavailable or stale'
                result['warnings'].append('Saved model is incompatible, incomplete or stale. Run setup to retrain.')
        expected_reports=['metrics.json','evaluation_metadata.json','folds.csv','calibration.csv','predictions.csv','failures.json']
        if (paths.reports/'metrics.json').exists():
            try:
                import hashlib
                if not all((paths.reports/name).exists() for name in expected_reports):
                    raise ValueError('Incomplete evaluation export')
                meta=json.loads((paths.reports/'evaluation_metadata.json').read_text())
                metrics=json.loads((paths.reports/'metrics.json').read_text())
                for model in ['dixon_coles','poisson','frequency']:
                    if not all(np.isfinite(metrics[model][key]) for key in ['log_loss','rps','brier']):
                        raise ValueError('Invalid evaluation metrics')
                if meta.get('data_sha256')!=hashlib.sha256(matches.to_csv(index=False).encode()).hexdigest():
                    raise ValueError('Evaluation belongs to different data')
                result['evaluation_status']='calculated'
            except (ValueError,OSError,KeyError,TypeError):
                result['warnings'].append('Saved evaluation is incomplete, invalid or belongs to different data and will not be displayed. Rerun setup with evaluation enabled.')
    except (ValueError,OSError,KeyError,TypeError) as exc:
        result['data_status']='invalid'
        result['errors'].append(f'Could not validate stored data: {exc}')
    return result


def setup(paths: RuntimePaths, config: dict, *, competition: int = 2, season: int = 27,
          cutoff: str = '2016-02-01', evaluate: bool = False,
          cutoffs: list[str] | None = None, progress=None) -> dict:
    """Fetch and validate real matches/lineups, then fit all required models.

    Safe on rerun: valid downloads are cached and compatible fitted models are reused.
    Evaluation is optional and separately statused; failure does not masquerade as success.
    """
    if not _SETUP_LOCK.acquire(blocking=False):
        raise ValueError('Setup is already running in another session. Wait, then refresh diagnostics.')
    stage='ingestion'
    def announce(fraction, message):
        _write_status(paths,'running',stage,message)
        if progress: progress(fraction,message)
    try:
        announce(0.,'Downloading real StatsBomb matches and player lineups')
        quality=ingest_statsbomb(paths.data,competition,season,
                               progress=lambda fraction,message: announce(.6*fraction,message))
        stage='validation'; announce(.62,'Validating match and player-minute data')
        matches,appearances=load_data(paths.data)
        require_valid(matches,appearances)
        historical=matches[pd.to_datetime(matches.date)<pd.Timestamp(cutoff)]
        covered=historical[historical.match_id.isin(appearances.match_id)]
        if len(historical)<20:
            raise ValueError(f'Only {len(historical)} matches precede {cutoff}; at least 20 are required. Choose a later cutoff or a larger dataset.')
        if len(covered)<20 or covered.date.nunique()<2:
            raise ValueError(f'Only {len(covered)} historical matches have compatible player appearances. At least 20 across multiple dates are required. Match data were saved, but player impact/EPA/verdicts cannot be trained. Retry missing lineups or import the documented player CSV contract.')
        stage='training'; announce(.65,'Fitting player impact and score models with bootstrap uncertainty')
        bundle=None
        if paths.artifact.exists():
            try:
                candidate=load(paths.artifact); verify_dataset(candidate,matches,appearances)
                if (candidate.metadata['cutoff']==cutoff and candidate.metadata['seed']==config['seed']
                    and candidate.metadata['bootstrap_draws']==config['bootstrap_draws']
                    and candidate.impact.min_minutes==config['min_player_minutes']
                    and tuple(candidate.impact.alphas)==tuple(config['ridge_alphas'])
                    and candidate.match.decay_days==config['match_decay_days']):
                    bundle=candidate
            except (ValueError,OSError,KeyError,EOFError,ImportError): pass
        reused=bundle is not None
        if bundle is None:
            bundle=train(matches,appearances,cutoff,config['bootstrap_draws'],config['seed'],
                         config['min_player_minutes'],config['match_decay_days'],tuple(config['ridge_alphas']))
            save(bundle,paths.artifact)
            bundle.impact.report().to_csv(paths.artifact.with_name('player_impact.csv'),index=False)
        if evaluate:
            stage='evaluation'; announce(.85,'Running chronological held-out evaluation; no stored example metrics are used')
            walk_forward(matches,appearances,cutoffs or ['2015-12-01','2016-02-01','2016-04-01','2016-06-01'],paths.reports)
        stage='ready'; announce(1.,'Real data validated and player/score models trained')
        _write_status(paths,'complete',stage,'Transfer research pipeline ready; financial inputs remain user assumptions.')
        return {'quality':quality,'models_reused':reused,'training':bundle.metadata,
                'evaluation_requested':evaluate,'diagnostics':diagnostics(paths)}
    except Exception as exc:
        _write_status(paths,'failed',stage,str(exc) if isinstance(exc,(ValueError,OSError,RuntimeError)) else 'Unexpected setup failure; inspect server logs.')
        raise
    finally:
        _SETUP_LOCK.release()
