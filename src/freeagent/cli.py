"""Reproducible ingestion, training, evaluation and transfer scenario commands."""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from freeagent.paths import RuntimePaths
from freeagent.services.setup_service import setup, diagnostics
from pathlib import Path
import numpy as np
import pandas as pd
from freeagent.config import load_config
from freeagent.data.ingestion import ingest_statsbomb,load_data
from freeagent.data.validation import require_valid
from freeagent.services.analysis_service import train,save,load,evaluate,standings,verify_dataset
from freeagent.evaluation.backtesting import walk_forward
from freeagent.valuation.displacement import Displacement
from freeagent.valuation.net_value import Costs


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,pd.DataFrame): return value.reset_index().to_dict('records')
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def run():
    parser = argparse.ArgumentParser(description='FreeAgent — Buying Points, Not Names')
    parser.add_argument('--data',type=Path,default=RuntimePaths.discover().data)
    parser.add_argument('--config',type=Path,default=Path(os.getenv('FREEAGENT_CONFIG','config/default.yaml')))
    sub = parser.add_subparsers(dest='command',required=True)
    ing = sub.add_parser('ingest'); ing.add_argument('--competition',type=int,default=2); ing.add_argument('--season',type=int,default=27); ing.add_argument('--limit',type=int)
    sub.add_parser('validate')
    sub.add_parser('status')
    boot = sub.add_parser('setup')
    boot.add_argument('--competition',type=int,default=2)
    boot.add_argument('--season',type=int,default=27)
    boot.add_argument('--cutoff',default='2016-02-01')
    boot.add_argument('--evaluate',action='store_true')
    boot.add_argument('--cutoffs',nargs='+')
    tr = sub.add_parser('train'); tr.add_argument('--cutoff',required=True); tr.add_argument('--artifact',type=Path,default=RuntimePaths.discover().artifact); tr.add_argument('--bootstrap',type=int)
    bt = sub.add_parser('backtest'); bt.add_argument('--cutoffs',nargs='+',required=True); bt.add_argument('--out',type=Path,default=RuntimePaths.discover().reports)
    ev = sub.add_parser('evaluate'); ev.add_argument('--scenario',type=Path,required=True); ev.add_argument('--artifact',type=Path,default=RuntimePaths.discover().artifact); ev.add_argument('--out',type=Path,default=Path('reports/scenario.json'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    cfg = load_config(args.config)
    if args.command in ('setup','status'):
        paths=RuntimePaths.discover()
        if args.data.resolve()!=paths.data.resolve():
            parser.error('Use FREEAGENT_HOME for setup/status runtime storage; --data applies to individual pipeline commands.')
        result=diagnostics(paths) if args.command=='status' else setup(paths,cfg,competition=args.competition,season=args.season,cutoff=args.cutoff,evaluate=args.evaluate,cutoffs=args.cutoffs,progress=lambda fraction,message: logging.info('%3d%% %s',round(fraction*100),message))
    elif args.command=='ingest': result = ingest_statsbomb(args.data,args.competition,args.season,args.limit)
    else:
        m,a = load_data(args.data)
        if args.command=='validate': result = require_valid(m,a)
        elif args.command=='train':
            bundle = train(m,a,args.cutoff,args.bootstrap or cfg['bootstrap_draws'],cfg['seed'],cfg['min_player_minutes'],cfg['match_decay_days'],tuple(cfg['ridge_alphas']))
            save(bundle,args.artifact)
            bundle.impact.report().to_csv(args.artifact.with_name('player_impact.csv'),index=False)
            result = bundle.metadata
        elif args.command=='backtest': result = walk_forward(m,a,args.cutoffs,args.out)
        else:
            s = json.loads(args.scenario.read_text()); bundle = load(args.artifact)
            verify_dataset(bundle,m,a)
            if not s.get('price_per_point_basis','').strip(): raise ValueError('A price_per_point_basis is required')
            selected = m[m.season==s['season']]
            future = selected[pd.to_datetime(selected.date)>=pd.Timestamp(bundle.metadata['cutoff'])]
            past = selected[pd.to_datetime(selected.date)<pd.Timestamp(bundle.metadata['cutoff'])]
            fixture_cols = ['match_id','date','home_id','away_id']
            nf = int(((future.home_id==s['club'])|(future.away_id==s['club'])).sum())
            result = evaluate(bundle,future[fixture_cols],s['club'],Displacement(s['incoming'],s['incumbent'],s['minutes'],s['incumbent_minutes'],nf),
                              Costs(**s['costs']),s['price_per_point'],n=s.get('simulations',cfg['simulations']),seed=cfg['seed'],
                              qualification=cfg['qualification_places'],relegation=cfg['relegation_places'],
                              initial=standings(past,bundle.match.teams),availability=tuple(s.get('availability',[1.,1.])),
                              tolerance=cfg['financial_tolerance'],threshold=cfg['verdict_probability'])
            result['scenario_inputs'] = s
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(result,default=serializable,indent=2))
            result = {'epa_mean':result['epa_mean'],'net_value':result['finance']['mean'],'verdict':result['finance']['verdict'],'report':str(args.out)}
    print(json.dumps(result,indent=2,default=serializable))

def main():
    try:
        run()
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as exc:
        print(f'FreeAgent: {exc}',file=sys.stderr)
        raise SystemExit(1) from None


if __name__=='__main__': main()
