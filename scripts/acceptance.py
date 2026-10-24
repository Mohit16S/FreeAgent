"""Verify saved real-data artifacts without inventing transfer truth labels."""
from pathlib import Path
import json
import numpy as np
from freeagent.data.ingestion import load_data
from freeagent.data.validation import require_valid
from freeagent.services.analysis_service import load,verify_dataset

m,a=load_data(Path('data'))
b=load(Path('models/artifacts/core.joblib'))
quality=require_valid(m,a)
verify_dataset(b,m,a)
r=json.loads(Path('reports/scenario.json').read_text())
metrics=json.loads(Path('reports/backtest/metrics.json').read_text())
failures=json.loads(Path('reports/backtest/failures.json').read_text())
teams=set(m.home_id)|set(m.away_id)
expected={(h,v) for h in teams for v in teams if h!=v}
actual=set(zip(m.home_id,m.away_id))
paired=np.asarray(r['realised_points_delta_samples']).reshape(len(r['epa_samples']),-1)
mc_se=float(np.sqrt((paired.var(axis=1,ddof=1)/paired.shape[1]).sum())/len(paired))
checks={
 'documented_real_source':set(m.source)=={'StatsBomb Open Data'},
 'validated_data':not quality['errors'],
 'complete_double_round_robin':actual==expected and len(m)==len(expected),
 'trained_player_model':len(b.impact.players)>0,
 'converged_match_model':b.match.fit_info['converged'],
 'displacement_conserved':r['displacement']['total_before']==r['displacement']['total_after'],
 'finite_epa_and_net_value':bool(np.isfinite(r['epa_mean']) and np.isfinite(r['finance']['mean'])),
 'valid_verdict_probabilities':bool(np.isclose(sum(r['finance']['category_probabilities'].values()),1)),
 'paired_epa_agrees_with_exact_expectation':abs(r['epa_mean']-r['simulation_epa_mean'])<max(.03,4*mc_se),
 'walk_forward_completed':metrics['rows']>0 and not failures,
}
report={'checks':checks,'software_core_passed':all(checks.values()),'match_rows':len(m),'appearance_rows':len(a),
        'epa_mc_standard_error':mc_se,'scope':'Real historical football inputs; hypothetical financial/minute scenario. No causal or financial validation implied.',
        'research_blockers':['Transfer-level labels and contracts unavailable','Impact-to-rate mapping and uncertainty calibration not externally validated']}
Path('reports/acceptance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not report['software_core_passed']: raise SystemExit(1)
