"""Discrete additive candidate selection under budget and role constraints."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint


def optimise(candidates: pd.DataFrame, budget: float, role_caps: dict[str,int]) -> dict:
    """Binary MILP. Values must be incremental EPA or Net Value, in one consistent unit.

    Ignores interactions between signings; re-evaluate selected squad jointly before use.
    Cost is cash commitment when cash is the constrained budget, never amortised cost.
    """
    required={'player_id','role','cost','value','replacement_id'}
    if not required.issubset(candidates): raise ValueError('Candidate fields missing')
    if candidates.empty or not np.isfinite(budget) or budget<0: raise ValueError('Invalid budget/pool')
    if candidates.player_id.duplicated().any(): raise ValueError('Duplicate candidates')
    if not np.isfinite(candidates[['cost','value']].to_numpy()).all() or (candidates.cost<0).any(): raise ValueError('Invalid candidate values')
    if not set(candidates.role).issubset(role_caps) or any(v<0 or int(v)!=v for v in role_caps.values()): raise ValueError('Every role requires a nonnegative integer capacity')
    n=len(candidates); rows=[candidates.cost.to_numpy()]; caps=[budget]
    for role,cap in role_caps.items(): rows.append((candidates.role==role).astype(int).to_numpy()); caps.append(cap)
    # Prevent two signings from both claiming the same displaced incumbent.
    for incumbent in candidates.replacement_id.unique(): rows.append((candidates.replacement_id==incumbent).astype(int).to_numpy()); caps.append(1)
    result=milp(-candidates.value.to_numpy(),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                constraints=LinearConstraint(np.array(rows),np.full(len(rows),-np.inf),np.array(caps)))
    if not result.success: raise RuntimeError(f'Budget optimisation failed: {result.message}')
    chosen=candidates[result.x>.5].copy()
    return {'selected':chosen,'total_cost':float(chosen.cost.sum()),'total_value':float(chosen.value.sum()),
            'no_signing_value':0.,'warning':'Additive approximation; joint squad interactions are not optimised.'}
