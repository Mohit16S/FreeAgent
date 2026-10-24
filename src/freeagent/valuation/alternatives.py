"""Coverage-aware similarity on estimated impact and observed playing time."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def alternatives(report: pd.DataFrame, player_id: str, limit: int = 10) -> pd.DataFrame:
    if player_id not in set(report.player_id): raise ValueError('Unknown reference player')
    if limit<1: raise ValueError('Positive result limit required')
    ref=report.set_index('player_id').loc[player_id]
    pool=report[(report.position==ref.position)&report.eligible].copy()
    if player_id not in set(pool.player_id): raise ValueError('Reference player lacks coverage')
    pool['uncertainty_width']=pool.upper-pool.lower
    values=StandardScaler().fit_transform(pool[['impact','minutes','uncertainty_width']])
    ix=pool.player_id.tolist().index(player_id)
    pool['similarity_distance']=np.linalg.norm(values-values[ix],axis=1)
    return pool[pool.player_id!=player_id].sort_values('similarity_distance').head(limit)
