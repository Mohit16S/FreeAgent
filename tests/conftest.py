"""Synthetic fixtures ONLY for software validation; never dashboard data."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def data():
    rng = np.random.default_rng(9)
    rows,apps = [],[]
    teams = ['A','B','C','D']
    for i in range(60):
        h,a = rng.choice(teams,2,replace=False)
        mid = str(i)
        rows.append(dict(match_id=mid,date=str(date(2023,1,1)+timedelta(days=int(i))),
                         competition='SYNTHETIC TEST',season='2023',home_id=h,away_id=a,
                         home_goals=int(rng.poisson(1.6)),away_goals=int(rng.poisson(1.1)),source='SYNTHETIC TEST'))
        for t in [h,a]:
            for p in range(11):
                apps.append(dict(match_id=mid,player_id=f'{t}{p}',player_name=f'Test {t}{p}',team_id=t,
                                 minutes=90.,position='Test role',started=True))
    return pd.DataFrame(rows),pd.DataFrame(apps)
