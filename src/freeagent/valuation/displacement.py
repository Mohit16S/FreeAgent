"""One-role minutes conservation; caller specifies a realistic incumbent allocation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Displacement:
    incoming: str
    incumbent: str
    expected_minutes: float
    incumbent_minutes: float
    fixtures: int

    def validate(self) -> None:
        values = [self.expected_minutes,self.incumbent_minutes,self.fixtures]
        if not np.isfinite(values).all() or self.fixtures<=0: raise ValueError('Invalid horizon')
        if self.incoming==self.incumbent: raise ValueError('Incoming player must differ from incumbent')
        if not 0<=self.expected_minutes<=self.incumbent_minutes<=90*self.fixtures:
            raise ValueError('Incoming minutes must fit incumbent allocation and fixture horizon')

    def audit(self) -> dict:
        self.validate()
        return {'incoming_before':0.,'incoming_after':self.expected_minutes,
                'incumbent_before':self.incumbent_minutes,
                'incumbent_after':self.incumbent_minutes-self.expected_minutes,
                'total_before':self.incumbent_minutes,'total_after':self.incumbent_minutes,
                'minutes_share':self.expected_minutes/(90*self.fixtures)}


def minutes_sensitivity(impact, match, fixtures, club: str, displacement: Displacement) -> list[dict]:
    """Exact expected EPA at feasible minute allocations, holding match parameters fixed.

    This is a point-parameter sensitivity, explicitly separate from ensemble uncertainty.
    """
    from freeagent.match.dixon_coles import outcomes
    displacement.validate()
    difference=impact.model.coef_[impact.players.index(displacement.incoming)]-impact.model.coef_[impact.players.index(displacement.incumbent)]
    rows=[]
    for fraction in [0.,.25,.5,.75,1.]:
        minutes=displacement.incumbent_minutes*fraction
        shift=float(difference*minutes/(90*displacement.fixtures)); epa=0.
        for r in fixtures.itertuples():
            if club not in (r.home_id,r.away_id): continue
            before=outcomes(match.matrix(r.home_id,r.away_id))
            after=outcomes(match.matrix(r.home_id,r.away_id,{club:shift}))
            win=0 if r.home_id==club else 2
            epa+=3*(after[win]-before[win])+after[1]-before[1]
        rows.append({'minutes':minutes,'epa':epa})
    return rows
