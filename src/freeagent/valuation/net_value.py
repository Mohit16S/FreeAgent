"""Single-season accounting horizon and user-specified value per point."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Costs:
    fee: float
    annual_wage: float
    contract_years: float
    bonus: float = 0.
    transaction_cost: float = 0.
    horizon_years: float = 1.

    def breakdown(self) -> dict:
        vals = [self.fee,self.annual_wage,self.contract_years,self.bonus,self.transaction_cost,self.horizon_years]
        if not np.isfinite(vals).all() or min(vals)<0 or self.contract_years<=0 or not 0<self.horizon_years<=1:
            raise ValueError('Nonnegative costs, positive contract, and at most one sporting year required')
        if self.horizon_years>self.contract_years: raise ValueError('Contract shorter than analysis horizon')
        # Fee in the project equation is the fee allocated to this horizon, not full cash fee.
        amortised = (self.fee+self.bonus+self.transaction_cost)*self.horizon_years/self.contract_years
        wages = self.annual_wage*self.horizon_years
        return {'fee_allocated':amortised,'wages_allocated':wages,'total':amortised+wages,
                'total_contract_commitment':self.fee+self.bonus+self.transaction_cost+self.annual_wage*self.contract_years}


def net_value(epa: np.ndarray, costs: Costs, price_per_point: float,
              tolerance: float = 1e6, threshold: float = .75) -> dict:
    epa = np.asarray(epa)
    if epa.size==0 or not np.isfinite(epa).all() or not np.isfinite([price_per_point,tolerance,threshold]).all(): raise ValueError('Finite samples and assumptions required')
    if price_per_point<0 or tolerance<0 or not .5<threshold<=1: raise ValueError('Invalid price/tolerance/verdict threshold')
    breakdown = costs.breakdown()
    samples = epa*price_per_point-breakdown['total']
    probabilities = {'Best Buy':float((samples>tolerance).mean()),'Fair':float((abs(samples)<=tolerance).mean()),
                     'Overpaid':float((samples < -tolerance).mean())}
    verdict = 'Best Buy' if probabilities['Best Buy']>=threshold else 'Overpaid' if probabilities['Overpaid']>=threshold else 'Fair'
    return {'samples':samples,'mean':float(samples.mean()),'median':float(np.median(samples)),
            'interval':np.quantile(samples,[.05,.95]).tolist(),'positive_probability':float((samples>0).mean()),
            'loss_probability':float((samples<0).mean()),'category_probabilities':probabilities,'verdict':verdict,
            'verdict_note':'Fair includes inconclusive uncertainty; not a claim of accurate pricing.', 'costs':breakdown}


def financial_sensitivity(epa: np.ndarray, costs: Costs, price_per_point: float) -> list[dict]:
    """One-at-a-time financial stress grid, preserving the benefit/cost horizon."""
    from dataclasses import replace
    rows=[]
    for factor in [.5,.75,1.,1.25,1.5]:
        for parameter in ['fee','annual_wage','price_per_point']:
            c=replace(costs,**{parameter:getattr(costs,parameter)*factor}) if parameter!='price_per_point' else costs
            price=price_per_point*factor if parameter=='price_per_point' else price_per_point
            r=net_value(epa,c,price)
            rows.append({'parameter':parameter,'factor':factor,'mean_net_value':r['mean'],'positive_probability':r['positive_probability']})
    return rows
