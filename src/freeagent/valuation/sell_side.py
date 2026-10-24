"""Compare retention and a fully evaluated replacement scenario."""
from __future__ import annotations
import numpy as np


def sell_or_retain(replacement_net_values: np.ndarray, sale_fee: float,
                   incumbent_book_value: float, saved_wages: float) -> dict:
    """Incremental one-horizon accounting P&L, not sale cash flow.

    replacement_net_values already include the replacement's allocated costs and
    sporting difference relative to retention. Saved wages use the same horizon.
    """
    values=np.asarray(replacement_net_values,dtype=float)
    if values.size==0 or not np.isfinite(values).all() or not np.isfinite([sale_fee,incumbent_book_value,saved_wages]).all() or min(sale_fee,incumbent_book_value,saved_wages)<0:
        raise ValueError('Finite samples and nonnegative accounting inputs required')
    incremental=values+sale_fee-incumbent_book_value+saved_wages
    return {'retain_baseline':0.,'sell_and_replace_samples':incremental,
            'mean_incremental_value':float(incremental.mean()),'probability_prefer_sale':float((incremental>0).mean()),
            'sale_accounting_profit':sale_fee-incumbent_book_value,'saved_wages':saved_wages}
