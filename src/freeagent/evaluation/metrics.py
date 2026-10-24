"""Proper scoring rules in home/draw/away order."""
import numpy as np
import pandas as pd

def probability_metrics(probabilities, labels) -> dict:
    p,y = np.asarray(probabilities),np.asarray(labels,dtype=int)
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Invalid probabilities')
    truth = np.eye(3)[y]
    return {'log_loss':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()),
            'rps':float(((np.cumsum(p,axis=1)[:,:2]-np.cumsum(truth,axis=1)[:,:2])**2).sum(axis=1).mean()/2),
            'brier':float(((p-truth)**2).sum(axis=1).mean())}


def calibration(probabilities, labels, bins: int = 5) -> pd.DataFrame:
    p,y = np.asarray(probabilities),np.asarray(labels)
    rows = []
    for c,name in enumerate(['home','draw','away']):
        for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
            mask = (p[:,c]>=lo)&((p[:,c]<hi) if hi<1 else (p[:,c]<=hi))
            if mask.any(): rows.append({'outcome':name,'lower':lo,'upper':hi,'count':int(mask.sum()),
                                       'predicted':float(p[mask,c].mean()),'observed':float((y[mask]==c).mean())})
    return pd.DataFrame(rows)
