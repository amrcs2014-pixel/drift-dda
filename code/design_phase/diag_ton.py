import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""Diagnostic (static detector only): is ToN near-chance because of the trace or because of an identifier
feature (DNS_QUERY_ID, a random transaction id)? Also an oracle configuration on benign flows sampled from
the test period itself, as an upper reference for what a fitted-once f-AnoGAN can separate."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np, torch
from core import configure, anomaly_scores, _auc_pair
from streams import RealStream
from config import real_cfg

torch.set_num_threads(3)
cfg = real_cfg()
st = RealStream('ton', 4096, 0)
names = st.meta['features']
def per_window_auc(prep, models, cols):
    aus, aps = [], []
    for t in range(0, st.n_windows, 3):
        w = st.window(t); X = w['X'][:, cols]
        s = anomaly_scores(models, prep.transform(X), cfg, 'cpu')
        a, p = _auc_pair(w['y'].astype(np.int64), s)
        if a == a: aus.append(a); aps.append(p)
    return float(np.mean(aus)), float(np.mean(aps)), len(aus)

allc = np.arange(len(names))
noid = np.array([i for i, n in enumerate(names) if n not in ('DNS_QUERY_ID',)])
res = {}
for lab, cols in [('all_features', allc), ('without_DNS_QUERY_ID', noid)]:
    torch.manual_seed(0)
    prep, models, tau = configure(st.config_X[:, cols], cfg, 'cpu')
    res[lab] = per_window_auc(prep, models, cols); print(lab, res[lab], flush=True)
# oracle: configure on benign flows from the test period
rng = np.random.RandomState(0)
ben = np.where(st.y == 0)[0]; pick = np.sort(rng.choice(ben, 10000, replace=False))
Xo = np.asarray(st.X[st.lo + pick], np.float32)[:, noid]
torch.manual_seed(0)
prep, models, tau = configure(Xo, cfg, 'cpu')
res['oracle_test_benign_without_id'] = per_window_auc(prep, models, noid); print('oracle', res['oracle_test_benign_without_id'], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'diag_ton.json'), 'w'), indent=1)
