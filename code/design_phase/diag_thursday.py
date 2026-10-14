import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""Diagnose CIC Thursday (insider port-scan from 192.168.10.8) for one policy: per-window share of the
insider, its Stage-I flag rate, threshold, tier, quarantine/blacklist status."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np, torch
import core
from core import configure, run_stream, anomaly_scores
from streams import RealStream, load_cache
from config import real_cfg
import pandas as pd

torch.set_num_threads(2)
pol = sys.argv[1] if len(sys.argv) > 1 else 'gated'
over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
cfg = real_cfg(ph_lambda=0.02, **over)
st = RealStream('cic17', 4096, 0)
# address id of the insider
arr, meta = load_cache('cic17')
d = os.path.join(os.path.dirname(core.__file__), '..', 'data', 'cic17')
# recover id: the most frequent attack source on Thursday (day 3)
y = st.y; src = st.src; day = st.day
att_thu = src[(y == 1) & (day == 3)]
ins = int(pd.Series(att_thu).value_counts().index[0])
print('insider id', ins, 'Thursday attack flows from it', int((att_thu == ins).sum()))
prep, models, tau = configure(st.config_X, cfg, 'cpu')

# wrap run_window to log per-window insider stats
orig = core.Dictionaries.run_window
LOG = []
def rw(self, s_, d_, anom):
    out = orig(self, s_, d_, anom)
    m = (np.asarray(s_) == ins)
    LOG.append(dict(share=float(m.mean()), flag1=float(anom[m].mean()) if m.any() else np.nan,
                    final=float(out[m].mean()) if m.any() else np.nan, black=ins in self.blacklist,
                    cnt=self.black.get(ins, 0), good=self.good.get(ins, 0)))
    return out
core.Dictionaries.run_window = rw
r = run_stream(pol, cfg, st, prep, models, tau, 'cpu', seed=0)
L = r['logs']
thu = [i for i in range(st.n_windows) if st.window_day[i] == 3]
print('Thursday windows', thu[0], '-', thu[-1])
for i in thu:
    x = L[i]; g = LOG[i]
    if g['share'] > 0.01 or x['tier'] > 0:
        print('w%3d share %.2f flagI %.2f final %.2f black %s cnt %6d good %6d | tier %d sev %.2f tau %.4f rec %.2f' % (
            i, g['share'], g['flag1'], g['final'], g['black'], g['cnt'], g['good'], x['tier'], x['sev'], x['tau'],
            x['tp'] / max(1, x['tp'] + x['fn'])))
