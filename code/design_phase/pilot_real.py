import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""Pilot: static vs gated on the real temporal traces (seed 0) - timing and natural-drift check."""
import sys, time
import numpy as np, torch
from core import configure, run_stream
from streams import RealStream
from config import real_cfg

torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
name = sys.argv[1]; W = int(sys.argv[2])
import json
opts = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
cfg = real_cfg(**(json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}))
t0 = time.time()
st = RealStream(name, W, seed=0)
print(name, 'windows', st.n_windows, 'drift points (day changes)', st.drift_points, 'load %.1fs' % (time.time() - t0))
t0 = time.time()
prep, models, tau = configure(st.config_X, cfg, 'cpu')
print('configured: dsel', len(prep.idx), 'tau %.4f' % tau, '%.1fs' % (time.time() - t0))
for pol in sys.argv[4].split(',') if len(sys.argv) > 4 else ['static', 'gated']:
    t0 = time.time()
    r = run_stream(pol, cfg, st, prep, models, tau, 'cpu', opts=opts, seed=0)
    L = r['logs']
    def pooled(sl):
        tp = sum(x['tp'] for x in sl); fp = sum(x['fp'] for x in sl); fn = sum(x['fn'] for x in sl); tn = sum(x['tn'] for x in sl)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); fpr = fp / max(1, fp + tn); rec = tp / max(1, tp + fn)
        return f1, fpr, rec
    print('== %s  time %.1fs  steps %d  tiers %s  black %d' % (pol, time.time() - t0, r['steps'],
          np.bincount([x['tier'] for x in L], minlength=4).tolist(), r['n_black']))
    print('   all: F1 %.3f FPR %.3f recall %.3f' % pooled(L))
    for dday in sorted(set(st.window_day.tolist())):
        sl = [L[i] for i in range(len(L)) if st.window_day[i] == dday]
        print('   day %d (%d windows): F1 %.3f FPR %.3f recall %.3f' % ((dday, len(sl)) + pooled(sl)))
    fpr_w = [x['fp'] / max(1, x['fp'] + x['tn']) for x in L]
    print('   FPR per 10-window block:', ' '.join('%.2f' % np.mean(fpr_w[i:i + 10]) for i in range(0, len(L), 10)))
    print('   tier windows:', [(i, x['tier'], round(x['arate'], 3)) for i, x in enumerate(L) if x['tier'] > 0][:40])
    cn = [x['contam'] for x in L if x['tier'] > 0]; cl = [x['cover_loss'] for x in L if x['tier'] > 0]
    if cn: print('   retrain contamination %.4f coverage loss %.3f' % (np.nanmean(cn), np.nanmean(cl)))
    au = [x['auroc'] for x in L if x['auroc'] == x['auroc']]
    print('   mean window AUROC %.3f (n=%d)' % (np.mean(au) if au else float('nan'), len(au)))
