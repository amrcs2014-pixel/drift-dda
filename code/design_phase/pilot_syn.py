import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""Pilot: calibrate synthetic difficulty / drift types (few seeds)."""
import sys, time, json
import numpy as np, torch
from core import configure, run_stream
from streams import SynthStream
from config import syn_cfg

torch.set_num_threads(2)
spec = json.loads(sys.argv[1])
pols = sys.argv[2].split(',')
seeds = range(int(sys.argv[3]) if len(sys.argv) > 3 else 3)
opts = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
cfg_over = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
cfg = syn_cfg(**cfg_over)
res = {p: [] for p in pols}
for s in seeds:
    st = SynthStream(spec, s)
    prep, models, tau = configure(st.config_X, cfg, 'cpu')
    for p in pols:
        t0 = time.time()
        r = run_stream(p, cfg, st, prep, models, tau, 'cpu', opts=opts, seed=s)
        L = r['logs']; dp = st.drift_points[0]
        def pooled(sl):
            tp = sum(x['tp'] for x in sl); fp = sum(x['fp'] for x in sl); fn = sum(x['fn'] for x in sl); tn = sum(x['tn'] for x in sl)
            return 2 * tp / max(1, 2 * tp + fp + fn), fp / max(1, fp + tn)
        f_pre, _ = pooled(L[:dp]); f_post, fpr_post = pooled(L[dp:])
        au = np.nanmean([x['auroc'] for x in L[dp:]])
        tp1 = sum(x['tp1'] for x in L[dp:]); fp1 = sum(x['fp1'] for x in L[dp:]); npos = sum(x['tp'] + x['fn'] for x in L[dp:])
        f1s1 = 2 * tp1 / max(1, tp1 + fp1 + npos)
        res[p].append((f_pre, f_post, fpr_post, au, f1s1, r['steps'], np.bincount([x['tier'] for x in L], minlength=4)[1:].tolist(), time.time() - t0))
for p in pols:
    a = np.array([x[:6] for x in res[p]], float)
    print('%-8s preF1 %.3f postF1 %.3f postFPR %.3f postAUROC %.3f postF1(stageI) %.3f steps %6.0f tiers %s  %.1fs' % (
        p, *a.mean(0), [x[6] for x in res[p]], np.mean([x[7] for x in res[p]])))
