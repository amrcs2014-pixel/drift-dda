"""Metrics over the compact per-window logs written by runner.py, and across-seed statistics."""
import json, glob, os
import numpy as np
from scipy.stats import t as tdist, wilcoxon, ttest_rel

def pooled(r, idx):
    idx = np.asarray(idx, int)
    g = lambda k: float(np.sum(np.asarray(r[k])[idx])) if len(idx) else 0.0
    tp, fp, fn, tn, tp1, fp1 = g('tp'), g('fp'), g('fn'), g('tn'), g('tp1'), g('fp1')
    npos = tp + fn; nneg = fp + tn
    out = dict(f1=2 * tp / max(1, 2 * tp + fp + fn), prec=tp / max(1, tp + fp), rec=tp / max(1, npos),
               fpr=fp / max(1, nneg), f1_s1=2 * tp1 / max(1, tp1 + fp1 + npos), fpr_s1=fp1 / max(1, nneg),
               rec_s1=tp1 / max(1, npos), npos=npos)
    au = np.asarray(r['auroc'], float)[idx]; ap = np.asarray(r['auprc'], float)[idx]
    out['auroc'] = float(np.nanmean(au)) if np.isfinite(au).any() else float('nan')
    out['auprc'] = float(np.nanmean(ap)) if np.isfinite(ap).any() else float('nan')
    return out

def window_fpr(r):
    fp = np.asarray(r['fp'], float); tn = np.asarray(r['tn'], float)
    return fp / np.maximum(1, fp + tn)

def onset_metrics(r, onsets, pre_len=5, eps_rec=0.02, eps_deg=0.05, hold=3):
    """Per change point: alarm delay, recovery time (rolling FPR back within eps of pre-onset level),
    degraded windows (FPR above pre-onset + eps_deg) until the next change point. Censored at segment end."""
    fpr = window_fpr(r); n = len(fpr)
    act = np.asarray(r['tier']) > 0
    alarm = np.asarray(r['alarm']) > 0
    res = []
    bounds = list(onsets) + [n]
    for i, dp in enumerate(onsets):
        end = bounds[i + 1]
        if dp >= n or dp < 1:
            continue
        pre = fpr[max(0, dp - pre_len):dp].mean()
        seg = np.arange(dp, end)
        a = np.where(alarm[seg] | act[seg])[0]
        delay = int(a[0]) if len(a) else len(seg)
        rec = len(seg)
        for k in range(len(seg)):
            w = fpr[seg[k]:min(end, seg[k] + hold)]
            if len(w) and w.mean() <= pre + eps_rec:
                rec = k; break
        deg = int((fpr[seg] > pre + eps_deg).sum())
        res.append(dict(onset=int(dp), alarm_delay=delay, recovery=rec, degraded=deg, seg_len=len(seg)))
    return res

def summarize_run(r, kind='syn'):
    n = len(r['tp']); dps = r.get('drift_points', [])
    s = dict(steps=r['steps_total'], wall=r['wall_total'], n_black=r['n_black'])
    tiers = np.asarray(r['tier'])
    for k in (1, 2, 3):
        s['t%d' % k] = int((tiers == k).sum())
    s['n_esc'] = int(np.sum(r.get('esc', [0])))
    s['all'] = pooled(r, range(n))
    if kind == 'real' and 'window_day' in r:
        wd = np.asarray(r['window_day'])
        s['days'] = {int(d): pooled(r, np.where(wd == d)[0]) for d in sorted(set(wd.tolist()))}
        onsets = [0] + list(dps)          # stream start (config -> test) is the first change
        s['onsets'] = onset_metrics(r, [o for o in onsets if o > 0] or list(dps))
    else:
        dp0 = dps[0] if dps else n
        s['pre'] = pooled(r, range(dp0)); s['post'] = pooled(r, range(dp0, n))
        s['onsets'] = onset_metrics(r, dps)
    c = np.asarray(r['contam'], float); cl = np.asarray(r['cover_loss'], float); pz = np.asarray(r['poison_in'], float)
    s['contam'] = float(np.nanmean(c)) if np.isfinite(c).any() else float('nan')
    s['cover_loss'] = float(np.nanmean(cl)) if np.isfinite(cl).any() else float('nan')
    s['poison_in'] = float(np.nanmean(pz)) if np.isfinite(pz).any() else float('nan')
    on = s['onsets']
    if on:
        for k in ('alarm_delay', 'recovery', 'degraded'):
            s[k] = float(np.mean([o[k] for o in on]))
    s['wdist'] = r.get('wdist', float('nan'))
    return s

def cat_recall(r, cat, w_from=0, w_to=None):
    tp = np.asarray(r['cat_tp_w'])[w_from:w_to, cat].sum(); n = np.asarray(r['cat_n_w'])[w_from:w_to, cat].sum()
    return float(tp / n) if n else float('nan')

def ci95(x):
    x = np.asarray([v for v in x if v == v], float); n = len(x)
    if n == 0:
        return float('nan'), float('nan')
    m = float(x.mean())
    if n < 2:
        return m, 0.0
    return m, float(tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))

def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = dict(mean_diff=float((a - b).mean()) if len(a) else float('nan'), n=int(len(a)))
    try:
        out['wilcoxon_p'] = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
    except Exception:
        out['wilcoxon_p'] = float('nan')
    try:
        out['ttest_p'] = float(ttest_rel(a, b).pvalue) if np.any(a != b) else 1.0
    except Exception:
        out['ttest_p'] = float('nan')
    return out

def load(exp, tag=None):
    pat = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'raw', exp,
                       ('%s__s*.json' % tag) if tag else '*.json')
    out = {}
    for f in glob.glob(pat):
        r = json.load(open(f))
        out.setdefault(r['job']['tag'], {})[r['job']['seed']] = r
    return out

def get(d, path):
    for k in path.split('.'):
        if isinstance(d, dict):
            d = d.get(int(k) if k.isdigit() and int(k) in d else k, float('nan'))
        else:
            return float('nan')
    return d
