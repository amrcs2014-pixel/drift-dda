"""Aggregate raw per-run logs into results/results_v7.json (every number in the manuscript comes from it)."""
import json, os, re
import numpy as np
from metrics import load, summarize_run, ci95, paired, get, cat_recall

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')

def S(v):
    m, h = ci95(v)
    return dict(mean=m, ci=h, n=len([x for x in v if x == x]), values=[float(x) for x in v])

def summarize_tag(runs, kind):
    return {s: summarize_run(r, kind) for s, r in sorted(runs.items())}

METRICS_REAL = ['all.f1', 'all.prec', 'all.rec', 'all.fpr', 'all.f1_s1', 'all.auroc', 'all.auprc', 'steps', 'wall',
                't1', 't2', 't3', 'n_esc', 'contam', 'cover_loss', 'recovery', 'degraded', 'alarm_delay', 'n_black']
METRICS_SYN = ['pre.f1', 'post.f1', 'post.prec', 'post.rec', 'post.fpr', 'post.f1_s1', 'post.auroc', 'post.auprc', 'all.f1',
               'steps', 'wall', 't1', 't2', 't3', 'n_esc', 'contam', 'cover_loss', 'alarm_delay', 'recovery', 'degraded', 'wdist']

def table(raw, kind, metrics, extra=None):
    out = {}
    for tag, runs in raw.items():
        sm = summarize_tag(runs, kind)
        row = {m: S([get(sm[s], m) for s in sm]) for m in metrics}
        row['seeds'] = list(sm.keys())
        if kind == 'real':
            days = sorted({d for s in sm for d in sm[s].get('days', {})})
            row['days'] = {str(d): {m: S([get(sm[s], 'days.%d.%s' % (d, m)) for s in sm]) for m in ['f1', 'fpr', 'rec', 'f1_s1', 'auroc']}
                           for d in days}
        if extra:
            row.update(extra(tag, runs))
        out[tag] = row
    return out

def tests(tab, ref_tag, others, metric):
    out = {}
    a = tab.get(ref_tag)
    if a is None:
        return out
    for o in others:
        if o in tab and o != ref_tag:
            out[o] = paired(a[metric]['values'], tab[o][metric]['values'])
    return out

def main():
    R = {}
    # E1 real
    raw = load('real')
    if raw:
        tab = table(raw, 'real', METRICS_REAL); R['real'] = tab
        R['real_tests'] = {}
        for ds in sorted({t.split('_')[0] for t in tab}):
            tags = [t for t in tab if t.startswith(ds + '_')]
            R['real_tests'][ds] = {m: tests(tab, ds + '_gated', tags, m) for m in ['all.f1', 'all.f1_s1', 'steps', 'all.fpr']}
    raw = load('heldout')
    if raw:
        tab = table(raw, 'real', METRICS_REAL); R['heldout'] = tab
        tags = [t for t in tab if t.startswith('ton_') and not t.startswith('ton_rot')]
        R['heldout_tests'] = {m: tests(tab, 'ton_gated', tags, m) for m in ['all.f1', 'all.f1_s1', 'steps', 'all.fpr', 'all.rec']}
        rt = [t for t in tab if t.startswith('ton_rot1.0_')]
        R['heldout_tests_rot'] = {m: tests(tab, 'ton_rot1.0_gated', rt, m) for m in ['all.f1', 'steps']}
    raw = load('rotation')
    if raw:
        R['rotation'] = table(raw, 'real', METRICS_REAL)
    raw = load('posthoc')
    if raw:
        real_part = {t: v for t, v in raw.items() if t.startswith(('unsw', 'cic17', 'ton'))}
        syn_part = {t: v for t, v in raw.items() if t not in real_part}
        def target(tag, runs):
            if not tag.startswith(('clean_', 'pois_')):
                return {}
            return {'target_recall_after': S([cat_recall(r, 0, 90) for r in runs.values()])}
        tab = table(real_part, 'real', METRICS_REAL)
        tab.update(table(syn_part, 'syn', METRICS_SYN + ['poison_in'], target))
        R['posthoc'] = tab
    # E3 synthetic
    raw = load('synth')
    if raw:
        def novel(tag, runs):
            if '_novel_' not in tag:
                return {}
            return {'novel_recall': S([cat_recall(r, 4, 60) for r in runs.values()])}
        tab = table(raw, 'syn', METRICS_SYN, novel); R['synth'] = tab
        R['synth_tests'] = {}
        for sc in sorted({t.rsplit('_', 1)[0] for t in tab}):
            tags = [t for t in tab if t.rsplit('_', 1)[0] == sc]
            R['synth_tests'][sc] = {m: tests(tab, sc + '_gated', tags, m) for m in ['post.f1', 'post.f1_s1', 'steps']}
    for exp, kind_of in [('ablation', None), ('frontier', None), ('sensitivity', None)]:
        raw = load(exp)
        if not raw:
            continue
        tabs = {}
        for tag, runs in raw.items():
            kind = 'real' if tag.startswith(('unsw', 'cic17', 'ton')) else 'syn'
            tabs.update(table({tag: runs}, kind, METRICS_REAL if kind == 'real' else METRICS_SYN))
        R[exp] = tabs
    raw = load('severity')
    if raw:
        R['severity'] = table(raw, 'syn', METRICS_SYN)
    raw = load('poison')
    if raw:
        def target(tag, runs):
            return {'target_recall_after': S([cat_recall(r, 0, 90) for r in runs.values()]),
                    'target_recall_before': S([cat_recall(r, 0, 20, 30) for r in runs.values()])}
        R['poison'] = table(raw, 'syn', METRICS_SYN + ['poison_in'], target)
    raw = load('injected')
    if raw:
        def inj(tag, runs):
            return {'wdist_std': S([r.get('wdist_std', float('nan')) for r in runs.values()])}
        R['injected'] = table(raw, 'syn', METRICS_SYN, inj)
    p = os.path.join(RES, 'tuning_selected.json')
    if os.path.exists(p):
        R['tuning'] = json.load(open(p))
    for f in ['cost_cpu1.json', 'cost_cpu6.json', 'cost_gpu.json']:
        q = os.path.join(RES, f)
        if os.path.exists(q):
            R.setdefault('cost', {})[f[5:-5]] = json.load(open(q))
    tmp = os.path.join(RES, 'results_v7.json.tmp%d' % os.getpid())
    json.dump(R, open(tmp, 'w'), indent=1)
    os.replace(tmp, os.path.join(RES, 'results_v7.json'))   # atomic: concurrent readers never see a partial file
    print('wrote results_v7.json with sections:', list(R))

if __name__ == '__main__':
    main()
