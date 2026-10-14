"""Held-out tuning of the DMAC gate parameters (lambda, rho2, rho3) on synthetic tuning seeds 100-104,
which never appear in any test stream. Scenario mix: {easy, hard} x {abrupt, gradual, recurring} x
attacker rotation {0, 1}. Selection rule (stated before running): among configurations whose mean
post-drift F1 over all tuning streams is within 0.005 of the best, pick the one with the lowest mean
adaptation cost (optimizer steps). Writes results/tuning_selected.json and results/tuning_grid.json."""
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU-only (an empty value is deleted on Windows)

import itertools, json, os, sys
import numpy as np
from config import syn_cfg, DMAC_TUNED
from runner import run_all, RES, out_path
from experiments import syn_data
from metrics import summarize_run

TUNE_SEEDS = [100, 101, 102, 103, 104]
LAMS = [0.02, 0.05, 0.10]
R2 = [1.25, 1.5, 2.0]
R3 = [2.0, 3.0, 4.0]

def grid():
    for lam, r2, r3 in itertools.product(LAMS, R2, R3):
        if r3 > r2:
            yield dict(ph_lambda=lam, rate_t2=r2, rate_t3=r3)

def scenarios():
    for diff in ['easy', 'hard']:
        for drift in ['abrupt', 'gradual', 'recurring']:
            for rot in [0.0, 1.0]:
                yield '%s_%s_rot%.0f' % (diff, drift, rot), syn_data(diff, drift, rot)

def jobs():
    J = []
    for g in grid():
        tag = 'lam%.2f_r2%.2f_r3%.1f' % (g['ph_lambda'], g['rate_t2'], g['rate_t3'])
        for sname, data in scenarios():
            for s in TUNE_SEEDS:
                c = syn_cfg(dict(DMAC_TUNED)); c.update(g)
                J.append(dict(exp='tuning', tag='%s__%s' % (sname, tag), data=data, seed=s, policy='gated', cfg=c, opts={}))
    return J

def select():
    J = jobs(); rows = {}
    for j in J:
        p = out_path(j)
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); sm = summarize_run(r)
        key = j['tag'].split('__')[1]
        rows.setdefault(key, dict(f1=[], cost=[], cfg={k: j['cfg'][k] for k in ['ph_lambda', 'rate_t2', 'rate_t3']}))
        rows[key]['f1'].append(sm['post']['f1']); rows[key]['cost'].append(sm['steps'])
    table = {k: dict(cfg=v['cfg'], f1=float(np.mean(v['f1'])), cost=float(np.mean(v['cost'])), n=len(v['f1'])) for k, v in rows.items()}
    best = max(v['f1'] for v in table.values())
    ok = [k for k, v in table.items() if v['f1'] >= best - 0.005]
    pick = min(ok, key=lambda k: table[k]['cost'])
    json.dump(dict(rule='cheapest within 0.005 of best mean post-drift F1 on tuning seeds 100-104',
                   best_f1=best, selected=table[pick]['cfg'], selected_key=pick, table=table),
              open(os.path.join(RES, 'tuning_selected.json'), 'w'), indent=1)
    for k in sorted(table, key=lambda k: -table[k]['f1']):
        print('%-26s F1 %.4f cost %7.0f n=%d %s' % (k, table[k]['f1'], table[k]['cost'], table[k]['n'], '<= selected' if k == pick else ''))

if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run_all(jobs(), int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    select()
