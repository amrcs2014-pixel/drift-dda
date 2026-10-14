"""Experiment definitions for DRIFT-DDA v7. Every job is built from the frozen configuration
(config.py + results/tuning_selected.json written by tune.py). Run:  python experiments.py <exp> [workers]"""
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU-only (an empty value is deleted on Windows)

import json, os, sys
from config import syn_cfg, real_cfg, DMAC_TUNED
from runner import run_all, RES

SEEDS = list(range(10))
POL_MAIN = ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated', 'ssf']
HARD = dict(difficulty='hard', sep_lo=1.0, sep_hi=1.5, attack_sigma=0.2)
LONG = dict(n_windows=120, window_size=512)
DRIFTS = {
    'none':      dict(drift_type='abrupt', drift_points=[30, 80], drift_magnitude=0.0),
    'abrupt':    dict(drift_type='abrupt', drift_points=[30, 80]),
    'gradual':   dict(drift_type='gradual', drift_points=[30, 80], gradual_len=20),
    'recurring': dict(drift_type='recurring', drift_points=[30], recur_period=20),
    'novel':     dict(drift_type='abrupt', drift_points=[30, 80], novel_at=60),
}

def dmac():
    p = os.path.join(RES, 'tuning_selected.json')
    d = dict(DMAC_TUNED)
    if os.path.exists(p):
        d.update(json.load(open(p))['selected'])
    return d

def S(**kw):
    c = syn_cfg(dmac()); c.update(kw); return c

def R(**kw):
    c = real_cfg(dmac()); c.update(kw); return c

def syn_data(difficulty='hard', drift='abrupt', rotation=0.0, **extra):
    sp = dict(LONG); sp.update(HARD if difficulty == 'hard' else dict(difficulty='easy')); sp.update(DRIFTS[drift])
    sp['rotation'] = rotation; sp.update(extra)
    return dict(kind='syn', spec=sp)

def real_data(name, rotation=0.0):
    return dict(kind='real', name=name, window=4096, rotation=rotation)

def job(exp, tag, data, seed, policy, cfg, opts=None):
    return dict(exp=exp, tag=tag, data=data, seed=seed, policy=policy, cfg=cfg, opts=opts or {})

REAL_SETS = ['unsw', 'cic17']

# ----------------------------------------------------------------------------- experiments
def exp_real():                      # E1: natural drift, real addresses
    J = []
    for name in REAL_SETS:
        for s in SEEDS:
            for p in POL_MAIN:
                J.append(job('real', '%s_%s' % (name, p), real_data(name), s, p, R()))
    return J

def exp_rotation():                  # E2: attacker-address rotation on the real traces
    J = []
    for name in REAL_SETS:
        for rot in [0.5, 1.0]:
            for s in SEEDS[:5]:
                for p in ['static', 'detect', 'tier1', 'tier2', 'gated']:
                    J.append(job('rotation', '%s_rot%.1f_%s' % (name, rot, p), real_data(name, rot), s, p, R()))
                J.append(job('rotation', '%s_rot%.1f_gated-norep' % (name, rot), real_data(name, rot), s, 'gated', R(),
                             dict(no_reputation=True)))
    return J

def exp_synth():                     # E3: drift types x difficulty x rotation (long streams)
    J = []
    for diff in ['easy', 'hard']:
        for drift in DRIFTS:
            for rot in [0.0, 1.0]:
                for s in SEEDS:
                    for p in POL_MAIN:
                        J.append(job('synth', '%s_%s_rot%.0f_%s' % (diff, drift, rot, p),
                                     syn_data(diff, drift, rot), s, p, S(periodic_k=10)))
    return J

ABLATIONS = {
    'full': {}, 'no_gating': dict(no_gating=True), 'no_escalation': dict(no_escalation=True),
    'no_cooldown': dict(no_cooldown=True), 'no_replay': dict(no_replay=True), 'no_reputation': dict(no_reputation=True),
    'no_majority': dict(no_majority=True),
    'rule_base': dict(rule='base'), 'tau_v6': dict(tau_rule='retrain'), 'severity_v6': dict(severity='rate'),
    'no_surge': dict(no_surge=True), 'no_defer': dict(no_defer=True),
}
V6_CTRL = dict(rebase=False, severity='rate', tau_rule='retrain', rule='base', bl_majority=False, src_cap_frac=0, surge=False)

def exp_ablation():                  # E4
    J = []
    sets = [('hard_recurring_rot1', syn_data('hard', 'recurring', 1.0), S(periodic_k=10)),
            ('hard_abrupt_rot1', syn_data('hard', 'abrupt', 1.0), S(periodic_k=10)),
            ('unsw', real_data('unsw'), R()), ('cic17', real_data('cic17'), R())]
    for name, data, cfg in sets:
        for s in (SEEDS if name.startswith('hard') else SEEDS[:5]):
            for a, o in ABLATIONS.items():
                J.append(job('ablation', '%s_%s' % (name, a), data, s, 'gated', cfg, o))
            c6 = dict(cfg); c6.update(V6_CTRL); c6.update(dict(rate_t2=1.6, rate_t3=2.5) if name.startswith('hard') else dict(rate_t2=1.5, rate_t3=3.0))
            # the v6 controller never re-baselines and retrains hundreds of times on the real traces (~1 h per
            # run): two configurations there, and the v6 Tier-3-only baseline only on the synthetic streams
            if name.startswith('hard') or s < 2:
                J.append(job('ablation', '%s_v6controller' % name, data, s, 'gated', c6))
            if name.startswith('hard'):
                J.append(job('ablation', '%s_v6controller-detect' % name, data, s, 'detect', c6))
    return J

def exp_severity():                  # E5: severity sweep + no-drift control (abrupt, both difficulties)
    J = []
    for diff in ['easy', 'hard']:
        for mag in [0.0, 0.2, 0.4, 0.6, 1.0]:
            for rot in [0.0, 1.0]:
                for s in SEEDS:
                    for p in ['static', 'detect', 'tier1', 'gated']:
                        J.append(job('severity', '%s_m%.1f_rot%.0f_%s' % (diff, mag, rot, p),
                                     syn_data(diff, 'abrupt', rot, drift_magnitude=mag), s, p, S()))
    return J

def exp_frontier():                  # E6: quality-cost frontier
    J = []
    sets = [('hard_abrupt_rot1', syn_data('hard', 'abrupt', 1.0), S, [5, 10, 20, 40], SEEDS, ['detect', 'tier1', 'tier2', 'gated']),
            ('unsw', real_data('unsw'), R, [25, 50, 100], SEEDS[:5], ['detect', 'tier1', 'gated'])]
    for name, data, C, ks, seeds, pols in sets:
        for s in seeds:
            J.append(job('frontier', '%s_static' % name, data, s, 'static', C()))
            for k in ks:
                J.append(job('frontier', '%s_periodic_k%d' % (name, k), data, s, 'periodic', C(periodic_k=k)))
            for lam in [0.01, 0.02, 0.05, 0.10]:
                for p in pols:
                    J.append(job('frontier', '%s_%s_lam%.2f' % (name, p, lam), data, s, p, C(ph_lambda=lam)))
    return J

def exp_poison():                    # E7: boiling-frog poisoning from fresh addresses
    J = []
    P = dict(t0=30, t1=90, rate=0.05, target=0)
    for s in SEEDS:
        for pois in [False, True]:
            data = syn_data('hard', 'abrupt', 0.0, drift_points=[30], poison=P if pois else None)
            for p, o, nm in [('static', {}, 'static'), ('tier1', {}, 'tier1'), ('detect', {}, 'detect'), ('gated', {}, 'gated'),
                             ('gated', dict(rule='probation', g_min=3), 'gated-probation')]:
                J.append(job('poison', '%s_%s' % ('pois' if pois else 'clean', nm), data, s, p, S(), o))
    return J

def exp_injected():                  # E8: injected shift on real UNSW records (continuous columns only)
    J = []
    for s in SEEDS:
        data = dict(kind='inj', spec=dict(n_windows=60, window_size=512, drift_points=[20, 40], magnitude=2.0))
        for p, o, nm in [('static', {}, 'static'), ('gated', {}, 'gated'), ('gated', dict(rule='base'), 'gated-base'),
                         ('gated', dict(no_reputation=True), 'gated-norep'), ('detect', {}, 'detect'), ('tier1', {}, 'tier1')]:
            J.append(job('injected', nm, data, s, p, R(periodic_k=10), o))
    return J

def exp_sensitivity():               # one-at-a-time sweeps around the frozen configuration
    J = []
    d = dmac()
    grid = {'ph_lambda': [0.01, 0.02, 0.05, 0.10], 'rate_t2': [1.25, 1.5, 2.0, 2.5], 'rate_t3': [2.0, 3.0, 4.0, 6.0],
            'cooldown': [1, 2, 4, 8], 'pi_w': [85, 90, 95, 99], 'clean_frac': [0.3, 0.5, 0.7, 0.9],
            'verify_m': [1, 2, 3, 5]}
    # real traces: 3 seeds, the non-default ends of each range (the default is the main 'real' run)
    grid_real = {'ph_lambda': [0.01, 0.05, 0.10], 'rate_t2': [1.25, 2.0], 'rate_t3': [2.0, 4.0], 'cooldown': [1, 4],
                 'pi_w': [90, 99], 'L': [500, 5000]}
    sets = [('hard_abrupt_rot1', syn_data('hard', 'abrupt', 1.0), S, SEEDS, grid),
            ('unsw', real_data('unsw'), R, SEEDS[:3], grid_real), ('cic17', real_data('cic17'), R, SEEDS[:3], grid_real)]
    for name, data, C, seeds, gr in sets:
        for par, vals in gr.items():
            for v in vals:
                for s in seeds:
                    J.append(job('sensitivity', '%s_%s=%s' % (name, par, v), data, s, 'gated', C(**{par: v})))
    return J

def exp_heldout():                  # E9: NF-ToN-IoT-v3, never used during design or tuning
    J = []
    for s in SEEDS:
        for p in POL_MAIN:
            J.append(job('heldout', 'ton_%s' % p, real_data('ton'), s, p, R()))
        if s < 5:
            for p in ['static', 'detect', 'tier1', 'tier2', 'gated']:
                J.append(job('heldout', 'ton_rot1.0_%s' % p, real_data('ton', 1.0), s, p, R()))
    return J

def exp_posthoc():                  # post-hoc (after seeing rotation results): fresh-source guard
    J = []
    for name in ['cic17', 'unsw']:
        for rot in [0.0, 0.5, 1.0]:
            for s in SEEDS[:5]:
                J.append(job('posthoc', '%s_rot%.1f_gated-fresh' % (name, rot), real_data(name, rot), s, 'gated', R(),
                             dict(fresh_guard=True)))
    for rot in [0.0, 1.0]:
        for s in SEEDS[:5]:
            J.append(job('posthoc', 'ton_rot%.1f_gated-fresh' % rot, real_data('ton', rot), s, 'gated', R(), dict(fresh_guard=True)))
    # post-hoc (after seeing the poisoning results): a source needs >= 3 verdicts before it can be trusted
    P = dict(t0=30, t1=90, rate=0.05, target=0)
    for s in SEEDS:
        for pois in [False, True]:
            data = syn_data('hard', 'abrupt', 0.0, drift_points=[30], poison=P if pois else None)
            J.append(job('posthoc', '%s_gated-trust3' % ('pois' if pois else 'clean'), data, s, 'gated', S(), dict(trust_min=3)))
    for name in ['unsw', 'cic17', 'ton']:
        for s in SEEDS[:5]:
            J.append(job('posthoc', '%s_rot0.0_gated-trust3' % name, real_data(name), s, 'gated', R(), dict(trust_min=3)))
    for s in SEEDS[:5]:
        J.append(job('posthoc', 'hard_abrupt_rot1_gated-trust3', syn_data('hard', 'abrupt', 1.0), s, 'gated', S(), dict(trust_min=3)))
    return J

EXPS = dict(posthoc=exp_posthoc, heldout=exp_heldout, real=exp_real, rotation=exp_rotation, synth=exp_synth, ablation=exp_ablation, severity=exp_severity,
            frontier=exp_frontier, poison=exp_poison, injected=exp_injected, sensitivity=exp_sensitivity)

if __name__ == '__main__':
    names = sys.argv[1].split(',')
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    jobs = []
    for n in names:
        jobs += EXPS[n]()
    run_all(jobs, workers)
