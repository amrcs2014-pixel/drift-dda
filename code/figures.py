"""All manuscript figures from results/results_v7.json and the raw logs (no hand-typed numbers).
Policy colours follow the entity (validated categorical palette; static = neutral reference grey);
every series also carries a marker / line style so identity never depends on colour alone."""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metrics import load, window_fpr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
OUT = os.environ.get('DDA_FIGS_DIR', os.path.join(HERE, '..', 'results', 'figures'))
os.makedirs(OUT, exist_ok=True)

POL = ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated', 'ssf']
LABEL = {'static': 'static', 'periodic': 'periodic', 'detect': 'Tier-3 only (detect)', 'tier1': 'Tier-1 only',
         'tier2': 'Tier-2 only', 'gated': 'DRIFT-DDA', 'ssf': 'SSF-style'}
SHORT = {'static': 'static', 'periodic': 'periodic', 'detect': 'T3-only', 'tier1': 'T1-only', 'tier2': 'T2-only',
         'gated': 'DRIFT-DDA', 'ssf': 'SSF-style'}
COL = {'static': '#8a8984', 'periodic': '#4a3aa7', 'detect': '#eb6834', 'tier1': '#1baf7a', 'tier2': '#eda100',
       'gated': '#2a78d6', 'ssf': '#e87ba4'}
MK = {'static': 'o', 'periodic': 'v', 'detect': 's', 'tier1': '^', 'tier2': 'D', 'gated': '*', 'ssf': 'P'}
LS = {'static': ':', 'periodic': '-.', 'detect': '--', 'tier1': '-', 'tier2': '-', 'gated': '-', 'ssf': ':'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DSNAME = {'unsw': 'UNSW-NB15 (real)', 'cic17': 'CICIDS2017 (real)', 'ton': 'NF-ToN-IoT-v3 (held out)'}

plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.6,
                     'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})

R = json.load(open(os.path.join(RES, 'results_v7.json')))

def m(row, key):
    v = row.get(key, {}); return v.get('mean', np.nan), v.get('ci', np.nan)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig); print('wrote', name)

def roll(x, k=5):
    x = np.asarray(x, float); out = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - k + 1):i + 1]; w = w[np.isfinite(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out

# ---------------------------------------------------------------- Fig 2: real-trace timelines
def fig_timelines():
    sets = [('real', 'unsw'), ('real', 'cic17'), ('heldout', 'ton')]
    sets = [s for s in sets if load(s[0], '%s_static' % s[1])]
    if not sets:
        return
    fig, axes = plt.subplots(2, len(sets), figsize=(3.4 * len(sets), 4.4), squeeze=False)
    for j, (exp, ds) in enumerate(sets):
        for p in ['static', 'tier1', 'detect', 'gated']:
            runs = load(exp, '%s_%s' % (ds, p)).get('%s_%s' % (ds, p), {})
            if not runs:
                continue
            fpr = np.nanmean([window_fpr(r) for r in runs.values()], 0)
            rec = []
            for r in runs.values():
                tp = np.asarray(r['tp'], float); fn = np.asarray(r['fn'], float)
                rr = np.where(tp + fn >= 50, tp / np.maximum(1, tp + fn), np.nan); rec.append(rr)
            rec = np.nanmean(rec, 0)
            x = np.arange(len(fpr))
            axes[0, j].plot(x, roll(fpr), color=COL[p], ls=LS[p], label=SHORT[p], lw=1.4 if p != 'gated' else 1.9)
            axes[1, j].plot(x, roll(rec), color=COL[p], ls=LS[p], lw=1.4 if p != 'gated' else 1.9)
            dps = next(iter(runs.values())).get('drift_points', [])
        for a in axes[:, j]:
            for d in dps:
                a.axvline(d, color=INK2, lw=0.8, ls='--')
        axes[0, j].set_title(DSNAME[ds], fontsize=9, color=INK)
        axes[0, j].set_ylim(0, 1); axes[1, j].set_ylim(0, 1.02)
        axes[1, j].set_xlabel('window (4096 flows, timestamp order)')
    axes[0, 0].set_ylabel('false-positive rate\n(5-window mean)'); axes[1, 0].set_ylabel('recall\n(windows with ≥50 attacks)')
    axes[0, 0].legend(ncol=2, fontsize=7.5, loc='upper left')
    fig.tight_layout(); save(fig, 'fig2_real_timelines')

# ---------------------------------------------------------------- Fig 3: main real-trace comparison
def fig_real_bars():
    rows = []
    for ds, sec in [('unsw', 'real'), ('cic17', 'real'), ('ton', 'heldout')]:
        tab = R.get(sec) or {}
        if any(t.startswith(ds + '_') for t in tab):
            rows.append((ds, tab))
    if not rows:
        return
    pols = ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated']
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    w = 0.8 / len(pols)
    for i, (ds, tab) in enumerate(rows):
        for k, p in enumerate(pols):
            row = tab.get('%s_%s' % (ds, p))
            if row is None:
                continue
            for a, key in zip(axes[:2], ['all.f1', 'all.f1_s1']):
                mu, ci = m(row, key)
                a.bar(i + (k - len(pols) / 2 + 0.5) * w, mu, w * 0.92, yerr=ci, color=COL[p], label=SHORT[p] if i == 0 else None,
                      error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2))
            mu, ci = m(row, 'steps')
            x = i + (k - len(pols) / 2 + 0.5) * w
            if mu > 0:
                axes[2].bar(x, mu, w * 0.92, color=COL[p])
            else:
                axes[2].text(x, 1.3, '0', ha='center', va='bottom', fontsize=7, color=INK2)
    for a, t in zip(axes, ['final F1 (with reputation)', 'Stage-I F1 (model only)', 'adaptation cost (optimizer steps, log)']):
        a.set_xticks(range(len(rows))); a.set_xticklabels([DSNAME[d].split(' (')[0] for d, _ in rows], fontsize=7.5)
        a.set_title(t, fontsize=9)
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1); axes[2].set_yscale('log'); axes[2].set_ylim(1, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(pols), fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'fig3_real_comparison')

# ---------------------------------------------------------------- Fig 4: regime map (synthetic)
def fig_regimes():
    tab = R.get('synth')
    if not tab:
        return
    pols = ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated']
    DO = ['none', 'abrupt', 'gradual', 'recurring', 'novel']
    scen = sorted({t.rsplit('_', 1)[0] for t in tab}, key=lambda s: (s.split('_')[0], s.split('_')[2], DO.index(s.split('_')[1])))
    for metric, name, ttl in [('post.f1', 'fig4_regimes_final', 'post-drift final F1'),
                              ('post.f1_s1', 'fig4b_regimes_stage1', 'post-drift Stage-I F1')]:
        M = np.array([[m(tab.get('%s_%s' % (s, p), {}), metric)[0] for p in pols] for s in scen])
        best = np.nanmax(M, 1, keepdims=True); D = best - M
        fig, ax = plt.subplots(figsize=(5.6, 0.26 * len(scen) + 1.2))
        im = ax.imshow(D, cmap='Blues', vmin=0, vmax=max(0.2, np.nanmax(D)), aspect='auto')
        for i in range(len(scen)):
            for j in range(len(pols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=6.5,
                            color='white' if D[i, j] > 0.12 else INK)
        ax.set_xticks(range(len(pols))); ax.set_xticklabels([SHORT[p] for p in pols], fontsize=7.5)
        ax.set_yticks(range(len(scen))); ax.set_yticklabels([s.replace('_rot0', ', fixed IPs').replace('_rot1', ', rotating IPs').replace('_', ' ') for s in scen], fontsize=7)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('gap to best policy in row', fontsize=7.5)
        ax.set_title(ttl, fontsize=9)
        fig.tight_layout(); save(fig, name)

# ---------------------------------------------------------------- Fig 5: address rotation
def fig_rotation():
    tabR, tabE = R.get('real') or {}, R.get('rotation') or {}
    if not tabE:
        return
    pols = ['static', 'detect', 'tier1', 'tier2', 'gated']
    tabH, tabP = R.get('heldout') or {}, R.get('posthoc') or {}
    dss = [d for d in ['unsw', 'cic17'] if any(t.startswith(d) for t in tabE)] + (['ton'] if tabH else [])
    fig, axes = plt.subplots(1, len(dss), figsize=(3.3 * len(dss), 2.9), squeeze=False)
    for j, ds in enumerate(dss):
        a = axes[0, j]
        for p in pols + ['gated-norep', 'gated-fresh']:
            xs, ys, es = [], [], []
            for rot in [0.0, 0.5, 1.0]:
                if p == 'gated-fresh':
                    row = tabP.get('%s_rot%.1f_gated-fresh' % (ds, rot))
                elif ds == 'ton':
                    row = tabH.get('ton_%s' % p) if rot == 0 else tabH.get('ton_rot%.1f_%s' % (rot, p))
                else:
                    row = tabR.get('%s_%s' % (ds, p)) if rot == 0 else tabE.get('%s_rot%.1f_%s' % (ds, rot, p))
                if row:
                    mu, ci = m(row, 'all.f1'); xs.append(rot); ys.append(mu); es.append(ci)
            if not xs:
                continue
            if p == 'gated-norep':
                a.errorbar(xs, ys, yerr=es, color=COL['gated'], marker='x', ls=':', lw=1.1, label='DRIFT-DDA w/o reputation', ms=5, capsize=2)
            elif p == 'gated-fresh':
                a.errorbar(xs, ys, yerr=es, color=COL['gated'], marker='o', mfc='white', ls='--', lw=1.1,
                           label='DRIFT-DDA + fresh-source guard (post hoc)', ms=5, capsize=2)
            else:
                a.errorbar(xs, ys, yerr=es, color=COL[p], marker=MK[p], ls=LS[p], label=SHORT[p], ms=5, capsize=2)
        a.set_title(DSNAME[ds], fontsize=9); a.set_xlabel('share of attack flows from fresh addresses'); a.set_ylim(0, 1)
        a.set_xticks([0, 0.5, 1.0])
    axes[0, 0].set_ylabel('final F1 (whole test stream)')
    h, l = [], []
    for a in axes[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, ncol=4, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.1, 1, 1)); save(fig, 'fig5_rotation')

# ---------------------------------------------------------------- Fig 6: quality-cost frontier
def fig_frontier():
    tab = R.get('frontier')
    if not tab:
        return
    sets = [s for s in ['hard_abrupt_rot1', 'unsw'] if any(t.startswith(s + '_') for t in tab)]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 3.0), squeeze=False)
    for j, s in enumerate(sets):
        a = axes[0, j]; key = 'all.f1' if s == 'unsw' else 'post.f1'
        groups = {}
        for t, row in tab.items():
            if not t.startswith(s + '_'):
                continue
            rest = t[len(s) + 1:]; p = rest.split('_')[0]
            groups.setdefault(p, []).append((m(row, 'steps')[0], m(row, key)[0], m(row, key)[1], rest))
        for p, pts in groups.items():
            pts.sort()
            a.errorbar([max(1, x[0]) for x in pts], [x[1] for x in pts], yerr=[x[2] for x in pts], color=COL.get(p, INK2),
                       marker=MK.get(p, 'o'), ls=LS.get(p, '-'), label=SHORT.get(p, p), ms=5, capsize=2, lw=1.2)
        a.set_xscale('log'); a.set_xlabel('adaptation cost (optimizer steps, log)')
        a.set_ylabel('final F1' + (' (post-drift)' if s != 'unsw' else '')); a.set_title({'unsw': DSNAME['unsw'], 'hard_abrupt_rot1': 'synthetic hard, rotating IPs'}[s], fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); save(fig, 'fig6_frontier')

# ---------------------------------------------------------------- Fig 7: severity sweep
def fig_severity():
    tab = R.get('severity')
    if not tab:
        return
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7))
    for j, (diff, rot) in enumerate([('easy', 0), ('easy', 1), ('hard', 0), ('hard', 1)]):
        a = axes[j]
        for p in ['static', 'detect', 'tier1', 'gated']:
            xs, ys, es = [], [], []
            for mag in [0.0, 0.2, 0.4, 0.6, 1.0]:
                row = tab.get('%s_m%.1f_rot%d_%s' % (diff, mag, rot, p))
                if row:
                    xs.append(m(row, 'wdist')[0]); mu, ci = m(row, 'all.f1'); ys.append(mu); es.append(ci)
            a.errorbar(xs, ys, yerr=es, color=COL[p], marker=MK[p], ls=LS[p], label=SHORT[p], ms=4.5, capsize=2)
        a.set_title('%s, %s IPs' % (diff, 'rotating' if rot else 'fixed'), fontsize=9); a.set_ylim(0, 1)
        a.set_xlabel('measured shift (mean 1-D Wasserstein)')
    axes[0].set_ylabel('final F1 (whole stream)'); axes[0].legend(fontsize=7)
    fig.tight_layout(); save(fig, 'fig7_severity')

# ---------------------------------------------------------------- Fig 8: ablation
ABL_ORDER = ['full', 'no_gating', 'no_escalation', 'no_cooldown', 'no_replay', 'no_reputation', 'no_majority',
             'no_surge', 'no_defer', 'rule_base', 'tau_v6', 'severity_v6', 'v6controller', 'v6controller-detect']
ABL_LABEL = {'full': 'full DRIFT-DDA', 'no_gating': 'no gating (always Tier 3)', 'no_escalation': 'no escalation',
             'no_cooldown': 'no cooldown', 'no_replay': 'no replay', 'no_reputation': 'no reputation in adaptation',
             'no_majority': 'count-only blacklist (GAN-DDA)', 'no_surge': 'no surge quarantine', 'no_defer': 'no representativeness guard',
             'rule_base': 'no widening (v6 retrain rule)',
             'tau_v6': 'v6 threshold reset', 'severity_v6': 'v6 rate severity', 'v6controller': 'v6 controller (all v6 rules)',
             'v6controller-detect': 'v6 detect baseline'}
def fig_ablation():
    tab = R.get('ablation')
    if not tab:
        return
    sets = [s for s in ['hard_abrupt_rot1', 'hard_recurring_rot1', 'unsw', 'cic17'] if ('%s_full' % s) in tab]
    fig, axes = plt.subplots(1, len(sets), figsize=(2.9 * len(sets), 3.6), sharey=True, squeeze=False)
    for j, s in enumerate(sets):
        a = axes[0, j]; key = 'all.f1' if s in ('unsw', 'cic17') else 'post.f1'
        vals = [(ABL_LABEL[v], *m(tab.get('%s_%s' % (s, v), {}), key)) for v in ABL_ORDER]
        y = np.arange(len(vals))
        a.barh(y, [v[1] for v in vals], xerr=[v[2] for v in vals], color=['#2a78d6' if i == 0 else '#9db9de' for i in range(len(vals))],
               error_kw=dict(lw=0.8, capsize=1.5, ecolor=INK2), height=0.7)
        a.axvline(vals[0][1], color='#2a78d6', lw=0.8, ls='--')
        a.set_yticks(y); a.set_yticklabels([v[0] for v in vals], fontsize=7); a.invert_yaxis(); a.set_xlim(0, 1)
        a.set_title({'unsw': 'UNSW-NB15', 'cic17': 'CICIDS2017', 'hard_abrupt_rot1': 'syn. hard abrupt, rot.',
                     'hard_recurring_rot1': 'syn. hard recurring, rot.'}[s], fontsize=8.5)
        a.set_xlabel(key.replace('all.', 'final ').replace('post.', 'post-drift '))
    fig.tight_layout(); save(fig, 'fig8_ablation')

# ---------------------------------------------------------------- Fig 9: wall-clock cost
def fig_cost():
    C = R.get('cost')
    if not C:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    style = {'cpu1': ('1 CPU thread', '-'), 'cpu6': ('6 CPU threads', '--'), 'gpu': ('GPU (GTX 1060)', ':')}
    for dev, res in C.items():
        rows = res['rows']; wds = [r['width'] for r in rows]
        lab, ls = style.get(dev, (dev, '-'))
        for tier, col, mk in [('tier2_s', '#eda100', 'D'), ('tier3_s', '#eb6834', 's')]:
            axes[0].plot(wds, [r[tier] for r in rows], color=col, ls=ls, marker=mk, ms=4,
                         label='%s, %s' % ('Tier 2' if tier == 'tier2_s' else 'Tier 3', lab))
        # exposure: flows scored by the stale model during one Tier-3 update at 10k flows/s
        axes[1].plot(wds, [r['tier3_s'] * 1e4 for r in rows], color='#eb6834', ls=ls, marker='s', ms=4, label=lab)
    axes[0].set_xscale('log', base=2); axes[0].set_yscale('log'); axes[0].set_xlabel('hidden width'); axes[0].set_ylabel('wall-clock per update (s)')
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log'); axes[1].set_xlabel('hidden width'); axes[1].set_ylabel('flows scored by stale model\nduring one Tier-3 update (10k flows/s)')
    axes[0].legend(fontsize=6.5); axes[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, 'fig9_cost')

# ---------------------------------------------------------------- Fig 10: poisoning
def fig_poison():
    tab = R.get('poison')
    if not tab:
        return
    pols = ['static', 'tier1', 'detect', 'gated', 'gated-probation']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, p in enumerate(pols):
        c = tab.get('clean_%s' % p); q = tab.get('pois_%s' % p)
        if not c or not q:
            continue
        col = COL.get(p.split('-')[0], INK2)
        a0, e0 = m(c, 'target_recall_after'); a1, e1 = m(q, 'target_recall_after')
        axes[0].plot([a0, a1], [k, k], color=col, lw=1.5); axes[0].plot(a0, k, 'o', color=col, mfc='white', ms=6)
        axes[0].plot(a1, k, 'o', color=col, ms=6)
        pz, pe = m(q, 'poison_in'); axes[1].barh(k, 0 if not np.isfinite(pz) else pz, xerr=pe, color=col, height=0.6)
    for a in axes:
        a.set_yticks(range(len(pols))); a.set_yticklabels([SHORT.get(p, p).replace('gated-probation', 'DRIFT-DDA + probation') if p in SHORT else 'DRIFT-DDA + probation' for p in pols], fontsize=7.5)
        a.invert_yaxis()
    axes[0].set_xlabel('recall of the targeted attack after the campaign\n(open = clean stream, filled = poisoned)'); axes[0].set_xlim(0, 1.02)
    axes[1].set_xlabel('poison share of retrain sets'); axes[1].set_yticklabels([])
    fig.tight_layout(); save(fig, 'fig10_poison')

if __name__ == '__main__':
    for f in [fig_timelines, fig_real_bars, fig_regimes, fig_rotation, fig_frontier, fig_severity, fig_ablation, fig_cost, fig_poison]:
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print('FAILED', f.__name__, e)
