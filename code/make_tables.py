"""LaTeX tables + numbers.tex macros from results/results_v7.json and results/data_table.json."""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
OUT = os.environ.get('DDA_TABLES_DIR', os.path.join(HERE, '..', 'results', 'tables'))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(RES, 'results_v7.json')))
D = json.load(open(os.path.join(RES, 'data_table.json'))) if os.path.exists(os.path.join(RES, 'data_table.json')) else {}
MACROS = {}

DIGITS = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
          '8': 'eight', '9': 'nine'}

def mac(name, val):
    name = re.sub(r'[^A-Za-z]', '', ''.join(DIGITS.get(ch, ch) for ch in name.replace('cic17', 'cic')))
    MACROS[name] = val

def g(row, key, f='%.3f', ci=True):
    v = (row or {}).get(key)
    if not v or v.get('mean') is None or not np.isfinite(v['mean']):
        return '--'
    s = f % v['mean']
    if ci and v.get('ci') and np.isfinite(v['ci']) and v['ci'] > 0:
        s += r'{\,\scriptsize$\pm$' + (f % v['ci']) + '}'
    return s

def gi(row, key):
    v = (row or {}).get(key)
    if not v or not np.isfinite(v.get('mean', np.nan)):
        return '--'
    return '{:,}'.format(int(round(v['mean'])))

def bold_best(cells, vals, higher=True):
    ok = [i for i, v in enumerate(vals) if np.isfinite(v)]
    if not ok:
        return cells
    b = max(ok, key=lambda i: vals[i]) if higher else min(ok, key=lambda i: vals[i])
    cells[b] = r'\textbf{' + cells[b] + '}'
    return cells

def write(name, s):
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write(s); print('wrote', name)

NAME = {'static': 'static', 'periodic': 'periodic', 'detect': 'Tier-3 only', 'tier1': 'Tier-1 only',
        'tier2': 'Tier-2 only', 'gated': r'\drift{}', 'ssf': 'SSF-style$^\dagger$'}
DS = {'unsw': 'UNSW-NB15', 'cic17': 'CICIDS2017', 'ton': 'NF-ToN-IoT-v3'}

# ------------------------------------------------------------------ dataset table
def tab_data():
    if not D:
        return
    rows = []
    for k in ['unsw', 'cic17', 'ton']:
        if k not in D:
            continue
        d = D[k]
        rows.append('%s & %s & %s & %.3f & %d / %d & %.2f & %.2f & %s \\\\' % (
            DS[k], '{:,}'.format(d['test_flows']), d['test_windows'], d['test_attack_share'], d['attacker_src'],
            d['benign_src'], d['top1_attack_share'], d['max_window_attack_share'], d['test_start'][:10] + ' -- ' + d['test_end'][:10]))
    s = (r'\begin{tabular}{@{}lrrrrrrl@{}}' '\n' r'\toprule' '\n'
         r'Trace & test flows & windows & attack share & attacker / benign src. & top-1 src. share & max window share & period \\' '\n'
         r'\midrule' '\n' + '\n'.join(rows) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_data.tex', s)

# ------------------------------------------------------------------ real-trace main table
def tab_real():
    blocks = []
    for ds, sec in [('unsw', 'real'), ('cic17', 'real'), ('ton', 'heldout')]:
        tab = R.get(sec) or {}
        pols = [p for p in ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated', 'ssf'] if '%s_%s' % (ds, p) in tab]
        if not pols:
            continue
        lines = []
        cols = ['all.f1', 'all.prec', 'all.rec', 'all.fpr', 'all.f1_s1', 'all.auroc']
        vals = {c: [tab['%s_%s' % (ds, p)].get(c, {}).get('mean', np.nan) if p != 'ssf' else np.nan for p in pols] for c in cols}
        for i, p in enumerate(pols):
            row = tab['%s_%s' % (ds, p)]
            cells = [g(row, c, ci=(c == 'all.f1')) for c in cols]
            for j, c in enumerate(cols):
                best = np.nanmax(vals[c]) if c != 'all.fpr' else np.nanmin(vals[c])
                if p != 'ssf' and np.isfinite(vals[c][i]) and abs(vals[c][i] - best) < 1e-12:
                    cells[j] = r'\textbf{' + cells[j] + '}'
            t = '%s/%s/%s' % (gi(row, 't1'), gi(row, 't2'), gi(row, 't3')) if p not in ('static', 'periodic') else ('--' if p == 'static' else '0/0/%s' % gi(row, 't3'))
            lines.append('%s & %s & %s & %s \\\\' % (NAME[p], ' & '.join(cells), gi(row, 'steps'), t))
            vals_f1 = [x for x in row.get('all.f1', {}).get('values', []) if x == x]
            if vals_f1:
                mac('%s%sfonemed' % (ds, p), '%.3f' % float(np.median(vals_f1)))
                mac('%s%sfonemin' % (ds, p), '%.3f' % float(np.min(vals_f1)))
                mac('%s%snlow' % (ds, p), str(int(sum(1 for x in vals_f1 if x < np.median(vals_f1) - 0.1))))
            w_ = row.get('wall', {}).get('mean')
            if w_ is not None and w_ == w_:
                mac('%s%swall' % (ds, p), '%.1f' % w_)
            for c in cols + ['steps']:
                v = row.get(c, {}).get('mean')
                if v is not None:
                    mac('%s%s%s' % (ds, p, c.replace('all.', '').replace('f1_s1', 'stageone').replace('f1', 'fone')),
                        ('%.3f' % v) if c != 'steps' else '{:,}'.format(int(round(v))))
        hdr = r'\multicolumn{9}{@{}l}{\emph{%s}%s} \\' % (DS[ds], ' (held out)' if ds == 'ton' else '')
        blocks.append(hdr + '\n' + '\n'.join(lines))
    if not blocks:
        return
    s = (r'\begin{tabular}{@{}lcccccccc@{}}' '\n' r'\toprule' '\n'
         r'Policy & $F_1$ & prec. & recall & FPR & St-I $F_1$ & AUROC & steps & T1/T2/T3 \\' '\n' r'\midrule' '\n'
         + '\n\\midrule\n'.join(blocks) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_real.tex', s)
    for ds, sec in [('unsw', 'real'), ('cic17', 'real'), ('ton', 'heldout')]:
        tab = R.get(sec) or {}
        a_, b_ = tab.get('%s_detect' % ds), tab.get('%s_gated' % ds)
        if a_ and b_ and b_['steps']['mean'] > 0:
            mac('%sstepratio' % ds, '%.1f' % (a_['steps']['mean'] / b_['steps']['mean']))
    # tests
    tt = dict(R.get('real_tests', {})); tt['ton'] = R.get('heldout_tests', {})
    for ds, t in tt.items():
        for metric, d in t.items():
            for other, res in d.items():
                pol = other.split('_', 1)[1]
                mac('p%s%svs%s' % (metric.replace('all.', '').replace('f1_s1', 'stageone').replace('f1', 'fone'), ds, pol),
                    '%.3f' % res['wilcoxon_p'] if res.get('wilcoxon_p') == res.get('wilcoxon_p') else '--')

def tab_days():
    lines = []
    for ds, sec in [('unsw', 'real'), ('cic17', 'real'), ('ton', 'heldout')]:
        tab = R.get(sec) or {}
        for p in ['static', 'detect', 'tier1', 'tier2', 'gated']:
            row = tab.get('%s_%s' % (ds, p))
            if not row or 'days' not in row:
                continue
            days = sorted(row['days'], key=int)
            for k, d in enumerate(days):
                for met in ['rec', 'fpr', 'f1']:
                    v = row['days'][d].get(met, {}).get('mean')
                    if v is not None and v == v:
                        mac('%s%sday%d%s' % (ds, p, k + 1, met.replace('f1', 'fone')), '%.3f' % v)
            cells = ['%s/%s' % ('--' if (ds == 'unsw' and k == 0) else g(row['days'][d], 'f1', ci=False),
                                g(row['days'][d], 'fpr', ci=False)) for k, d in enumerate(days)]
            lines.append('%s & %s & %s \\\\' % (DS[ds] if p == 'static' else '', NAME[p], ' & '.join(cells + [''] * (6 - len(cells)))))
        if any(('%s_static' % ds) in tab for _ in [0]):
            lines.append(r'\midrule')
    if lines:
        s = (r'\begin{tabular}{@{}llcccccc@{}}' '\n' r'\toprule' '\n'
             r'Trace & Policy & \multicolumn{6}{c}{$F_1$ / FPR per test day} \\' '\n' r'\midrule' '\n' + '\n'.join(lines[:-1]) +
             '\n' r'\bottomrule' '\n' r'\end{tabular}')
        write('tab_days.tex', s)

# ------------------------------------------------------------------ rotation
def tab_rotation():
    tabR, tabE = R.get('real') or {}, R.get('rotation') or {}
    if not tabE:
        return
    lines = []
    for ds in ['unsw', 'cic17']:
        for p in ['static', 'detect', 'tier1', 'tier2', 'gated', 'gated-norep']:
            cells = []
            for rot in [0.0, 0.5, 1.0]:
                if rot == 0:
                    row = tabR.get('%s_%s' % (ds, p))
                else:
                    row = tabE.get('%s_rot%.1f_%s' % (ds, rot, p))
                cells += [g(row, 'all.f1', ci=False), g(row, 'all.f1_s1', ci=False)]
                if row and rot > 0:
                    lab = 'half' if rot == 0.5 else 'full'
                    mac('rot%s%s%sfone' % (lab, ds, p.replace('-', '')), g(row, 'all.f1', ci=False))
                    mac('rot%s%s%scontam' % (lab, ds, p.replace('-', '')), g(row, 'contam', ci=False))
                    mac('rot%s%s%srec' % (lab, ds, p.replace('-', '')), g(row, 'all.rec', ci=False))
            lines.append('%s & %s & %s \\\\' % (DS[ds] if p == 'static' else '', NAME.get(p, r'\drift{} w/o reputation'), ' & '.join(cells)))
        lines.append(r'\midrule')
    tabH = R.get('heldout') or {}
    for p in ['static', 'detect', 'tier1', 'tier2', 'gated']:
        a = tabH.get('ton_%s' % p); b = tabH.get('ton_rot1.0_%s' % p)
        if a or b:
            lines.append('%s & %s & %s & %s & -- & -- & %s & %s \\\\' % ('NF-ToN-IoT-v3' if p == 'static' else '', NAME[p],
                         g(a, 'all.f1', ci=False), g(a, 'all.f1_s1', ci=False), g(b, 'all.f1', ci=False), g(b, 'all.f1_s1', ci=False)))
            if b:
                mac('rotfullton%sfone' % p, g(b, 'all.f1', ci=False)); mac('rotfullton%sfpr' % p, g(b, 'all.fpr', ci=False))
    while lines and lines[-1] == r'\midrule':
        lines.pop()
    s = (r'\begin{tabular}{@{}llcccccc@{}}' '\n' r'\toprule' '\n'
         r' & & \multicolumn{2}{c}{$p=0$ (real addresses)} & \multicolumn{2}{c}{$p=0.5$} & \multicolumn{2}{c}{$p=1$ (all fresh)} \\' '\n'
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}' '\n'
         r'Trace & Policy & $F_1$ & St-I $F_1$ & $F_1$ & St-I $F_1$ & $F_1$ & St-I $F_1$ \\' '\n' r'\midrule' '\n'
         + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_rotation.tex', s)

# ------------------------------------------------------------------ synthetic summary (selected scenarios)
def tab_synth():
    tab = R.get('synth')
    if not tab:
        return
    pols = ['static', 'periodic', 'detect', 'tier1', 'tier2', 'gated']
    lines = []
    scen = sorted({t.rsplit('_', 1)[0] for t in tab}, key=lambda s: (s.split('_')[0], s.split('_')[2], ['none', 'abrupt', 'gradual', 'recurring', 'novel'].index(s.split('_')[1])))
    for sc in scen:
        diff, drift, rot = sc.split('_')
        f = [tab.get('%s_%s' % (sc, p), {}).get('post.f1', {}).get('mean', np.nan) for p in pols]
        s1 = [tab.get('%s_%s' % (sc, p), {}).get('post.f1_s1', {}).get('mean', np.nan) for p in pols]
        for p in pols + ['ssf']:
            row = tab.get('%s_%s' % (sc, p), {})
            nr = row.get('novel_recall', {}).get('mean')
            if nr is not None and nr == nr:
                mac('novrec%s%s%s' % (diff, 'rot' if rot == 'rot1' else 'fix', p), '%.2f' % nr)
            for key, nm in [('post.f1', 'fone'), ('post.f1_s1', 'stageone'), ('steps', 'steps')]:
                v = row.get(key, {}).get('mean')
                if v is not None and v == v:
                    mac('syn%s%s%s%s%s' % (diff, drift, 'rot' if rot == 'rot1' else 'fix', p, nm),
                        '%.2f' % v if nm != 'steps' else '{:,}'.format(int(round(v))))
        cells = ['%.2f/%.2f' % (a, b) for a, b in zip(f, s1)]
        cells = bold_best(cells, list(f))
        st = tab.get('%s_detect' % sc, {}).get('steps', {}).get('mean', np.nan); sg = tab.get('%s_gated' % sc, {}).get('steps', {}).get('mean', np.nan)
        ratio = st / sg if sg and sg > 0 else np.nan
        lines.append('%s & %s & %s & %s & %s \\\\' % (diff, drift, 'rot.' if rot == 'rot1' else 'fixed', ' & '.join(cells),
                                                      '%.1f' % ratio if np.isfinite(ratio) else '--'))
    # regime summaries over the four drift types (none excluded)
    for diff in ['easy', 'hard']:
        for rot, rl in [('rot0', 'fix'), ('rot1', 'rot')]:
            for p in pols + ['ssf']:
                for key, nm in [('post.f1', 'fone'), ('post.f1_s1', 'stageone'), ('steps', 'steps')]:
                    v = [tab.get('%s_%s_%s_%s' % (diff, d, rot, p), {}).get(key, {}).get('mean', np.nan)
                         for d in ['abrupt', 'gradual', 'recurring', 'novel']]
                    v = [x for x in v if x == x]
                    if v:
                        mac('reg%s%s%s%s' % (diff, rl, p, nm), '%.2f' % np.mean(v) if nm != 'steps' else '{:,}'.format(int(round(np.mean(v)))))
            a = [tab.get('%s_%s_%s_detect' % (diff, d, rot), {}).get('steps', {}).get('mean', np.nan) for d in ['abrupt', 'gradual', 'recurring', 'novel']]
            b = [tab.get('%s_%s_%s_gated' % (diff, d, rot), {}).get('steps', {}).get('mean', np.nan) for d in ['abrupt', 'gradual', 'recurring', 'novel']]
            if all(x == x for x in a + b):
                mac('reg%s%sstepratio' % (diff, rl), '%.1f' % (np.mean(a) / np.mean(b)))
    s = (r'\begin{tabular}{@{}lll' + 'c' * len(pols) + r'c@{}}' '\n' r'\toprule' '\n'
         r'Diff. & Drift & IPs & ' + ' & '.join(NAME[p] for p in pols) + r' & $\frac{\text{T3-only steps}}{\text{\drift{} steps}}$ \\' '\n'
         r'\midrule' '\n' + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_synth.tex', s)

# ------------------------------------------------------------------ ablation
ABL = ['full', 'no_gating', 'no_escalation', 'no_cooldown', 'no_replay', 'no_reputation', 'no_majority', 'no_surge', 'no_defer',
       'rule_base', 'tau_v6', 'severity_v6', 'v6controller', 'v6controller-detect']
ABLN = {'full': r'full \drift{}', 'no_gating': 'no gating (always Tier 3)', 'no_escalation': 'no escalation',
        'no_cooldown': 'no cooldown', 'no_replay': 'no replay', 'no_reputation': 'no reputation in adaptation',
        'no_majority': 'count-only blacklist (GAN-DDA)', 'no_surge': 'no surge quarantine', 'no_defer': 'no representativeness guard',
        'rule_base': 'no widening (v6 retrain rule)',
        'tau_v6': 'v6 threshold reset', 'severity_v6': 'v6 rate-based severity',
        'v6controller': r'v6 \drift{} (all v6 rules)', 'v6controller-detect': 'v6 Tier-3-only baseline'}
def tab_ablation():
    tab = R.get('ablation')
    if not tab:
        return
    sets = [s for s in ['hard_abrupt_rot1', 'hard_recurring_rot1', 'unsw', 'cic17'] if '%s_full' % s in tab]
    lines = []
    for a in ABL:
        cells = []
        for s in sets:
            row = tab.get('%s_%s' % (s, a)); key = 'all.f1' if s in ('unsw', 'cic17') else 'post.f1'
            cells += [g(row, key, ci=False), gi(row, 'steps')]
            if row:
                mac('abl%s%sfone' % (s, a), g(row, key, ci=False)); mac('abl%s%ssteps' % (s, a), gi(row, 'steps'))
        lines.append('%s & %s \\\\' % (ABLN[a], ' & '.join(cells)))
    hdr = ' & '.join(r'\multicolumn{2}{c}{%s}' % {'hard_abrupt_rot1': 'hard abrupt, rot.', 'hard_recurring_rot1': 'hard recurring, rot.',
                                                   'unsw': 'UNSW-NB15', 'cic17': 'CICIDS2017'}[s] for s in sets)
    sub = ' & '.join(['$F_1$ & steps'] * len(sets))
    s = (r'\begin{tabular}{@{}l' + 'cc' * len(sets) + r'@{}}' '\n' r'\toprule' '\n' r'Configuration & ' + hdr + r' \\' '\n'
         r' & ' + sub + r' \\' '\n' r'\midrule' '\n' + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_ablation.tex', s)

# ------------------------------------------------------------------ poison / injected / sensitivity / tuning / cost
def tab_poison():
    tab = R.get('poison')
    if not tab:
        return
    lines = []
    for p in ['static', 'tier1', 'detect', 'gated', 'gated-probation']:
        c = tab.get('clean_%s' % p); q = tab.get('pois_%s' % p)
        if not c or not q:
            continue
        lines.append('%s & %s & %s & %s & %s & %s \\\\' % (NAME.get(p, r'\drift{} + probation'), g(c, 'target_recall_after'),
                     g(q, 'target_recall_after'), g(q, 'poison_in'), g(c, 'post.f1', ci=False), g(q, 'post.f1', ci=False)))
        mac('pois%sclean' % p.replace('-', ''), g(c, 'target_recall_after', ci=False)); mac('pois%spois' % p.replace('-', ''), g(q, 'target_recall_after', ci=False))
        mac('pois%sin' % p.replace('-', ''), g(q, 'poison_in', ci=False))
    s = (r'\begin{tabular}{@{}lccccc@{}}' '\n' r'\toprule' '\n'
         r'Policy & target recall (clean) & target recall (poisoned) & poison share of $\mathcal R_t$ & post $F_1$ clean & post $F_1$ poisoned \\' '\n'
         r'\midrule' '\n' + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_poison.tex', s)

def tab_injected():
    tab = R.get('injected')
    if not tab:
        return
    lines = []
    for p, n in [('static', 'static'), ('tier1', 'Tier-1 only'), ('detect', 'Tier-3 only'), ('gated-base', r'\drift{}, no widening'),
                 ('gated', r'\drift{}'), ('gated-norep', r'\drift{}, no reputation')]:
        row = tab.get(p)
        if not row:
            continue
        lines.append('%s & %s & %s & %s & %s & %s \\\\' % (n, g(row, 'pre.f1'), g(row, 'post.f1'), g(row, 'contam'), g(row, 'cover_loss'), gi(row, 'steps')))
        mac('inj%spost' % p.replace('-', ''), g(row, 'post.f1', ci=False)); mac('inj%scontam' % p.replace('-', ''), g(row, 'contam', ci=False))
        mac('inj%scover' % p.replace('-', ''), g(row, 'cover_loss', ci=False))
    ws = tab.get('gated', {}).get('wdist_std', {}).get('mean')
    if ws:
        mac('injwdist', '%.2f' % ws)
    s = (r'\begin{tabular}{@{}lccccc@{}}' '\n' r'\toprule' '\n'
         r'Policy & pre-drift $F_1$ & post-drift $F_1$ & contamination & coverage loss & steps \\' '\n' r'\midrule' '\n'
         + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_injected.tex', s)

def tab_sensitivity():
    tab = R.get('sensitivity')
    if not tab:
        return
    sets = [s for s in ['hard_abrupt_rot1', 'unsw', 'cic17'] if any(t.startswith(s + '_') for t in tab)]
    lines = []
    params = ['ph_lambda', 'rate_t2', 'rate_t3', 'cooldown', 'pi_w', 'clean_frac', 'verify_m', 'L']
    pn = {'ph_lambda': r'$\lambda$', 'rate_t2': r'$\rho_2$', 'rate_t3': r'$\rho_3$', 'cooldown': r'$\kappa$', 'pi_w': r'$\pi_w$',
          'clean_frac': 'trust share', 'verify_m': '$v$', 'L': '$L$'}
    default = {'ph_lambda': 0.02, 'rate_t2': 1.5, 'rate_t3': 3.0, 'cooldown': 2, 'pi_w': 95, 'clean_frac': 0.5,
               'verify_m': 2, 'L': 2000}
    main = R.get('real') or {}
    def default_row(s, k=3):
        """Frozen-default values on the real traces come from the main runs, restricted to seeds 0..k-1."""
        r = main.get('%s_gated' % s)
        if not r:
            return '--', '--'
        f = r['all.f1']['values'][:k]; st = r['steps']['values'][:k]
        return '%.3f' % np.mean(f), '{:,}'.format(int(round(np.mean(st))))
    for par in params:
        vals = sorted({t.split('=')[1] for t in tab if ('_%s=' % par) in t} | ({str(default[par])} if par == 'L' else set()),
                      key=float)
        for v in vals:
            cells = []
            for s in sets:
                row = tab.get('%s_%s=%s' % (s, par, v)); key = 'all.f1' if s in ('unsw', 'cic17') else 'post.f1'
                if row is None and s in ('unsw', 'cic17') and abs(float(v) - default[par]) < 1e-9:
                    cells += list(default_row(s))
                else:
                    cells += [g(row, key, ci=False), gi(row, 'steps')]
                mac('sens%s%s%s' % ({'hard_abrupt_rot1': 'syn'}.get(s, s), par.replace('_', ''), v.replace('.', 'p')), cells[-2])
            vtxt = v + ('$^\\ast$' if abs(float(v) - default[par]) < 1e-9 else '')
            lines.append('%s & %s & %s \\\\' % (pn[par] if v == vals[0] else '', vtxt, ' & '.join(cells)))
        lines.append(r'\addlinespace[2pt]')
    hdr = ' & '.join(r'\multicolumn{2}{c}{%s}' % {'hard_abrupt_rot1': 'syn. hard abrupt, rot.', 'unsw': 'UNSW-NB15', 'cic17': 'CICIDS2017'}[s] for s in sets)
    s = (r'\begin{tabular}{@{}ll' + 'cc' * len(sets) + r'@{}}' '\n' r'\toprule' '\n' r'Parameter & value & ' + hdr + r' \\' '\n'
         r' & & ' + ' & '.join(['$F_1$ & steps'] * len(sets)) + r' \\' '\n' r'\midrule' '\n' + '\n'.join(lines[:-1]) + '\n'
         r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_sensitivity.tex', s)

def severity_macros():
    tab = R.get('severity')
    if not tab:
        return
    for diff in ['easy', 'hard']:
        for rot in [0, 1]:
            rl = 'rot' if rot else 'fix'
            for mag in [0.0, 0.2, 0.4, 0.6, 1.0]:
                for p in ['static', 'detect', 'tier1', 'gated']:
                    row = tab.get('%s_m%.1f_rot%d_%s' % (diff, mag, rot, p))
                    if not row:
                        continue
                    mac('sev%s%sm%d%sfone' % (diff, rl, int(round(mag * 10)), p), g(row, 'all.f1', ci=False))
                    mac('sev%s%sm%d%ssteps' % (diff, rl, int(round(mag * 10)), p), gi(row, 'steps'))
                    if p == 'gated':
                        mac('sevwdist%s%sm%d' % (diff, rl, int(round(mag * 10))), '%.2f' % row['wdist']['mean'])

def frontier_macros():
    tab = R.get('frontier')
    if not tab:
        return
    for t, row in tab.items():
        s = 'hardrot' if t.startswith('hard_abrupt_rot1_') else ('unsw' if t.startswith('unsw_') else None)
        if s is None:
            continue
        rest = t.split('_', 3)[-1] if s == 'hardrot' else t.split('_', 1)[1]
        key = 'post.f1' if s == 'hardrot' else 'all.f1'
        if rest.startswith(('detect_lam', 'tier1_lam', 'tier2_lam', 'gated_lam')):
            p, lam = rest.split('_lam')
            name = 'fr%s%slam%d' % (s, p, int(round(float(lam) * 100)))
        elif rest.startswith('periodic_k'):
            name = 'fr%speriodick%s' % (s, rest.split('_k')[1])
        else:
            name = 'fr%s%s' % (s, rest)
        mac(name + 'fone', g(row, key, ci=False)); mac(name + 'steps', gi(row, 'steps'))

def posthoc_macros():
    P = R.get('posthoc')
    if not P:
        return
    M, E, H, A, PO = R.get('real', {}), R.get('rotation', {}), R.get('heldout', {}), R.get('ablation', {}), R.get('poison', {})
    def base(ds, rot):
        if ds == 'ton':
            return H.get('ton_gated' if rot == 0 else 'ton_rot%.1f_gated' % rot)
        return M.get('%s_gated' % ds) if rot == 0 else E.get('%s_rot%.1f_gated' % (ds, rot))
    lab = {0.0: 'zero', 0.5: 'half', 1.0: 'full'}
    for ds in ['cic17', 'unsw', 'ton']:
        for rot in [0.0, 0.5, 1.0]:
            r = P.get('%s_rot%.1f_gated-fresh' % (ds, rot)); b = base(ds, rot)
            if r:
                mac('ph%s%sfreshfone' % (ds, lab[rot]), g(r, 'all.f1', ci=False))
                mac('ph%s%sfreshfpr' % (ds, lab[rot]), g(r, 'all.fpr', ci=False))
                mac('ph%s%sfreshcontam' % (ds, lab[rot]), g(r, 'contam', ci=False))
                mac('ph%s%sfreshsteps' % (ds, lab[rot]), gi(r, 'steps'))
            if b:
                v = b['all.f1']['values'][:5]
                mac('ph%s%sbasefone' % (ds, lab[rot]), '%.3f' % np.mean(v))
        r = P.get('%s_rot0.0_gated-trust3' % ds)
        if r:
            mac('ph%strustfone' % ds, g(r, 'all.f1', ci=False))
    for c in ['clean', 'pois']:
        r = P.get('%s_gated-trust3' % c)
        if r:
            mac('ph%strusttarget' % c, g(r, 'target_recall_after', ci=False)); mac('ph%strustpost' % c, g(r, 'post.f1', ci=False))
    r = P.get('hard_abrupt_rot1_gated-trust3'); b = A.get('hard_abrupt_rot1_full')
    if r and b:
        mac('phhardtrustfone', g(r, 'post.f1', ci=False)); mac('phhardbasefone', '%.3f' % np.mean(b['post.f1']['values'][:5]))

def tuning_macros():
    t = R.get('tuning')
    if not t:
        return
    sel = t['selected']
    mac('tunelambda', '%g' % sel['ph_lambda']); mac('tunerhotwo', '%g' % sel['rate_t2']); mac('tunerhothree', '%g' % sel['rate_t3'])
    mac('tunebestfone', '%.3f' % t['best_f1']); mac('tuneselfone', '%.3f' % t['table'][t['selected_key']]['f1'])
    mac('tunencfg', str(len(t['table'])))

def cost_macros():
    C = R.get('cost')
    if not C:
        return
    lines = []
    for dev, res in C.items():
        for r in res['rows']:
            lines.append('%s & %d & %s & %.3f & %.2f & %.2f \\\\' % ({'cpu1': '1 CPU thread', 'cpu6': '6 CPU threads', 'gpu': 'GPU'}.get(dev, dev),
                         r['width'], '{:,}'.format(r['params']), r['tier1_s'], r['tier2_s'], r['tier3_s']))
            mac('cost%sw%dtthree' % (dev, r['width']), '%.2f' % r['tier3_s']); mac('cost%sw%dttwo' % (dev, r['width']), '%.2f' % r['tier2_s'])
            mac('cost%sw%dtone' % (dev, r['width']), '%.3f' % r['tier1_s'])
            mac('cost%sw%dexpo' % (dev, r['width']), '{:,}'.format(int(round(r['tier3_s'] * 1e4))))   # flows at 10k flows/s
    s = (r'\begin{tabular}{@{}lrrrrr@{}}' '\n' r'\toprule' '\n' r'Device & width & parameters & Tier 1 (s) & Tier 2 (s) & Tier 3 (s) \\' '\n'
         r'\midrule' '\n' + '\n'.join(lines) + '\n' r'\bottomrule' '\n' r'\end{tabular}')
    write('tab_cost.tex', s)

if __name__ == '__main__':
    for f in [tab_data, tab_real, tab_days, tab_rotation, tab_synth, tab_ablation, tab_poison, tab_injected, tab_sensitivity, severity_macros, frontier_macros, posthoc_macros,
              tuning_macros, cost_macros]:
        try:
            f()
        except Exception as e:
            import traceback; traceback.print_exc(); print('FAILED', f.__name__)
    with open(os.path.join(OUT, 'numbers.tex'), 'w') as fh:
        fh.write('%% generated by v7/make_tables.py from results_v7.json -- do not edit\n')
        for k, v in sorted(MACROS.items()):
            fh.write('\\newcommand{\\%s}{%s}\n' % (k, v))
    print('numbers.tex: %d macros' % len(MACROS))
