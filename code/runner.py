"""Resumable parallel runner. A job = (exp, data spec, seed, policy, opts, cfg overrides, tag).
Jobs that share (data spec, seed, model config) are grouped so the stream is built and the detector
configured once; configured base models are cached on disk. Each job writes one JSON file
results/raw/<exp>/<tag>__s<seed>.json and is skipped if that file exists."""
import os as _os
_os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU-only (an empty value is deleted on Windows)

import hashlib, json, os, sys, time, traceback
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
MODELS = os.path.join(RES, 'models')

def _h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]

def out_path(job):
    return os.path.join(RES, 'raw', job['exp'], '%s__s%d.json' % (job['tag'], job['seed']))

def build_stream(data, seed):
    from streams import SynthStream, RealStream, InjectedStream
    k = data['kind']
    if k == 'syn':
        return SynthStream(data['spec'], seed)
    if k == 'real':
        return RealStream(data['name'], data.get('window', 4096), seed, rotation=data.get('rotation', 0.0),
                          max_windows=data.get('max_windows'))
    if k == 'inj':
        return InjectedStream(seed, **data.get('spec', {}))
    raise ValueError(k)

def model_family(data):
    """Configuration windows are identical within a family (see streams.py)."""
    if data['kind'] == 'syn':
        sp = data['spec']
        keys = ['difficulty', 'sep_lo', 'sep_hi', 'attack_sigma', 'n_features', 'n_benign_clusters', 'n_attack_types',
                'benign_sigma', 'config_size', 'config_contam']
        return 'syn_' + _h({k: sp.get(k) for k in keys})
    if data['kind'] == 'real':
        return 'real_%s_noid' % data['name']      # _noid: identifier fields removed (streams.IDENTIFIERS)
    return 'real_unsw_noid'      # injected stream shares the UNSW configuration split

MODEL_KEYS = ['latent_dim', 'hidden', 'lr', 'batch', 'n_critic', 'gp_lambda', 'alpha', 'wgan_epochs', 'enc_epochs', 'percentile']

def get_base(stream, data, seed, cfg):
    from core import configure, make_models
    os.makedirs(MODELS, exist_ok=True)
    p = os.path.join(MODELS, '%s_s%d_%s.pt' % (model_family(data), seed, _h({k: cfg[k] for k in MODEL_KEYS})))
    if os.path.exists(p):
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
            prep = ck['prep']; models = make_models(len(prep.idx), cfg, 'cpu')
            for m, sd in zip(models, ck['sd']):
                m.load_state_dict(sd)
            return prep, models, ck['tau']
        except Exception:
            pass
    torch.manual_seed(seed); np.random.seed(seed)
    prep, models, tau = configure(stream.config_X, cfg, 'cpu')
    tmp = p + '.tmp%d' % os.getpid()
    torch.save(dict(prep=prep, sd=[m.state_dict() for m in models], tau=tau), tmp)
    os.replace(tmp, p)
    return prep, models, tau

LOG_KEYS = ['tp', 'fp', 'fn', 'tn', 'tp1', 'fp1', 'auroc', 'auprc', 'arate', 'alarm', 'tier', 'esc', 'sev', 'qfrac', 'skip', 'steps',
            'wall', 'contam', 'cover_loss', 'poison_in', 'nsel', 'tau']

def compact(r, stream):
    L = r['logs']
    out = {k: [x[k] for x in L] for k in LOG_KEYS}
    out['cat_n'] = np.array([x['cat_n'] for x in L]).sum(0).tolist()
    out['cat_tp_w'] = [x['cat_tp'] for x in L]
    out['cat_n_w'] = [x['cat_n'] for x in L]
    out['steps_total'] = r['steps']; out['wall_total'] = r['wall']; out['n_black'] = r['n_black']
    out['drift_points'] = list(getattr(stream, 'drift_points', []))
    if hasattr(stream, 'window_day'):
        out['window_day'] = [int(x) for x in stream.window_day]
    if hasattr(stream, 'benign_shift'):
        try:
            out['wdist'] = stream.benign_shift()
        except Exception:
            pass
    if hasattr(stream, 'wdist_std'):
        out['wdist_std'] = stream.wdist_std
    return out

def run_group(group):
    """group: list of jobs sharing data spec + seed + model config."""
    from core import run_stream
    torch.set_num_threads(1)
    todo = [j for j in group if not os.path.exists(out_path(j))]
    if not todo:
        return 0
    j0 = todo[0]
    try:
        stream = build_stream(j0['data'], j0['seed'])
        base_cfg = j0['cfg']
        prep, models, tau = get_base(stream, j0['data'], j0['seed'], base_cfg)
        n = 0
        for j in todo:
            cfg = dict(j['cfg']); cfg['n_cat'] = getattr(stream, 'n_cat', 16)
            t0 = time.time()
            r = run_stream(j['policy'], cfg, stream, prep, models, tau, 'cpu', opts=j.get('opts', {}), seed=j['seed'])
            out = compact(r, stream)
            out['job'] = {k: j[k] for k in ['exp', 'tag', 'seed', 'policy', 'opts', 'data']}
            out['cfg'] = {k: v for k, v in cfg.items() if not isinstance(v, (list, dict))}
            out['runtime'] = time.time() - t0
            p = out_path(j); os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + '.tmp'
            json.dump(out, open(tmp, 'w'))
            os.replace(tmp, p); n += 1
        return n
    except Exception:
        err = os.path.join(RES, 'errors.log'); os.makedirs(RES, exist_ok=True)
        with open(err, 'a') as f:
            f.write('%s %s s%d\n%s\n' % (time.ctime(), j0['exp'], j0['seed'], traceback.format_exc()))
        return -1

def group_jobs(jobs):
    groups = {}
    for j in jobs:
        key = (json.dumps(j['data'], sort_keys=True), j['seed'], _h({k: j['cfg'][k] for k in MODEL_KEYS}))
        groups.setdefault(key, []).append(j)
    # real-data groups first (longest), then synthetic
    return sorted(groups.values(), key=lambda g: 0 if g[0]['data']['kind'] == 'real' else 1)

def run_all(jobs, workers=6):
    import multiprocessing as mp
    groups = [g for g in group_jobs(jobs) if any(not os.path.exists(out_path(j)) for j in g)]
    print('%d jobs in %d groups pending (%d done already)' % (sum(len(g) for g in groups), len(groups),
          sum(os.path.exists(out_path(j)) for j in jobs)), flush=True)
    if not groups:
        return
    t0 = time.time(); done = 0
    if workers <= 1:
        for g in groups:
            done += max(0, run_group(g)); print('  %d jobs done %.0fs' % (done, time.time() - t0), flush=True)
        return
    with mp.get_context('spawn').Pool(workers, maxtasksperchild=4) as pool:
        for k in pool.imap_unordered(run_group, groups):
            done += max(0, k)
            print('  %d jobs done  %.0fs elapsed' % (done, time.time() - t0), flush=True)
