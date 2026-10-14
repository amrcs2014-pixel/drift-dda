"""DRIFT-DDA v7 core: f-AnoGAN detector, GAN-DDA Stage-II dictionaries, Page-Hinkley DMAC,
label-free retrain-set rules, adaptation policies and per-window logging.

Every policy shares the same detector; they differ only in what happens on an alarm.
Labels are never used for adaptation - only for metrics and retrain-set composition probes.
"""
import os as _os
if _os.environ.get('DDA_ALLOW_GPU') != '1':
    _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # CPU-only (an empty value is deleted on Windows)

import time
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- preprocessing
class Preprocessor:
    """Variance + correlation filter on config benign data, then min-max scaling to [-1, 1]."""
    def __init__(self, var_thr=0.01, corr_thr=0.95, clip=3.0):
        self.var_thr, self.corr_thr, self.clip = var_thr, corr_thr, clip
        self.idx = self.mn = self.rng = None

    def fit(self, X):
        sd = X.std(0)
        mu = X.mean(0)
        Z = (X - mu) / (sd + 1e-9)
        keep = np.where((sd > 1e-9) & (Z.var(0) >= self.var_thr))[0]
        if len(keep) == 0:
            keep = np.arange(X.shape[1])
        C = np.nan_to_num(np.corrcoef(X[:, keep], rowvar=False))
        if C.ndim == 0:
            C = C.reshape(1, 1)
        drop = set()
        for a in range(len(keep)):
            if a in drop:
                continue
            for b in range(a + 1, len(keep)):
                if b not in drop and abs(C[a, b]) > self.corr_thr:
                    drop.add(b)
        self.idx = np.array([keep[a] for a in range(len(keep)) if a not in drop], dtype=int)
        Xs = X[:, self.idx]
        self.mn = Xs.min(0)
        self.rng = Xs.max(0) - self.mn
        self.rng[self.rng == 0] = 1.0
        return self

    def transform(self, X):
        Z = 2 * (X[:, self.idx] - self.mn) / self.rng - 1
        if self.clip is not None:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)


# ----------------------------------------------------------------------------- f-AnoGAN
class Gen(nn.Module):
    def __init__(s, z, h, d):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(z, h), nn.LeakyReLU(.2), nn.Linear(h, h), nn.LeakyReLU(.2), nn.Linear(h, d), nn.Tanh()) \
            if h >= 256 else nn.Sequential(nn.Linear(z, h), nn.LeakyReLU(.2), nn.Linear(h, d), nn.Tanh())
    def forward(s, x):
        return s.n(x)

class Disc(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.LeakyReLU(.2))
        s.o = nn.Linear(h, 1)
    def forward(s, x):
        f = s.h(x)
        return s.o(f), f

class Enc(nn.Module):
    def __init__(s, d, h, z):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d, h), nn.LeakyReLU(.2), nn.Linear(h, z), nn.Tanh())
    def forward(s, x):
        return s.n(x)

def _gp(disc, r, f):
    a = torch.rand(r.size(0), 1, device=r.device)
    it = (a * r + (1 - a) * f).requires_grad_(True)
    s, _ = disc(it)
    g = torch.autograd.grad(s.sum(), it, create_graph=True)[0]
    return ((g.norm(2, 1) - 1) ** 2).mean()

def train_wgan(gen, disc, data, cfg, epochs, device):
    og = torch.optim.Adam(gen.parameters(), cfg['lr'], betas=(0.0, 0.9))
    od = torch.optim.Adam(disc.parameters(), cfg['lr'], betas=(0.0, 0.9))
    X = torch.tensor(data, device=device)
    n = X.size(0); bs = min(cfg['batch'], n); steps = 0
    if bs < 2:
        return 0
    gen.train(); disc.train()
    for _ in range(epochs):
        p = torch.randperm(n, device=device)
        for i in range(0, n - bs + 1, bs):
            xb = X[p[i:i + bs]]
            for _c in range(cfg['n_critic']):
                z = torch.randn(bs, cfg['latent_dim'], device=device)
                fk = gen(z).detach(); sr, _ = disc(xb); sf, _ = disc(fk)
                ld = sf.mean() - sr.mean() + cfg['gp_lambda'] * _gp(disc, xb, fk)
                od.zero_grad(); ld.backward(); od.step(); steps += 1
            z = torch.randn(bs, cfg['latent_dim'], device=device)
            lg = -disc(gen(z))[0].mean()
            og.zero_grad(); lg.backward(); og.step(); steps += 1
    return steps

def train_encoder(enc, gen, disc, data, cfg, epochs, device):
    for p in gen.parameters(): p.requires_grad_(False)
    for p in disc.parameters(): p.requires_grad_(False)
    oe = torch.optim.Adam(enc.parameters(), cfg['lr'], betas=(0.0, 0.9))
    X = torch.tensor(data, device=device)
    n = X.size(0); bs = min(cfg['batch'], n); steps = 0
    enc.train(); gen.eval(); disc.eval()
    if bs >= 2:
        for _ in range(epochs):
            p = torch.randperm(n, device=device)
            for i in range(0, n - bs + 1, bs):
                xb = X[p[i:i + bs]]
                rec = gen(enc(xb)); _, fx = disc(xb); _, fr = disc(rec)
                loss = ((xb - rec) ** 2).mean() + cfg['alpha'] * ((fx - fr) ** 2).mean()
                oe.zero_grad(); loss.backward(); oe.step(); steps += 1
    for p in gen.parameters(): p.requires_grad_(True)
    for p in disc.parameters(): p.requires_grad_(True)
    return steps

@torch.no_grad()
def anomaly_scores(models, X, cfg, device, chunk=65536):
    enc, gen, disc = models
    enc.eval(); gen.eval(); disc.eval()
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.tensor(X[i:i + chunk], device=device)
        rec = gen(enc(xb)); _, fx = disc(xb); _, fr = disc(rec)
        out.append((((xb - rec) ** 2).mean(1) + cfg['alpha'] * ((fx - fr) ** 2).mean(1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)

def make_models(dsel, cfg, device):
    return (Enc(dsel, cfg['hidden'], cfg['latent_dim']).to(device),
            Gen(cfg['latent_dim'], cfg['hidden'], dsel).to(device),
            Disc(dsel, cfg['hidden']).to(device))

def clone_models(models, dsel, cfg, device):
    new = make_models(dsel, cfg, device)
    for a, b in zip(new, models):
        a.load_state_dict(b.state_dict())
    return new

def configure(cX, cfg, device):
    """Offline configuration (fit preprocessor + f-AnoGAN on the configuration window)."""
    prep = Preprocessor().fit(cX)
    Xc = prep.transform(cX)
    models = make_models(Xc.shape[1], cfg, device)
    e, g, d = models
    train_wgan(g, d, Xc, cfg, cfg['wgan_epochs'], device)
    train_encoder(e, g, d, Xc, cfg, cfg['enc_epochs'], device)
    sc = anomaly_scores(models, Xc, cfg, device)
    tau = float(np.percentile(sc, cfg['percentile']))
    prep.c0 = tau / max(1e-12, float(np.median(sc)))   # threshold-to-median ratio of the configured model
    return prep, models, tau


def median_tau(sc, bad, c0):
    """Median-anchored threshold: c0 x median score of flows not from known-bad sources. The median is
    robust while attacks are a minority of those flows and is not truncated by score."""
    x = sc[~bad] if (~bad).sum() >= 50 else sc
    return float(c0 * np.median(x))


# ----------------------------------------------------------------------------- Stage II (GAN-DDA dictionaries)
class Dictionaries:
    """Safe/block dictionaries with a T-step timeout, a suspicion counter with blacklist threshold L
    (GAN-DDA, Ruffo et al. 2026, Alg. 1), plus a persistent per-source count of benign verdicts
    (used only by the optional trusted / probation retrain rules)."""
    __slots__ = ('T', 'L', 'safe', 'block', 'black', 'blacklist', 'good', 'step', 'majority', 'quarantine', 'min_hist')

    def __init__(s, T, L, majority=False):
        s.T, s.L, s.majority = T, L, majority
        s.safe, s.block, s.black, s.good = {}, {}, {}, {}
        s.blacklist = set(); s.step = 0; s.quarantine = set(); s.min_hist = 1

    def run_window(s, src, dst, anom):
        """Process one window of flows in arrival order; returns final verdicts (bool array)."""
        safe, block, black, good, bl = s.safe, s.block, s.black, s.good, s.blacklist
        T, L, maj = s.T, s.L, s.majority
        step = s.step
        out = np.zeros(len(src), dtype=bool)
        for i in range(len(src)):
            step += 1
            a = int(src[i]); b = int(dst[i])
            if not anom[i]:
                if block.get(a, 0) <= step:
                    safe[a] = step + T
                    good[a] = good.get(a, 0) + 1
                if block.get(b, 0) <= step:
                    safe[b] = step + T
            else:
                if safe.get(a, 0) <= step:
                    block[a] = step + T
                    c = black.get(a, 0) + 1; black[a] = c
                    if c >= L and (not maj or c >= good.get(a, 0)):
                        bl.add(a)        # majority rule: suspicious verdicts must also outnumber benign ones
                if block.get(b, 0) <= step:
                    safe[b] = step + T
            out[i] = (block.get(a, 0) > step) or (a in bl)
        s.step = step
        return out

    def prune(s):
        st = s.step
        s.safe = {k: v for k, v in s.safe.items() if v > st}
        s.block = {k: v for k, v in s.block.items() if v > st}

    def discount(s, gamma):
        """Evidence discounting: scale every suspicion counter by gamma and lift blacklist entries that fall
        below L. Applied when the detector adapts, i.e. when it concedes that the verdicts that produced
        the evidence came from a miscalibrated model; persistent attackers re-accumulate quickly."""
        if gamma >= 1.0:
            return
        s.black = {k: v * gamma for k, v in s.black.items() if v * gamma >= 1.0}
        s.blacklist = {a for a in s.blacklist if s.black.get(a, 0) >= s.L}

    def bad_mask(s, src):
        """Sources excluded from every adaptation estimate: currently blocked, blacklisted, or quarantined
        (volume surge, see SurgeTracker). Final verdicts do not use the quarantine."""
        st, block, bl, q = s.step, s.block, s.blacklist, s.quarantine
        return np.fromiter(((block.get(int(a), 0) > st) or (int(a) in bl) or (int(a) in q) for a in src),
                           dtype=bool, count=len(src))

    def trusted_mask(s, src, g_min):
        good, black = s.good, s.black
        return np.fromiter(((good.get(int(a), 0) >= g_min) and (black.get(int(a), 0) == 0) for a in src),
                           dtype=bool, count=len(src))

    def clean_mask(s, src, frac):
        """Sources with history whose benign-verdict share is at least `frac`."""
        good, black = s.good, s.black
        out = np.zeros(len(src), bool)
        for i, a in enumerate(src):
            a = int(a); gd = good.get(a, 0); bk = black.get(a, 0)
            out[i] = (gd + bk) >= s.min_hist and (gd + bk) > 0 and gd >= frac * (gd + bk)
        return out

    def earned_mask(s, src, g_min):
        good = s.good
        return np.fromiter((good.get(int(a), 0) >= g_min for a in src), dtype=bool, count=len(src))


class SurgeTracker:
    """Volume-surge quarantine. A source whose share of the current window is at least `min_share` and at least
    `gamma` times its own exponentially weighted historical share is quarantined: excluded from the trusted
    proxy, the threshold reset, the severity estimate, the retrain set and the reservoir (final verdicts are
    unaffected). While a source surges its history is frozen, so a long burst stays anomalous; after `max_q`
    consecutive surge windows it is released and its new level accepted. Drift changes what traffic looks
    like, not who sends it: a host that suddenly dominates the traffic (scan, flood, exfiltration) should not
    define the new normal, however good its history."""
    def __init__(s, gamma=5.0, min_share=0.05, alpha=0.1, track_share=0.005, warm=3, max_q=50):
        s.gamma, s.min_share, s.alpha, s.track, s.warm, s.max_q = gamma, min_share, alpha, track_share, warm, max_q
        s.hist = {}; s.run = {}; s.t = 0
    def _h(s, a):
        v = s.hist.get(a)
        return 0.0 if v is None else v[0] * (1 - s.alpha) ** (s.t - v[1])
    def update(s, src):
        u, c = np.unique(np.asarray(src), return_counts=True)
        share = c / max(1, len(src))
        surge = set()
        if s.t >= s.warm:
            for a, sh in zip(u, share):
                a = int(a)
                if sh >= s.min_share and sh >= s.gamma * s._h(a) and s.run.get(a, 0) < s.max_q:
                    surge.add(a)
        for a, sh in zip(u, share):
            a = int(a)
            if a in surge:
                s.run[a] = s.run.get(a, 0) + 1          # history frozen while surging
                continue
            s.run.pop(a, None)
            if sh >= s.track or a in s.hist:
                s.hist[a] = ((1 - s.alpha) * s._h(a) + s.alpha * sh, s.t + 1)
        for a in list(s.run):                           # a source absent from this window ends its surge
            if a not in surge:
                s.run.pop(a, None)
        s.t += 1
        return surge


# ----------------------------------------------------------------------------- DMAC
class DriftMonitor:
    """One-sided Page-Hinkley test on the monitored anomaly rate."""
    def __init__(s, delta, lam):
        s.delta, s.lam = delta, lam
        s.ref_mean = 0.0; s.mT = 0.0; s.PH_min = 0.0; s.ready = False
    def set_reference(s, vals):
        s.ref_mean = float(np.mean(vals)); s.mT = 0.0; s.PH_min = 0.0; s.ready = True
    def reset_ph(s):
        s.mT = 0.0; s.PH_min = 0.0
    def update(s, v):
        s.mT += (v - s.ref_mean - s.delta); s.PH_min = min(s.PH_min, s.mT)
        return s.mT - s.PH_min, v / (s.ref_mean + 1e-6)

class Reservoir:
    def __init__(s, size, d, rng):
        s.size, s.d, s.rng = size, d, rng
        s.buf = np.zeros((size, d), np.float32); s.n = 0; s.seen = 0
    def add_many(s, X):
        """Vectorised Algorithm R (Vitter 1985): identical distribution to one-at-a-time insertion."""
        X = np.asarray(X)
        if len(X) == 0:
            return
        free = min(s.size - s.n, len(X))
        if free > 0:
            s.buf[s.n:s.n + free] = X[:free]; s.n += free; s.seen += free
            X = X[free:]
        if len(X) == 0:
            return
        seen_idx = s.seen + 1 + np.arange(len(X))           # 1-based arrival index of each item
        j = (s.rng.random_sample(len(X)) * seen_idx).astype(np.int64)
        hit = j < s.size
        s.buf[j[hit]] = X[hit]                                # later items overwrite earlier ones
        s.seen += len(X)
    def sample(s, k):
        if s.n == 0:
            return np.zeros((0, s.d), np.float32)
        return s.buf[s.rng.randint(0, s.n, size=min(k, s.n))]

class SSFReservoir:
    """Exponential-forgetting buffer used by the SSF-style baseline."""
    def __init__(s, size, d, rng, decay=0.98):
        s.size, s.d, s.decay, s.rng = size, d, decay, rng
        s.buf = np.zeros((size, d), np.float32); s.age = np.zeros(size, np.float32); s.n = 0
    def add_many(s, X):
        if len(X) == 0:
            return
        s.age[:s.n] *= s.decay ** len(X)
        for x in X:
            if s.n < s.size:
                s.buf[s.n] = x; s.age[s.n] = 1.0; s.n += 1
            else:
                w = 1.0 / (1e-6 + s.age[:s.n])
                j = int(s.rng.choice(s.n, p=w / w.sum()))
                s.buf[j] = x; s.age[j] = 1.0
    def sample(s, k):
        if s.n == 0:
            return np.zeros((0, s.d), np.float32)
        w = s.age[:s.n] / (s.age[:s.n].sum() + 1e-8)
        return s.buf[s.rng.choice(s.n, size=min(k, s.n), replace=False, p=w)]

def ssf_select(Xw, sc, bad, k):
    med = np.median(sc)
    pool = np.where((sc <= med) & (~bad))[0]
    if len(pool) < 10:
        pool = np.where(sc <= med)[0]
    if len(pool) <= k:
        return pool
    P = Xw[pool]
    first = int(np.argmin(sc[pool])); idx = [pool[first]]
    d2 = np.sum((P - P[first]) ** 2, axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d2)); idx.append(pool[j])
        d2 = np.minimum(d2, np.sum((P - P[j]) ** 2, axis=1))
    return np.asarray(idx)


# ----------------------------------------------------------------------------- metrics helpers
def _auc_pair(y, s):
    """AUROC and AUPRC of Stage-I scores in one window (NaN if a class has < 5 samples)."""
    npos = int(y.sum()); nneg = len(y) - npos
    if npos < 5 or nneg < 5:
        return float('nan'), float('nan')
    order = np.argsort(-s, kind='mergesort')
    ys = y[order]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    # AUROC via rank statistic
    ranks = np.empty(len(s)); ranks[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    auroc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    prec = tps / np.arange(1, len(ys) + 1)
    auprc = float((prec * ys).sum() / npos)
    return float(auroc), auprc


def source_cap(src, mask, k):
    """Bounded influence per address: keep at most k of the masked flows from any one source
    (the first k in arrival order), so a single heavy host cannot dominate a percentile or a retrain set."""
    idx = np.where(mask)[0]
    if k is None or k <= 0 or len(idx) == 0:
        return mask
    s = np.asarray(src)[idx]
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    start = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    rank = np.arange(len(ss)) - np.repeat(start, np.diff(np.r_[start, len(ss)]))
    keep = np.zeros(len(mask), bool)
    keep[idx[order[rank < k]]] = True
    return keep


def _cap_k(cfg, n):
    f = cfg.get('src_cap_frac', 0.02)
    return None if not f else max(20, int(f * n))


# ----------------------------------------------------------------------------- threshold reset
def recalibrate_tau(models, Xw, recent, src, bad, dda, tau, cfg, device, opts):
    """Reset the Stage-I threshold after an adaptation.

    'clean' (v7): the configured percentile of the (updated) model's scores on flows whose sources are not
    currently known-bad and have a mostly-benign verdict history - a benign proxy that is *not* truncated by
    score. 'retrain' (v6): the percentile of the scores on the retrain set, which keeps only the lowest pi% of
    the window and is therefore biased low by construction."""
    rule = opts.get('tau_rule', cfg.get('tau_rule', 'median'))
    pct = cfg['percentile']
    if rule == 'median':
        sc = anomaly_scores(models, Xw, cfg, device)
        return median_tau(sc, np.zeros(len(sc), bool) if opts.get('no_reputation') else bad, cfg['c0'])
    if rule == 'clean':
        sc = anomaly_scores(models, Xw, cfg, device)
        if opts.get('no_reputation'):          # reputation ablated: untruncated window percentile
            return float(np.percentile(sc, pct))
        C = (~bad) & dda.clean_mask(src, opts.get('clean_frac', cfg.get('clean_frac', 0.5)))
        if C.sum() < max(50, 0.05 * len(Xw)):
            C = ~bad
        if not opts.get('no_cap'):
            C = source_cap(src, C, _cap_k(cfg, len(Xw)))
        if C.sum() >= 50:
            return float(np.percentile(sc[C], pct))
    if len(recent) == 0:
        return tau
    return float(np.percentile(anomaly_scores(models, recent, cfg, device), pct))


def severity_tau(sc, src, dda, cfg, opts):
    """Threshold the current window would need (same rule as the threshold reset), before any update."""
    if opts.get('tau_rule', cfg.get('tau_rule', 'median')) == 'median':
        bad = np.zeros(len(sc), bool) if opts.get('no_reputation') else dda.bad_mask(src)
        return median_tau(sc, bad, cfg['c0'])
    return trusted_percentile(sc, src, dda, cfg, opts)


def trusted_percentile(sc, src, dda, cfg, opts):
    """Configured percentile of the current window's scores over the trusted proxy (not known-bad, mostly
    benign verdict history); falls back to not-known-bad flows, then to the whole window."""
    bad = dda.bad_mask(src)
    C = (~bad) & dda.clean_mask(src, opts.get('clean_frac', cfg.get('clean_frac', 0.5)))
    if C.sum() < max(50, 0.05 * len(sc)):
        C = ~bad
    if C.sum() < 50:
        C = np.ones(len(sc), bool)
    if not opts.get('no_cap'):
        C = source_cap(src, C, _cap_k(cfg, len(sc)))
    return float(np.percentile(sc[C], cfg['percentile']))


# ----------------------------------------------------------------------------- streaming loop
FIXED_TIER = {'detect': 3, 'tier1': 1, 'tier2': 2, 'ssf': 3}

def run_stream(policy, cfg, stream, prep, base_models, base_tau, device, opts=None, seed=0):
    """Prequential (test-then-adapt) run of one adaptation policy over a windowed stream.

    stream: object with .n_windows and .window(t) -> dict(X, y, src, dst, kind, cat)
      y    : binary ground truth (1 = malicious), used only for metrics
      kind : 0 benign, 1 attack, 2 poison  (poison flows are excluded from detection metrics)
      cat  : attack-category id (-1 benign) for per-category recall
    opts (all optional):
      monitor        'raw' (anomaly rate over all flows) | 'rep' (over flows from sources not already known-bad)
      rule           'base' | 'widen' | 'trusted' | 'probation'
      pi_w           widened inclusion percentile for rule='widen'
      g_min          earned-benign count for 'trusted'/'probation'
      no_reputation, no_replay, no_cooldown, no_gating   ablation switches
    """
    opts = opts or {}
    rng = np.random.RandomState(seed + 7919)
    torch.manual_seed(seed + 104729)
    dsel = len(prep.idx)
    models = clone_models(base_models, dsel, cfg, device)
    e, g, d = models
    tau = base_tau
    dda = Dictionaries(cfg['T'], cfg['L'], majority=cfg.get('bl_majority', True) and not opts.get('no_majority'))
    dda.min_hist = int(opts.get('trust_min', cfg.get('trust_min', 1)))   # verdicts needed before a source can be trusted
    buf = SSFReservoir(cfg['buffer_size'], dsel, rng) if policy == 'ssf' else Reservoir(cfg['buffer_size'], dsel, rng)
    dm = DriftMonitor(cfg['ph_delta'], cfg['ph_lambda'])
    monitor = opts.get('monitor', cfg.get('monitor', 'raw'))
    rule = opts.get('rule', cfg.get('rule', 'tau'))
    cfg = dict(cfg); cfg.setdefault('c0', getattr(prep, 'c0', 3.0))
    warm = []; last_t3 = -10 ** 9; last_act = -10 ** 9; last_tier = 0
    verify_left = 0; post = []; ref_old = 0.0; tau_ref = base_tau
    seen = set()
    surge = SurgeTracker(cfg.get('surge_gamma', 5.0), cfg.get('surge_min_share', 0.05))         if (cfg.get('surge', False) and not opts.get('no_surge')) else None
    logs = []
    usteps = 0; utime = 0.0
    n_cat = cfg.get('n_cat', 16)
    for t in range(stream.n_windows):
        w = stream.window(t)
        Xw = prep.transform(w['X']); yw = w['y']; kind = w['kind']; src = w['src']; dst = w['dst']
        sc = anomaly_scores(models, Xw, cfg, device)
        s1 = sc > tau
        bad_pre = dda.bad_mask(src) if monitor == 'rep' else None
        pred = dda.run_window(src, dst, s1)
        dda.prune()
        if surge is not None:
            dda.quarantine = surge.update(src)
        ev = kind != 2
        ye, pe, s1e = yw[ev], pred[ev], s1[ev]
        tp = int(((pe == 1) & (ye == 1)).sum()); fp = int(((pe == 1) & (ye == 0)).sum())
        fn = int(((pe == 0) & (ye == 1)).sum()); tn = int(((pe == 0) & (ye == 0)).sum())
        tp1 = int(((s1e == 1) & (ye == 1)).sum()); fp1 = int(((s1e == 1) & (ye == 0)).sum())
        auroc, auprc = _auc_pair(ye.astype(np.int64), sc[ev])
        # per-category detection counts (final verdicts)
        cat = w['cat']
        cat_n = np.bincount(cat[cat >= 0], minlength=n_cat)[:n_cat]
        cat_tp = np.bincount(cat[(cat >= 0) & pred], minlength=n_cat)[:n_cat]
        pois = kind == 2
        if monitor == 'rep':
            keep = ~bad_pre
            arate = float(s1[keep].mean()) if keep.sum() >= 20 else (dm.ref_mean if dm.ready else float(s1.mean()))
        else:
            arate = float(s1.mean())
        okbuf = ~pred
        qfrac = 0.0
        if opts.get('fresh_guard'):
            # post-hoc extension (not part of the frozen design): never-seen sources are treated like
            # quarantined ones for adaptation (excluded from retrain sets, thresholds and the reservoir, and
            # counted by the representativeness guard)
            fresh_src = {int(a) for a in np.unique(src)} - seen
            if fresh_src:
                dda.quarantine = set(dda.quarantine) | fresh_src
            seen.update(fresh_src)
        if dda.quarantine:
            inq = np.fromiter((int(a) in dda.quarantine for a in src), dtype=bool, count=len(src))
            okbuf &= ~inq; qfrac = float(inq.mean())
        buf.add_many(Xw[okbuf][:cfg.get('buf_add_cap', 10 ** 9)])
        tier = 0; alarm = False; escalated = False; sev = float('nan')
        rebase = cfg.get('rebase', True)
        sev_mode = opts.get('severity', cfg.get('severity', 'score'))
        gated_esc = (policy == 'gated' and cfg.get('escalate', True) and not opts.get('no_escalation')
                     and not opts.get('no_gating'))
        # representativeness guard: a window dominated by quarantined (surging) sources says nothing about
        # normal traffic -> no monitor update, no verification step, no adaptation (the current model is kept)
        skip = qfrac > cfg.get('defer_frac', 0.5) and not opts.get('no_defer')
        if policy == 'periodic':
            tier = 3 if (t > 0 and t % cfg['periodic_k'] == 0) else 0
        elif skip:
            pass
        elif policy != 'static':
            if t < cfg['warmup']:
                warm.append(arate)
                if t == cfg['warmup'] - 1:
                    dm.set_reference(warm); ref_old = dm.ref_mean
            elif rebase and verify_left > 0:
                # post-action verification: collect the rate, then either escalate or re-baseline
                post.append(arate); verify_left -= 1
                if verify_left == 0:
                    if sev_mode == 'score':
                        # does the (updated) model fit trusted traffic at its reference resolution again?
                        fail = (tau / tau_ref) >= cfg['rate_t2']
                    else:
                        fail = float(np.mean(post)) >= cfg.get('rho_v', cfg['rate_t2']) * ref_old
                    if gated_esc and last_tier < 3 and fail:
                        tier = last_tier + 1; escalated = True
                        if tier == 3 and (t - last_t3) < cfg['cooldown'] and not opts.get('no_cooldown'):
                            tier = 0          # Tier 3 blocked by cooldown: accept the current state
                    if tier == 0:
                        dm.set_reference(post)   # residual level after the response is the new normal
            elif dm.ready:
                ph, ratio = dm.update(arate)
                if ph > cfg['ph_lambda']:
                    alarm = True
                    if policy in FIXED_TIER:
                        tier = FIXED_TIER[policy]
                    else:  # gated (DRIFT-DDA): cheapest tier expected to suffice
                        if sev_mode == 'score':
                            # threshold inflation: how far the trusted-traffic score percentile has risen
                            # relative to the resolution of the last full fit
                            sev = severity_tau(sc, src, dda, cfg, opts) / tau_ref
                        else:
                            sev = ratio
                        tier = 1 if sev < cfg['rate_t2'] else (2 if sev < cfg['rate_t3'] else 3)
                        if opts.get('no_gating'):
                            tier = 3
                        if tier == 3 and (t - last_t3) < cfg['cooldown'] and not opts.get('no_cooldown'):
                            tier = 2
        rec = dict(tp=tp, fp=fp, fn=fn, tn=tn, tp1=tp1, fp1=fp1, auroc=auroc, auprc=auprc,
                   arate=arate, alarm=int(alarm), tier=int(tier), esc=int(escalated), sev=float(sev), qfrac=qfrac, skip=int(skip), steps=0, wall=0.0,
                   contam=float('nan'), cover_loss=float('nan'), poison_in=float('nan'), nsel=0,
                   cat_n=cat_n.tolist(), cat_tp=cat_tp.tolist(), tau=float(tau))
        if tier > 0:
            t0 = time.perf_counter(); steps = 0
            bad = np.zeros(len(src), bool) if opts.get('no_reputation') else dda.bad_mask(src)
            if policy == 'ssf':
                sel_idx = ssf_select(Xw, sc, bad, k=min(len(sc), max(20, int(0.3 * len(sc)))))
                sel = np.zeros(len(sc), bool); sel[sel_idx] = True
            else:
                pi = cfg['adapt_incl_pct']
                if rule == 'tau':
                    # flows the recalibrated (median-anchored) threshold deems benign, minus known-bad sources
                    inc = sc <= median_tau(sc, bad, cfg['c0'])
                    sel = inc & ~bad
                    if sel.sum() <= 10:
                        sel = inc
                if rule != 'tau':
                    if rule == 'widen' and tier >= 2:
                        pi = opts.get('pi_w', 90)
                    inc = sc <= np.percentile(sc, pi)
                    sel = inc & ~bad
                if rule == 'repwiden' and tier >= 2:
                    # widened band admitted only from sources whose verdict history is mostly benign;
                    # with reputation ablated the band is admitted from every source
                    wide = sc <= np.percentile(sc, opts.get('pi_w', cfg.get('pi_w', 95)))
                    if opts.get('no_reputation'):
                        sel = sel | wide
                    else:
                        sel = sel | (wide & ~bad & dda.clean_mask(src, opts.get('clean_frac', cfg.get('clean_frac', 0.5))))
                if rule == 'trusted':
                    sel = sel | (dda.trusted_mask(src, opts.get('g_min', 3)) & ~bad)
                elif rule == 'probation':
                    sel = sel & dda.earned_mask(src, opts.get('g_min', 3))
                if sel.sum() <= 10:
                    sel = inc & ~bad
                if sel.sum() <= 10:
                    sel = inc
                if not opts.get('no_cap'):
                    sel = source_cap(src, sel, _cap_k(cfg, len(sel)))
            recent = Xw[sel]
            ben = (yw == 0) & (kind == 0)
            rec['nsel'] = int(sel.sum())
            rec['contam'] = float(((yw == 1) & sel).sum() / max(1, sel.sum()))
            rec['poison_in'] = float((pois & sel).sum() / max(1, sel.sum()))
            rec['cover_loss'] = float((ben & ~sel).sum() / max(1, ben.sum()))
            buf.add_many(recent)
            train_X = recent
            if not opts.get('no_replay'):
                rep = buf.sample(cfg['replay_k'])
                if len(rep) > 0:
                    train_X = np.concatenate([recent, rep], 0)
            if tier >= 3:
                steps += train_wgan(g, d, train_X, cfg, cfg['adapt_gan_epochs'], device)
            if tier >= 2:
                steps += train_encoder(e, g, d, train_X, cfg, cfg['adapt_enc_epochs'], device)
            tau = recalibrate_tau(models, Xw, recent, src, bad, dda, tau, cfg, device, opts)
            wall = time.perf_counter() - t0
            rec['steps'] = steps; rec['wall'] = wall
            usteps += steps; utime += wall
            if tier == 3:
                last_t3 = t
                tau_ref = tau                    # a full re-fit defines the new reference resolution
            last_act = t; last_tier = tier
            if not opts.get('no_discount'):
                dda.discount(cfg.get('amnesty', 1.0))
            dm.reset_ph()
            if rebase and policy != 'periodic':
                if not escalated:
                    ref_old = dm.ref_mean        # reference that the response is judged against
                verify_left = cfg.get('verify_m', 2); post = []
        logs.append(rec)
    return dict(logs=logs, steps=usteps, wall=utime, n_black=len(dda.blacklist))
