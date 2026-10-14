"""Streams for DRIFT-DDA v7.

* SynthStream   controlled synthetic stream: easy (v6) or hard (overlapping) attacks; abrupt, gradual,
                recurring or no drift; optional novel attack family, bursty attack rate, attacker IP
                rotation and slow ("boiling-frog") poisoning from fresh addresses.
* RealStream    a real trace (UNSW-NB15 full records or improved CICIDS2017) replayed in timestamp order
                with its real source/destination addresses; optional attacker IP rotation.
* InjectedStream  windows sampled from the real UNSW-NB15 records (real addresses kept), with an injected
                covariate shift on continuous columns only (one-hot indicators are never shifted).
"""
import json, os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FRESH0 = 300_000_000          # fresh (never-seen) addresses are allocated above this id


class _Windows:
    n_windows = 0
    def window(self, t):
        raise NotImplementedError


# ============================================================================ synthetic
SYN_DEFAULT = dict(
    n_features=16, n_benign_clusters=3, n_attack_types=4, benign_sigma=0.14,
    n_benign_ips=4000, n_attacker_ips=60,
    config_size=6000, config_contam=0.01,
    n_windows=40, window_size=512, drift_points=[15, 28], drift_type='abrupt', drift_magnitude=0.6,
    stream_attack_rate=0.25, bursty=False, burst_rates=(0.05, 0.45), burst_p=0.08,
    difficulty='easy', sep_lo=0.45, sep_hi=0.9, attack_sigma=0.10,
    gradual_len=20, recur_period=30,
    novel_at=None, novel_share=0.5, novel_sep=0.55,
    rotation=0.0,
    poison=None,       # dict(t0, t1, rate, target)  -> poison flows ramp from benign toward attack `target`
)


class SynthStream(_Windows):
    def __init__(self, cfg, seed):
        c = dict(SYN_DEFAULT); c.update(cfg); self.c = c
        # independent random streams: structure, drift, configuration window, stream sampling - so the
        # configuration window (and hence the configured model) is identical across drift settings
        r_struct = np.random.RandomState(1000 + 17 * seed + (0 if c['difficulty'] == 'easy' else 7))
        r_drift = np.random.RandomState(2000 + 17 * seed)
        r_cfg = np.random.RandomState(3000 + 17 * seed)
        rng = np.random.RandomState(4000 + 17 * seed)
        d, K, A = c['n_features'], c['n_benign_clusters'], c['n_attack_types']
        means = r_struct.uniform(-0.5, 0.5, size=(K, d))
        if c['difficulty'] == 'easy':
            am = r_struct.uniform(-1.0, 1.0, size=(A, d)); am += np.sign(am) * 0.8
            asig = np.full(A, 0.05)
        else:
            am = np.zeros((A, d))
            for a in range(A):
                k = r_struct.randint(K); u = r_struct.randn(d); u /= np.linalg.norm(u)
                am[a] = means[k] + u * r_struct.uniform(c['sep_lo'], c['sep_hi'])
            asig = np.full(A, c['attack_sigma'])
        sig = c['benign_sigma']
        # ---- drift trajectory of benign cluster means, one (K,d) array per window
        nw, W = c['n_windows'], c['window_size']
        dp = list(c['drift_points']); mag = c['drift_magnitude']
        M = np.zeros((nw, K, d))
        cur = means.copy()
        steps = []
        for _ in range(max(1, len(dp))):
            m = r_drift.rand(K) < 0.6
            if not m.any():
                m[r_drift.randint(K)] = True
            steps.append(r_drift.uniform(-1, 1, size=(K, d)) * mag * m[:, None])
        if c['drift_type'] == 'recurring':
            B = means + steps[0]
            for t in range(nw):
                if t < dp[0]:
                    M[t] = means
                else:
                    M[t] = B if ((t - dp[0]) // c['recur_period']) % 2 == 0 else means
        else:
            start = means.copy(); target = means.copy(); t_on = -1; seg = 0
            for t in range(nw):
                if seg < len(dp) and t == dp[seg]:
                    start = cur.copy(); target = cur + steps[seg]; t_on = t; seg += 1
                if t_on >= 0:
                    if c['drift_type'] == 'gradual':
                        frac = min(1.0, (t - t_on + 1) / c['gradual_len'])
                    else:
                        frac = 1.0
                    cur = start + (target - start) * frac
                M[t] = cur
        # novel attack family placed next to the drifted benign manifold
        novel_mean = None
        if c['novel_at'] is not None:
            k = r_drift.randint(K); u = r_drift.randn(d); u /= np.linalg.norm(u)
            novel_mean = M[min(nw - 1, c['novel_at'])][k] + u * c['novel_sep']
        # ---- configuration window (drift-free, ~1% attacks)
        nc = c['config_size']
        cy = (r_cfg.rand(nc) < c['config_contam']).astype(np.int8)
        ck = r_cfg.randint(K, size=nc); ca = r_cfg.randint(A, size=nc)
        cX = means[ck] + r_cfg.randn(nc, d) * sig
        cX[cy == 1] = am[ca[cy == 1]] + r_cfg.randn(int(cy.sum()), d) * asig[ca[cy == 1]][:, None]
        self.config_X = cX.astype(np.float32); self.config_y = cy
        # ---- stream
        b_ips = np.arange(c['n_benign_ips']); a_ips = 10_000_000 + np.arange(c['n_attacker_ips'])
        nov_ips = 20_000_000 + np.arange(20)
        X = np.zeros((nw, W, d), np.float32); Y = np.zeros((nw, W), np.int8)
        S = np.zeros((nw, W), np.int64); D = np.zeros((nw, W), np.int64)
        KIND = np.zeros((nw, W), np.int8); CAT = np.full((nw, W), -1, np.int16)
        fresh = FRESH0
        rate = c['stream_attack_rate']; burst_state = 0
        P = c['poison']
        for t in range(nw):
            if c['bursty']:
                if rng.rand() < c['burst_p']:
                    burst_state = 1 - burst_state
                rate = c['burst_rates'][burst_state]
            isatt = rng.rand(W) < rate
            k = rng.randint(K, size=W)
            X[t] = M[t][k] + rng.randn(W, d) * sig
            S[t] = b_ips[rng.randint(len(b_ips), size=W)]; D[t] = b_ips[rng.randint(len(b_ips), size=W)]
            ai = np.where(isatt)[0]
            a = rng.randint(A, size=len(ai))
            if novel_mean is not None and t >= c['novel_at']:
                nov = rng.rand(len(ai)) < c['novel_share']
            else:
                nov = np.zeros(len(ai), bool)
            X[t, ai] = am[a] + rng.randn(len(ai), d) * asig[a][:, None]
            CAT[t, ai] = a
            S[t, ai] = a_ips[rng.randint(len(a_ips), size=len(ai))]
            if nov.any():
                ni = ai[nov]
                X[t, ni] = novel_mean + rng.randn(len(ni), d) * c['attack_sigma']
                CAT[t, ni] = A; S[t, ni] = nov_ips[rng.randint(len(nov_ips), size=len(ni))]
            Y[t, ai] = 1; KIND[t, ai] = 1
            if c['rotation'] > 0:
                rot = ai[rng.rand(len(ai)) < c['rotation']]
                S[t, rot] = fresh + np.arange(len(rot)); fresh += len(rot)
            if P is not None:
                tgt = P['target']
                if P['t0'] <= t < P['t1']:
                    npz = int(P['rate'] * W)
                    pi = rng.choice(W, size=npz, replace=False)
                    beta = (t - P['t0']) / max(1, P['t1'] - P['t0'] - 1)
                    kk = int(np.argmin(((M[t] - am[tgt]) ** 2).sum(1)))
                    X[t, pi] = (1 - beta) * M[t][kk] + beta * am[tgt] + rng.randn(npz, d) * sig
                    Y[t, pi] = 1; KIND[t, pi] = 2; CAT[t, pi] = -1
                    S[t, pi] = fresh + np.arange(npz); fresh += npz
                if t >= P['t1']:   # the poisoned-for attack then comes from fresh addresses
                    ti = np.where(CAT[t] == tgt)[0]
                    S[t, ti] = fresh + np.arange(len(ti)); fresh += len(ti)
        self.X, self.Y, self.S, self.D, self.KIND, self.CAT = X, Y, S, D, KIND, CAT
        self.n_windows = nw
        self.drift_points = dp
        self.n_cat = A + 1

    def window(self, t):
        return dict(X=self.X[t], y=self.Y[t], src=self.S[t], dst=self.D[t], kind=self.KIND[t], cat=self.CAT[t])

    def benign_shift(self):
        """Mean per-feature 1-D Wasserstein distance between pre- and post-first-drift benign flows (data units)."""
        from scipy.stats import wasserstein_distance
        dp = self.drift_points[0]
        pre = self.X[:dp][self.Y[:dp] == 0]; post = self.X[dp:][self.Y[dp:] == 0]
        return float(np.mean([wasserstein_distance(pre[:, j], post[:, j]) for j in range(pre.shape[1])]))


# ============================================================================ real traces
REAL_SPEC = {
    # configuration period, 1% attack contamination drawn from `contam_from`; test stream follows it
    'unsw':  dict(config_until_h=2.2, test_from_h=2.2, contam_from='config'),
    'cic17': dict(config_days=[0], contam_days=[1], test_days=[2, 3, 4]),
    'ton':   dict(contam_until_h=15.0, config_until_h=27.0),     # hours after midnight of 23 Apr 2019 (UTC)
}

# Connection identifiers are never model inputs (like addresses, ports and timestamps): TCP initial sequence
# numbers (UNSW-NB15 stcpb/dtcpb) and DNS transaction ids (NetFlow-v3 DNS_QUERY_ID) are random per flow.
IDENTIFIERS = {'stcpb', 'dtcpb', 'DNS_QUERY_ID'}

def feature_cols(meta):
    return np.array([i for i, f in enumerate(meta['features']) if f not in IDENTIFIERS])

def load_cache(name):
    d = os.path.join(DATA, name)
    arr = {k: np.load(os.path.join(d, k + '.npy'), mmap_mode='r') for k in ['X', 'y', 'src', 'dst', 'cat', 't', 'day']}
    meta = json.load(open(os.path.join(d, 'meta.json')))
    return arr, meta

def real_split(name, config_size, seed):
    """Indices of the configuration sample (benign + 1% attacks) and the contiguous test range."""
    arr, meta = load_cache(name)
    y = np.asarray(arr['y']); t = np.asarray(arr['t']); day = np.asarray(arr['day'])
    rng = np.random.RandomState(5000 + seed)
    if name == 'unsw':
        t0 = t[0]; sp = REAL_SPEC['unsw']
        cfg_idx = np.where(t < t0 + sp['config_until_h'] * 3600)[0]
        test = np.where(t >= t0 + sp['test_from_h'] * 3600)[0]
        ben = cfg_idx[y[cfg_idx] == 0]; att = cfg_idx[y[cfg_idx] == 1]
    elif name == 'ton':
        # 23 Apr 2019: 14:09-15:00 has attacks (contamination pool); 15:00-03:00 is 99.7% benign (configuration);
        # the test stream starts 24 Apr 03:00 and runs to 29 Apr 14:45
        sp = REAL_SPEC['ton']; d0 = 86400 * np.floor(t[0] / 86400)
        a, b, c = (d0 + h * 3600 for h in (sp['contam_until_h'], sp['config_until_h'], sp['config_until_h']))
        ben = np.where((t >= a) & (t < b) & (y == 0))[0]
        att = np.where((t < a) & (y == 1))[0]
        test = np.where(t >= c)[0]
    else:
        sp = REAL_SPEC['cic17']
        ben = np.where(np.isin(day, sp['config_days']) & (y == 0))[0]
        att = np.where(np.isin(day, sp['contam_days']) & (y == 1))[0]
        test = np.where(np.isin(day, sp['test_days']))[0]
    n_att = int(round(0.01 * config_size))
    ci = np.concatenate([rng.choice(ben, config_size - n_att, replace=False), rng.choice(att, n_att, replace=False)])
    rng.shuffle(ci)
    return arr, meta, np.sort(ci), int(test[0]), int(test[-1]) + 1


class RealStream(_Windows):
    def __init__(self, name, window_size, seed, rotation=0.0, max_windows=None, config_size=10000):
        arr, meta, ci, lo, hi = real_split(name, config_size, seed)
        self.meta = meta; self.name = name
        self.cols = feature_cols(meta)
        self.config_X = np.asarray(arr['X'][ci], np.float32)[:, self.cols]; self.config_y = np.asarray(arr['y'][ci])
        self.lo, self.W = lo, window_size
        n = (hi - lo) // window_size
        if max_windows:
            n = min(n, max_windows)
        self.n_windows = n
        hi = lo + n * window_size
        self.X = arr['X']; self.y = np.asarray(arr['y'][lo:hi]); self.cat = np.asarray(arr['cat'][lo:hi])
        self.src = np.asarray(arr['src'][lo:hi]).astype(np.int64); self.dst = np.asarray(arr['dst'][lo:hi]).astype(np.int64)
        self.day = np.asarray(arr['day'][lo:hi]); self.t = np.asarray(arr['t'][lo:hi])
        if rotation > 0:
            rng = np.random.RandomState(9000 + seed)
            m = (self.y == 1) & (rng.rand(len(self.y)) < rotation)
            self.src[m] = FRESH0 + np.arange(int(m.sum()))
        self.kind = self.y.astype(np.int8)
        self.n_cat = len(meta['cat_names'])
        # natural change points: first window of each new day in the test range
        wd = self.day[::window_size][:n]
        self.drift_points = [int(i) for i in np.where(np.diff(wd) != 0)[0] + 1]
        self.window_day = wd

    def window(self, t):
        a, b = t * self.W, (t + 1) * self.W
        return dict(X=np.asarray(self.X[self.lo + a:self.lo + b], np.float32)[:, self.cols], y=self.y[a:b], src=self.src[a:b],
                    dst=self.dst[a:b], kind=self.kind[a:b], cat=self.cat[a:b])


class InjectedStream(_Windows):
    """Windows sampled at random from the real UNSW-NB15 records outside the configuration period, with
    their real addresses, and a cumulative covariate shift of +/- m standard deviations applied to a random
    half of the *continuous* columns of benign flows at each drift point (v6 protocol without the one-hot artefact)."""
    def __init__(self, seed, n_windows=60, window_size=512, drift_points=(20, 40), magnitude=2.0,
                 attack_rate=0.25, config_size=10000):
        arr, meta, ci, lo, hi = real_split('unsw', config_size, seed)
        self.meta = meta
        cols = feature_cols(meta)
        self.config_X = np.asarray(arr['X'][ci], np.float32)[:, cols]; self.config_y = np.asarray(arr['y'][ci])
        rng = np.random.RandomState(7000 + seed)
        y = np.asarray(arr['y'][lo:hi])
        ben = lo + np.where(y == 0)[0]; att = lo + np.where(y == 1)[0]
        is_oh = np.array(meta['is_onehot'])[cols]
        sd = self.config_X[self.config_y == 0].std(0)
        d = self.config_X.shape[1]
        self.n_windows = n_windows; self.drift_points = list(drift_points); self.W = window_size
        shift = np.zeros(d, np.float32); self.wins = []
        Xall = arr['X']
        for t in range(n_windows):
            if t in drift_points:
                u = rng.uniform(-magnitude, magnitude, size=d).astype(np.float32)
                m = (rng.rand(d) < 0.5) & (~is_oh)
                shift = shift + u * m * sd
            isatt = rng.rand(window_size) < attack_rate
            idx = np.where(isatt, att[rng.randint(len(att), size=window_size)], ben[rng.randint(len(ben), size=window_size)])
            X = np.asarray(Xall[np.sort(idx)], np.float32)[:, cols]
            order = np.argsort(np.argsort(idx)); X = X[order]   # restore sampled order
            yy = np.asarray(arr['y'][idx]); X[yy == 0] += shift
            self.wins.append(dict(X=X, y=yy, src=np.asarray(arr['src'][idx]).astype(np.int64),
                                  dst=np.asarray(arr['dst'][idx]).astype(np.int64), kind=yy.astype(np.int8),
                                  cat=np.asarray(arr['cat'][idx])))
        self.n_cat = len(meta['cat_names'])
        pre = np.concatenate([w['X'][w['y'] == 0] for w in self.wins[:drift_points[0]]])
        post = np.concatenate([w['X'][w['y'] == 0] for w in self.wins[drift_points[0]:]])
        from scipy.stats import wasserstein_distance
        keep = np.where(pre.std(0) > 1e-8)[0]
        s = pre.std(0)
        self.wdist_std = float(np.mean([wasserstein_distance(pre[:, j] / s[j], post[:, j] / s[j]) for j in keep]))

    def window(self, t):
        return self.wins[t]
