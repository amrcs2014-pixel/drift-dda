"""Convert the two real-IP datasets into memory-mappable caches (one-time).

UNSW-NB15 full records (srcip/dstip/Stime, Argus+Bro features; 2.28M flows, 22 Jan and 17-18 Feb 2015)
improved CICIDS2017 (Engelen et al. 2021; Src/Dst IP + Timestamp; 5 days, 3-7 Jul 2017)

Each cache directory holds X.npy (float32, signed-log1p numeric + one-hot categoricals), y.npy (int8),
src.npy/dst.npy (int32 address ids), cat.npy (int16 attack category, -1 benign), t.npy (float64 seconds),
day.npy (int8) and meta.json (feature names, one-hot mask, category names, address-id map size).
Rows are sorted by flow start time. Ports, addresses and timestamps are never model inputs.
"""
import glob, json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
UNSW = sorted(glob.glob(os.path.join(os.environ.get('DDA_UNSW_DIR', os.path.join(OUT, 'raw', 'unswnb15')), '*.parquet')))
CIC = os.environ.get('DDA_CIC_DIR', os.path.join(OUT, 'raw', 'cicids2017'))

def slog(a):
    return (np.sign(a) * np.log1p(np.abs(a))).astype(np.float32)

def save(name, X, y, src, dst, cat, t, day, meta):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    for k, v in dict(X=X, y=y, src=src, dst=dst, cat=cat, t=t, day=day).items():
        np.save(os.path.join(d, k + '.npy'), v)
    json.dump(meta, open(os.path.join(d, 'meta.json'), 'w'), indent=1)
    print(name, X.shape, 'attack frac %.3f' % y.mean(), 'days', np.bincount(day))

def onehot(series, levels):
    s = series.where(series.isin(levels), '__OTHER__')
    cols = list(levels) + ['__OTHER__']
    return pd.get_dummies(pd.Categorical(s, categories=cols)).values.astype(np.float32), cols

def prep_unsw():
    df = pd.concat([pq.read_table(f).to_pandas() for f in UNSW], ignore_index=True)
    df = df.sort_values(['Stime', 'Ltime'], kind='mergesort').reset_index(drop=True)
    t = df['Stime'].astype(np.float64).values
    day = (pd.to_datetime(df['Stime'], unit='s').dt.date != pd.to_datetime(df['Stime'].iloc[0], unit='s').date()).astype(np.int8).values
    y = df['label'].astype(np.int8).values
    cats = df['attack_cat'].fillna('').astype(str).str.strip().replace({'Backdoors': 'Backdoor'})
    cat_names = sorted(c for c in cats.unique() if c not in ('', 'Normal'))
    cmap = {c: i for i, c in enumerate(cat_names)}
    cat = np.array([cmap.get(c, -1) for c in cats], np.int16); cat[y == 0] = -1
    ips = pd.Index(pd.unique(pd.concat([df['srcip'], df['dstip']]).astype(str)))
    src = ips.get_indexer(df['srcip'].astype(str)).astype(np.int32)
    dst = ips.get_indexer(df['dstip'].astype(str)).astype(np.int32)
    drop = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'attack_cat', 'label', 'proto', 'state', 'service']
    num = df.drop(columns=drop).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    names = list(num.columns)
    parts = [slog(num.values.astype(np.float64))]
    is_oh = [False] * len(names)
    early = t < t[0] + 2.2 * 3600            # category levels taken from the configuration period only
    for c in ['proto', 'state', 'service']:
        col = df[c].astype(str).str.strip()
        levels = list(col[early].value_counts().index[:15])
        oh, cols = onehot(col, levels)
        parts.append(oh); names += ['%s=%s' % (c, v) for v in cols]; is_oh += [True] * len(cols)
    X = np.concatenate(parts, 1)
    meta = dict(features=names, is_onehot=is_oh, cat_names=cat_names, n_ips=len(ips),
                note='UNSW-NB15 full records, sorted by Stime; day 0 = 22 Jan 2015, day 1 = 17-18 Feb 2015')
    save('unsw', X, y, src, dst, cat, t, day, meta)

def prep_cic():
    meta_cols = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Label',
                 'Attempted Category']
    Xs, labs, srcs, dsts, ts, days, protos = [], [], [], [], [], [], []
    names = None
    for di, dname in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
        f = pd.read_csv(os.path.join(CIC, dname + '.csv'), low_memory=False)
        f = f[~f['Label'].str.contains('Attempted')]           # failed attempts carry no attack payload (Engelen et al.)
        feat = f.drop(columns=meta_cols)
        if names is None:
            names = list(feat.columns)
        A = np.empty((len(f), len(names)), np.float32)
        for j, c in enumerate(names):
            v = pd.to_numeric(feat[c], errors='coerce').values.astype(np.float64)
            v[~np.isfinite(v)] = 0.0
            A[:, j] = slog(v)
        Xs.append(A); labs.append(f['Label'].astype(str).values)
        srcs.append(f['Src IP'].astype(str).values); dsts.append(f['Dst IP'].astype(str).values)
        ts.append(((pd.to_datetime(f['Timestamp']) - pd.Timestamp('1970-01-01')).dt.total_seconds()).values)
        days.append(np.full(len(f), di, np.int8)); protos.append(f['Protocol'].astype(int).astype(str).values)
        del f, feat
        print('read', dname, A.shape)
    t = np.concatenate(ts); order = np.argsort(t, kind='mergesort')
    X = np.concatenate(Xs)[order]; del Xs
    lab = np.concatenate(labs)[order]; srcs = np.concatenate(srcs)[order]; dsts = np.concatenate(dsts)[order]
    day = np.concatenate(days)[order]; proto = pd.Series(np.concatenate(protos)[order]); t = t[order]
    y = (lab != 'BENIGN').astype(np.int8)
    cat_names = sorted(c for c in set(lab) if c != 'BENIGN')
    cmap = {c: i for i, c in enumerate(cat_names)}
    cat = np.array([cmap.get(c, -1) for c in lab], np.int16)
    ips = pd.Index(pd.unique(np.concatenate([srcs, dsts])))
    src = ips.get_indexer(srcs).astype(np.int32); dst = ips.get_indexer(dsts).astype(np.int32)
    is_oh = [False] * len(names)
    oh, cols = onehot(proto, ['6', '17', '0'])
    X = np.concatenate([X, oh], 1); names = names + ['Protocol=%s' % c for c in cols]; is_oh += [True] * len(cols)
    meta = dict(features=names, is_onehot=is_oh, cat_names=cat_names, n_ips=len(ips),
                day_names=['Mon 3 Jul', 'Tue 4 Jul', 'Wed 5 Jul', 'Thu 6 Jul', 'Fri 7 Jul'],
                note='improved CICIDS2017 (Engelen et al. 2021), sorted by Timestamp; "Attempted" flows removed')
    save('cic17', X, y, src, dst, cat, t.astype(np.float64), day, meta)

TON = os.environ.get('DDA_TON_ZIP', os.path.join(OUT, 'raw', 'NF-ToN-IoT-v3.zip'))

def prep_ton(every=10):
    """NF-ToN-IoT-v3 (NetFlow v3; 6 days, 23-29 Apr 2019; 27.5M flows already sorted by start time).
    A 1-in-`every` systematic sample is kept (time order and per-address mix preserved)."""
    import zipfile
    zf = zipfile.ZipFile(TON)
    name = [n for n in zf.namelist() if n.endswith('NF-ToN-IoT-v3.csv')][0]
    parts = []; off = 0
    meta_cols = {'IPV4_SRC_ADDR': str, 'IPV4_DST_ADDR': str, 'Attack': str}
    for ch in pd.read_csv(zf.open(name), chunksize=400_000, dtype=meta_cols, low_memory=True):
        keep = (np.arange(off, off + len(ch)) % every) == 0
        sub = ch[keep].copy(); off += len(ch); del ch
        for c in sub.columns:
            if c not in meta_cols and c not in ('FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'):
                sub[c] = pd.to_numeric(sub[c], errors='coerce').astype(np.float32)
        parts.append(sub)
        if off % 4_000_000 == 0:
            print('read', off, flush=True)
    df = pd.concat(parts, ignore_index=True); del parts
    t = df['FLOW_START_MILLISECONDS'].values.astype(np.float64) / 1000.0
    day = ((t - 86400 * np.floor(t[0] / 86400)) // 86400).astype(np.int8)
    y = df['Label'].astype(np.int8).values
    lab = df['Attack'].astype(str)
    cat_names = sorted(c for c in lab.unique() if c != 'Benign')
    cmap = {c: i for i, c in enumerate(cat_names)}
    cat = np.array([cmap.get(c, -1) for c in lab], np.int16)
    ips = pd.Index(pd.unique(pd.concat([df['IPV4_SRC_ADDR'], df['IPV4_DST_ADDR']]).astype(str)))
    src = ips.get_indexer(df['IPV4_SRC_ADDR'].astype(str)).astype(np.int32)
    dst = ips.get_indexer(df['IPV4_DST_ADDR'].astype(str)).astype(np.int32)
    drop = ['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR',
            'L4_DST_PORT', 'PROTOCOL', 'Label', 'Attack']
    feat = df.drop(columns=drop)
    names = list(feat.columns)
    A = np.empty((len(df), len(names)), np.float32)
    for j, c in enumerate(names):
        v = pd.to_numeric(feat[c], errors='coerce').values.astype(np.float64)
        v[~np.isfinite(v)] = 0.0
        A[:, j] = slog(v)
    is_oh = [False] * len(names)
    oh, cols = onehot(df['PROTOCOL'].astype(int).astype(str), ['6', '17', '1'])
    X = np.concatenate([A, oh], 1); names += ['PROTOCOL=%s' % c for c in cols]; is_oh += [True] * len(cols)
    meta = dict(features=names, is_onehot=is_oh, cat_names=cat_names, n_ips=len(ips), sample_every=every,
                note='NF-ToN-IoT-v3, 1-in-%d systematic sample in start-time order; day 0 = 23 Apr 2019' % every)
    save('ton', X, y, src, dst, cat, t, day, meta)

if __name__ == '__main__':
    for w in sys.argv[1:]:
        {'unsw': prep_unsw, 'cic': prep_cic, 'ton': prep_ton}[w]()
