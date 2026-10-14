"""Descriptive statistics of the real traces (paper Table: datasets) -> results/data_table.json."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json, sys
import numpy as np
from streams import load_cache, real_split, RealStream

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def describe(name):
    arr, meta = load_cache(name)
    _, _, ci, lo, hi = real_split(name, 10000, 0)
    st = RealStream(name, 4096, 0)
    y = st.y; src = st.src; t = st.t
    att_src = np.unique(src[y == 1]); ben_src = np.unique(src[y == 0])
    vc = np.bincount(np.searchsorted(att_src, src[y == 1])) if len(att_src) else np.array([0])
    top = np.sort(vc)[::-1]
    import pandas as pd
    out = dict(
        name=name, total_flows=int(len(arr['y'])), n_features_raw=len(meta['features']),
        config_flows=int(len(ci)), config_attack_share=float(np.asarray(arr['y'])[ci].mean()),
        test_flows=int(st.n_windows * 4096), test_windows=int(st.n_windows), window=4096,
        test_attack_share=float(y.mean()), attack_categories=meta['cat_names'],
        attacker_src=int(len(att_src)), benign_src=int(len(ben_src)), src_both=int(len(np.intersect1d(att_src, ben_src))),
        top1_attack_share=float(top[0] / max(1, top.sum())), top3_attack_share=float(top[:3].sum() / max(1, top.sum())),
        test_start=str(pd.to_datetime(t[0], unit='s')), test_end=str(pd.to_datetime(t[-1], unit='s')),
        change_points=st.drift_points,
        windows_with_attacks=float(np.mean([y[i * 4096:(i + 1) * 4096].any() for i in range(st.n_windows)])),
        max_window_attack_share=float(max(y[i * 4096:(i + 1) * 4096].mean() for i in range(st.n_windows))),
    )
    return out

if __name__ == '__main__':
    names = sys.argv[1:] or ['unsw', 'cic17']
    p = os.path.join(RES, 'data_table.json')
    tab = json.load(open(p)) if os.path.exists(p) else {}
    for n in names:
        tab[n] = describe(n); print(json.dumps(tab[n], indent=1))
    json.dump(tab, open(p, 'w'), indent=1)
