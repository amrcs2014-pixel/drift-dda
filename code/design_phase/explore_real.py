import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""One-off profiling of the two real-IP datasets (temporal span, attack mix, IP concentration)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, sys

from prep_real import UNSW, CIC   # raw-data locations (environment variables DDA_UNSW_DIR, DDA_CIC_DIR)

def prof_unsw():
    df = pd.concat([pq.read_table(f, columns=['srcip', 'dstip', 'Stime', 'attack_cat', 'label']).to_pandas() for f in UNSW])
    print('UNSW rows', len(df), 'dup check')
    df = df.sort_values('Stime')
    t = pd.to_datetime(df['Stime'], unit='s')
    print('time span', t.min(), t.max())
    df['hour'] = t.dt.floor('h')
    g = df.groupby('hour').agg(n=('label', 'size'), att=('label', 'mean'))
    print(g.to_string())
    att = df[df.label == 1]
    vc = att['srcip'].value_counts()
    print('attack src IPs', len(vc), 'top10 share', vc.head(10).sum() / len(att))
    print(vc.head(12))
    ben = df[df.label == 0]
    vb = ben['srcip'].value_counts()
    print('benign src IPs', len(vb), vb.head(8))
    print('overlap attack/benign src', len(set(vc.index) & set(vb.index)))
    print(df['attack_cat'].value_counts())

def prof_cic():
    for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']:
        df = pd.read_csv(f"{CIC}\\{day}.csv", usecols=['Src IP', 'Dst IP', 'Timestamp', 'Label', 'Attempted Category'])
        t = pd.to_datetime(df['Timestamp'])
        lab = (df['Label'] != 'BENIGN')
        print(f'== {day}: rows {len(df)} span {t.min()} .. {t.max()} attack frac {lab.mean():.3f}')
        print(df['Label'].value_counts().head(10).to_string())
        att = df[lab]
        if len(att):
            vc = att['Src IP'].value_counts(); print('  attack src IPs', len(vc), 'top3 share', round(vc.head(3).sum() / len(att), 3), vc.head(5).to_dict())
        vb = df[~lab]['Src IP'].value_counts(); print('  benign src IPs', len(vb))
        print('  attempted (label!=BENIGN?)', df['Attempted Category'].value_counts().head(5).to_dict())

if __name__ == '__main__':
    which = sys.argv[1]
    prof_unsw() if which == 'unsw' else prof_cic()
