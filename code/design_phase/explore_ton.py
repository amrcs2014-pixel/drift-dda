import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), ".."))
"""Profile NF-ToN-IoT-v3 (time span per hour, attack share, attacker-IP concentration) without loading it all."""
import zipfile, io, collections
import pandas as pd, numpy as np

from prep_real import TON as Z   # raw-data location (environment variable DDA_TON_ZIP)
zf = zipfile.ZipFile(Z)
name = [n for n in zf.namelist() if n.endswith('NF-ToN-IoT-v3.csv')][0]
hour = collections.Counter(); hatt = collections.Counter(); asrc = collections.Counter(); bsrc = collections.Counter()
cats = collections.Counter(); n = 0; tmin, tmax = 1e20, 0
sorted_flag = True; last = 0
for ch in pd.read_csv(zf.open(name), usecols=['FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'Label', 'Attack'], chunksize=2_000_000):
    t = ch['FLOW_START_MILLISECONDS'].values
    if t[0] < last: sorted_flag = False
    if (np.diff(t) < 0).any(): sorted_flag = False
    last = t[-1]
    tmin = min(tmin, t.min()); tmax = max(tmax, t.max())
    h = (t // 3_600_000)
    hour.update(h.tolist()); hatt.update(h[ch['Label'].values == 1].tolist())
    asrc.update(ch.loc[ch['Label'] == 1, 'IPV4_SRC_ADDR'].tolist()); bsrc.update(ch.loc[ch['Label'] == 0, 'IPV4_SRC_ADDR'].tolist())
    cats.update(ch['Attack'].tolist()); n += len(ch)
    print('rows', n, flush=True)
print('total', n, 'sorted by start time:', sorted_flag)
print('span', pd.to_datetime(tmin, unit='ms'), pd.to_datetime(tmax, unit='ms'))
for h in sorted(hour):
    print(pd.to_datetime(h * 3600, unit='s'), hour[h], round(hatt[h] / hour[h], 3))
print('attack src IPs', len(asrc), asrc.most_common(12))
print('benign src IPs', len(bsrc), bsrc.most_common(8))
print(cats.most_common())
