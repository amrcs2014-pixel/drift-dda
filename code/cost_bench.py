"""Wall-clock cost of each adaptation tier vs model width and device.

Runs unchanged on a Raspberry Pi (pip install torch numpy scipy scikit-learn; copy this folder):
    python cost_bench.py --device cpu --threads 4 --out pi5.json
On the workstation:
    python cost_bench.py --device cpu --threads 1 --out cpu1.json
    python cost_bench.py --device cpu --threads 6 --out cpu6.json
    python cost_bench.py --device cuda --out gpu.json

Each measurement times one Tier-2 (encoder fine-tune) and one Tier-3 (warm WGAN-GP + encoder) update on a
retrain set the size used on the real traces (4096-flow window + 2000 replayed flows, 8 epochs), plus the
Tier-1 threshold reset (scoring one window). Input width d=50 matches the real traces after filtering.
"""
import argparse, json, os, platform, sys, time
if 'cuda' in sys.argv:
    os.environ['DDA_ALLOW_GPU'] = '1'
import numpy as np, torch
from core import make_models, train_wgan, train_encoder, anomaly_scores

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--widths', default='48,128,512,1024'); ap.add_argument('--reps', type=int, default=3)
ap.add_argument('--d', type=int, default=50); ap.add_argument('--n', type=int, default=6096)
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--out', default='cost.json')
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = a.device
rng = np.random.RandomState(0)
X = np.tanh(rng.randn(a.n, a.d)).astype(np.float32)
W = X[:4096]
res = dict(device=dev, threads=a.threads, machine=platform.machine(), processor=platform.processor(),
           torch=torch.__version__, d=a.d, n=a.n, epochs=a.epochs, rows=[])
if dev == 'cuda':
    res['gpu'] = torch.cuda.get_device_name(0)

def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()

for h in [int(x) for x in a.widths.split(',')]:
    cfg = dict(latent_dim=16, hidden=h, lr=2e-4, batch=256, n_critic=5, gp_lambda=10.0, alpha=1.0)
    t1, t2, t3, st2, st3 = [], [], [], 0, 0
    for r in range(a.reps):
        torch.manual_seed(r)
        e, g, d = make_models(a.d, cfg, dev)
        sync(); t0 = time.perf_counter()
        s = anomaly_scores((e, g, d), W, cfg, dev); np.percentile(s, 95)
        sync(); t1.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); st2 = train_encoder(e, g, d, X, cfg, a.epochs, dev); sync(); t2.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        st3 = train_wgan(g, d, X, cfg, a.epochs, dev) + train_encoder(e, g, d, X, cfg, a.epochs, dev)
        sync(); t3.append(time.perf_counter() - t0)
    nparam = sum(p.numel() for m in (e, g, d) for p in m.parameters())
    row = dict(width=h, params=int(nparam), tier1_s=float(np.median(t1)), tier2_s=float(np.median(t2)),
               tier3_s=float(np.median(t3)), tier2_steps=int(st2), tier3_steps=int(st3))
    res['rows'].append(row)
    print(row, flush=True)
json.dump(res, open(a.out, 'w'), indent=1)
print('wrote', a.out)
