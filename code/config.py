"""Configurations. Model sizes follow v6 (synthetic adaptation epochs raised from 3 to 10 so that a Tier-3
update is a genuine warm re-fit); dictionary parameters on real traces follow GAN-DDA's published values
(T=10, L=2000); the DMAC gate parameters (lambda, rho2, rho3) are selected on held-out synthetic tuning
streams by tune.py (results/tuning_selected.json) and frozen for every test stream."""

# v7 controller design (fixed before tuning; each element is ablated in experiments.exp_ablation)
V7_DESIGN = dict(
    rebase=True, verify_m=2,          # PH reference re-estimated after every response (all alarm-driven policies)
    severity='score',                 # threshold inflation of trusted traffic (v6: 'rate' = anomaly-rate ratio)
    escalate=True,                    # Tier-2 -> Tier-3 when the update leaves inflation >= rho2
    tau_rule='clean',                 # threshold reset on the trusted proxy (v6: 'retrain', score-truncated)
    rule='repwiden', pi_w=95, clean_frac=0.5,   # widened retrain band only from mostly-benign sources
    bl_majority=True,                 # blacklist needs >= L suspicious verdicts AND a suspicious majority
    src_cap_frac=0,                   # per-source influence cap: rejected in pilots (rare sources dominate)
    amnesty=1.0,                      # evidence discounting off (rejected in pilots: feedback loop)
    surge=True, surge_gamma=5.0, surge_min_share=0.05,   # volume-surge quarantine (history frozen while surging)
    defer_frac=0.5,                   # skip windows dominated (>50%) by quarantined sources
    monitor='raw',
)

DMAC_V6 = dict(warmup=3, ph_delta=0.005, ph_lambda=0.05, rate_t2=1.6, rate_t3=2.5, cooldown=2, adapt_incl_pct=70)

DMAC_TUNED = dict(DMAC_V6); DMAC_TUNED.update(V7_DESIGN); DMAC_TUNED.update(rate_t2=1.5, rate_t3=3.0)

MODEL_SYN = dict(latent_dim=8, hidden=48, lr=1e-3, batch=128, n_critic=3, gp_lambda=10.0, alpha=1.0,
                 wgan_epochs=10, enc_epochs=10, adapt_gan_epochs=10, adapt_enc_epochs=10, percentile=95,
                 buffer_size=3000, replay_k=1000, periodic_k=5)

MODEL_REAL = dict(latent_dim=16, hidden=128, lr=2e-4, batch=256, n_critic=5, gp_lambda=10.0, alpha=1.0,
                  wgan_epochs=30, enc_epochs=30, adapt_gan_epochs=8, adapt_enc_epochs=8, percentile=95,
                  buffer_size=5000, replay_k=2000, periodic_k=25)

DICT_SYN = dict(T=10, L=20)
DICT_REAL = dict(T=10, L=2000)

def syn_cfg(dmac=None, **over):
    c = dict(MODEL_SYN); c.update(DICT_SYN); c.update(dmac or DMAC_TUNED); c.update(over); return c

def real_cfg(dmac=None, **over):
    c = dict(MODEL_REAL); c.update(DICT_REAL); c.update(dmac or DMAC_TUNED); c.update(over); return c
