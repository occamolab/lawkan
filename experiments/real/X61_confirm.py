#!/usr/bin/env python3
"""X61 -- branch-closing confirmation.

3 configs x seeds {1,2,3}, N=60, 200 epochs; raw trace + EMA(beta=0.99) trace.
cfg: base (f30 two-slot) | gonly (q14 global-only) | kernel (q14 two-slot).
T1 (registered): gonly_final - base_final within TOST margin +-0.11.
T2 (registered): |EMA final - raw final| <= 0.02.
"""
import argparse, copy, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument('--cfg', choices=['base', 'gonly', 'kernel'], required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--N', type=int, default=60)
ap.add_argument('--epochs', type=int, default=200)
a = ap.parse_args()
os.environ['FEAT'] = 'fourier' if a.cfg == 'base' else 'quat'

import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import X7_robot_calibration as X7, X24_fr3_gate as X24, X33_fr3_twoslot as X33
import X34_fr3_kernel as X34, X7_gn_ls as GN
from scipy.cluster.vq import kmeans2
DT = torch.float64
BETA, CK = 0.99, 25


def train_trace(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr, seed):
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi.Wk.requires_grad_(True)
    m.psi.global_lin.weight.requires_grad_(True)
    m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    ema = [p.detach().clone() for p in params]
    raw, em = {}, {}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        qp, tp = m(q_tr)
        dp = (tp - ot_tr).norm(dim=-1)
        dot = (qp * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema, params):
                e.mul_(BETA).add_(p, alpha=1 - BETA)
        if ep % CK == 0:
            raw[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
            saved = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e)
            em[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
    return raw, em


q_np, obs_np = X7.load_philnad_dataset(os.path.join(_HERE, 'data', 'calib_data.pickle'))
C, E = X7.extract_nominal_chain()
q_all = torch.tensor(q_np, dtype=DT)
oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
perm = np.random.default_rng(1000 + a.seed).permutation(len(q_np))
tr, te = perm[:a.N], perm[a.N:]
q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, a.seed)
feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=a.seed)
centers = torch.tensor(cen, dtype=DT)
d = torch.cdist(centers, centers)
bw = float(torch.median(d[d > 0]))
m = X33.make_two_slot(exact, centers, bw)
if a.cfg == 'gonly':
    m.psi.use_kernel = False
raw, em = train_trace(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, a.epochs, 5e-3, a.seed)
print(json.dumps({'cfg': a.cfg, 'seed': a.seed, 'bw': bw,
                  'raw': raw, 'ema': em}), flush=True)
