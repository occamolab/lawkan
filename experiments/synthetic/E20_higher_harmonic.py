# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E20 — HIGHER HARMONICS: when does K>1 help?
#
# The paper's E13 shows that on the canonical comp2 task K=2 is slightly WORSE
# than K=1: the exact solution lives on the K=1 analytic support, so extra
# harmonics only add variance. This script isolates the complementary regime:
# tasks on which the exact per-group rotation vector is NOT K=1-representable,
# so extra harmonics are capacity, not variance.
#
# Task family. Each group's angle depends on its axis:  theta_i = theta0_i +
# eps * sin(beta_i),  with beta_i the polar angle of the axis. The exact edge
# target is v_i = (theta_i/2) a_i. Its components are
#     (theta0/2) sin b cos a  +  (eps/2) sin^2 b cos a
#   = (theta0/2) sin b cos a  +  (eps/4) cos a  -  (eps/4) cos 2b cos a ...
# i.e. they contain SECOND-order (alpha,beta) harmonics. K=1 features
# {e_m(alpha) e_n(beta): |m|,|n|<=1} cannot represent the cos(2beta) terms;
# K=2 features can. So:
#   * eps = 0  -> the canonical task: K=1 exact, K=2 adds variance (E13);
#   * eps > 0  -> K=1 has a representational floor, K=2 is (near) exact.
# A noiseless representational check separates representation from optimization.
#
# Protocol identical to run_main_experiments.py: 800/4000, 5 seeds (3 for the
# noiseless check), 3000 epochs, geodesic loss, lr=1e-2, grad clip 1.0.
#
# Run:  python E20_higher_harmonic.py            (~3-5 min on an RTX 2080 SUPER)
#       python E20_higher_harmonic.py --quick    (CPU smoke test)
# ============================================================
import argparse, json, math
import numpy as np
import torch, torch.nn as nn
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--seed_start', type=int, default=10)
ap.add_argument('--epochs', type=int, default=3000)
args, _unknown = ap.parse_known_args()
NSEEDS = 2 if args.quick else args.seeds
SEED_START = args.seed_start
EPOCHS = 300 if args.quick else args.epochs

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,3) for v in vals]}")

def reg_text(key, text):
    RESULTS[key] = dict(text=text)
    print(f"[{key}] {text}")

def paired_t(a, b):
    d = np.array(a) - np.array(b); n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float('nan')
    return float(t), n-1, float(p)

# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

def axis_angle_to_quat(axis, angle):
    half = angle.unsqueeze(-1)/2
    return torch.cat([torch.cos(half), torch.sin(half)*axis], dim=-1)

def rotate(q, v):
    qv = torch.cat([torch.zeros_like(q[...,:1]), v], dim=-1)
    qi = q.clone(); qi[...,1:] = -qi[...,1:]
    return quat_mul(quat_mul(q, qv), qi)[...,1:]

def vec_to_angles(v):
    return torch.stack([torch.atan2(v[...,1], v[...,0]),
                        torch.acos(torch.clamp(v[...,2], -1+1e-6, 1-1e-6))], dim=-1)

def angles_to_vec(a):
    al, be = a[...,0], a[...,1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)

def geo_loss(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return (1-cos).mean()

def geo_deg(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(cos))

PROBE = torch.tensor([0.,0.,1.], device=device)
ANG2 = (0.7, 1.1)

def n_params(ctor):
    return sum(p.numel() for p in ctor().parameters())

# ------------------------------------------------------------
# Data: axis-dependent angle  theta_i = theta0_i + eps*sin(beta_i)
# ------------------------------------------------------------
def make_compn_angaxis(N, n=2, noise=0.03, seed=1, angles=ANG2, eps=0.0):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N, 3, generator=g); a = a/a.norm(dim=-1, keepdim=True)
        axes.append(a)
    def theta_of(i):
        beta = torch.acos(torch.clamp(axes[i][:,2], -1.0, 1.0))
        return float(angles[i]) + eps*torch.sin(beta)
    prod = axis_angle_to_quat(axes[-1], theta_of(n-1))
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[i], theta_of(i)))
    v = rotate(prod, torch.tensor([0.,0.,1.]).expand(N,3))
    v = v / v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N,2*n,generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N,2,generator=g)
    return X.to(device), Y.to(device)

# ------------------------------------------------------------
# Model: LawKAN dense edge of order K (same as run_main)
# ------------------------------------------------------------
class GroupedQuatExpNet(nn.Module):
    def __init__(self, n_groups=2, K=1):
        super().__init__()
        self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3) * 0.1)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa = [torch.ones_like(al)]; fb = [torch.ones_like(be)]
        for m in range(1, self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        qs = []
        for gi in range(self.n_groups):
            f = self.group_feats(x[:, 2*gi:2*gi+2])
            v = torch.einsum('bf,fc->bc', f, self.w[gi])
            nv = v.norm(dim=-1, keepdim=True)
            qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1))
        prod = qs[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = quat_mul(prod, qs[gi])
        v = rotate(prod, PROBE.expand(x.shape[0], 3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True) + 1e-6))

def train_eval(ctor, Xtr, Ytr, Xte, Yte, lr, epochs):
    net = ctor().to(device)   # init seeded by the caller (run_seeds: torch.manual_seed(100+s))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        geo_loss(net(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()

def run_seeds(ctor, lr, n_seeds, epochs, **data_kw):
    vals = []
    for s in range(SEED_START, SEED_START + n_seeds):
        Xtr, Ytr = make_compn_angaxis(800, seed=s, **data_kw)
        Xte, Yte = make_compn_angaxis(4000, seed=s+100, **data_kw)
        torch.manual_seed(100+s)
        vals.append(train_eval(ctor, Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

# ------------------------------------------------------------
# Monte-Carlo Bayes floor for the axis-dependent-angle task
# (posterior mean direction given input+target noise; see run_main E10)
# ------------------------------------------------------------
def bayes_floor(n=2, angles=ANG2, eps=0.0, N=2000, noise=0.03, M=256, seed=100):
    X, Y = make_compn_angaxis(N, n=n, noise=noise, seed=seed, angles=angles, eps=eps)
    g = torch.Generator().manual_seed(seed + 777)
    draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
    al = draws[..., 0::2]; be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
    w = torch.sin(be).prod(-1); w = w / w.sum(1, keepdim=True)
    axes = torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)
    prod = axis_angle_to_quat(axes[:, :, -1],
                              float(angles[-1]) + eps*torch.sin(be[:, :, -1]))
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i],
                        float(angles[i]) + eps*torch.sin(be[:, :, i])))
    v = rotate(prod, PROBE.expand(N, M, 3))
    v = v / v.norm(dim=-1, keepdim=True)
    mean_dir = (v * w.unsqueeze(-1)).sum(1)
    mean_dir = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + 1e-8)
    return geo_deg(vec_to_angles(mean_dir), Y).mean().item()

# ============================================================
# Run
# ============================================================
EPS = 0.3
print("\n" + "="*72)
print(f"E20 / higher harmonics: theta_i = theta0_i + eps*sin(beta_i), eps={EPS}")
print("="*72)

for tag, eps in [('eps=0 (canonical)', 0.0), (f'eps={EPS} (harmonic)', EPS)]:
    print(f"\n--- task {tag} ---")
    for K in [1, 2, 3]:
        ctor = lambda K=K: GroupedQuatExpNet(2, K=K)
        vals = run_seeds(ctor, 1e-2, NSEEDS, EPOCHS, angles=ANG2, eps=eps)
        reg(f'E20:{tag}:K{K}', vals, params=n_params(ctor))
    k1 = RESULTS[f'E20:{tag}:K1']['per_seed']; k2 = RESULTS[f'E20:{tag}:K2']['per_seed']
    t, df, p = paired_t(k2, k1)
    reg_text(f'E20:{tag}:ttest K2 vs K1', f"t={t:.2f}, df={df}, p={p:.2e}")
    bf = bayes_floor(2, angles=ANG2, eps=eps)
    reg_text(f'E20:{tag}:bayes', f"bayes_floor={bf:.3f} deg")

# ------------------------------------------------------------
# Noiseless representational check: K=1 should plateau, K=2 -> ~0
# ------------------------------------------------------------
print("\n--- noiseless representational check (3 seeds, eps=0.3) ---")
for K in [1, 2]:
    ctor = lambda K=K: GroupedQuatExpNet(2, K=K)
    vals = run_seeds(ctor, 1e-2, 3, EPOCHS, angles=ANG2, eps=EPS, noise=0.0)
    reg(f'E20:noiseless:K{K}', vals, params=n_params(ctor))

save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=NSEEDS,
          seed_start=SEED_START, epochs=EPOCHS)
print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=1, default=float))
print("\nE20 COMPLETE")
