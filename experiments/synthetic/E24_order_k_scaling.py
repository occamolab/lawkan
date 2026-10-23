# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E24 — ORDER-K SCALING (registered blind test RT-2)
#
# Registered prediction (the experiment journal, commit d2af621, BEFORE this run):
# on the noiseless harmonic task theta = theta0 + eps*sin(beta):
#   (1) the K=1 representational floor is LINEAR in eps:
#       floor_K1(eps) = 1.779 * (eps/0.3) deg  -> 0.89, 1.78, 2.67, 3.56 (+-15%)
#   (2) ratio floor_K1(0.60)/floor_K1(0.15) = 4.0 +- 0.5
#   (3) floor_K2(eps) < 0.3 deg for all eps
#   (4) floor_K1(0.30) ~ 1.78 deg (reproduces the earlier value)
#
# The residual of the K=1 approximation is (eps/4) * (unit vector field), i.e.
# exactly proportional to eps, so the floor is predicted to scale linearly.
# Held-out seeds pass via --seed_start.
#
# Run:  python E24_order_k_scaling.py --seed_start 20 --seeds 5
# ============================================================
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--seed_start', type=int, default=20)
ap.add_argument('--epochs', type=int, default=3000)
args, _unknown = ap.parse_known_args()
NSEEDS = 2 if args.quick else args.seeds
SEED_START = args.seed_start
EPOCHS = 300 if args.quick else args.epochs

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,3) for v in vals]}")
def reg_text(key, text):
    RESULTS[key] = dict(text=text); print(f"[{key}] {text}")

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
    return torch.stack([torch.sin(be)*torch.cos(al), torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)
def geo_loss(pred, target):
    c = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6); return (1-c).mean()
def geo_deg(pred, target):
    c = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6); return torch.rad2deg(torch.acos(c))
PROBE = torch.tensor([0.,0.,1.], device=device)
ANG2 = (0.7, 1.1)

def make_data(N, seed=1, eps=0.0, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for _ in range(2):
        a = torch.randn(N,3,generator=g); a = a/a.norm(dim=-1,keepdim=True); axes.append(a.to(device))
    def theta(i):
        be = torch.acos(torch.clamp(axes[i][:,2], -1.0, 1.0)); return float(ANG2[i]) + eps*torch.sin(be)
    prod = axis_angle_to_quat(axes[1], theta(1))
    prod = quat_mul(prod, axis_angle_to_quat(axes[0], theta(0)))
    v = rotate(prod, torch.tensor([0.,0.,1.], device=device).expand(N,3)); v = v/v.norm(dim=-1,keepdim=True)
    X = torch.cat([vec_to_angles(axes[0]), vec_to_angles(axes[1])], -1)
    Y = vec_to_angles(v)
    if noise>0:
        X = X + noise*torch.randn_like(X); Y = Y + noise*torch.randn_like(Y)
    return X.float().to(device), Y.float().to(device)

class QK(nn.Module):
    def __init__(self, n_groups=2, K=1):
        super().__init__(); self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3) * 0.1)
    def feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1,self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]; fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], -1)
    def forward(self, x):
        qs=[]
        for gi in range(self.n_groups):
            v = torch.einsum('bf,fc->bc', self.feats(x[:,2*gi:2*gi+2]), self.w[gi])
            nv = v.norm(dim=-1,keepdim=True)
            qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)],-1))
        prod = qs[-1]
        for gi in range(self.n_groups-2,-1,-1): prod = quat_mul(prod, qs[gi])
        v = rotate(prod, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v/(v.norm(dim=-1,keepdim=True)+1e-6))

def run(K, eps):
    vals=[]
    for s in range(SEED_START, SEED_START+NSEEDS):
        Xtr,Ytr = make_data(800, seed=s, eps=eps, noise=0.0)
        Xte,Yte = make_data(4000, seed=s+100, eps=eps, noise=0.0)
        torch.manual_seed(100+s)
        net = QK(2,K).to(device); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(EPOCHS):
            opt.zero_grad(); geo_loss(net(Xtr),Ytr).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        with torch.no_grad(): vals.append(geo_deg(net(Xte),Yte).mean().item())
    return vals

print("\nE24 / K-scaling on the noiseless harmonic task")
EPS = [0.15, 0.30, 0.45, 0.60]
for eps in EPS:
    reg(f'E24:floor_K1:eps{eps}', run(1, eps))
    reg(f'E24:floor_K2:eps{eps}', run(2, eps))

f1 = [RESULTS[f'E24:floor_K1:eps{e}']['mean'] for e in EPS]
f2 = [RESULTS[f'E24:floor_K2:eps{e}']['mean'] for e in EPS]
pred = [1.779*(e/0.3) for e in EPS]
for e, m, p in zip(EPS, f1, pred):
    reg_text(f'E24:check:eps{e}', f"K1 measured {m:.3f} vs predicted {p:.3f} ({(m/p-1)*100:+.1f}%)")
reg_text('E24:ratio_0.60_over_0.15', f"{f1[-1]/f1[0]:.2f} (predicted 4.0±0.5)")
reg_text('E24:K2_max', f"max K2 floor {max(f2):.3f} (predicted <0.3)")

print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=1, default=float))
print("\nE24 COMPLETE")
