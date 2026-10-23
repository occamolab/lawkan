# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — E25: state-conditioned LawKAN crosses the rigid-body boundary
#
# Registered prediction (the experiment journal, commit a08797a), held-out
# seeds 30-34. Task identical to E17: input = initial angular velocity w0,
# target = orientation after 20 Euler steps (I=diag(1,2,3), dt=0.05).
#
# E17 negative result: the FEED-FORWARD LawKAN (GroupedQuatExpNet, 2 groups)
# composes learned functions of w0 and cannot represent the ODE solution map
# (12.66 +- 5.34 deg, seed-bimodal); MLP h=32 gets 0.27 deg.
#
# E25 model: state-conditioned edge + the SAME exact composition node.
#   w_{t+1} = w_t + G_theta(w_t) dt
#   q_{t+1} = q_t (x) exp(w_{t+1} dt)          (semi-implicit, as the data)
# G_theta is a polynomial edge (linear / quad / cubic). The node is exact.
#
# Run: python E25_state_rigid.py
# ============================================================
import torch, torch.nn as nn, math, time, json
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,4) for v in vals]}")

# ---------------- geometry (identical to run_extensions.py) ----------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)
def axis_angle_to_quat(axis, angle):
    half = angle.unsqueeze(-1)/2
    return torch.cat([torch.cos(half), torch.sin(half)*axis], dim=-1)
def rotate(q, v):
    u, w = q[...,1:], q[...,:1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)
def vec_to_angles(v):
    al = torch.atan2(v[...,1], v[...,0])
    be = torch.acos(torch.clamp(v[...,2], -1+1e-6, 1-1e-6))
    return torch.stack([al, be], -1)
def angles_to_vec(a):
    al, be = a[...,0], a[...,1]
    return torch.stack([torch.sin(be)*torch.cos(al), torch.sin(be)*torch.sin(al), torch.cos(be)], -1)
def geo_loss(pred, target):
    vp, vt = angles_to_vec(pred), angles_to_vec(target)
    return (1 - (vp*vt).sum(-1).clamp(-1+1e-6, 1-1e-6)).mean()
def geo_deg(pred, target):
    vp, vt = angles_to_vec(pred), angles_to_vec(target)
    return torch.rad2deg(torch.acos((vp*vt).sum(-1).clamp(-1+1e-6, 1-1e-6)))
def n_params(m): return sum(p.numel() for p in m.parameters())
PROBE = torch.tensor([0.0, 0.0, 1.0]).to(device)
STEPS, DT = 20, 0.05
I_TENSOR = torch.tensor([1.0, 2.0, 3.0]).to(device)

# ---------------- data (verbatim from E17) ----------------
def make_rigid_data(N, steps=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    I = torch.tensor([1.0, 2.0, 3.0]).to(device)
    w = ((torch.rand(N,3,generator=g)*1.0 + 0.5) * torch.sign(torch.randn(N,3,generator=g))).to(device)
    w0 = w.clone()
    q = torch.tensor([1.0,0,0,0]).expand(N,4).clone().to(device)
    dt = 0.05
    for s in range(steps):
        Iw = w * I
        dw = -torch.cross(w, Iw, dim=-1) / I
        w = w + dw*dt
        wn = w.norm(dim=-1).clamp_min(1e-8)
        dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*dt)
        q = quat_mul(q, dq); q = q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(N,3))
    Y = vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))
    return w0.to(device), Y.to(device)

def rigid_rollout(w0, steps=20):
    w = w0.clone()
    q = torch.tensor([1.0,0,0,0]).expand(w0.shape[0],4).clone().to(device)
    for s in range(steps):
        dw = -torch.cross(w, w*I_TENSOR, dim=-1)/I_TENSOR
        w = w + dw*DT
        wn = w.norm(dim=-1).clamp_min(1e-8)
        dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*DT)
        q = quat_mul(q, dq); q = q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(w0.shape[0],3))
    return vec_to_angles(v/(v.norm(dim=-1,keepdim=True)+1e-6))

# ---------------- feed-forward LawKAN (E17 negative baseline) ----------------
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

# ---------------- state-conditioned LawKAN ----------------
def poly_features(w, mode):
    one = torch.ones_like(w[..., :1])
    if mode == 'linear':
        return torch.cat([one, w], -1)                                   # 4
    outer = (w.unsqueeze(-1) * w.unsqueeze(-2)).reshape(-1, 9)           # 9
    if mode == 'quad':
        return torch.cat([one, w, outer], -1)                            # 13
    cube = (outer.unsqueeze(-1) * w.unsqueeze(-2)).reshape(-1, 27)       # 27
    return torch.cat([one, w, outer, cube], -1)                          # 40

class StateKAN(nn.Module):
    """Edge = polynomial in the state w -> dw; node = exp + Hamilton product."""
    def __init__(self, mode='quad'):
        super().__init__(); self.mode = mode
        d = {'linear':4, 'quad':13, 'wcubic':40}[mode]
        self.W = nn.Parameter(torch.zeros(d, 3).normal_(std=0.01))
    def forward(self, w0):
        B = w0.shape[0]; w = w0
        q = torch.tensor([1.,0,0,0], device=device).expand(B,4).clone()
        for s in range(STEPS):
            dw = poly_features(w, self.mode) @ self.W
            w = w + dw*DT
            wn = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            q = quat_mul(q, axis_angle_to_quat(w/wn, (wn*DT).squeeze(-1)))
            q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = rotate(q, PROBE.expand(B,3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True) + 1e-6))

class MLPFlat(nn.Module):
    def __init__(self, in_dim, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        q = self.net(x); q = q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v/(v.norm(dim=-1,keepdim=True)+1e-6))

# ---------------- train / eval ----------------
def train_eval(ctor, lr, seed, epochs=3000, do_pad=False):
    torch.manual_seed(100+seed)
    Xtr, Ytr = make_rigid_data(800, seed=seed)
    Xte, Yte = make_rigid_data(4000, seed=seed+100)
    def pad(X): return torch.cat([X, torch.zeros(X.shape[0],1).to(device)], -1)
    if do_pad: Xtr, Xte = pad(Xtr), pad(Xte)
    m = ctor().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    diverged = False
    for ep in range(epochs):
        opt.zero_grad(); loss = geo_loss(m(Xtr), Ytr); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            diverged = True; break
    with torch.no_grad():
        err = geo_deg(m(Xte), Yte).mean().item()
    return (float('nan') if diverged else err), n_params(m)

def main():
    t0 = time.time()
    # control: chaos divergence
    gc = torch.Generator().manual_seed(0)
    w0c = ((torch.rand(2000,3,generator=gc)*1.0 + 0.5) * torch.sign(torch.randn(2000,3,generator=gc))).to(device)
    pert = 1e-3 * torch.randn(2000,3,generator=gc).to(device)
    div = geo_deg(rigid_rollout(w0c), rigid_rollout(w0c + pert)).mean().item()
    print(f"control: 1e-3 perturbation of omega0 -> {div:.4f} deg after 20 steps")
    RESULTS['E25:control'] = dict(chaos_deg=div)

    # analytic floor
    Xte_ref, Yte_ref = make_rigid_data(4000, seed=30+100)
    ana = geo_deg(rigid_rollout(Xte_ref), Yte_ref).mean().item()
    reg('E25:analytic', [ana], params=0)

    configs = [
        ('FeedForward-QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2, True),
        ('StateKAN-linear',     lambda: StateKAN('linear'),        1e-2, False),
        ('StateKAN-quad',       lambda: StateKAN('quad'),          1e-2, False),
        ('StateKAN-cubic',      lambda: StateKAN('wcubic'),        1e-3, False),
        ('MLP h=32',            lambda: MLPFlat(3, 32),            3e-3, False),
        ('MLP h=128',           lambda: MLPFlat(3, 128),           3e-3, False),
    ]
    for name, ctor, lr, do_pad in configs:
        vals = []; ps = 0
        for seed in range(30, 35):
            e, ps = train_eval(ctor, lr, seed, do_pad=do_pad)
            vals.append(e)
        reg(f'E25:{name}', vals, params=ps, lr=lr, held_out_seeds=[30,31,32,33,34])
    RESULTS['E25:wall_s'] = time.time()-t0
    save_dump(RESULTS, __file__, seeds=5, seed_start=30, epochs=3000)
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
