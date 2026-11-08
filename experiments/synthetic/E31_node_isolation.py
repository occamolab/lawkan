# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E31: node vs edge isolation (Tier 1 control)
#
# Fix the NODE (exp per group + Hamilton product in the data's composition
# order, probe readout) and vary only the EDGE; add controls without the node.
#   (a) node + KAN/Fourier edge (ours, GroupedQuatExpNet K=1)
#   (b) node + MLP edge (per group: phases -> cos/sin -> MLP -> so(3))
#   (c) node + linear edge (per group: cos/sin features -> 3, matched params)
# Controls: plain MLP-Quat head (no node), single-exp MLP-Lie head (no node).
#
# Protocol (as E1): comp2 (n=2, angles 0.7/1.1, sigma=0.03), 800/4000,
# geodesic loss, Adam, 3000 epochs, seeds 0-4, geodesic deg, paired t.
# Pre-registration: P49 in the experiment journal (2026-09-27).
#
# Run: python E31_node_isolation.py [--quick]
# Output: repro/json/e31.<ts>.json (smoke: -smoke suffix)
# ============================================================
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------- geometry (as run_main_experiments.py) ----------------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def rotate(q, v):
    # q*v*q* sandwich (30 vs 56 scalar ops)
    u, w = q[..., 1:], q[..., :1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)


def vec_to_angles(v):
    return torch.stack([torch.atan2(v[..., 1], v[..., 0]),
                        torch.acos(torch.clamp(v[..., 2], -1+1e-6, 1-1e-6))], dim=-1)


def angles_to_vec(a):
    al, be = a[..., 0], a[..., 1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al),
                        torch.cos(be)], dim=-1)


def geo_loss(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return (1 - cos).mean()


def geo_deg(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(cos))


PROBE = torch.tensor([0., 0., 1.], device=device)
ANG2 = (0.7, 1.1)


def make_comp2(N, noise=0.03, seed=1, angles=ANG2):
    """Composition of n=2 rotations, random axes, noisy phase inputs (E1 task)."""
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(2):
        a = torch.randn(N, 3, generator=g)
        axes.append(a/a.norm(dim=-1, keepdim=True))
    ths = [torch.full((N,), float(angles[i])) for i in range(2)]
    prod = quat_mul(angle_axis(axes[1], ths[1]), angle_axis(axes[0], ths[0]))
    v = rotate(prod, torch.tensor([0., 0., 1.]).expand(N, 3))
    v = v/v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N, 4, generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N, 2, generator=g)
    return X.to(device), Y.to(device)


def angle_axis(axis, theta):
    half = (theta/2).unsqueeze(-1) if theta.dim() == 1 else theta/2
    s = torch.sin(half)
    return torch.cat([torch.cos(half)*torch.ones_like(axis[..., :1]), s*axis], dim=-1)


# ---------------- models ----------------
def quat_from_so3(v):
    nv = v.norm(dim=-1, keepdim=True)
    return torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv + 1e-8)], dim=-1)


def probe_readout(prod):
    v = rotate(prod, PROBE.expand(prod.shape[0], 3))
    return vec_to_angles(v/(v.norm(dim=-1, keepdim=True) + 1e-6))


class NodeModel(nn.Module):
    """Fixed node: per-group edge -> so(3) -> exp -> Hamilton product -> probe."""
    def __init__(self, n_groups=2):
        super().__init__()
        self.n_groups = n_groups

    def edge_so3(self, a):
        raise NotImplementedError

    def forward(self, x):
        qs = []
        for gi in range(self.n_groups):
            self._gi = gi
            qs.append(quat_from_so3(self.edge_so3(x[:, 2*gi:2*gi+2])))
        prod = qs[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = quat_mul(prod, qs[gi])
        return probe_readout(prod)


class NodeKANEdge(NodeModel):
    """(a) KAN/Fourier tensor-product edge (ours), K=1."""
    def __init__(self, n_groups=2, K=1):
        super().__init__(n_groups)
        self.K = K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3) * 0.1)

    def group_feats(self, a):
        al, be = a[:, 0], a[:, 1]
        fa = [torch.ones_like(al)]
        fb = [torch.ones_like(be)]
        for m in range(1, self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)

    def edge_so3(self, a):
        f = self.group_feats(a)
        return torch.einsum('bf,fc->bc', f, self.w[self._gi])


class NodeLinearEdge(NodeModel):
    """(c) linear edge over cos/sin features (matched params to K=1)."""
    def __init__(self, n_groups=2):
        super().__init__(n_groups)
        self.lin = nn.ModuleList([nn.Linear(4, 3) for _ in range(n_groups)])

    def edge_so3(self, a):
        f = torch.cat([torch.cos(a), torch.sin(a)], -1)
        return self.lin[self._gi](f)


class NodeMLPEdge(NodeModel):
    """(b) MLP edge per group; one hidden layer if hidden_layers=1 (matched)."""
    def __init__(self, n_groups=2, h=8, hidden_layers=2):
        super().__init__(n_groups)
        nets = []
        for _ in range(n_groups):
            layers = [nn.Linear(4, h), nn.SiLU()]
            for _ in range(hidden_layers-1):
                layers += [nn.Linear(h, h), nn.SiLU()]
            layers += [nn.Linear(h, 3)]
            nets.append(nn.Sequential(*layers))
        self.nets = nn.ModuleList(nets)

    def edge_so3(self, a):
        f = torch.cat([torch.cos(a), torch.sin(a)], -1)
        return self.nets[self._gi](f)


class MLPQuatHead(nn.Module):
    """Control: plain MLP -> quaternion, probe readout; NO node."""
    def __init__(self, in_phases=4, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 4))

    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        q = self.net(f)
        q = q/q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = rotate(q, PROBE.expand(x.shape[0], 3))
        return vec_to_angles(v/(v.norm(dim=-1, keepdim=True) + 1e-6))


class MLPLieHead(nn.Module):
    """Control: MLP -> so(3) -> one exp; single rotation, NO node/product."""
    def __init__(self, in_phases=4, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))

    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return probe_readout(quat_from_so3(self.net(f)))


# ---------------- protocol ----------------
def n_params(ctor):
    return sum(p.numel() for p in ctor().parameters())


def train_eval(ctor, seed, lr, epochs=3000, ntr=800, nte=4000):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    net = ctor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = geo_loss(net(Xtr), Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()


def run_seeds(ctor, lr, n_seeds, epochs, ntr, nte, seed_start=0):
    return [train_eval(ctor, seed_start+s, lr, epochs, ntr, nte) for s in range(n_seeds)]


def paired_t(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    t = d.mean()/(d.std(ddof=1)/math.sqrt(n) + 1e-12)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float('nan')
    return float(t), n-1, float(p)


MODELS = [
    ('E31:node+KAN (K=1)', lambda: NodeKANEdge(2, K=1), 1e-2),
    ('E31:node+linear', lambda: NodeLinearEdge(2), 3e-3),
    ('E31:node+MLP h=2 (1L)', lambda: NodeMLPEdge(2, h=2, hidden_layers=1), 3e-3),
    ('E31:node+MLP h=8', lambda: NodeMLPEdge(2, h=8), 3e-3),
    ('E31:node+MLP h=32', lambda: NodeMLPEdge(2, h=32), 3e-3),
    ('E31:plain MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3),
    ('E31:single-exp MLP-Lie h=32', lambda: MLPLieHead(4, h=32), 3e-3),
]
CONTROLS = ['E31:plain MLP-Quat h=32', 'E31:single-exp MLP-Lie h=32']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='smoke run (never canonical)')
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=0)
    args = ap.parse_args()

    epochs, ns = args.epochs, args.seeds
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, ntr, nte = 50, 1, 64, 128

    print(f"Device: {device} | E31 node/edge isolation | epochs={epochs} seeds={ns} "
          f"seed_start={args.seed_start}", flush=True)
    t0 = time.time()
    results = {}
    for name, ctor, lr in MODELS:
        vals = run_seeds(ctor, lr, ns, epochs, ntr, nte, args.seed_start)
        results[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                             per_seed=vals, params=int(n_params(ctor)), lr=lr,
                             held_out_seeds=list(range(args.seed_start, args.seed_start+ns)))
        print(f"[{name:30s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f} deg | "
              f"params {n_params(ctor):5d} | lr {lr:g}", flush=True)

    node_names = [n for n, _, _ in MODELS if n not in CONTROLS]
    for nname in node_names:
        for cname in CONTROLS:
            t, df, p = paired_t(results[nname]['per_seed'], results[cname]['per_seed'])
            results[f'E31:ttest:{nname} vs {cname}'] = dict(
                text=f"t={t:.2f}, df={df}, p={p:.2e}")
            print(f"[ttest] {nname} vs {cname}: t={t:.2f}, p={p:.2e}", flush=True)

    kn = results['E31:node+KAN (K=1)']['mean']
    others = [results[n]['mean'] for n in node_names if n != 'E31:node+KAN (K=1)']
    results['E31:P49.4'] = dict(text=f"node+KAN {kn:.4f} vs min(others) {min(others):.4f} "
                                     f"(diff {kn-min(others):+.4f} deg)")

    results['E31:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs)
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
