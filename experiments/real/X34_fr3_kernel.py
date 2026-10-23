#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X34_fr3_kernel.py — does the local kernel psi head improve the MAIN FR3 result?
(registered before run)

X32 showed the kernel head beats the global linear head on the Kinova socket and
removes the collapse without L2. Here we test the same locality on the FR3 arm
isolated correction (the paper's main positive result, 1.14/1.17 mm), replacing the
Fourier psi with a local RBF head over the pair features (centres = k-means of the
feature vectors seen on the frozen exact edge).

Protocol: FR3, X7 splits rng(1000+seed), seeds 0-4, N=20/40/60/80, isolation on the
frozen exact dense edge, E34 recipe. Variants: base (Fourier psi) and kernel (RBF).

Registered prediction (plan 2026-10-14, before run):
  P-FRK: on FR3 the kernel psi is not worse than the Fourier psi at N>=60 (paired)
         and improves or is neutral at small N; if the kernel is worse, locality
         does not transfer from the socket to the arm.

Output: repro/json/X34_fr3_kernel.<ts>.json.
Run:  python X34_fr3_kernel.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X14_fourier_law as X14               # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X24_fr3_gate as X24                  # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


FEAT = os.environ.get('FEAT', 'fourier')
QF = FEAT == 'quat'
HY = FEAT == 'hybrid'


def feats30(A, B):
    feats = []
    for p in (A, B):
        th, ax, ay, az, ts = X14.chart_feats(p)
        for x in (th, ax, ay, az):
            feats += [torch.ones_like(x), torch.sin(x), torch.cos(x)]
        feats += [ts[..., 0:1], ts[..., 1:2], ts[..., 2:3]]
    return torch.cat(feats, -1)


def raw_feats(A, B):
    if QF or HY:
        fs = []
        for p in (A, B):
            q, t = p
            fs += [q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4],
                   t[..., 0:1], t[..., 1:2], t[..., 2:3]]
        return torch.cat(fs, -1)   # (...,14)
    feats = []
    for p in (A, B):
        th, ax, ay, az, ts = X14.chart_feats(p)
        for x in (th, ax, ay, az):
            feats += [torch.ones_like(x), torch.sin(x), torch.cos(x)]
        feats += [ts[..., 0:1], ts[..., 1:2], ts[..., 2:3]]
    return torch.cat(feats, -1)   # (...,30)


class KernelFourierPsi(nn.Module):
    def __init__(self, centers, bw):
        super().__init__()
        self.register_buffer('centers', centers.clone())
        self.bw = float(bw)
        self.W = nn.Parameter(torch.zeros(centers.shape[0], 6, dtype=DT))

    def forward(self, A, B):
        f = raw_feats(A, B).reshape(-1, 14 if QF else 30)
        d2 = (torch.minimum(torch.cdist(f, self.centers),
                             torch.cdist(f, -self.centers)) ** 2) if QF else (torch.cdist(f, self.centers) ** 2)
        phi = torch.exp(-d2 / (2 * self.bw ** 2))
        return 0.1 * (phi @ self.W)


def collect_feats(m, q):
    fs = []

    def hook(mod, inp, out):
        A, B = inp
        fs.append(raw_feats(A, B).reshape(-1, 14 if QF else 30).detach().cpu())
    h = m.psi.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.tensor(q, dtype=DT))
    h.remove()
    return torch.cat(fs, 0)


def phase2(m, q_tr, oq_tr, ot_tr, epochs, lr, seed, l2=0.0):
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if l2:
            loss = loss + l2 * sum((p ** 2).sum() for p in m.psi.parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--centers', type=int, default=20)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.epochs = '20', 1, 300
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = [int(s) for s in args.sizes.split(',')]

    from scipy.cluster.vq import kmeans2
    variants = ('base', 'kernel')
    res = {n: {v: [] for v in variants} for n in sizes}

    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)

            # base: Fourier psi
            mb = copy.deepcopy(exact)
            phase2(mb, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            res[n_tr]['base'].append(GN.eval_pos(mb, q_te, oq_te, ot_te)[0])

            # kernel: collect feats from a frozen-edge model, k-means, RBF
            probe = copy.deepcopy(exact)
            feats = collect_feats(probe, q_tr)
            cen, _ = kmeans2(feats.numpy(), args.centers, minit='++', seed=s)
            centers = torch.tensor(cen, dtype=DT)
            dat = torch.cdist(centers, centers)
            auto_bw = float(torch.median(dat[dat > 0])) if (dat > 0).any() else 1.0
            mk = copy.deepcopy(exact)
            mk.psi = KernelFourierPsi(centers, auto_bw)
            phase2(mk, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            res[n_tr]['kernel'].append(GN.eval_pos(mk, q_te, oq_te, ot_te)[0])
            print(f'[X34 seed {s} n={n_tr}] base {res[n_tr]["base"][-1]:.3f} '
                  f'kernel {res[n_tr]["kernel"][-1]:.3f} (auto_bw {auto_bw:.3f})',
                  flush=True)
        print(f'[X34] seed {s} done', flush=True)

    from scipy import stats
    for n in sizes:
        for v in variants:
            a = np.array(res[n][v])
            RESULTS[f'X34:n{n}:{v}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        a = np.array(res[n]['kernel']); b = np.array(res[n]['base'])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'X34:paired:n{n}:kernel_vs_base'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    RESULTS['X34:protocol'] = dict(sizes=sizes, seeds=args.seeds, epochs=args.epochs,
                                   lr=args.lr, centers=args.centers, psi='rbf-kernel')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
