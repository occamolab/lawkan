#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X32_kernel.py — local (kernel/RBF) psi head vs the global linear/MLP head
(registered before run).

The Kinova collapse is controlled by capacity (X26a: weight decay lambda=1e-4,
a narrow window). X32 replaces the global psi head with a local kernel head
psi(f) = sum_k w_k * exp(-||f-c_k||^2 / 2 bw^2), centers c_k = k-means clusters of
the pair-feature vectors f seen on the (frozen-edge) training sockets. Outside the
data the kernel decays to zero automatically (conservatism by construction), so no
bound and no L2 are needed.

Registered predictions (plan 2026-10-13):
  X32-1: the kernel head removes the collapse on seeds 8,9 WITHOUT L2;
  X32-2: the healthy-seed cost is smaller than the L2 cost (+0.12 mm at 1e-4);
  X32-3: the trade-off disappears / the window widens;
  X32-4: the best bandwidth scales with pose density (bw* ~ mean inter-pose
         distance), checked by a small bw sweep.
  Falsifier: kernel worse than base on healthy seeds -> locality is not a panacea.

Output: repro/json/X32_kernel.<ts>.json.
Run:  python X32_kernel.py [--quick]
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

import MK1_mukca as MK              # noqa: E402
import S2_ur5gtpn_relaxed as S2     # noqa: E402
import X10_fk_calib_real as X10     # noqa: E402
from repro.dump import save_dump    # noqa: E402

DT = torch.float64
RESULTS = {}
PLAT = 'kinova'


class KernelPsi(nn.Module):
    """Local head: psi(f) = phi(f) @ W, phi_k(f)=exp(-||f-c_k||^2/(2 bw^2))."""
    def __init__(self, centers, bw):
        super().__init__()
        self.register_buffer('centers', centers.clone())
        self.bw = float(bw)
        self.w = nn.Parameter(torch.zeros(centers.shape[0], 6, dtype=DT))

    def features(self, f):
        d2 = torch.cdist(f, self.centers) ** 2
        return torch.exp(-d2 / (2 * self.bw ** 2))

    def forward(self, f):
        return self.features(f) @ self.w


def build_iso(xi, M0, edge_state, seed, kernel=None):
    torch.manual_seed(seed)
    iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
    iso.load_state_dict(copy.deepcopy(edge_state), strict=False)
    if kernel is not None:
        centers, bw = kernel
        iso.psi = KernelPsi(centers, bw)
    for p in iso.parameters():
        p.requires_grad_(False)
    for p in iso.psi.parameters():
        p.requires_grad_(True)
    return iso


def collect_features(m, q):
    fs = []

    def hook(module, inp, out):
        fs.append(inp[0].detach().cpu())
    h = m.psi.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.tensor(q, dtype=DT))
    h.remove()
    return torch.cat(fs, 0)          # (B*n_joints, 14)


def fit(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    q0t = torch.tensor(q0_tr, dtype=DT)
    q1t = torch.tensor(q1_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        loss = MK.socket_loss(m, q0t, q1t)
        if l2:
            loss = loss + l2 * sum((p ** 2).sum() for p in m.psi.parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-list', default='0,1,2,8,9')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--centers', type=int, default=20)
    ap.add_argument('--bw', type=float, default=0.0,
                    help='0 -> auto = median center separation')
    ap.add_argument('--l2', type=float, default=0.0)
    ap.add_argument('--bw-sweep', default='',
                    help='comma list of bw factors of auto (for X32-4)')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seed_list, args.steps = '0,8', 300
    t0 = time.time()

    xi, M0 = MK.platform_tensors(MK.PLATFORMS[PLAT])
    q0, q1 = MK.load_sockets(PLAT)
    seeds = [int(x) for x in args.seed_list.split(',')]
    bw_factors = [float(x) for x in args.bw_sweep.split(',')] if args.bw_sweep else [1.0]
    variants = ['base'] + [f'kr_{c:g}' if bw_factors == [1.0] else f'kr_{c:g}'
                           for c in bw_factors]
    rms = {v: [] for v in ('edge_b',) + tuple(variants)}

    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        MK.fit(eb, q0_tr, q1_tr, args.steps, args.lr, s)
        rms['edge_b'].append(MK.evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)[0])
        edge_state = copy.deepcopy(eb.state_dict())

        # collect features on a frozen-edge isolated model, cluster -> centers
        probe = build_iso(xi, M0, edge_state, s)
        feats = collect_features(probe, np.concatenate([q0_tr, q1_tr], 0))
        from scipy.cluster.vq import kmeans2
        np.random.seed(s)
        cen, _ = kmeans2(feats.numpy(), args.centers, minit='++', seed=s)
        centers = torch.tensor(cen, dtype=DT)
        # auto bandwidth = median center separation
        dc = torch.cdist(centers, centers)
        auto_bw = float(torch.median(dc[dc > 0])) if (dc > 0).any() else 1.0

        # base
        m = build_iso(xi, M0, edge_state, s)
        fit(m, q0_tr, q1_tr, args.steps, args.lr, s)
        rms['base'].append(MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)[0])

        # kernel variants (bw = factor * auto)
        for f in bw_factors:
            v = f'kr_{f:g}'
            mk = build_iso(xi, M0, edge_state, s, kernel=(centers, f * auto_bw))
            fit(mk, q0_tr, q1_tr, args.steps, args.lr, s, l2=args.l2)
            rms[v].append(MK.evaluate(mk, q0_tr, q1_tr, q0_te, q1_te)[0])
        print(f'[X32 seed {s}] auto_bw {auto_bw:.3f} | edge_b {rms["edge_b"][-1]:.3f} '
              f'base {rms["base"][-1]:.3f} | ' +
              ' '.join(f'{v} {rms[v][-1]:.3f}' for v in variants[1:]), flush=True)

    for k, a in rms.items():
        a = np.array(a)
        RESULTS[f'X32:{k}:rms_mm'] = dict(mean=float(a.mean()), std=float(a.std()),
                                          per_seed=[float(x) for x in a])
    RESULTS['X32:n_collapsed'] = {v: int((np.array(rms[v]) > np.array(rms['edge_b'])).sum())
                                  for v in variants}
    RESULTS['X32:protocol'] = dict(platform=PLAT, seeds=seeds, steps=args.steps,
                                   lr=args.lr, centers=args.centers,
                                   bw_factors=bw_factors)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
