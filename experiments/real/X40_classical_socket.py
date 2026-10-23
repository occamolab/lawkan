#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X40_classical_socket.py — a STANDALONE classical socket estimator (plain product-of-
exponentials calibration), trained with the SAME optimizer as our models (Adam,
cosine schedule, 3000 steps). (registered before run)

Motivation. Even with psi=0 our recipe is a different implementation (a KAN-style
node carrying the exact law, edges carrying increments), so comparing our models with
"our dense" is not a classical-vs-ours comparison. This script implements the classical
estimator independently --- the standard PoE kinematic model, no KAN node, no psi ---
and fits it with the same optimizer we use, so the only difference to our models is the
architecture.

Model (n joints, per joint a screw xi_j = xi_nom,j + dxi_j):
    T(q) = M0 . prod_j exp( xi_j (q_j + b_j) ) . exp(tool)
  classical_dense   : dxi, b, tool free (55p);
  classical_reduced : dxi = 0, b, tool free (13p);
  classical_nominal : dxi = 0, b = 0, tool free (6p).
Loss: MK1 socket_loss (Var(p0)+Var(p1)+lam*(||c0-c1||-0.05)^2), identical to ours.
Optimizer: Adam lr 1e-3, cosine to 0, 3000 steps (identical to our runs).

Registered prediction (plan 2026-10-16):
  P-CLS-2: with a matched optimizer, the classical reduced matches our reduced edge_b,
           and the classical dense matches our dense (same class, same optimizer);
           any gap is architecture, not optimizer.

Output: repro/json/X40_classical_socket.<ts>.json via repro/dump.
Run:  python X40_classical_socket.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import MK1_mukca as MK                      # noqa: E402  (data + metric only)
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}
SOCKET = ['kuka14', 'panda_1', 'fr3', 'kinova']
MODELS = ('nominal', 'reduced', 'dense')


def se3_exp(xi):
    """xi: (...,6) -> (...,4,4), series near zero (standalone)."""
    w = xi[..., :3]
    v = xi[..., 3:]
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = torch.zeros(*xi.shape[:-1], 3, 3, dtype=xi.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = w[..., 2], -w[..., 1], w[..., 0]
    I = torch.eye(3, dtype=xi.dtype).expand_as(K)
    R = I + (torch.sin(th) / th)[..., None] * K + ((1 - torch.cos(th)) / th ** 2)[..., None] * (K @ K)
    V = (I + ((1 - torch.cos(th)) / th ** 2)[..., None] * K
         + ((th - torch.sin(th)) / th ** 3)[..., None] * (K @ K))
    t = (V @ v[..., None])[..., 0]
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


class ClassicalPoE(torch.nn.Module):
    def __init__(self, xi0, M0, mode):
        super().__init__()
        self.n = xi0.shape[0]
        self.register_buffer('xi0', xi0.clone())
        self.register_buffer('M0', M0.clone())
        if mode == 'dense':
            self.dxi = torch.nn.Parameter(torch.zeros(self.n, 6, dtype=DT))
        else:
            self.register_buffer('dxi', torch.zeros(self.n, 6, dtype=DT))
        if mode == 'nominal':
            self.register_buffer('b', torch.zeros(self.n, dtype=DT))
        else:
            self.b = torch.nn.Parameter(torch.zeros(self.n, dtype=DT))
        self.tool = torch.nn.Parameter(torch.zeros(6, dtype=DT))

    def forward(self, q):
        T = self.M0.unsqueeze(0).expand(q.shape[0], 4, 4).clone()
        tool = se3_exp(self.tool)
        for j in range(self.n):
            ang = q[:, j] + self.b[j]
            xi = self.xi0[j] + self.dxi[j]
            T = T @ se3_exp(xi * ang[:, None])
        return T @ tool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default=','.join(SOCKET))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--fr3_seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.fr3_seeds, args.steps = 1, 1, 300
    t0 = time.time()

    for plat in args.platforms.split(','):
        xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
        q0, q1 = MK.load_sockets(plat)
        nseed = args.fr3_seeds if plat == 'fr3' else args.seeds
        print(f'[X40 {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}, '
              f'{nseed} seeds', flush=True)
        rms = {k: [] for k in MODELS}
        for s in range(nseed):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]
            q0t = torch.tensor(q0_tr, dtype=DT)
            q1t = torch.tensor(q1_tr, dtype=DT)

            line = f'[X40 {plat} seed {s}]'
            for mode in MODELS:
                torch.manual_seed(s)
                m = ClassicalPoE(xi, M0, mode)
                opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=0)
                for _ in range(args.steps):
                    opt.zero_grad()
                    loss = MK.socket_loss(m, q0t, q1t)
                    loss.backward()
                    opt.step()
                    sch.step()
                r, dd = MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)
                rms[mode].append(r)
                line += f' {mode} {r:.3f}'
            print(line + ' mm', flush=True)

        for k in MODELS:
            a = np.array(rms[k])
            RESULTS[f'X40:{plat}:{k}:rms_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                per_seed=[float(x) for x in a])

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=args.steps,
              extra_meta={'method': 'standalone classical PoE socket calibration',
                          'optimizer': 'Adam lr 1e-3, cosine, 3000 steps (same as our models)',
                          'note': 'P-CLS-2: matched-optimizer classical reference vs our edges'})


if __name__ == '__main__':
    main()
