#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X33_fr3_twoslot.py — two-slot psi head on FR3 (global linear trend + local
kernel), joint vs sequential-on-residual (registered before run).

The X34 L-curve: the local kernel wins the data-starved corner (N=20), the global
linear head the data-rich corner (N>=60). The two-slot head sums them. Both slots
are zero-initialized; L2 only on the global slot (X32-5: kernel+L2 hurts).

Variants: base (global linear = Fourier psi), kernel (X32/X34 local), mixed_joint
(both trained jointly), mixed_seq (kernel first, then global on the residual).
Diagnostics: global-slot weight norm vs N; correction-off per slot.

Registered (plan 2026-10-14, before run):
  X33-1: mixed dominates both corners; X33-2: the global-slot weight grows with N
  (crossover at the X34 bend, N~40-60); X33-4: joint slots are non-identifiable
  (compensation), sequential-on-residual avoids it.

Output: repro/json/X33_fr3_twoslot.<ts>.json.
Run:  python X33_fr3_twoslot.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

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
import X34_fr3_kernel as X34                # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


class TwoSlotPsi(nn.Module):
    def __init__(self, centers, bw):
        super().__init__()
        self.global_lin = nn.Linear(30 if (X34.HY or not X34.QF) else 14, 6).double()
        with torch.no_grad():
            self.global_lin.weight.zero_()
            self.global_lin.bias.zero_()
        self.register_buffer('centers', centers.clone())
        self.bw = float(bw)
        self.Wk = nn.Parameter(torch.zeros(centers.shape[0], 6, dtype=DT))
        self.use_global = True
        self.use_kernel = True

    def forward(self, A, B):
        f = X34.raw_feats(A, B).reshape(-1, 14 if (X34.QF or X34.HY) else 30)
        if X34.HY:
            f30 = X34.feats30(A, B).reshape(-1, 30)
        out = torch.zeros(f.shape[0], 6, dtype=DT, device=f.device)
        if self.use_global:
            out = out + self.global_lin(f30 if X34.HY else f)
        if self.use_kernel:
            d2 = torch.cdist(f, self.centers) ** 2
            out = out + torch.exp(-d2 / (2 * self.bw ** 2)) @ self.Wk
        return 0.1 * out


def train(m, q_tr, oq_tr, ot_tr, epochs, lr, seed, slots, l2_global=0.0):
    """slots: subset of {'global','kernel'} to keep trainable (others frozen)."""
    for p in m.parameters():
        p.requires_grad_(False)
    if 'kernel' in slots:
        m.psi.Wk.requires_grad_(True)
    if 'global' in slots:
        m.psi.global_lin.weight.requires_grad_(True)
        m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if l2_global:
            loss = loss + l2_global * (m.psi.global_lin.weight ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
    return m


def make_two_slot(exact, centers, bw):
    m = copy.deepcopy(exact)
    m.psi = TwoSlotPsi(centers, bw)
    return m


def global_norm(m):
    with torch.no_grad():
        return float(m.psi.global_lin.weight.abs().sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--seq-split', type=float, default=0.5,
                    help='fraction of epochs for the kernel stage (sequential)')
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
    # X26c dose of L2 on the global slot, per training size
    DOSE = {20: 3e-3, 40: 3e-4, 60: 1e-4, 80: 0.0}

    variants = ('mixed_joint', 'mixed_seq')
    res = {n: {v: [] for v in variants} for n in sizes}
    gnorm = {n: [] for n in sizes}

    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)

            feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
            cen, _ = kmeans2(feats.numpy(), args.centers, minit='++', seed=s)
            centers = torch.tensor(cen, dtype=DT)
            dat = torch.cdist(centers, centers)
            bw = float(torch.median(dat[dat > 0])) if (dat > 0).any() else 1.0

            # mixed_joint (global slot dosed per N, X26c)
            dose = DOSE.get(int(n_tr), 0.0)
            mj = make_two_slot(exact, centers, bw)
            train(mj, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s,
                  slots={'global', 'kernel'}, l2_global=dose)
            res[n_tr]['mixed_joint'].append(GN.eval_pos(mj, q_te, oq_te, ot_te)[0])

            # mixed_seq: FULL budget kernel, then global on residual (dosed per N)
            ms = make_two_slot(exact, centers, bw)
            train(ms, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s, slots={'kernel'})
            train(ms, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s,
                  slots={'global'}, l2_global=dose)
            res[n_tr]['mixed_seq'].append(GN.eval_pos(ms, q_te, oq_te, ot_te)[0])
            gnorm[n_tr].append(global_norm(ms))
            print(f'[X33 seed {s} n={n_tr}] mixed_joint {res[n_tr]["mixed_joint"][-1]:.3f} '
                  f'mixed_seq {res[n_tr]["mixed_seq"][-1]:.3f} |g_slot| {gnorm[n_tr][-1]:.3f}',
                  flush=True)
        print(f'[X33] seed {s} done', flush=True)

    for n in sizes:
        for v in variants:
            a = np.array(res[n][v])
            RESULTS[f'X33:n{n}:{v}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        RESULTS[f'X33:n{n}:global_slot_norm'] = float(np.mean(gnorm[n]))
    RESULTS['X33:protocol'] = dict(sizes=sizes, seeds=args.seeds, epochs=args.epochs,
                                   lr=args.lr, centers=args.centers,
                                   seq_order='kernel(full epochs)->global(full epochs)',
                                   dose=DOSE)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
