#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
S8_ur10_relaxed.py — relaxed composition node on the UR10 laser-tracker dataset
(IPA / christianlandgraf ipa-robot-accuracy, Series A, 6D), the third real
calibration platform and the natural complement of the FR3 positive (X13/X14).

Same relaxation as X13/X14/S2, placed in the X10 screw model:
    T <- T * exp(v_j) * exp(gain * psi(f(T, exp(v_j))))  after every joint,
psi a small MLP (14 -> 16 -> 6, ZERO last layer, gain 0.1) shared across joints,
so at init the model is exactly X10.ScrewCalib (dense, 48p).

Protocol identical to the canonical X10/X10b runs (references from their dumps):
IPA Series A, splits rng(1000 + seed), seed_start = 50 (seeds 50-54), N=20/40/80,
full 6D loss (position + rotation) as X10.train_screwcalib, Adam 1e-2, cosine,
2000 steps; metric: pos_mean in mm on the held-out poses.

Registered predictions (plan 2026-09-29, before the run):
  P-S8a: relaxed <= exact (screw_dense) at N>=40 (paired over seeds);
  P-S8b: relaxed <= GN-LS (poe_ls) at N>=40;
  P-S8c: at N=20 relaxed is not better than exact (exact is seed-unstable there);
  P-S8d (boundary): relaxed ~ exact at all N -> the law is not deformed on UR10
        and the FR3 win is platform-specific.

Output: repro/json/S8_ur10_relaxed.<ts>.json via repro/dump.
Run:  python S8_ur10_relaxed.py [--quick]
"""

import argparse
import json
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

import X10_fk_calib_real as X10          # noqa: E402
import X7_robot_calibration as X7        # noqa: E402
from repro.dump import save_dump
from repro.canon import load as load_canon         # noqa: E402

DT = torch.float64
RESULTS = {}

CANON = load_canon('x10_p48')
GN = load_canon('x10b')


class ScrewCalibRelax(X10.ScrewCalib):
    """X10.ScrewCalib with a zero-initialized learned correction after every joint."""

    def __init__(self, *a, psi_hidden=16, psi_gain=0.1, **kw):
        super().__init__(*a, **kw)
        self.psi_gain = float(psi_gain)
        self.psi = nn.Sequential(nn.Linear(14, psi_hidden), nn.Tanh(),
                                 nn.Linear(psi_hidden, 6)).double()
        with torch.no_grad():
            self.psi[-1].weight.zero_()
            self.psi[-1].bias.zero_()

    @staticmethod
    def _canon(q):
        return torch.where(q[..., :1] >= 0, q, -q)

    def forward(self, q):
        qd = q.double()
        g = qd + self.b.double().unsqueeze(0)
        v = g.unsqueeze(-1) * self.twists().double().unsqueeze(0)
        B, n = qd.shape
        T = self.M0.double().unsqueeze(0).expand(B, 4, 4).clone()
        for j in range(n):
            Tj = X10.se3_exp(v[:, j, :])
            T = T @ Tj
            f = torch.cat([self._canon(X7.mat_to_quat(T[:, :3, :3])),
                           T[:, :3, 3],
                           self._canon(X7.mat_to_quat(Tj[:, :3, :3])),
                           Tj[:, :3, 3]], -1)
            T = T @ X10.se3_exp((self.psi_gain * self.psi(f)).double())
        T = T @ X10.se3_exp(self.tool.double().unsqueeze(0))
        return T


def fit_relaxed(xi, M0, q_tr, T_obs2_tr, tool_init, steps=2000, seed=0):
    """Mirror of X10.train_screwcalib (full 6D loss) for the relaxed model."""
    torch.manual_seed(100 + seed)
    m = ScrewCalibRelax(xi, M0, mode='dense')
    with torch.no_grad():
        m.tool.copy_(torch.tensor(tool_init, dtype=m.tool.dtype))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    qt = torch.tensor(q_tr, dtype=DT)
    Tt = torch.tensor(T_obs2_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - Tt[:, :3, 3]) ** 2).sum(-1).mean()
        tr = (Tp[:, :3, :3].transpose(-1, -2) @ Tt[:, :3, :3]).diagonal(
            dim1=-2, dim2=-1).sum(-1)
        loss = loss + (1.0 - ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
    return m


def load_refs():
    canon = CANON['ipa_A']
    gn = GN['ipa_A']
    return canon, gn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--series', default='A')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=50)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 200
    t0 = time.time()

    data = X10.load_ipa(args.data_dir, args.series)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    sizes = [int(s) for s in args.sizes.split(',')]
    canon, gn = load_refs()
    n_psi = sum(p.numel() for p in ScrewCalibRelax(xi, M0, mode='dense').parameters())
    print(f'[S8] relaxed params {n_psi} (exact dense 48; +psi 342)', flush=True)

    for s in range(args.seed_start, args.seed_start + args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            goal_tr = data['goal'][tr]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], goal_tr, full=True)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=True, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            m = fit_relaxed(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C),
                            steps=args.steps, seed=s)
            rec = X10.pose_metrics(X10.apply_sc(m, q_all[te]), T_obs2_te, full=True)
            exact = canon['per_seed']['seed%d' % s]['N%d' % N]['screw_dense']['pos_mean']
            gnm = gn['per_seed']['seed%d' % s]['N%d' % N]['poe_ls']['pos_mean']
            gap = canon['per_seed']['seed%d' % s]['N%d' % N]['nominal_gap']['pos_mean']
            RESULTS[f'S8:seed{s}:N{N}:relaxed'] = float(rec['pos_mean'])
            RESULTS[f'S8:seed{s}:N{N}:rot'] = float(rec['rot_mean'])
            RESULTS[f'S8:seed{s}:N{N}:exact'] = float(exact)
            RESULTS[f'S8:seed{s}:N{N}:gn'] = float(gnm)
            RESULTS[f'S8:seed{s}:N{N}:nominal_gap'] = float(gap)
            print(f'[S8 seed {s} N={N}] relaxed {rec["pos_mean"]:.3f} '
                  f'exact {exact:.3f} gn {gnm:.3f} gap {gap:.3f}', flush=True)
        print(f'  seed {s} done', flush=True)

    RESULTS['_meta'] = dict(
        experiment='S8 relaxed node on UR10 (IPA Series A, 6D)',
        date=time.strftime('%Y-%m-%d %H:%M'), sizes=sizes, seeds=args.seeds,
        seed_start=args.seed_start, steps=args.steps,
        refs=dict(exact='x10_p48', gn='x10b'),
        device='cpu', torch=torch.__version__,
        wall_seconds=float(time.time() - t0))
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, seed_start=args.seed_start,
                    epochs=args.steps)
    print('\n===== SUMMARY-JSON =====')
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False, default=float))
    print('dump:', out)


if __name__ == '__main__':
    main()
