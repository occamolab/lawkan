#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
S2_ur5gtpn_relaxed.py — relaxed composition node on the independent UR5 dataset
(Zenodo 18954650, position-only), the second real calibration platform.

Same relaxation as X13/X14, but placed in the X10/X11 screw model:
T <- T * exp(v_j) * exp(psi(features(T, exp(v_j)))) at every joint, psi a small
MLP (14 -> 16 -> 6, ZERO last layer, gain 0.1) shared across joints, so at init
the model is exactly X10.ScrewCalib (dense, joint zeros included). Protocol
identical to X11: N=20..80, seeds 0-4, position-only, 2000 Adam steps, lr 1e-2.

References (same splits/seeds): X11 exact/GN/const per-seed from its dump.

Registered predictions (plan 2026-09-29, before the run):
  P-S2a: relaxed <= exact at N>=60 (paired over seeds);
  P-S2b: relaxed <= GN-LS at N>=60;
  P-S2c: at N=20 relaxed is not better than exact (overfit, as on FR3);
  P-S2d (boundary): if relaxed ~ exact at all N, the law is fine on this dataset.

Output: repro/json/S2_ur5gtpn_relaxed.<ts>.json via repro/dump.
Run:  python S2_ur5gtpn_relaxed.py [--quick]
"""

import argparse
import glob
import json
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
import X10_fk_calib_real as X10              # noqa: E402
import X11_ur5_gtpn_sweep as X11             # noqa: E402
from repro.dump import save_dump             # noqa: E402

DT = torch.float64
RESULTS = {}


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
    torch.manual_seed(seed)
    m = ScrewCalibRelax(xi, M0, mode='dense')
    with torch.no_grad():
        m.tool.copy_(torch.tensor(tool_init, dtype=m.tool.dtype))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    qt = torch.tensor(q_tr, dtype=DT)
    pt = torch.tensor(T_obs2_tr[:, :3, 3], dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - pt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m


def load_ref():
    cands = [p for p in glob.glob(os.path.join(
        _REPO, 'repro', 'json', 'X11_ur5_gtpn_sweep.*.json')) if 'smoke' not in p]
    d = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
    return {n: {k: d[f'X11:n{n}:{k}']['per_seed'] for k in ('ours', 'gn', 'const')}
            for n in (20, 40, 60, 80)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=X11.DATA)
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 200
    t0 = time.time()

    data = X11.load_gtpn(args.data)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    sizes = [int(s) for s in args.sizes.split(',')]
    ref = load_ref()
    params = sum(p.numel() for p in ScrewCalibRelax(xi, M0, mode='dense').parameters())
    print(f'[S2] relaxed params {params} (exact 48)', flush=True)

    res = {N: [] for N in sizes}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], None, full=False)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=False, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            m = fit_relaxed(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C),
                            steps=args.steps, seed=s)
            cur = X10.pose_metrics(X10.apply_sc(m, q_all[te]), T_obs2_te,
                                   full=False)['pos_mean']
            res[N].append(float(cur))
            exact = ref[N]['ours'][s] if s < len(ref[N]['ours']) else float('nan')
            print(f'[S2 seed {s} N={N}] relaxed {cur:.3f} exact {exact:.3f} '
                  f'gn {ref[N]["gn"][s]:.3f}', flush=True)
        print(f'[S2] seed {s} done', flush=True)

    for N in sizes:
        a = np.array(res[N])
        RESULTS[f'S2:n{N}:relaxed'] = dict(pos_mm_mean=float(a.mean()),
                                           pos_mm_std=float(a.std()),
                                           per_seed=[float(x) for x in a],
                                           params=int(params))
        for rname in ('ours', 'gn', 'const'):
            rr = np.array(ref[N][rname])
            if len(a) != len(rr) or len(a) < 2:
                continue
            t, p = stats.ttest_rel(a, rr)
            RESULTS[f'S2:paired:n{N}:relaxed_vs_{rname}'] = dict(
                diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.steps)
    print('dump ->', p)


if __name__ == '__main__':
    main()
