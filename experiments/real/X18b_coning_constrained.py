#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X18b_coning_constrained.py — P-CON2b: a constrained coning correction.

X18 showed that an unconstrained per-step psi destroys the coning floor
(7.8 deg vs 0.17 deg).  Here the correction is a SINGLE end-of-window map,
linear in the coning statistic c = sum_k u_k x u_{k+1}:
    q = prod_k exp(u_k) . exp(A c),   A in R^{3x3} (9 parameters, zero at init).
Predictions (plan 2026-10-05): removes >= 80% of the left-scheme floor
(0.167 deg); inert on non-coning (TOST); no effect on the mid scheme (already
0.005 deg).
Output: repro/json/X18b_coning_constrained.<ts>.json
Run: python X18b_coning_constrained.py [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import X18_coning as X18                     # noqa: E402
from repro.dump import save_dump             # noqa: E402

DT = torch.float64
RESULTS = {}


class ConstrainedConing(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(3, 3, dtype=DT))

    def forward(self, U):
        q = X18.coarse_integrate(U)
        c = torch.zeros(U.shape[0], 3, dtype=DT)
        for k in range(U.shape[1] - 1):
            c = c + torch.linalg.cross(U[:, k], U[:, k + 1])
        return X18.quat_mul(q, X18.exp_so3(c @ self.A.T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=3000)
    ap.add_argument('--W', type=int, default=64)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.windows, args.W, args.steps = 300, 32, 300
    t0 = time.time()
    dt = 1.0 / 286.0
    a, om, b = 3.0, 30.0, 0.5

    for tag, kw in (('coning_left', dict(a=a, om=om, b=b, scheme='left')),
                    ('coning_mid', dict(a=a, om=om, b=b, scheme='mid')),
                    ('nonconing', dict(a=0.0, om=om, b=b, scheme='left'))):
        U, q_rel, D, theta = X18.make_windows(args.windows, args.W, dt, seed=0, **kw)
        floor = float(X18.geo_deg(X18.coarse_integrate(U), q_rel).mean())
        ntr = int(0.7 * len(U))
        torch.manual_seed(0)
        model = ConstrainedConing()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for it in range(args.steps):
            opt.zero_grad()
            loss = (X18.geo_deg(model(U[:ntr]), q_rel[:ntr]) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            err = float(X18.geo_deg(model(U[ntr:]), q_rel[ntr:]).mean())
        gain = 100.0 * (1 - err / max(floor, 1e-12))
        RESULTS[f'X18b:{tag}:floor_deg'] = floor
        RESULTS[f'X18b:{tag}:test_deg'] = err
        RESULTS[f'X18b:{tag}:gain_pct'] = gain
        RESULTS[f'X18b:{tag}:A'] = model.A.detach().cpu().numpy().tolist()
        print(f'[{tag}] floor {floor:.4f} test {err:.4f} deg gain {gain:.1f}%', flush=True)

    RESULTS['X18b:protocol'] = dict(W=args.W, windows=args.windows, steps=args.steps,
                                   lr=args.lr)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'done {time.time()-t0:.0f}s; dump: {out}')


if __name__ == '__main__':
    main()
