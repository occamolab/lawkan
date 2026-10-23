#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X18_coning.py — P-CONING: an analytically known violator of first-order composition.

Coning motion: the body rate has two orthogonal harmonic components,
    omega(t) = a (cos(Omega t) e1 + sin(Omega t) e2) + b e3,
so the true finite rotation over a window differs from the composition of the
coarse per-sample increments (each exp(omega_k dt)) by the classical coning term,
the second-order BCH integral
    D = (1/2) \\int_{t0}^{t} [Theta(s), omega(s)] ds,   Theta(s) = \\int_{t0}^{s} omega.
The coarse first-order composition q_c = prod exp(omega_k dt) is exact for each
step but blind to D; its geodesic error against the fine-grained reference is the
coning floor.  Prediction P-CON1: the floor matches |D| (analytic, quadrature)
within 20%.  A zero-initialized correction on the frozen coarse increments with
physics features (step increment, cross-product coning statistic, accumulated
log) is trained in isolation.  P-CON2: it removes >= 80% of the floor.
P-CON3: on non-coning motion (fixed axis) the floor vanishes and the correction
stays inert (TOST-able zero effect).

Output: repro/json/X18_coning.<ts>.json via repro/dump.
Run:  python X18_coning.py [--quick]
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.dump import save_dump  # noqa: E402

DT = torch.float64
RESULTS = {}


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def quat_conj(q):
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], -1)


def exp_so3(v):
    th = v.norm(dim=-1, keepdim=True)
    half = 0.5 * th
    small = th < 1e-9
    s = torch.where(small, torch.ones_like(th), torch.sin(half) / th.clamp_min(1e-12))
    return torch.cat([torch.cos(half), v * s], -1)


def geo_deg(q_pred, q_tgt):
    d = (q_pred * q_tgt).sum(-1).abs().clamp(max=1 - 1e-9)
    return torch.rad2deg(2.0 * torch.acos(d))


def omega_fn(t, a, om, b):
    """(..., 3) body rate of circular coning; t scalar tensor."""
    return torch.stack([a * torch.cos(om * t), a * torch.sin(om * t),
                        b * torch.ones_like(t)], -1)


def make_windows(n_win, W, dt, a, om, b, fine=50, seed=0, scheme='left'):
    """Coarse increments (n,W,3), fine reference (n,4), BCH D (n,3), coarse log (n,3)."""
    g = torch.Generator().manual_seed(seed)
    t0 = torch.rand(n_win, generator=g) * 10.0
    k = torch.arange(W, dtype=DT)
    t = t0[:, None] + k[None, :] * dt                       # (n,W)
    ts = t if scheme == 'left' else t + 0.5 * dt
    U = omega_fn(ts, a, om, b) * dt                          # coarse increments
    # fine reference (midpoint, sub-steps)
    h = dt / fine
    t_f = (t0[:, None, None] + (k[None, :, None] * fine + torch.arange(fine, dtype=DT)[None, None, :]) * h)
    tf = t_f.reshape(n_win, W * fine)
    w_f = omega_fn(tf + 0.5 * h, a, om, b) * h
    q = torch.zeros(n_win, 4, dtype=DT); q[:, 0] = 1.0
    for j in range(W * fine):
        q = quat_mul(q, exp_so3(w_f[:, j]))
    q_rel = q / q.norm(dim=-1, keepdim=True)
    # analytic BCH term D = 1/2 Int [Theta(s), omega(s)] ds (quadrature over fine grid)
    theta = torch.zeros(n_win, 3, dtype=DT)
    D = torch.zeros(n_win, 3, dtype=DT)
    for j in range(W * fine):
        wj = omega_fn(tf[:, j] + 0.5 * h, a, om, b)
        D = D + 0.25 * h * torch.linalg.cross(theta, wj) + 0.25 * h * torch.linalg.cross(theta + h * wj, wj)
        theta = theta + h * wj
    # D above approximates 1/2 Int [Theta, omega]: trapezoid of 0.5*[Theta,omega]
    return U, q_rel, D, theta


def coarse_integrate(U):
    q = torch.zeros(U.shape[0], 4, dtype=DT); q[:, 0] = 1.0
    for k in range(U.shape[1]):
        q = quat_mul(q, exp_so3(U[:, k]))
    return q / q.norm(dim=-1, keepdim=True)


class ConingPsi(nn.Module):
    """Zero-init per-step correction with explicit physics features."""

    def __init__(self, hidden=16, physics=True):
        super().__init__()
        self.physics = physics
        dim = 9 if physics else 6
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3)).double()
        with torch.no_grad():
            self.net[-1].weight.zero_(); self.net[-1].bias.zero_()

    def forward(self, U):
        n, W, _ = U.shape
        c = torch.zeros_like(U)
        c[:, 1:] = torch.linalg.cross(U[:, :-1], U[:, 1:])
        q = torch.zeros(n, 4, dtype=DT); q[:, 0] = 1.0
        out = []
        for k in range(W):
            logacc = 2.0 * torch.atan2(q[:, 1:].norm(dim=-1), q[:, 0].clamp(min=1e-12))
            axis = q[:, 1:] / q[:, 1:].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            loc = axis * logacc.unsqueeze(-1)
            feats = [U[:, k], c[:, k], loc] if self.physics else [U[:, k], loc]
            f = torch.cat(feats, -1)
            q = quat_mul(q, exp_so3(U[:, k] + self.net(f)))
            out.append(q)
        return q / q.norm(dim=-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', type=int, default=3000)
    ap.add_argument('--W', type=int, default=64)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.windows, args.W, args.steps = 200, 32, 300
    t0 = time.time()
    dt = 1.0 / 286.0
    a, om, b = 3.0, 30.0, 0.5

    for tag, kw in (('coning_left', dict(a=a, om=om, b=b, scheme='left')),
                    ('coning_mid', dict(a=a, om=om, b=b, scheme='mid')),
                    ('nonconing', dict(a=0.0, om=om, b=b, scheme='left'))):
        U, q_rel, D, theta = make_windows(args.windows, args.W, dt, seed=0, **kw)
        q_c = coarse_integrate(U)
        floor = float(geo_deg(q_c, q_rel).mean())
        # decomposition (left scheme): sampling quadrature |Theta_c - Theta_fine|
        # and the coning BCH term |D|; at 286 Hz the quadrature term dominates.
        theta_c = U.sum(1)
        bch = float((2.0 * torch.asin((D.norm(dim=-1) / 2).clamp(max=1))).mean() * 180 / np.pi)
        quad = float((theta_c - theta).norm(dim=-1).mean() * 180 / np.pi)
        print(f'[{tag}] floor {floor:.4f} deg | BCH |D| {bch:.4f} | quad {quad:.4f}',
              flush=True)
        RESULTS[f'X18:{tag}:floor_deg'] = floor
        RESULTS[f'X18:{tag}:bch_deg'] = bch
        RESULTS[f'X18:{tag}:quad_deg'] = quad

        ntr = int(0.7 * len(U))
        for physics in (True, False):
            torch.manual_seed(0)
            model = ConingPsi(physics=physics)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            Utr, qtr = U[:ntr], q_rel[:ntr]
            for it in range(args.steps):
                opt.zero_grad()
                qp = model(Utr)
                loss = (geo_deg(qp, qtr) ** 2).mean()
                loss.backward(); opt.step()
            with torch.no_grad():
                qp = model(U[ntr:])
                err = float(geo_deg(qp, q_rel[ntr:]).mean())
                pn = float(sum(p.abs().sum() for p in model.parameters()))
            key = 'physics' if physics else 'generic'
            RESULTS[f'X18:{tag}:psi_{key}_test_deg'] = err
            RESULTS[f'X18:{tag}:psi_{key}_gain_pct'] = 100.0 * (1 - err / max(floor, 1e-12))
            RESULTS[f'X18:{tag}:psi_{key}_norm'] = pn
            print(f'[{tag}] psi-{key}: held-out {err:.4f} deg '
                  f'(gain {100 * (1 - err / max(floor, 1e-12)):.1f}%, |psi| {pn:.2f})',
                  flush=True)

    RESULTS['X18:protocol'] = dict(dt=dt, a=a, om=om, b=b, W=args.W,
                                  windows=args.windows, steps=args.steps)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'done {time.time()-t0:.0f}s; dump: {out}')


if __name__ == '__main__':
    main()
