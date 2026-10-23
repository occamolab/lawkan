#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X6b_tinygc_style.py — P-X6b: the TinyGC-Net calibration subnet (27 parameters)
as a *measured* competitor under the X6 gyro-calibration protocol.

TinyGC-Net (Cui et al., 2024; arXiv:2403.02618) ships a 27-parameter calibration
subnet = affine(12) + PReLU(3) + affine(12).  Their Sec. 3.1 notes that the
subnet reduces to the classical linear measurement model (Eq. (1)) when the
parameters are constant, and Remark 4.1.5(2) reports that adding more LBNs and
activations gave minimal accuracy improvement.  This script trains exactly that
27-parameter net in place of the affine (M, b) correction, with the same
windows, split, seeds, optimizer and geodesic metric as X6, so Table 15 gains a
measured row instead of a construction note.

Init: identity (PReLU slope 1, A1 = A2 = I, c = 0), i.e. the raw integrator at
step 0 — the same zero/identity start as the rest of the paper.

Prediction P-X6b: TinyGC-style ~ LM / learned (M, b) within TOST(10%); no
significant gain over the classical LM fit.  Falsifier: if it significantly
beats LM, the nonlinearity matters and the "affine suffices" reading must be
revised.

Run:  python X6b_tinygc_style.py --data_dir /path/to/broad/data_hdf5 [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X6_broad_calibration as X6      # noqa: E402
from repro.dump import save_dump
from repro.data_guard import skip_if_missing       # noqa: E402

DT = torch.float64
RESULTS = {}


def reg(key, vals, **meta):
    vals = [float(v) for v in vals] if isinstance(vals, (list, tuple, np.ndarray)) else [float(vals)]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), vals=vals, **meta)


class TinyGCStyleIntegrator(torch.nn.Module):
    """27p calibration subnet (affine + PReLU + affine), then strapdown integration."""

    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.A1 = torch.nn.Parameter(torch.eye(3))
        self.c1 = torch.nn.Parameter(torch.zeros(3))
        self.slope = torch.nn.Parameter(torch.ones(3))
        self.A2 = torch.nn.Parameter(torch.eye(3))
        self.c2 = torch.nn.Parameter(torch.zeros(3))

    def forward(self, omega_raw):
        h = torch.einsum('ij,bsj->bsi', self.A1, omega_raw) + self.c1
        h = torch.where(h > 0, h, self.slope * h)
        corr = torch.einsum('ij,bsj->bsi', self.A2, h) + self.c2
        return X6.integrate(corr, self.dt)


def run(name, data_dir, W, stride, groups, seeds=(0, 1, 2), epochs=2000, lr=1e-2):
    print(f"\n===== X6b:{name}  (W={W}, TinyGC-style 27p) =====")
    X, Y, tidx, names, fs = X6.assemble(data_dir, W, stride, groups)
    dt = 1.0 / fs
    n = len(X)

    g = np.random.default_rng(12345)
    perm = g.permutation(n)
    ntr = int(0.7 * n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]

    q_raw = X6.integrate(Xte, dt)
    reg(f'X6b:{name}:Raw', X6.geo_deg(q_raw, Yte).cpu().numpy())

    p_lm = X6.lm_fit(Xtr.cpu().numpy(), Ytr.cpu().numpy(), dt)
    M_lm = torch.tensor(p_lm[:9], dtype=DT).reshape(3, 3).to(X6.device)
    b_lm = torch.tensor(p_lm[9:12], dtype=DT).to(X6.device)
    q_lm = X6.integrate(torch.einsum('ij,bsj->bsi', M_lm, Xte.to(DT) - b_lm), dt)
    lm_mean = float(X6.geo_deg(q_lm, Yte).mean())
    reg(f'X6b:{name}:LM', [lm_mean])

    errs = []
    for s in seeds:
        torch.manual_seed(100 + s)
        model = TinyGCStyleIntegrator(dt).to(X6.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            loss = (X6.geo_rad(model(Xtr), Ytr) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            errs.append(float(X6.geo_deg(model(Xte), Yte).mean()))
        reg(f'X6b:{name}:TinyGC[seed{s}]', [errs[-1]])
        if s == seeds[0]:
            print(f"    A1 diag={np.diag(model.A1.detach().cpu().numpy()).round(4).tolist()} "
                  f"slope={model.slope.detach().cpu().numpy().round(4).tolist()}")
            print(f"    c1={model.c1.detach().cpu().numpy().round(5).tolist()} "
                  f"c2={model.c2.detach().cpu().numpy().round(5).tolist()}")

    reg(f'X6b:{name}:TinyGC', errs)
    d = np.array(errs) - lm_mean
    if len(d) >= 2:
        t, p = stats.ttest_1samp(d, 0.0)
        RESULTS[f'X6b:{name}:paired:TinyGC_vs_LM'] = dict(
            diff=float(d.mean()), t=float(t), p=float(p))
    tol = 0.10 * lm_mean
    if len(d) >= 2:
        se = d.std(ddof=1) / np.sqrt(len(d))
        half = stats.t.ppf(0.95, len(d) - 1) * se
        RESULTS[f'X6b:{name}:TOST10'] = dict(
            margin=float(tol), ci_lo=float(d.mean() - half), ci_hi=float(d.mean() + half),
            pass_=bool(abs(d.mean()) + half < tol))
    print(f"    -> TinyGC-style {np.mean(errs):.4f} deg (seeds {np.round(errs, 4).tolist()}) "
          f"vs LM {lm_mean:.4f}  diff {d.mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--groups', default='undisturbed')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    if args.quick:
        args.seeds, args.steps = 1, 200
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    t0 = time.time()

    blocks = [('a-short', dict(W=20, stride=20)),
              ('b-035s', dict(W=100, stride=100)),
              ('c-1s', dict(W=286, stride=286))]
    if args.quick:
        blocks = blocks[:1]
    for name, kw in blocks:
        run(name, args.data_dir, groups=groups, seeds=tuple(range(args.seeds)),
            epochs=args.steps, lr=args.lr, **kw)

    RESULTS['_meta'] = dict(wall_s=time.time() - t0, device=str(X6.device),
                            version='v1.1 protocol of X6; model=TinyGC-style 27p')
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print(f"\nTotal wall time: {time.time()-t0:.1f} s; dump: {out}")


if __name__ == '__main__':
    main()
