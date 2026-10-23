#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
E26c_error_law.py — deterministic error-accumulation law on the edge (real Baxter).

Idea: the node is an exact integrator; instead of a memoryless edge qdd = f(x),
the edge carries an error state with its own accumulation law
    e_{k+1} = e_k + (u(x_k) - lam * e_k),   qdd_k = e_{k+1},   e_0 = 0,
where u(x) = A phi(x) is linear (21 -> 7; 154 parameters, exactly the memoryless
linear edge's budget) and lam is FIXED (not learned): lam = 0 (pure accumulation)
and lam = 1/tau with tau = 20 steps (decay).

Registered predictions (plan 2026-09-29, before the run):
  P-A1: circle error of errlaw <= memoryless lin at both dt (paired over seeds);
  P-A2: errlaw <= MLP h=64 on circle;
  P-A3: errlaw <= MLP h=2/h=4 on circle (structure vs capacity);
  P-A4: in-dist errlaw not worse than lin by more than 5%;
  P-A5: errlaw0 vs errlaw20 — accumulation vs decay (first look at the nature).

Protocol: same as E26b (squared+random 70/30, transfer circle, seeds 0-2, 100
epochs, max_train 6000, H=50, dt=1 and 2 ms; metric = RMSE of the 14-dim state
increment). Controls: zero, mean, lin, MLP h=2/4/64, phys, flat.

Output: repro/json/E26c_error_law.<ts>.json via repro/dump.
Run:  python E26c_error_law.py            # 3 seeds, dt=1/2 ms, 100 epochs
      python E26c_error_law.py --quick    # smoke: 1 seed, dt=1 ms, 60 epochs
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

import E26b_baxter_dynamics as B              # noqa: E402
from repro.dump import save_dump              # noqa: E402
from repro.data_guard import skip_if_missing  # noqa: E402

DEV = torch.float64
RESULTS = {}


def integrate_errlaw(q0, v0, tau, edge, mu, sd, dt, H, lam):
    """Semi-implicit Euler with an error state e living by its own law."""
    q, v = q0, v0
    mu_t = torch.tensor(mu, dtype=DEV)
    sd_t = torch.tensor(sd, dtype=DEV)
    e = torch.zeros_like(q0)
    for _ in range(H):
        x = torch.cat([q, v, tau], -1)
        u = edge((x - mu_t) / sd_t)
        e = e + (u - lam * e)
        v = v + dt * e
        q = q + dt * v
    return q - q0, v - v0


def fit_errlaw(Xtr, Ytr, H, dt, lam, epochs=100, seed=0):
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdims=True)
    sd = np.where(Xtr.std(0, keepdims=True) < 1e-8, 1.0, Xtr.std(0, keepdims=True))
    Xt = torch.tensor(Xtr, dtype=DEV)
    Yt = torch.tensor(Ytr, dtype=DEV)
    edge = nn.Linear(Xt.shape[1], 7).double()
    with torch.no_grad():                # start from zero dynamics (safe e_0=0)
        edge.weight.zero_()
        edge.bias.zero_()
    opt = torch.optim.Adam(edge.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        dq, dv = integrate_errlaw(Xt[:, :7], Xt[:, 7:14], Xt[:, 14:], edge,
                                  mu, sd, dt, H, lam)
        loss = ((torch.cat([dq, dv], -1) - Yt) ** 2).mean()
        loss.backward()
        opt.step()
    return mu, sd, edge


def evaluate_errlaw(Xev, Yev, H, dt, mu, sd, edge, lam):
    Xe = torch.tensor(Xev, dtype=DEV)
    with torch.no_grad():
        dq, dv = integrate_errlaw(Xe[:, :7], Xe[:, 7:14], Xe[:, 14:], edge,
                                  mu, sd, dt, H, lam)
        pred = torch.cat([dq, dv], -1).numpy()
    return B.rmse(pred, Yev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--dt', default='1,2', help='comma-separated dt in ms')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--max-train', type=int, default=6000)
    ap.add_argument('--tau', type=float, default=20.0, help='decay time in steps')
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    dts = tuple(float(x) / 1000.0 for x in args.dt.split(','))
    epochs, max_train = args.epochs, args.max_train
    if args.quick:
        seeds, dts, epochs, max_train = (0,), (1e-3,), 60, 3000
    if skip_if_missing(B.D, 'Baxter CSV directory (set BAXTER)'):
        return
    t0 = time.time()

    Xtr, Ytr, Xte, Yte = [], [], [], []
    for k in ('squared', 'random'):
        q, v, t = B.load(k)
        X, Y = B.windows(q, v, t, B.H)
        c = int(0.7 * len(X))
        Xtr.append(X[:c]); Ytr.append(Y[:c])
        Xte.append(X[c:]); Yte.append(Y[c:])
    Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
    Xte, Yte = np.concatenate(Xte), np.concatenate(Yte)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr))[:min(max_train, len(Xtr))]
    Xtr, Ytr = Xtr[idx], Ytr[idx]
    qc, vc, tc = B.load('circle')
    Xc, Yc = B.windows(qc, vc, tc, B.H)

    def zero(X, dt):
        v = X[:, 7:14]
        return np.concatenate([v * B.H * dt, np.zeros_like(v)], -1)

    kinds = ('lin', 'mlp2', 'mlp4', 'mlp64', 'phys')
    for dt in dts:
        rows = {}
        rows['zero'] = (B.rmse(zero(Xte, dt), Yte), B.rmse(zero(Xc, dt), Yc))
        rows['mean'] = (B.rmse(np.tile(Ytr.mean(0), (len(Yte), 1)), Yte),
                        B.rmse(np.tile(Ytr.mean(0), (len(Xc), 1)), Yc))
        for kind in kinds:
            h = 64 if kind == 'mlp64' else 4 if kind == 'mlp4' else 2
            base = 'mlp' if kind.startswith('mlp') else kind
            it, ic = [], []
            for s in seeds:
                mu, sd, edge = B.fit(base, Xtr, Ytr, B.H, dt, epochs=epochs,
                                     seed=s, h=h)
                it.append(B.evaluate(base, Xte, Yte, B.H, dt, mu, sd, edge, h))
                ic.append(B.evaluate(base, Xc, Yc, B.H, dt, mu, sd, edge, h))
            rows[kind] = (float(np.mean(it)), float(np.mean(ic)))
            rows[kind + ':per_seed'] = [[float(a), float(b)] for a, b in zip(it, ic)]
        ft, fc = [], []
        for s in seeds:
            ft.append(B.fit_flat(Xtr, Ytr, Xte, Yte, seed=s, epochs=epochs))
            fc.append(B.fit_flat(Xtr, Ytr, Xc, Yc, seed=s, epochs=epochs))
        rows['flat'] = (float(np.mean(ft)), float(np.mean(fc)))
        rows['flat:per_seed'] = [[float(a), float(b)] for a, b in zip(ft, fc)]
        for lam_name, lam in (('errlaw0', 0.0),
                              ('errlaw%d' % int(args.tau), 1.0 / args.tau)):
            it, ic = [], []
            for s in seeds:
                mu, sd, edge = fit_errlaw(Xtr, Ytr, B.H, dt, lam,
                                          epochs=epochs, seed=s)
                it.append(evaluate_errlaw(Xte, Yte, B.H, dt, mu, sd, edge, lam))
                ic.append(evaluate_errlaw(Xc, Yc, B.H, dt, mu, sd, edge, lam))
            rows[lam_name] = (float(np.mean(it)), float(np.mean(ic)))
            rows[lam_name + ':per_seed'] = [[float(a), float(b)]
                                            for a, b in zip(it, ic)]
        RESULTS['E26c:dt%d' % round(dt * 1000)] = rows
        print(f'\n==== dt = {dt*1000:.1f} ms ====')
        print(f'{"model":12s} {"in-dist":>9s} {"circle":>9s}')
        for k in ('zero', 'mean', 'lin', 'mlp2', 'mlp4', 'mlp64', 'phys', 'flat',
                  'errlaw0', 'errlaw%d' % int(args.tau)):
            a, b = rows[k]
            print(f'{k:12s} {a:9.4f} {b:9.4f}', flush=True)

    RESULTS['_meta'] = dict(dt_effective=B.rate_check(), H=B.H, K_tau=B.KTAU,
                            tau_steps=args.tau, errlaw_params=154,
                            wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=len(seeds), epochs=epochs)
    print('\ndump ->', p)


if __name__ == '__main__':
    main()
