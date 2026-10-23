#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
E26b_baxter_dynamics.py — audit/reproduction of E26 (real Baxter dynamics).
E26 was exploratory, its code was never committed; this script rebuilds the
protocol with the full model set so the documented boundary can be checked.

Data: baxter_dl CSVs (Zenodo 10.5281/zenodo.17035193, CC-BY-4.0), 21 columns
= 7 joint angles (rad), 7 velocities (rad/s), 7 torques (N m).

Protocol (journal E26): state (q, qdot), input tau; a state-conditioned edge
learns qdd = f(q, qdot, tau); the exact node integrates the state over the
horizon (semi-implicit Euler); target = state increment over H steps.
Train: squared + random (first 70%); in-dist test: last 30%; transfer test:
the circle trajectory. Metric: RMSE over the 14-dim state increment.
Sampling: the journal assumed 1 kHz; the data suggest ~2 ms/step (see audit
line printed at runtime), so both dt are reported.

Models:
  zero      constant-velocity integration (classical, training-free)
  mean      train-mean increment (training-free)
  lin       linear edge qdd = W[q,qdot,tau]+b, exact integration
  phys      linear edge over {1, tau, sin q, cos q, qdot_i qdot_j}, integration
  mlp       MLP edge (q,qdot,tau) -> qdd, integration
  flat      MLP (q,qdot,tau) -> increment directly, no integration
  hidden:   inputs are (q_t, tau history) with velocities hidden; the
            classical constant-velocity baseline is unavailable there, so
            zero becomes hold (increment = 0) and the MLP above is the test.

Output: repro/json/E26b_baxter_dynamics.<ts>.json via repro/dump.
Run:  python E26b_baxter_dynamics.py            # 3 seeds, dt=1/2 ms, 200 epochs
      python E26b_baxter_dynamics.py --quick    # smoke: 1 seed, dt=1 ms, 60 epochs
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from repro.dump import save_dump  # noqa: E402
from repro.data_guard import skip_if_missing  # noqa: E402

D = os.environ.get('BAXTER',
                   os.path.join(os.path.dirname(_REPO), 'baxter_dl'))
FILES = {'squared': 'right_squared_p15_t140.csv',
         'random': 'right_random_p15_t70.csv',
         'circle': 'right_circle_p15_t140.csv'}
DT = 1e-3
H = 50
KTAU = 25
RESULTS = {}
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def load(name):
    df = pd.read_csv(os.path.join(D, FILES[name]), sep=r'\s+')
    q = df.filter(like='ang_').values.astype(np.float64)
    v = df.filter(like='vel_').values.astype(np.float64)
    t = df.filter(like='torq_').values.astype(np.float64)
    return q, v, t


def rate_check():
    """Effective step from TV(q)/sum|v| on the active joints."""
    es = []
    for k in FILES:
        q, v, _ = load(k)
        r = np.abs(np.diff(q, axis=0)).sum(0) / np.abs(v).sum(0)
        r = r[(r > 0.5e-3) & (r < 5e-3)]
        es.append(np.median(r))
    return float(np.median(es))


def windows(q, v, t, H):
    s = np.concatenate([q, v], -1)
    X = np.concatenate([q, v, t], -1)
    return X[:-H], s[H:] - s[:-H]


def windows_hidden(q, v, t, H, K=KTAU):
    s = np.concatenate([q, v], -1)
    X = []
    for i in range(K - 1, len(q) - H):
        X.append(np.concatenate([q[i], t[i - K + 1:i + 1].ravel()]))
    X = np.array(X)
    Y = s[K - 1 + H:] - s[K - 1:-H]
    return X, Y


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def integrate(q0, v0, tau, edge, kind, mu, sd, dt, H):
    """Semi-implicit Euler with the edge re-evaluated on the evolving state
    (tau is held fixed over the horizon)."""
    q, v = q0, v0
    mu_t = torch.tensor(mu, dtype=torch.float64)
    sd_t = torch.tensor(sd, dtype=torch.float64)
    for _ in range(H):
        x = torch.cat([q, v, tau], -1)
        if kind == 'phys':
            qdd = edge(phys_feats((x - mu_t) / sd_t))
        else:
            qdd = edge((x - mu_t) / sd_t)
        v = v + dt * qdd
        q = q + dt * v
    return q - q0, v - v0


def make_edge(kind, din, h=64):
    if kind == 'lin':
        return nn.Linear(din, 7).double()
    if kind == 'phys':
        return nn.Linear(71, 7).double()
    if kind == 'mlp':
        return nn.Sequential(nn.Linear(din, h), nn.Tanh(),
                             nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 7)).double()
    raise ValueError(kind)


def phys_feats(x):
    q, v, t = x[:, :7], x[:, 7:14], x[:, 14:]
    sq, cq = torch.sin(q), torch.cos(q)
    vv = (v.unsqueeze(-1) * v.unsqueeze(-2)).reshape(x.shape[0], -1)
    return torch.cat([torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device),
                      t, sq, cq, vv], -1)


def fit(kind, Xtr, Ytr, H, dt, epochs=300, seed=0, h=64):
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdims=True)
    sd = np.where(Xtr.std(0, keepdims=True) < 1e-8, 1.0, Xtr.std(0, keepdims=True))
    Xt = torch.tensor(Xtr, dtype=torch.float64)   # raw; integrate normalizes
    Yt = torch.tensor(Ytr, dtype=torch.float64)
    edge = make_edge(kind, Xt.shape[1], h)
    opt = torch.optim.Adam(edge.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        dq, dv = integrate(Xt[:, :7], Xt[:, 7:14], Xt[:, 14:], edge, kind,
                           mu, sd, dt, H)
        loss = ((torch.cat([dq, dv], -1) - Yt) ** 2).mean()
        loss.backward()
        opt.step()
    return mu, sd, edge


def evaluate(kind, Xev, Yev, H, dt, mu, sd, edge, h=64):
    Xe = torch.tensor(Xev, dtype=torch.float64)   # raw; integrate normalizes
    with torch.no_grad():
        dq, dv = integrate(Xe[:, :7], Xe[:, 7:14], Xe[:, 14:], edge, kind,
                           mu, sd, dt, H)
        pred = torch.cat([dq, dv], -1).numpy()
    return rmse(pred, Yev)


def fit_flat(Xtr, Ytr, Xev, Yev, epochs=300, seed=0, h=64):
    torch.manual_seed(seed)
    mu = Xtr.mean(0, keepdims=True)
    sd = np.where(Xtr.std(0, keepdims=True) < 1e-8, 1.0, Xtr.std(0, keepdims=True))
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float64)
    Yt = torch.tensor(Ytr, dtype=torch.float64)
    Xe = torch.tensor((Xev - mu) / sd, dtype=torch.float64)
    net = nn.Sequential(nn.Linear(Xt.shape[1], h), nn.Tanh(),
                        nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 14)).double()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt) - Yt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(Xe).numpy()
    return rmse(pred, Yev)


def run(seeds=(0, 1, 2), dt=DT, H=H, epochs=200, max_train=20000):
    Xtr, Ytr, Xte, Yte = [], [], [], []
    for k in ('squared', 'random'):
        q, v, t = load(k)
        X, Y = windows(q, v, t, H)
        c = int(0.7 * len(X))
        Xtr.append(X[:c]); Ytr.append(Y[:c])
        Xte.append(X[c:]); Yte.append(Y[c:])
    Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
    Xte, Yte = np.concatenate(Xte), np.concatenate(Yte)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xtr))[:min(max_train, len(Xtr))]
    Xtr, Ytr, Xte, Yte = Xtr[idx], Ytr[idx], Xte, Yte
    qc, vc, tc = load('circle')
    Xc, Yc = windows(qc, vc, tc, H)

    def zero(X):
        v = X[:, 7:14]
        return np.concatenate([v * H * dt, np.zeros_like(v)], -1)

    rows = {}
    rows['zero'] = (rmse(zero(Xte), Yte), rmse(zero(Xc), Yc))
    rows['mean'] = (rmse(np.tile(Ytr.mean(0), (len(Yte), 1)), Yte),
                    rmse(np.tile(Ytr.mean(0), (len(Xc), 1)), Yc))
    for kind in ('lin', 'phys', 'mlp'):
        it, ic = [], []
        for s in seeds:
            mu, sd, edge = fit(kind, Xtr, Ytr, H, dt, seed=s, epochs=epochs)
            it.append(evaluate(kind, Xte, Yte, H, dt, mu, sd, edge))
            ic.append(evaluate(kind, Xc, Yc, H, dt, mu, sd, edge))
        rows[kind] = (float(np.mean(it)), float(np.mean(ic)))
    ft, fc = [], []
    for s in seeds:
        ft.append(fit_flat(Xtr, Ytr, Xte, Yte, seed=s, epochs=epochs))
        fc.append(fit_flat(Xtr, Ytr, Xc, Yc, seed=s, epochs=epochs))
    rows['flat'] = (float(np.mean(ft)), float(np.mean(fc)))

    # hidden-velocity variant (true no-classical cell)
    Xtrh, Ytrh, Xteh, Yteh = [], [], [], []
    for k in ('squared', 'random'):
        q, v, t = load(k)
        X, Y = windows_hidden(q, v, t, H)
        c = int(0.7 * len(X))
        Xtrh.append(X[:c]); Ytrh.append(Y[:c])
        Xteh.append(X[c:]); Yteh.append(Y[c:])
    Xtrh, Ytrh = np.concatenate(Xtrh), np.concatenate(Ytrh)
    Xteh, Yteh = np.concatenate(Xteh), np.concatenate(Yteh)
    idxh = rng.permutation(len(Xtrh))[:min(max_train, len(Xtrh))]
    Xtrh, Ytrh = Xtrh[idxh], Ytrh[idxh]
    qc, vc, tc = load('circle')
    Xch, Ych = windows_hidden(qc, vc, tc, H)
    rows['hidden:hold'] = (rmse(np.zeros_like(Yteh), Yteh), rmse(np.zeros_like(Ych), Ych))
    rows['hidden:mean'] = (rmse(np.tile(Ytrh.mean(0), (len(Yteh), 1)), Yteh),
                           rmse(np.tile(Ytrh.mean(0), (len(Xch), 1)), Ych))
    ht, hc = [], []
    for s in seeds:
        ht.append(fit_flat(Xtrh, Ytrh, Xteh, Yteh, seed=s, epochs=epochs))
        hc.append(fit_flat(Xtrh, Ytrh, Xch, Ych, seed=s, epochs=epochs))
    rows['hidden:flat'] = (float(np.mean(ht)), float(np.mean(hc)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='smoke: 1 seed, dt=1 ms only, 60 epochs, 3000 windows')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--dt', default='1,2', help='comma-separated dt in ms')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--max-train', type=int, default=20000)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    dts = tuple(float(x) / 1000.0 for x in args.dt.split(','))
    epochs, max_train = args.epochs, args.max_train
    if args.quick:
        seeds, dts, epochs, max_train = (0,), (1e-3,), 60, 3000
    if skip_if_missing(D, 'Baxter CSV directory (set BAXTER)'):
        return
    for name in FILES.values():
        if skip_if_missing(os.path.join(D, name), f'Baxter file {name}'):
            return
    t0 = time.time()
    eff = rate_check()
    print(f'[E26b] effective step from data: {eff*1000:.2f} ms '
          f'(~{1/eff:.0f} Hz); journal assumed {DT*1000:.1f} ms')
    print(f'[E26b] horizon H={H} steps -> {H*DT*1000:.0f} ms at dt={DT*1000:.1f} ms')
    out = {}
    for dt in dts:
        rows = run(seeds=seeds, dt=dt, H=H, epochs=epochs, max_train=max_train)
        out[f'dt{dt*1000:.0f}'] = rows
        print(f'\n==== dt = {dt*1000:.1f} ms ====')
        print(f'{"model":14s} {"in-dist":>9s} {"circle":>9s}   verdict')
        for k, (a, b) in rows.items():
            z = rows['zero'][1] if not k.startswith('hidden') else rows['hidden:hold'][1]
            tag = 'BEATS zero' if b < z else 'worse'
            print(f'{k:14s} {a:9.3f} {b:9.3f}   {tag}')
    RESULTS['E26b'] = out
    RESULTS['_meta'] = dict(dt_journal=DT, dt_effective=eff, H=H, K_tau=KTAU,
                            wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=len(seeds), epochs=epochs)
    print(f'\ndump -> {p}')


if __name__ == '__main__':
    main()
