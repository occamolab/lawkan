#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X39_classical_socket.py — the CLASSICAL socket reference: the per-joint screw fit by
nonlinear least squares (Levenberg--Marquardt), independent of our Adam/relaxed-node
architecture. (registered before run)

Motivation. Table 8's `dense screw` reference is currently our gradient-descent fit of
the per-joint-screw class (MK1_mukca `ScrewCalib` + `MK.fit`), i.e. our own equivalent
architecture, not a distinct classical algorithm --- so it cannot serve as the
"classical vs LawKAN" reference. Here the SAME model class is fit classically:
`scipy.optimize.least_squares` (LM) on the identical socket residual, giving an
architecture-independent reference. The point is the practical equivalence check: the
classical LM screw should match our LawKAN screw (X38 `dense`).

Residual (sum of squares == `MK1_mukca.socket_loss`):
  [ (p0 - mean0)/sqrt(n0), (p1 - mean1)/sqrt(n1), sqrt(lam)*(||mean0-mean1|| - 0.05) ].

Models (by which parameters are free):
  nominal  : tool only (6p);
  edge_b   : tool + joint biases b (13p);
  dense    : full per-joint screw (55p) --- the classical reference.

Registered prediction (plan 2026-10-16):
  P-CLS-1: on every platform the classical LM `dense` matches our Adam `dense`
           (X38) within noise.

Output: repro/json/X39_classical_socket.<ts>.json via repro/dump.
Run:  python X39_classical_socket.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy import stats
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import X10_fk_calib_real as X10            # noqa: E402
import MK1_mukca as MK                      # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}
SOCKET = ['kuka14', 'panda_1', 'fr3', 'kinova']
MODELS = ('nominal', 'edge_b', 'dense')


def free_names(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def get_flat(model, names):
    return np.concatenate([model.get_parameter(n).detach().numpy().reshape(-1) for n in names])


def set_flat(model, names, theta):
    i = 0
    with torch.no_grad():
        for n in names:
            p = model.get_parameter(n)
            k = p.numel()
            p.copy_(torch.tensor(theta[i:i + k], dtype=DT).reshape(p.shape))
            i += k


def make_residual(model, names, q0t, q1t, n0, n1, lam):
    def res(theta):
        set_flat(model, names, theta)
        with torch.no_grad():
            p0 = model(q0t)[:, :3, 3]
            p1 = model(q1t)[:, :3, 3]
            c0, c1 = p0.mean(0), p1.mean(0)
            r0 = ((p0 - c0) / np.sqrt(n0)).reshape(-1)
            r1 = ((p1 - c1) / np.sqrt(n1)).reshape(-1)
            rd = (np.sqrt(lam) * ((c0 - c1).norm() - 0.05)).reshape(1)
        return torch.cat([r0, r1, rd]).detach().numpy()
    return res


def build_model(xi, M0, kind, seed):
    torch.manual_seed(seed)
    m = X10.ScrewCalib(xi, M0, mode='dense')
    if kind == 'dense':
        pass
    elif kind == 'edge_b':
        m.dxi.requires_grad_(False)
    elif kind == 'nominal':
        m.dxi.requires_grad_(False)
        m.b.requires_grad_(False)
    else:
        raise ValueError(kind)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default=','.join(SOCKET))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--fr3_seeds', type=int, default=5)
    ap.add_argument('--lam', type=float, default=100.0)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.fr3_seeds = 1, 1
    t0 = time.time()

    for plat in args.platforms.split(','):
        xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
        q0, q1 = MK.load_sockets(plat)
        nseed = args.fr3_seeds if plat == 'fr3' else args.seeds
        print(f'[X39 {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}, '
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

            line = f'[X39 {plat} seed {s}]'
            for kind in MODELS:
                m = build_model(xi, M0, kind, s)
                names = free_names(m)
                th0 = get_flat(m, names)
                sol = least_squares(make_residual(m, names, q0t, q1t, n0, n1, args.lam),
                                    th0, method='lm', max_nfev=20000)
                set_flat(m, names, sol.x)
                r, _ = MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)
                rms[kind].append(r)
                line += f' {kind} {r:.3f}'
            print(line + ' mm', flush=True)

        for k in MODELS:
            a = np.array(rms[k])
            RESULTS[f'X39:{plat}:{k}:rms_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                per_seed=[float(x) for x in a])

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=0,
              extra_meta={'method': 'classical nonlinear-LS (LM) fit of the per-joint screw',
                          'note': 'P-CLS-1: classical LM dense matches our Adam dense (X38)'})


if __name__ == '__main__':
    main()
