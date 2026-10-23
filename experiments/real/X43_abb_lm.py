#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X43_abb_lm.py — OUR model (edge + global psi) fitted with the CLASSICAL optimizer
(scipy least_squares, LM) on the SAME residual as the `gn` reference (ABB range-only).

Motivation. In Table 7 (top, ABB) the LawKAN rows use Adam while `reference` is the
classical `gn` (scipy LM). To remove the "it's your optimizer/loss" objection, give
BOTH sides the SAME (best) solver: fit our model (RC1.RangeScrew relax, global psi
head = the `global` column / X36 `mlp`) with scipy least_squares on the same residual
`norm(p - anchor) - L`. Any residual gap to `gn` is then architecture (psi), not Adam.

Protocol: identical to RC1/X36 (splits rng(1000+seed), seeds 0-4, N=80/160/320,
metric mean |dL| mm). x0 = zero init (as `gn`).

Output: repro/json/X43_abb_lm.<ts>.json via repro/dump.
Run:  python X43_abb_lm.py [--quick]
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
torch.set_num_threads(1)
from scipy import stats
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import RC1_robotcali as RC1                 # noqa: E402
from repro.dump import save_dump
from repro.canon import load as load_canon            # noqa: E402

DT = torch.float64
RESULTS = {}


def pack(params):
    return np.concatenate([p.detach().numpy().ravel() for p in params])


def unpack(params, x):
    i = 0
    for p in params:
        n = p.numel()
        with torch.no_grad():
            p.copy_(torch.tensor(x[i:i + n], dtype=DT).view_as(p))
        i += n


def fit_lm(m, q_tr, L_tr, max_nfev=400, params=None):
    params = list(params) if params is not None else list(m.parameters())
    qt = torch.tensor(q_tr, dtype=DT)
    Lt = np.asarray(L_tr, dtype=np.float64)
    x0 = pack(params)

    def fun(x):
        unpack(params, x)
        with torch.no_grad():
            pred = m.range(qt).numpy().astype(np.float64)
        return pred - Lt

    sol = least_squares(fun, x0, method='trf', max_nfev=max_nfev, x_scale='jac')
    unpack(params, sol.x)
    return m


def load_gn():
    d = load_canon('rc1')
    return {N: np.array(d[f'RC1:N{N}:gn']['per_seed']) for N in (80, 160, 320)}, \
           {N: np.array(d[f'RC1:N{N}:exact']['per_seed']) for N in (80, 160, 320)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=RC1.DATA)
    ap.add_argument('--sizes', default='80,160,320')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--max-nfev', type=int, default=400)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.max_nfev = '80', 1, 100
    t0 = time.time()

    robot, xi, M0 = RC1.build_robot()
    q_all, L_all = RC1.load_robotcali(args.data)
    Tf_all = RC1.X10.fk_np(robot, q_all)
    p_nom, R_nom = Tf_all[:, :3, 3], Tf_all[:, :3, :3]
    sizes = [int(s) for s in args.sizes.split(',')]
    print('[X43] our model (edge + global psi) on the classical LM, ABB', flush=True)

    res = {(N, k): [] for N in sizes for k in ('psi_lm', 'exact')}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(q_all))
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            a0 = RC1.anchor_init(p_nom[tr], L_all[tr])
            ex = RC1.RangeScrew(xi, M0, a0)
            RC1.fit_adam(ex, q_all[tr], L_all[tr], 3000, 1e-2, s)
            res[(N, 'exact')].append(RC1.eval_dL_range(ex, q_all[te], L_all[te]))
            ps = RC1.RangeScrew(xi, M0, a0, relax=True)
            ps.load_state_dict(copy.deepcopy(ex.state_dict()), strict=False)
            for q in ps.parameters():
                q.requires_grad_(False)
            psi_params = list(ps.sc.psi.parameters())
            for q in psi_params:
                q.requires_grad_(True)
            ps = fit_lm(ps, q_all[tr], L_all[tr], args.max_nfev, params=psi_params)
            r = RC1.eval_dL_range(ps, q_all[te], L_all[te])
            res[(N, 'psi_lm')].append(r)
            print(f'[X43 seed {s} N={N}] psi_lm {r:.4f} exact {res[(N,"exact")][-1]:.4f}', flush=True)

    gn, exact = load_gn()
    for N in sizes:
        for k in ('psi_lm', 'exact'):
            a = np.array(res[(N, k)])
            RESULTS[f'X43:N{N}:{k}'] = dict(
                dL_mm_mean=float(a.mean()), dL_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        a = np.array(res[(N, 'psi_lm')])
        for name, ref in (('gn', gn), ('exact', res[(N, 'exact')])):
            b = np.asarray(ref, dtype=np.float64)[:len(a)]
            if len(b) < 2:
                continue
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'X43:paired:N{N}:psi_lm_vs_{name}'] = dict(
                diff=float((a - b).mean()), t=float(t), p=float(p))
        print(f'N={N}: psi_lm {a.mean():.4f} gn {np.asarray(gn[N])[:len(a)].mean():.4f} '
              f'exact {np.array(res[(N,"exact")]).mean():.4f}')

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds,
                  extra_meta={'model': 'isolated global psi on the frozen Adam exact edge',
                              'optimizer': 'scipy least_squares trf (LM), matched to gn',
                              'note': 'P-X43: our model at the classical optimizer'})
    print('dump ->', p)


if __name__ == '__main__':
    main()
