#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X25b_weight_corr.py — cross-seed consistency of the learned psi function
(mechanism check for X25; registered P-B5).

Hypothesis (the "winding" mechanism): with the periodic/unbounded parameterization
different seeds land on different windings of exp(psi), so the *learned function*
psi differs across seeds; an injective (bounded) chart makes psi one-to-one and
the functions should agree across seeds. We fingerprint the psi function on a
fixed probe set of features (raw MLP weights are permutation-ambiguous, so we use
the function itself) and report the mean pairwise Pearson correlation across
seeds for base vs chart.

Registered prediction (P-B5, plan 2026-10-13, before run):
  corr(chart) > corr(base) on healthy Kinova seeds; if not, periodicity was not
  the main cause.

Output: repro/json/X25b_weight_corr.<ts>.json.
Run:  python X25b_weight_corr.py [--quick]
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

import MK1_mukca as MK              # noqa: E402
import X10_fk_calib_real as X10     # noqa: E402
import X25_psi_bound as X25         # noqa: E402
from repro.dump import save_dump    # noqa: E402

DT = torch.float64
RESULTS = {}
PLAT = 'kinova'


def mean_pairwise_corr(fps):
    """fps: list of 1-D np arrays (one per seed); mean Pearson r over pairs."""
    rs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            a, b = fps[i], fps[j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                continue
            rs.append(stats.pearsonr(a, b)[0])
    return (float(np.mean(rs)) if rs else float('nan')), [float(x) for x in rs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-list', default='0,1,2')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wmax', type=float, default=1.5)
    ap.add_argument('--probes', type=int, default=64)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seed_list, args.steps, args.probes = '0,1', 300, 16
    t0 = time.time()

    xi, M0 = MK.platform_tensors(MK.PLATFORMS[PLAT])
    q0, q1 = MK.load_sockets(PLAT)
    seeds = [int(x) for x in args.seed_list.split(',')]

    g = torch.Generator().manual_seed(0)
    u_probe = (torch.randn(args.probes, 14, generator=g, dtype=DT) * np.pi)

    fp = {'base': [], 'chart': []}
    rms = {'edge_b': [], 'base': [], 'chart': []}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        MK.fit(eb, q0_tr, q1_tr, args.steps, args.lr, s)
        rms['edge_b'].append(MK.evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)[0])
        edge_state = eb.state_dict()

        for v in ('base', 'chart'):
            m = X25.build_iso(xi, M0, edge_state, s,
                              bound=(args.wmax if v == 'chart' else None))
            X25.fit(m, q0_tr, q1_tr, args.steps, args.lr, s)
            rms[v].append(MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)[0])
            with torch.no_grad():
                out = m.psi(u_probe).detach().cpu().numpy().ravel()
            fp[v].append(out)
        print(f'[X25b seed {s}] edge_b {rms["edge_b"][-1]:.3f} '
              f'base {rms["base"][-1]:.3f} chart {rms["chart"][-1]:.3f}', flush=True)

    for v in ('base', 'chart'):
        r, pairs = mean_pairwise_corr(fp[v])
        RESULTS[f'X25b:{v}:cross_seed_corr'] = dict(mean=r, pairs=pairs)
    RESULTS['X25b:P-B5'] = dict(
        corr_base=RESULTS['X25b:base:cross_seed_corr']['mean'],
        corr_chart=RESULTS['X25b:chart:cross_seed_corr']['mean'],
        holds=bool(RESULTS['X25b:chart:cross_seed_corr']['mean']
                   > RESULTS['X25b:base:cross_seed_corr']['mean']))
    RESULTS['X25b:protocol'] = dict(platform=PLAT, seeds=seeds, steps=args.steps,
                                    lr=args.lr, wmax=args.wmax,
                                    probes=args.probes, fingerprint='psi(u_probe)')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
