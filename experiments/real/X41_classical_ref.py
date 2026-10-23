#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X41_classical_ref.py — the CLASSICAL reference for ABB (range-only) at the SAME
optimizer as our models (Adam). (registered before run)

Motivation. In RC1 the classical reference (`gn`) is a scipy Gauss-Newton/LM fit of the
same 51p class, while our models use Adam --- so the "parity"/"win" is confounded with
the optimizer. Here the SAME classical class is fit with the SAME Adam schedule
(RC1's own `fit_adam`: lr 1e-2, cosine, 3000 steps), giving an optimizer-matched
classical reference. (On a well-conditioned class the optimizer barely matters, so this
should stay close to `exact`/`gn`; the point is to have it on record.)

Protocol: identical to RC1 (splits rng(1000+seed), seeds 0-4, N=80/160/320,
metric mean |dL| mm on held-out poses).

Registered prediction (plan 2026-10-16):
  P-CLS-3: the Adam-fitted classical reference matches RC1's `exact` (same class, same
           optimizer) --- parity up to the optimizer.

Output: repro/json/X41_classical_ref.<ts>.json via repro/dump.
Run:  python X41_classical_ref.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import RC1_robotcali as RC1                 # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=RC1.DATA)
    ap.add_argument('--sizes', default='80,160,320')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 300
    t0 = time.time()

    robot, xi, M0 = RC1.build_robot()
    q_all, L_all = RC1.load_robotcali(args.data)
    Tf_all = RC1.X10.fk_np(robot, q_all)
    p_nom, R_nom = Tf_all[:, :3, 3], Tf_all[:, :3, :3]
    sizes = [int(s) for s in args.sizes.split(',')]
    print('[X41] classically-fitted reference (Adam, matched), ABB range-only', flush=True)

    res = {(N, k): [] for N in sizes for k in ('classical',)}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(q_all))
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            a0 = RC1.anchor_init(p_nom[tr], L_all[tr])
            m = RC1.RangeScrew(xi, M0, a0)          # classical class, no law node, no psi
            RC1.fit_adam(m, q_all[tr], L_all[tr], args.steps, args.lr, s)
            r = RC1.eval_dL_range(m, q_all[te], L_all[te])
            res[(N, 'classical')].append(r)
            print(f'[X41 seed {s} N={N}] classical {r:.4f} mm', flush=True)

    for N in sizes:
        a = np.array(res[(N, 'classical')])
        RESULTS[f'X41:N{N}:classical'] = dict(
            dL_mm_mean=float(a.mean()), dL_mm_std=float(a.std()),
            per_seed=[float(x) for x in a])

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=args.steps,
              extra_meta={'method': 'classical 51p class, Adam (matched optimizer)',
                          'note': 'P-CLS-3: Adam classical reference ~ RC1 exact'})


if __name__ == '__main__':
    main()
