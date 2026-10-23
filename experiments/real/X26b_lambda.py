#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X26b_lambda.py — L2 (weight-decay) sweep on the psi head (registered before run).

X26 showed the collapse is neither a magnitude nor a periodicity problem: the bound
never engages (sat<1) and only the L2 term removes the collapse (seed 8/9), but
lambda=1e-2 also suppresses the healthy correction (seed 0). Here we sweep a small
lambda to look for a value that removes the collapse without killing the healthy gain.

Protocol: Kinova socket (MK1), isolation, seeds 0 (healthy), 8, 9 (collapse),
3000 steps, lr 1e-3; variants base (no L2) and l2 in {1e-3, 3e-3, 1e-2}.

Registered prediction (plan 2026-10-13, before run):
  P-LAM: a small lambda (<=3e-3) removes the collapse on seeds 8,9 (rms <= edge_b)
         while leaving seed 0 at base level (within 0.05 mm); if instead every
         lambda that fixes 8,9 also degrades seed 0 by >0.05 mm, the collapse and
         the healthy gain are not separable by L2 alone (-> X27 curriculum).

Output: repro/json/X26b_lambda.<ts>.json.
Run:  python X26b_lambda.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import MK1_mukca as MK              # noqa: E402
import S2_ur5gtpn_relaxed as S2     # noqa: E402
import X10_fk_calib_real as X10     # noqa: E402
from repro.dump import save_dump    # noqa: E402

DT = torch.float64
RESULTS = {}
PLAT = 'kinova'


def build_iso(xi, M0, edge_state, seed):
    torch.manual_seed(seed)
    iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
    iso.load_state_dict(copy.deepcopy(edge_state), strict=False)
    for p in iso.parameters():
        p.requires_grad_(False)
    for p in iso.psi.parameters():
        p.requires_grad_(True)
    return iso


def fit(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    q0t = torch.tensor(q0_tr, dtype=DT)
    q1t = torch.tensor(q1_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        loss = MK.socket_loss(m, q0t, q1t)
        if l2:
            loss = loss + l2 * sum((p ** 2).sum() for p in m.psi.parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-list', default='0,8,9')
    ap.add_argument('--l2-list', default='1e-3,3e-3,1e-2')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seed_list, args.steps = '0,8', 300
    t0 = time.time()

    xi, M0 = MK.platform_tensors(MK.PLATFORMS[PLAT])
    q0, q1 = MK.load_sockets(PLAT)
    seeds = [int(x) for x in args.seed_list.split(',')]
    l2s = [float(x) for x in args.l2_list.split(',')]

    variants = ['base'] + [f'l2_{x:g}' for x in l2s]
    rms = {v: [] for v in ('edge_b',) + tuple(variants)}

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
        edge_state = copy.deepcopy(eb.state_dict())

        for v, l2 in [('base', 0.0)] + [(f'l2_{x:g}', x) for x in l2s]:
            m = build_iso(xi, M0, edge_state, s)
            fit(m, q0_tr, q1_tr, args.steps, args.lr, s, l2=l2)
            rms[v].append(MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)[0])
        print(f'[X26b seed {s}] edge_b {rms["edge_b"][-1]:.3f} | ' +
              ' '.join(f'{v} {rms[v][-1]:.3f}' for v in variants), flush=True)

    for k, a in rms.items():
        a = np.array(a)
        RESULTS[f'X26b:{k}:rms_mm'] = dict(mean=float(a.mean()), std=float(a.std()),
                                           per_seed=[float(x) for x in a])
    RESULTS['X26b:n_collapsed'] = {v: int((np.array(rms[v]) > np.array(rms['edge_b'])).sum())
                                   for v in variants}
    RESULTS['X26b:protocol'] = dict(platform=PLAT, seeds=seeds, l2=l2s,
                                    steps=args.steps, lr=args.lr)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
