#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X21_fric_proxy.py — P-FRIC-proxy on IPA Series A (UR10, laser tracker).

Real trajectories are unavailable in most sets; IPA Series A has real Timestamps
(11911 poses), so the approach direction d_i = sign(q_i - q_{i-1}) and a speed
proxy |dq|/dt are available from the recording order.  Residuals of a dense
screw model (fitted on 4000 rows) are evaluated on 2000 held-out rows.

Registered test (plan 2026-10-05): for each joint j, the correlation between
d_ij and the residual projection onto the Jacobian column direction u_ij is
predicted POSITIVE (backlash direction signature); pass = >= 5/6 positive signs
AND permutation p < 0.05 (shuffled d), else INCONCLUSIVE (coarse order).
Speed/dwell correlation is reported as a preview only.

Output: repro/json/X21_fric_proxy.<ts>.json
Run: python X21_fric_proxy.py [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import X10_fk_calib_real as X10     # noqa: E402
from repro.dump import save_dump    # noqa: E402

RESULTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--train', type=int, default=4000)
    ap.add_argument('--test', type=int, default=2000)
    ap.add_argument('--steps-gap', type=int, default=1500)
    ap.add_argument('--steps-sc', type=int, default=3000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.train, args.test, args.steps_gap, args.steps_sc = 800, 400, 200, 300
    t0 = time.time()

    df = pd.read_csv(os.path.join(args.data, 'ipa', 'Series_A', 'measurements_grid.csv'),
                     sep=';', decimal=',')
    ts = df['Timestamp'].to_numpy(dtype=np.float64)
    d = X10.load_ipa(args.data, 'A')
    q, T, goal = d['q'], d['T'], d['goal']
    n = len(q)
    print(f'[X21] IPA A: {n} rows, timestamps {int(ts.min())}..{int(ts.max())}', flush=True)

    robot, xi, M0 = X10.get_robot_tensors(d['robot'])
    Tf = X10.fk_np(robot, q)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr = perm[:args.train]
    te = perm[args.train:args.train + args.test]

    B0, C0 = X10.gap_init(Tf[tr], T[tr], goal[tr], full=True)
    B, C = X10.fit_gaps_torch(Tf[tr], T[tr], B0, C0, steps=args.steps_gap,
                              full=True, seed=0)
    invB = np.linalg.inv(B)
    T_obs2 = np.array([invB @ t for t in T[tr]])
    model = X10.train_screwcalib(xi, M0, q[tr], T_obs2, X10.log_se3_np(C),
                                 mode='dense', steps=args.steps_sc, seed=0, full=True)
    pred = X10.apply_sc(model, q[te])
    meas = np.array([invB @ t for t in T[te]])
    res = (pred[:, :3, 3] - meas[:, :3, 3]) * 1000.0            # mm
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    print(f'[X21] held-out position RMS {rms:.3f} mm', flush=True)

    # approach directions and speed from the GLOBAL recording order
    dq = np.diff(q, axis=0)                       # row i-1 -> i
    dt = np.diff(ts)
    dsign = np.sign(dq)
    speed = np.linalg.norm(dq, axis=1) / np.clip(dt, 1e-3, None)

    # Jacobian column directions per test row
    Jv = np.zeros((len(te), 3, q.shape[1]))
    for k, i in enumerate(te):
        J = robot.jacobe(q[i])
        Jv[k] = J[3:6, :]

    valid = (te > 0)
    idx = np.where(valid)[0]
    # mask: moved enough and sane dt
    keep = (np.linalg.norm(dq[te[idx] - 1], axis=1) > 0.02) & (dt[te[idx] - 1] > 0.5)
    idx = idx[keep]
    rows = te[idx]
    R = res[idx]
    D = dsign[rows - 1]                            # (m,6)
    SP = speed[rows - 1]

    # per-joint signed projections onto Jacobian unit directions
    rhos = []
    for j in range(q.shape[1]):
        u = Jv[idx, :, j]
        u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-9)
        s = (R * u).sum(1)                          # mm along +q_j
        if D[:, j].std() > 0 and s.std() > 0:
            rho = float(np.corrcoef(D[:, j], s)[0, 1])
        else:
            rho = 0.0
        rhos.append(rho)
    rhos = np.array(rhos)
    pos = int((rhos > 0).sum())

    # permutation: shuffle direction rows
    rng2 = np.random.default_rng(1)
    null = []
    for _ in range(1000):
        p = rng2.permutation(len(rows))
        DD = D[p]
        r = []
        for j in range(q.shape[1]):
            u = Jv[idx, :, j]
            u = u / np.linalg.norm(u, axis=1, keepdims=True).clip(1e-9)
            s = (R * u).sum(1)
            r.append(float(np.corrcoef(DD[:, j], s)[0, 1]))
        null.append(np.mean(r))
    obs_mean = float(rhos.mean())
    p_perm = float(np.mean(np.array(null) >= obs_mean))

    # speed preview: Spearman correlation of |res| with speed
    from scipy import stats
    rho_sp, p_sp = stats.spearmanr(np.linalg.norm(R, axis=1), SP)

    passed = bool(pos >= 5 and p_perm < 0.05)
    RESULTS['X21:rows'] = dict(total=int(n), evaluated=int(len(rows)))
    RESULTS['X21:heldout_rms_mm'] = rms
    RESULTS['X21:joint_rhos'] = rhos.tolist()
    RESULTS['X21:positive_signs'] = pos
    RESULTS['X21:mean_rho'] = obs_mean
    RESULTS['X21:perm_p'] = p_perm
    RESULTS['X21:speed_spearman'] = dict(rho=float(rho_sp), p=float(p_sp))
    RESULTS['X21:verdict'] = 'signal' if passed else 'inconclusive'
    print(f'[X21] joint rhos {np.round(rhos, 3).tolist()} -> {pos}/6 positive; '
          f'mean {obs_mean:+.4f}, perm p={p_perm:.4f} -> '
          f'{"SIGNAL" if passed else "INCONCLUSIVE"}', flush=True)
    print(f'[X21] speed preview: spearman {rho_sp:+.3f} (p={p_sp:.3g})', flush=True)

    RESULTS['X21:protocol'] = dict(train=args.train, test=args.test,
                                   steps_gap=args.steps_gap, steps_sc=args.steps_sc)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'done {time.time()-t0:.0f}s; dump: {out}')


if __name__ == '__main__':
    main()
