#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X11_ur5_gtpn_sweep.py — external UR5 dataset (Zenodo record 18954650, CC-BY-4.0):
position-only FK-calibration sweep with the X10 model class, our protocol.

Data: newdata/ur5_gtpn.csv, 727 poses: q (deg) + target XYZ/RxRyRz + measured
XYZ (mm). No orientation metrology -> position only. Provenance is recorded in
the co-author real-data documentation (branch NP_master).

Protocol (registered in the plan 2026-09-29, predictions U1-U4):
  ours : X10.ScrewCalib dense (48p const edge), X10 training (2000 steps,
         Adam 1e-2, cosine; position loss), same as the X10 harness
  gn   : scipy least_squares (trf, x_scale='jac') of the same model class
  const: 12p base/tool gap fit (X10 fit_gaps_torch)
  theta: Fourier K=1 edge (120p) — BEYOND-PARITY control (U3)
  N in {20,40,60,80}, seeds 0-4 (rng 1000+seed), test = held-out poses,
  metric = held-out position error (mm); paired t and TOST 10% vs ours.

Output: repro/json/X11_ur5_gtpn_sweep.<ts>.json via repro/dump.
Run:  python X11_ur5_gtpn_sweep.py            # 5 seeds, N=20..80
      python X11_ur5_gtpn_sweep.py --quick    # smoke: 1 seed, N=80
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X10_fk_calib_real as X10          # noqa: E402
from repro.dump import save_dump          # noqa: E402

DATA = os.environ.get(
    'GTPN', os.path.join(os.path.dirname(_REPO), 'newdata', 'ur5_gtpn.csv'))
RESULTS = {}


def load_gtpn(path):
    df = pd.read_csv(path)
    q = np.deg2rad(df.iloc[:, :6].values.astype(np.float64))
    p = df[['x-real', 'y-real', 'z-real']].values.astype(np.float64) / 1000.0
    T = np.tile(np.eye(4), (len(df), 1, 1))
    T[:, :3, 3] = p
    return dict(name='ur5_gtpn', robot='ur5', q=q, T=T, full=False, goal=None,
                note='Zenodo 18954650 (CC-BY-4.0), position-only, 727 poses')


def fit_gn(xi, M0, q_tr, T_obs2_tr, tool_init, max_nfev=300):
    """Least squares of the same X10.ScrewCalib class, position residuals."""
    from scipy.optimize import least_squares
    m = X10.ScrewCalib(xi, M0, mode='dense')
    with torch.no_grad():
        m.tool.copy_(torch.tensor(tool_init, dtype=m.tool.dtype))
    n = int(xi.shape[0])
    x0 = np.concatenate([m.dxi.detach().numpy().ravel(),
                         m.b.detach().numpy(), m.tool.detach().numpy()])
    qt = torch.tensor(q_tr, dtype=torch.float64)
    Pt = torch.tensor(T_obs2_tr[:, :3, 3], dtype=torch.float64)

    def set_from(x):
        i = 0
        with torch.no_grad():
            m.dxi.copy_(torch.tensor(x[i:i + 6 * n],
                                     dtype=torch.float64).view(n, 6))
            i += 6 * n
            m.b.copy_(torch.tensor(x[i:i + n], dtype=torch.float64))
            i += n
            m.tool.copy_(torch.tensor(x[i:i + 6], dtype=torch.float64))

    def residual(x):
        set_from(x)
        with torch.no_grad():
            Tp = m(qt)
        return (Tp[:, :3, 3] - Pt).numpy().ravel()

    r = least_squares(residual, x0, method='trf', x_scale='jac',
                      max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)
    set_from(r.x)
    return m


def tost(a, b, margin):
    d = np.asarray(a) - np.asarray(b)
    _, p1 = stats.ttest_1samp(d, -margin, alternative='greater')
    _, p2 = stats.ttest_1samp(d, margin, alternative='less')
    return float(max(p1, p2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 200
    t0 = time.time()

    data = load_gtpn(args.data)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    sizes = [int(s) for s in args.sizes.split(',')]
    raw = X10.pose_metrics(Tf_all, T_all, full=False)
    print(f'[X11] {data["name"]}: {n} poses, nominal_raw pos {raw["pos_mean"]:.3f} mm',
          flush=True)

    cells = {(N, k): [] for N in sizes for k in ('ours', 'gn', 'theta', 'const')}
    rows = {}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], None, full=False)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=False, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            models = {
                'ours': X10.train_screwcalib(xi, M0, q_all[tr], T_obs2_tr,
                                             X10.log_se3_np(C), mode='dense',
                                             steps=args.steps, seed=s, full=False),
                'theta': X10.train_screwcalib(xi, M0, q_all[tr], T_obs2_tr,
                                              X10.log_se3_np(C), mode='fourier',
                                              K=1, steps=args.steps, seed=s, full=False),
            }
            models['gn'] = fit_gn(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C))
            cells[(N, 'const')].append(
                X10.pose_metrics(B @ Tf_all[te] @ C, T_all[te], full=False)['pos_mean'])
            for k in ('ours', 'gn', 'theta'):
                cells[(N, k)].append(
                    X10.pose_metrics(X10.apply_sc(models[k], q_all[te]),
                                     T_obs2_te, full=False)['pos_mean'])
            print(f'[X11 seed {s} N={N}] ours {cells[(N,"ours")][-1]:.3f} '
                  f'gn {cells[(N,"gn")][-1]:.3f} '
                  f'theta {cells[(N,"theta")][-1]:.3f} '
                  f'const {cells[(N,"const")][-1]:.3f}', flush=True)
        print(f'[X11] seed {s} done', flush=True)

    for N in sizes:
        for k in ('ours', 'gn', 'theta', 'const'):
            a = np.array(cells[(N, k)])
            rows[f'X11:n{N}:{k}'] = dict(pos_mm_mean=float(a.mean()),
                                         pos_mm_std=float(a.std()),
                                         per_seed=[float(x) for x in a])
        a = np.array(cells[(N, 'ours')])
        for ref in ('gn', 'theta', 'const'):
            b = np.array(cells[(N, ref)])
            t, p = stats.ttest_rel(a, b)
            rows[f'X11:paired:n{N}:ours_vs_{ref}'] = dict(
                diff_pos=float(a.mean() - b.mean()), t=float(t), p=float(p),
                tost10=float(tost(a, b, 0.10 * abs(a.mean()))))

    RESULTS.update(rows)
    RESULTS['X11:nominal_raw'] = raw
    RESULTS['X11:params'] = dict(ours=48, gn=48, theta=120, const=12)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.steps)
    print('dump ->', p)


if __name__ == '__main__':
    main()
