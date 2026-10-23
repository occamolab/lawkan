#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X7_gn_ls.py — the strongest classical baseline in the X7 harness: a Gauss-Newton
least-squares fit of the SAME model class as QuatKANArm (base + per-joint SE(3)
corrections + tool, 54 parameters in exp coordinates) on the same splits.

Why: X7's robotkinecal PoE-LS (1.52 mm at n=80) is a weaker classical reference
than the Gauss-Newton fit used in the X10 harness (1.25 mm). The learned model
with a position-balanced loss reaches 1.40 mm (5 seeds, paired p=0.017 vs
PoE-LS). Whether the real-data gap is closed can only be read against this
strongest classical fit on the very same chain/splits/metric.

Models:
  nominal      X7.NominalFK, no fit
  gn_pos       least squares on position residuals (mm)
  gn_bal       least squares on [dp, 0.5*ang] (mirrors ang_weight=0.25)

Output: repro/json/X7_gn_ls.<ts>.json via repro/dump.
Run:  python X7_gn_ls.py                      # seeds 50-54 (default)
      python X7_gn_ls.py --seed-start 0 --quick  # smoke
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402  (ScrewArm, compose)
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def pack(m):
    return np.concatenate([p.detach().numpy().ravel() for p in
                           (m.base_r, m.base_t, m.tool_r, m.tool_t, m.xi_c)])


def unpack(m, x):
    i = 0
    for p in (m.base_r, m.base_t, m.tool_r, m.tool_t, m.xi_c):
        n = p.numel()
        with torch.no_grad():
            p.copy_(torch.tensor(x[i:i+n], dtype=DT).view_as(p))
        i += n


def residuals(x, m, q, oq, ot, bal):
    unpack(m, x)
    with torch.no_grad():
        dp, ang = X7.pose_error(m(q), oq, ot)
    if bal:
        return torch.cat([dp, 0.5 * ang]).numpy()
    return dp.numpy()


def fit(C, E, q_tr, oq_tr, ot_tr, bal):
    m = E34.ScrewArm(C, E, mode='const')
    x0 = pack(m)
    r = least_squares(residuals, x0, args=(m, q_tr, oq_tr, ot_tr, bal),
                      method='trf', max_nfev=300, x_scale='jac',
                      xtol=1e-12, ftol=1e-12)
    return m, r


@torch.no_grad()
def eval_pos(m, q, oq, ot):
    dp, ang = X7.pose_error(m(q), oq, ot)
    return float(dp.mean()) * 1000.0, float(np.degrees(ang.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-start', type=int, default=50)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds = 1
    t0 = time.time()
    q_np, obs_np = X7.load_philnad_dataset(
        os.path.join(_REPO, 'data', 'calib_data.pickle'))
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = (20, 40, 60, 80)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    res = {(s, k): [] for s in sizes for k in ('nominal', 'gn_pos', 'gn_bal')}
    for seed in seeds:
        perm = np.random.default_rng(1000 + seed).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            nom = X7.NominalFK(C, E)
            res[(n_tr, 'nominal')].append(eval_pos(nom, q_te, oq_te, ot_te))
            for bal, key in ((False, 'gn_pos'), (True, 'gn_bal')):
                m, r = fit(C, E, q_tr, oq_tr, ot_tr, bal)
                res[(n_tr, key)].append(eval_pos(m, q_te, oq_te, ot_te))
            line = ' | '.join(f'{k} {np.mean([x[0] for x in res[(n_tr, k)]]):.3f}mm'
                              for k in ('nominal', 'gn_pos', 'gn_bal'))
            print(f'[seed {seed} n={n_tr:3d}] {line}', flush=True)
        torch.save({'res': {f'{a}_{b}': v for (a, b), v in res.items()}},
                   'repro/json/X7_gn_ls.partial.pt')
    for s in sizes:
        for k in ('nominal', 'gn_pos', 'gn_bal'):
            a = np.array(res[(s, k)])
            RESULTS[f'X7gn:n{s}:{k}'] = dict(
                pos_mm_mean=float(a[:, 0].mean()), pos_mm_std=float(a[:, 0].std()),
                rot_deg_mean=float(a[:, 1].mean()),
                per_seed=[[float(v) for v in row] for row in a])
    RESULTS['_meta'] = dict(wall_s=time.time() - t0, model='ScrewArm const (54p)',
                            optimizer='scipy least_squares trf, 300 nfev',
                            seeds=list(seeds), sizes=list(sizes))
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, seed_start=args.seed_start)
    print('dump ->', p)
    for s in sizes:
        row = {k: round(RESULTS[f'X7gn:n{s}:{k}']['pos_mm_mean'], 3)
               for k in ('nominal', 'gn_pos', 'gn_bal')}
        print(f'n={s}: {row}')


if __name__ == '__main__':
    main()
