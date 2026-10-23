#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
G2a_ur10_isolated.py — isolated psi on UR10 (frozen exact edge).

S8 (joint training) degenerated on UR10: the edge drifted away from the exact
fit and psi compensated it (test 1.024 vs exact 0.434 mm; psi off -> 11.7 mm).
Here the correction is isolated: first fit the exact dense model (X10 protocol),
freeze it, then train only the zero-initialized psi (second phase, same loss and
steps per phase).

Protocol: IPA Series A, splits rng(1000+seed), seed_start=50 (seeds 50-54),
N=20/40/80, full 6D loss, Adam 1e-2, cosine, 2000 steps per phase.
References: canonical exact (screw_dense) and GN (poe_ls) from the X10/X10b
dumps; the joint-trained S8 numbers are reported for contrast.

Registered predictions (plan 2026-09-29, before the run):
  P-G2a1: isolated test <= own-exact + 0.05 mm at all N;
  P-G2a2: at N=20 no significant harm (paired vs own exact, p>0.05);
  P-G2a3: mean per-joint |corr| <= 1e-3 (vs 3.8e-3 joint at N=80);
  P-G2a4 (boundary): if isolated is still clearly worse than exact, isolation
          does not fix it and the gate/regularization route is required.

Output: repro/json/G2a_ur10_isolated.<ts>.json via repro/dump.
Run:  python G2a_ur10_isolated.py [--quick]
"""

import argparse
import glob
import json
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

import X10_fk_calib_real as X10          # noqa: E402
import X7_robot_calibration as X7        # noqa: E402
import S8_ur10_relaxed as S8             # noqa: E402
from repro.dump import save_dump
from repro.canon import load as load_canon         # noqa: E402

DT = torch.float64
RESULTS = {}

CANON = load_canon('x10_p48')
GN = load_canon('x10b')


def fit_isolated(xi, M0, q_tr, T_obs2_tr, tool_init, steps, seed):
    """Phase 1: exact edge (psi frozen); phase 2: psi only (rest frozen)."""
    ex = X10.train_screwcalib(xi, M0, q_tr, T_obs2_tr, tool_init,
                              mode='dense', steps=steps, seed=seed, full=True)
    m = S8.ScrewCalibRelax(xi, M0, mode='dense')
    m.load_state_dict(ex.state_dict(), strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(100 + seed)
    opt = torch.optim.Adam(m.psi.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    qt = torch.tensor(q_tr, dtype=DT)
    Tt = torch.tensor(T_obs2_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - Tt[:, :3, 3]) ** 2).sum(-1).mean()
        tr = (Tp[:, :3, :3].transpose(-1, -2) @ Tt[:, :3, :3]).diagonal(
            dim1=-2, dim2=-1).sum(-1)
        loss = loss + (1.0 - ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
    return ex, m


@torch.no_grad()
def corr_stats(m, q_tr):
    qt = torch.tensor(q_tr, dtype=DT)
    g = qt + m.b.double().unsqueeze(0)
    v = g.unsqueeze(-1) * m.twists().double().unsqueeze(0)
    T = m.M0.double().unsqueeze(0).expand(qt.shape[0], 4, 4).clone()
    norms = []
    for j in range(qt.shape[1]):
        Tj = X10.se3_exp(v[:, j, :])
        T = T @ Tj
        f = torch.cat([m._canon(X7.mat_to_quat(T[:, :3, :3])), T[:, :3, 3],
                       m._canon(X7.mat_to_quat(Tj[:, :3, :3])), Tj[:, :3, 3]], -1)
        norms.append((m.psi_gain * m.psi(f)).norm(dim=-1).mean().item())
    return float(np.mean(norms))


def load_refs():
    canon = CANON['ipa_A']
    gn = GN['ipa_A']
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json',
                                               'S8_ur10_relaxed.*.json'))
             if 'smoke' not in p]
    s8 = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
    return canon, gn, s8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--series', default='A')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=50)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 200
    t0 = time.time()

    data = X10.load_ipa(args.data_dir, args.series)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    sizes = [int(s) for s in args.sizes.split(',')]
    canon, gn, s8 = load_refs()
    print('[G2a] isolated psi on UR10 (frozen exact edge)', flush=True)

    rec = {(s, N): {} for s in range(args.seed_start, args.seed_start + args.seeds)
           for N in sizes}
    for s in range(args.seed_start, args.seed_start + args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], data['goal'][tr], full=True)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=True, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            ex, m = fit_isolated(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C),
                                 args.steps, s)
            ex_tr = X10.pose_metrics(X10.apply_sc(ex, q_all[tr]), T_obs2_tr,
                                     full=True)['pos_mean']
            ex_te = X10.pose_metrics(X10.apply_sc(ex, q_all[te]), T_obs2_te,
                                     full=True)['pos_mean']
            iso_tr = X10.pose_metrics(X10.apply_sc(m, q_all[tr]), T_obs2_tr,
                                      full=True)['pos_mean']
            iso = X10.pose_metrics(X10.apply_sc(m, q_all[te]), T_obs2_te,
                                   full=True)
            corr = corr_stats(m, q_all[tr])
            cex = canon['per_seed']['seed%d' % s]['N%d' % N]['screw_dense']['pos_mean']
            cgn = gn['per_seed']['seed%d' % s]['N%d' % N]['poe_ls']['pos_mean']
            joint = s8.get(f'S8:seed{s}:N{N}:relaxed', float('nan'))
            rec[(s, N)] = dict(exact_own=ex_te, isolated=iso['pos_mean'],
                               exact_train=ex_tr, isolated_train=iso_tr,
                               rot=iso['rot_mean'], corr=corr,
                               canon=cex, gn=cgn, joint=joint)
            print(f'[G2a seed {s} N={N}] exact {ex_te:.3f} isolated '
                  f'{iso["pos_mean"]:.3f} (train {iso_tr:.3f}) joint {joint:.3f} '
                  f'canon {cex:.3f} gn {cgn:.3f} |corr| {corr:.2e}', flush=True)
        print(f'  seed {s} done', flush=True)

    for N in sizes:
        for key in ('exact_own', 'isolated', 'joint', 'canon', 'gn', 'corr'):
            vals = [rec[(s, N)][key] for s in range(args.seed_start,
                                                    args.seed_start + args.seeds)]
            RESULTS[f'G2a:N{N}:{key}'] = dict(
                mean=float(np.nanmean(vals)), std=float(np.nanstd(vals)),
                per_seed=[float(v) for v in vals])
        a = np.array([rec[(s, N)]['isolated'] for s in range(args.seed_start,
                                                             args.seed_start + args.seeds)])
        b = np.array([rec[(s, N)]['exact_own'] for s in range(args.seed_start,
                                                              args.seed_start + args.seeds)])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'G2a:paired:N{N}:isolated_vs_exact_own'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    RESULTS['G2a:refs'] = dict(canon='x10_p48', gn='x10b')
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, seed_start=args.seed_start,
                    epochs=args.steps)
    print('\n===== SUMMARY-JSON =====')
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False, default=float))
    print('dump:', out)


if __name__ == '__main__':
    main()
