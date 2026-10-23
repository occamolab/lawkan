#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X44_fk_matched.py — the applied FK-calibration table (tab:x10) under a MATCHED
optimizer+loss, plus the same-class classical estimator (LM).

Motivation. In tab:x10 the `const base/tool` baseline is solved with LBFGS
(X10.fit_gaps_torch) while our screw model uses Adam; the two are not on the same
solver. This probe refits the baseline with the SAME Adam schedule + SAME loss as
our model, and adds the same-class classical estimator (per-joint screw via scipy
least_squares / LM on the position residual).

Models (per platform, same splits/seeds as X10, rng(1000+seed)):
  ours        X10.ScrewCalib dense, Adam (existing path)
  const_adam  base B + tool C (12p), Adam + same loss (was LBFGS)
  gn          X10.ScrewCalib dense, scipy least_squares (LM), position residual

Output: repro/json/X44_fk_matched.<ts>.json via repro/dump.
Run:  python X44_fk_matched.py [--quick]
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import X10_fk_calib_real as X10               # noqa: E402
from repro.dump import save_dump             # noqa: E402

DT = torch.float64
RESULTS = {}
CONST_STEPS = 10000


def train_gaps_adam(Tf_tr, T_obs_tr, B0, C0, steps, seed, full):
    """The const base/tool (12p) baseline under OUR Adam schedule + OUR loss."""
    torch.manual_seed(100 + seed)
    pB = torch.tensor(X10.log_se3_np(B0), dtype=DT, requires_grad=True)
    pC = torch.tensor(X10.log_se3_np(C0), dtype=DT, requires_grad=True)
    Tft = torch.tensor(Tf_tr, dtype=DT)
    Tmt = torch.tensor(T_obs_tr, dtype=DT)
    opt = torch.optim.Adam([pB, pC], lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    for _ in range(steps):
        opt.zero_grad()
        B = X10.se3_exp(pB.unsqueeze(0))[0]
        C = X10.se3_exp(pC.unsqueeze(0))[0]
        Tp = B @ Tft @ C
        loss = 10.0 * ((Tp[:, :3, 3] - Tmt[:, :3, 3]) ** 2).sum(-1).mean()
        if full:
            tr = (Tp[:, :3, :3].transpose(-1, -2) @ Tmt[:, :3, :3]).diagonal(
                dim1=-2, dim2=-1).sum(-1)
            loss = loss + (1.0 - ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        B = X10.se3_exp(pB.unsqueeze(0))[0].numpy()
        C = X10.se3_exp(pC.unsqueeze(0))[0].numpy()
    return B, C


def fit_gn(xi, M0, q_tr, T_tr, tool_init, full=True, max_nfev=400):
    """Same-class classical: per-joint screw via scipy least_squares (LM)."""
    m = X10.ScrewCalib(xi, M0, 'dense')
    with torch.no_grad():
        m.tool.copy_(torch.tensor(tool_init, dtype=m.tool.dtype))
    params = [m.dxi, m.b, m.tool]

    def pack():
        return np.concatenate([p.detach().numpy().ravel() for p in params])

    def unpack(x):
        i = 0
        for p in params:
            n = p.numel()
            with torch.no_grad():
                p.copy_(torch.tensor(x[i:i + n], dtype=p.dtype).view_as(p))
            i += n

    qt = torch.tensor(q_tr, dtype=DT)
    Tt = np.asarray(T_tr, dtype=np.float64)

    def fun(x):
        unpack(x)
        with torch.no_grad():
            Tp = m(qt).numpy().astype(np.float64)
        dp = np.linalg.norm(Tp[:, :3, 3] - Tt[:, :3, 3], axis=1)
        if not full:
            return np.sqrt(10.0) * dp
        tr = np.trace(Tp[:, :3, :3].transpose(0, 2, 1) @ Tt[:, :3, :3], axis1=1, axis2=2)
        ang = np.arccos(np.clip((tr - 1) / 2, -1, 1))
        return np.concatenate([np.sqrt(10.0) * dp, ang])

    sol = least_squares(fun, pack(), method='trf', max_nfev=max_nfev, x_scale='jac')
    unpack(sol.x)
    return m


def run_full_pose(name, data, sizes, seeds, steps, out):
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    full = data['full']
    res = {(N, k): [] for N in sizes for k in ('ours', 'const_adam', 'gn')}
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            if N >= n:
                continue
            tr, te = perm[:N], perm[N:]
            goal_tr = data['goal'][tr] if data['goal'] is not None else None
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], goal_tr, full=full)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0, steps=steps,
                                      full=full, seed=s)
            Ba, Ca = train_gaps_adam(Tf_all[tr], T_all[tr], B0, C0, CONST_STEPS, s, full)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])

            m_ours = X10.train_screwcalib(xi, M0, q_all[tr], T_obs2_tr,
                                          X10.log_se3_np(C), mode='dense',
                                          steps=steps, seed=s, full=full)
            m_gn = fit_gn(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C), full=full)
            res[(N, 'ours')].append(
                X10.pose_metrics(X10.apply_sc(m_ours, q_all[te]), T_obs2_te, full=full)['pos_mean'])
            res[(N, 'gn')].append(
                X10.pose_metrics(X10.apply_sc(m_gn, q_all[te]), T_obs2_te, full=full)['pos_mean'])
            res[(N, 'const_adam')].append(
                X10.pose_metrics(Ba @ Tf_all[te] @ Ca, T_all[te], full=full)['pos_mean'])
            print(f'[{name} s{s} N={N}] ours {res[(N,"ours")][-1]:.3f} '
                  f'const_adam {res[(N,"const_adam")][-1]:.3f} gn {res[(N,"gn")][-1]:.3f}',
                  flush=True)
    for N in sizes:
        for k in ('ours', 'const_adam', 'gn'):
            if res[(N, k)]:
                a = np.array(res[(N, k)])
                RESULTS[f'X44:{name}:N{N}:{k}'] = dict(
                    pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                    per_seed=[float(v) for v in a])
        for k in ('const_adam', 'gn'):
            a = np.array(res[(N, 'ours')]); b = np.array(res[(N, k)])
            if len(a) > 1:
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'X44:{name}:paired:N{N}:ours_vs_{k}'] = dict(
                    diff=float((a - b).mean()), t=float(t), p=float(p))


def run_pauljac(name, data, sizes, seeds, steps, out):
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    qg, Tg = data['q_grid'], data['T_grid']
    res = {(N, k): [] for N in sizes for k in ('ours', 'const_adam', 'gn')}
    Tf_g = X10.fk_np(robot, qg)
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(qg))
        for N in sizes:
            if N >= len(qg):
                continue
            tr, te = perm[:N], perm[N:]
            B0, C0 = X10.gap_init(Tf_g[tr], Tg[tr], None, full=False)
            B, C = X10.fit_gaps_torch(Tf_g[tr], Tg[tr], B0, C0, steps=steps,
                                      full=False, seed=s)
            Ba, Ca = train_gaps_adam(Tf_g[tr], Tg[tr], B0, C0, CONST_STEPS, s, False)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in Tg[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in Tg[te]])
            m_ours = X10.train_screwcalib(xi, M0, qg[tr], T_obs2_tr,
                                          X10.log_se3_np(C), mode='dense',
                                          steps=steps, seed=s, full=False)
            m_gn = fit_gn(xi, M0, qg[tr], T_obs2_tr, X10.log_se3_np(C), full=False)
            res[(N, 'ours')].append(
                X10.pose_metrics(X10.apply_sc(m_ours, qg[te]), T_obs2_te, full=False)['pos_mean'])
            res[(N, 'gn')].append(
                X10.pose_metrics(X10.apply_sc(m_gn, qg[te]), T_obs2_te, full=False)['pos_mean'])
            res[(N, 'const_adam')].append(
                X10.pose_metrics(Ba @ Tf_g[te] @ Ca, Tg[te], full=False)['pos_mean'])
            print(f'[{name} s{s} N={N}] ours {res[(N,"ours")][-1]:.3f} '
                  f'const_adam {res[(N,"const_adam")][-1]:.3f} gn {res[(N,"gn")][-1]:.3f}',
                  flush=True)
    for N in sizes:
        for k in ('ours', 'const_adam', 'gn'):
            if res[(N, k)]:
                a = np.array(res[(N, k)])
                RESULTS[f'X44:{name}:N{N}:{k}'] = dict(
                    pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                    per_seed=[float(v) for v in a])
        for k in ('const_adam', 'gn'):
            a = np.array(res[(N, 'ours')]); b = np.array(res[(N, k)])
            if len(a) > 1:
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'X44:{name}:paired:N{N}:ours_vs_{k}'] = dict(
                    diff=float((a - b).mean()), t=float(t), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--datasets', default='philnad,ipa,pauljac')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--const-steps', type=int, default=10000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.sizes, args.steps, args.const_steps = 1, '20', 200, 2000
        args.datasets = 'philnad'
    global CONST_STEPS
    CONST_STEPS = args.const_steps
    sizes = [int(s) for s in args.sizes.split(',')]
    t0 = time.time()
    todo = args.datasets.split(',')
    if 'philnad' in todo:
        print('== philnad FR3 ==', flush=True)
        run_full_pose('philnad', X10.load_philnad(args.data_dir), sizes,
                      args.seeds, args.steps, None)
    if 'ipa' in todo:
        print('== ipa UR10 ==', flush=True)
        run_full_pose('ipa_A', X10.load_ipa(args.data_dir, 'A'), sizes,
                      args.seeds, args.steps, None)
    if 'pauljac' in todo:
        print('== pauljac UR5 ==', flush=True)
        run_pauljac('pauljac_UR5', X10.load_pauljac(args.data_dir, 'UR5'), sizes,
                    args.seeds, args.steps, None)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds,
                  epochs=args.steps,
                  extra_meta={'const': 'base/tool 12p under our Adam (was LBFGS)',
                              'gn': 'same-class screw via scipy LM (position)',
                              'note': 'P-X44: matched optimizer on tab:x10'})
    print('dump ->', p)


if __name__ == '__main__':
    main()
