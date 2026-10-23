# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - X10b v3: classical trust-region LS baseline on the same splits (review Q6)
#
# The X10 baseline is a 12-parameter constant base/tool fit that cannot absorb
# per-joint errors. This script fits the SAME model class (nominal screws +
# per-joint SE(3) corrections + joint bias + tool, dense) with a deterministic
# classical solver, on the identical platforms/splits/seeds as X10
# (seeds 50-54, N in {20,40,80}).
#
# v3 changes (registered before the run, 2026-09-28):
#   * solver: scipy.optimize.least_squares(method='trf', x_scale='jac',
#     analytic Jacobian via torch.func.jacrev, xtol=ftol=gtol=1e-12,
#     max_nfev=--iters). v2 fixed-lambda Gauss-Newton (lambda=1e-6, no step
#     acceptance) blew up on UR10 N=20 seed 52 (60.4 mm); fixed-lambda GN is
#     not a fair classical baseline (audit 2026-09-28).
#   * residual (exact parity with the X10 loss 10*pos^2 + (1-cos)):
#       [sqrt(10)*dp ; sqrt(2)*vect(q_err)]   (full pose)
#       [sqrt(10)*dp]                         (position only)
#     because ||sqrt(2)*vect(q_err)||^2 = 2 sin^2(theta/2) = 1 - cos(theta)
#     exactly, and the vector part of the error quaternion is smooth at 0
#     (no kink of sqrt(1-cos), no 1/(2 sqrt(x)) Jacobian blow-up).
#   * dump records nfev/cost/optimality/status, rank/cond(J) at the solution,
#     and train position/rotation diagnostics per cell.
#
# Output: repro/json/X10b_poe_ls_baseline.<ts>.json  (smoke: ...-smoke.json)
# Run: python X10b_poe_ls_baseline.py [--quick] [--datasets philnad,ipa,pauljac]
#       [--sizes 20,40,80] [--seeds 5] [--seed_start 50] [--iters 300] [--steps 2000]
# ============================================================
import argparse
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, jacrev
from scipy.optimize import least_squares

from X10_fk_calib_real import (ScrewCalib, apply_sc, fit_gaps_torch, fk_np,
                               gap_init, get_robot_tensors, load_ipa,
                               load_pauljac, load_philnad, log_se3_np,
                               pose_metrics)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

POS_WEIGHT = 10.0
SQRT2 = math.sqrt(2.0)


def quat_vec(R):
    """Vector part (x, y, z) of the unit quaternion of R (..., 3, 3), w >= 0.

    Smooth for rotation angles < pi.  ||sqrt(2) * v||^2 = 1 - cos(theta)
    exactly, so the LS residual matches the X10 rotation term (1 - cos) with
    no square-root kink.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    w = 0.5 * torch.sqrt((1.0 + tr).clamp_min(1e-12))
    vx = (R[..., 2, 1] - R[..., 1, 2]) / (4.0 * w)
    vy = (R[..., 0, 2] - R[..., 2, 0]) / (4.0 * w)
    vz = (R[..., 1, 0] - R[..., 0, 1]) / (4.0 * w)
    return torch.stack([vx, vy, vz], dim=-1)


def fit_poe_trf(xi, M0, q, T, tool_init, full=True, max_nfev=300):
    """Classical trust-region LS on the dense screw model (deterministic).

    Returns (model_with_solution, solver_info).
    """
    model = ScrewCalib(xi, M0, 'dense')
    with torch.no_grad():
        model.tool.copy_(torch.tensor(tool_init, dtype=model.tool.dtype))
    qt = torch.tensor(q, dtype=torch.float64)
    Tt = torch.tensor(T, dtype=torch.float64)
    names = [k for k, _ in model.named_parameters()]
    shapes = {k: tuple(v.shape) for k, v in model.named_parameters()}
    x0 = np.concatenate([getattr(model, k).detach().numpy().reshape(-1)
                         for k in names])

    def unpack(x):
        p = {}
        off = 0
        for k in names:
            sz = int(np.prod(shapes[k]))
            p[k] = torch.tensor(x[off:off + sz], dtype=torch.float64).reshape(shapes[k])
            off += sz
        return p

    def resid(p):
        Tp = functional_call(model, p, (qt,))
        dp = (Tp[:, :3, 3] - Tt[:, :3, 3]).reshape(-1)
        parts = [math.sqrt(POS_WEIGHT) * dp]
        if full:
            R = Tp[:, :3, :3].transpose(-1, -2) @ Tt[:, :3, :3]
            parts.append(SQRT2 * quat_vec(R).reshape(-1))
        return torch.cat(parts)

    def fun(x):
        return resid(unpack(x)).detach().numpy()

    def jac(x):
        J = jacrev(resid)(unpack(x))
        m = J[names[0]].shape[0]
        Jm = torch.cat([J[k].reshape(m, -1) for k in names], dim=1)
        return Jm.detach().numpy()

    res = least_squares(fun, x0, jac=jac, method='trf', x_scale='jac',
                        xtol=1e-12, ftol=1e-12, gtol=1e-12,
                        max_nfev=max_nfev)

    sv = np.linalg.svd(jac(res.x), compute_uv=False)
    p = unpack(res.x)
    fitted = ScrewCalib(xi, M0, 'dense')
    with torch.no_grad():
        for k in names:
            fitted.get_parameter(k).copy_(p[k])
        Tp_tr = fitted(qt).numpy()
    dp_tr = np.linalg.norm(Tp_tr[:, :3, 3] - T[:, :3, 3], axis=1).mean() * 1000.0
    info = dict(cost=float(res.cost), nfev=int(res.nfev),
                optimality=float(res.optimality), status=int(res.status),
                message=str(res.message), rank=int((sv > 1e-8 * sv[0]).sum()),
                cond=float(sv[0] / max(float(sv[-1]), 1e-300)),
                train_pos_mm=float(dp_tr))
    if full:
        R = Tp_tr[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)
        ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2,
                                           -1, 1)))
        info['train_rot_deg'] = float(ang.mean())
    return fitted, info


def run_dataset(data, sizes, seeds, seed_start, results, iters, steps=2000):
    robot, xi, M0 = get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = fk_np(robot, q_all)
    full = data['full']
    per_seed = {}
    for s in range(seed_start, seed_start + seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        per_seed['seed%d' % s] = {}
        for N in sizes:
            if N >= n:
                continue
            tr, te = perm[:N], perm[N:]
            goal_tr = data['goal'][tr] if data['goal'] is not None else None
            B0, C0 = gap_init(Tf_all[tr], T_all[tr], goal_tr, full=full)
            B, C = fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0, steps=steps,
                                  full=full, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            model, info = fit_poe_trf(xi, M0, q_all[tr], T_obs2_tr,
                                      log_se3_np(C), full=full, max_nfev=iters)
            rec = {'nominal_gap': pose_metrics(B @ Tf_all[te] @ C, T_all[te], full=full)}
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            rec['poe_ls'] = pose_metrics(apply_sc(model, q_all[te]), T_obs2_te, full=full)
            rec['solver'] = info
            for k, v in rec.items():
                v['n_test'] = int(len(te))
            per_seed['seed%d' % s]['N%d' % N] = rec
            print('   seed %d N=%d: poe_ls %.3f mm  (nfev=%d, cost=%.3e, opt=%.1e, rank=%d)'
                  % (s, N, rec['poe_ls']['pos_mean'], info['nfev'], info['cost'],
                     info['optimality'], info['rank']), flush=True)
    results[data['name']] = dict(robot=data['robot'], n_poses=int(n),
                                 full_pose=bool(full), per_seed=per_seed,
                                 params={'dense': ScrewCalib(xi, M0, 'dense').n_params(),
                                         'method': 'scipy trf LS (deterministic, x_scale=jac)'})


def run_pauljac(data, sizes, seeds, seed_start, results, iters, steps=2000):
    robot, xi, M0 = get_robot_tensors(data['robot'])
    qg, Tg = data['q_grid'], data['T_grid']
    qr, Tr = data['q_rand'], data['T_rand']
    Tf_g = fk_np(robot, qg)
    per_seed = {}
    for s in range(seed_start, seed_start + seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(qg))
        per_seed['seed%d' % s] = {}
        for N in sizes:
            if N >= len(qg):
                continue
            tr, te = perm[:N], perm[N:]
            B0, C0 = gap_init(Tf_g[tr], Tg[tr], None, full=False)
            B, C = fit_gaps_torch(Tf_g[tr], Tg[tr], B0, C0, steps=steps,
                                  full=False, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in Tg[tr]])
            model, info = fit_poe_trf(xi, M0, qg[tr], T_obs2_tr,
                                      log_se3_np(C), full=False, max_nfev=iters)
            rec = {'nominal_gap_random': pose_metrics(B @ fk_np(robot, qr) @ C, Tr, full=False),
                   'nominal_gap_gridhold': pose_metrics(B @ Tf_g[te] @ C, Tg[te], full=False)}
            T_obs2_r = np.array([np.linalg.inv(B) @ t for t in Tr])
            T_obs2_g = np.array([np.linalg.inv(B) @ t for t in Tg[te]])
            rec['poe_ls_random'] = pose_metrics(apply_sc(model, qr), T_obs2_r, full=False)
            rec['poe_ls_gridhold'] = pose_metrics(apply_sc(model, qg[te]), T_obs2_g, full=False)
            rec['solver'] = info
            per_seed['seed%d' % s]['N%d' % N] = rec
            print('   seed %d N=%d: poe_ls_random %.3f mm  (nfev=%d, cost=%.3e, opt=%.1e, rank=%d)'
                  % (s, N, rec['poe_ls_random']['pos_mean'], info['nfev'], info['cost'],
                     info['optimality'], info['rank']), flush=True)
    results[data['name']] = dict(robot='ur5', n_poses=int(len(qg)), full_pose=False,
                                 per_seed=per_seed,
                                 params={'dense': ScrewCalib(xi, M0, 'dense').n_params(),
                                         'method': 'scipy trf LS (deterministic, x_scale=jac)'})


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--datasets', default='philnad,ipa,pauljac')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=50)
    ap.add_argument('--iters', type=int, default=1000,
                    help='scipy least_squares max_nfev (v3; in v2 this was GN iterations)')
    ap.add_argument('--steps', type=int, default=2000,
                    help='gap-fit LBFGS schedule (shared with X10)')
    ap.add_argument('--out', default=None, help='explicit dump path (sparingly; smoke/tests)')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.sizes, args.iters, args.datasets = 1, '20', 100, 'philnad'
    sizes = [int(s) for s in args.sizes.split(',')]
    print('X10b v3 classical scipy-trf LS baseline | max_nfev=%d seeds=%d-%d'
          % (args.iters, args.seed_start, args.seed_start + args.seeds - 1), flush=True)
    t0 = time.time()
    results = {}
    if 'philnad' in args.datasets:
        run_dataset(load_philnad(args.data_dir), sizes, args.seeds,
                    args.seed_start, results, args.iters, steps=args.steps)
    if 'ipa' in args.datasets:
        run_dataset(load_ipa(args.data_dir, 'A'), sizes, args.seeds,
                    args.seed_start, results, args.iters, steps=args.steps)
    if 'pauljac' in args.datasets:
        run_pauljac(load_pauljac(args.data_dir, 'UR5'), sizes, args.seeds,
                    args.seed_start, results, args.iters, steps=args.steps)
    results['wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=args.seeds,
              seed_start=args.seed_start, out=args.out, extra_meta=dict(
                  method='scipy.optimize.least_squares(trf, x_scale=jac)',
                  residual='[sqrt(10)*dp ; sqrt(2)*vect(q_err)]',
                  max_nfev=args.iters, nodes='X10b-v3 (registered 2026-09-28)'))
    print('\n===== SUMMARY-JSON =====')
    print('Wall: %.1f s' % results['wall_s'])


if __name__ == '__main__':
    main()
