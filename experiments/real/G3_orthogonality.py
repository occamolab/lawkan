#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
G3_orthogonality.py — is the learned correction orthogonal to the edge?

Diagnosis (30.09): joint training of edge + psi creates a compensation on both
UR5 and UR10 (psi off: 2.1 / 11.7 mm; |psi| 2e-2 / 3.8e-3), but with different
generalization. Hypothesis: the useful part of psi is the one orthogonal (over
configurations) to the family of corrections reproducible by retuning the edge;
the compensations are edge-absorbable.

Measured here for trained models (joint and isolated) on FR3 (Fourier psi),
UR5 (MLP), UR10 (MLP), seed 0 / UR10 seed 50, N=80:
  * psi field  Psi_i = log(T_off_i^{-1} T_on_i) over the training configurations
    (psi active vs psi zeroed);
  * output Jacobian A w.r.t. all non-psi ("edge") parameters, central finite
    differences with psi frozen at its trained function;
  * absorbable share = 1 - ||Psi - A lstsq||^2 / ||Psi||^2.

Registered predictions (plan 2026-09-29, before the run):
  P-G3a1: joint UR5/UR10 — absorbable share >= 0.9;
  P-G3a2: isolated UR5/UR10 — |Psi| much smaller, absorbable share <= 0.5;
  P-G3a3: isolated FR3 — orthogonal share substantial (>= 0.5);
  P-G3a4 (boundary): if FR3 orthogonal share is small, the orthogonality story
          needs revision.

Output: repro/json/G3_orthogonality.<ts>.json via repro/dump.
Run:  python G3_orthogonality.py [--platforms fr3,ur5,ur10] [--quick]
"""

import argparse
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

import X10_fk_calib_real as X10          # noqa: E402
import X7_robot_calibration as X7        # noqa: E402
import E34_x7_recipe as E34              # noqa: E402
import X11_ur5_gtpn_sweep as X11         # noqa: E402
import X14_fourier_law as X14            # noqa: E402
import X7_gn_ls as GN                    # noqa: E402
import S2_ur5gtpn_relaxed as S2          # noqa: E402
import S8_ur10_relaxed as S8             # noqa: E402
import G2a_ur10_isolated as G2a          # noqa: E402
import G2bc_isolated_psi as G2bc         # noqa: E402
from repro.dump import save_dump         # noqa: E402

DT = torch.float64
RESULTS = {}


def rightlog_diffs(Ta, Tb):
    out = np.empty((len(Ta), 6))
    for i, (a, b) in enumerate(zip(Ta, Tb)):
        out[i] = X10.log_se3_np(np.linalg.inv(a) @ b)
    return out


def quat_to_mat_batch(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def poses_screw(m, q):
    return X10.apply_sc(m, q)


def poses_fr3(m, q):
    with torch.no_grad():
        qp, tp = m(q)
    qp, tp = qp.numpy(), tp.numpy()
    T = np.tile(np.eye(4), (len(qp), 1, 1))
    T[:, :3, :3] = quat_to_mat_batch(qp)
    T[:, :3, 3] = tp
    return T


def psi_field(forward, m, q):
    T_on = forward(m, q)
    sd = {k: v.detach().clone() for k, v in m.psi.state_dict().items()}
    with torch.no_grad():
        for v in m.psi.state_dict().values():
            v.zero_()
    T_off = forward(m, q)
    m.psi.load_state_dict(sd)
    return T_on, T_off, rightlog_diffs(T_off, T_on)


def edge_jacobian(forward, m, q, eps=1e-6):
    T0 = forward(m, q)
    cols = []
    for name, p in m.named_parameters():
        if name.startswith('psi'):
            continue
        flat = p.view(-1)
        for idx in range(flat.numel()):
            with torch.no_grad():
                flat[idx] += eps
            Tp = forward(m, q)
            with torch.no_grad():
                flat[idx] -= eps
            cols.append((rightlog_diffs(T0, Tp) / eps).reshape(-1))
    return np.stack(cols, axis=1)


def eval_psi_off(m, eval_fn, q):
    sd = {k: v.clone() for k, v in m.psi.state_dict().items()}
    with torch.no_grad():
        for p in m.psi.parameters():
            p.zero_()
    val = eval_fn(m, q)
    m.psi.load_state_dict(sd)
    return float(val)


def analyze(tag, forward, m, q, eval_fn=None, q_te=None):
    _, _, Psi = psi_field(forward, m, q)
    A = edge_jacobian(forward, m, q)
    Psi_v = Psi.reshape(-1)
    sol, _, rank, _ = np.linalg.lstsq(A, Psi_v, rcond=None)
    resid = Psi_v - A @ sol
    tot = float(Psi_v @ Psi_v)
    res = float(resid @ resid)
    abs_share = 1.0 - res / tot if tot > 0 else float('nan')
    per_cfg = np.linalg.norm(Psi, axis=1)
    rec = dict(
        psi_norm_mean=float(per_cfg.mean()),
        psi_norm_max=float(per_cfg.max()),
        edge_params=int(A.shape[1]), edge_rank=int(rank),
        absorbable_share=float(abs_share),
        orthogonal_share=float(1.0 - abs_share),
        n_configs=int(len(q)))
    if eval_fn is not None and q_te is not None:
        rec['pos_test_on'] = float(eval_fn(m, q_te))
        rec['pos_test_off'] = eval_psi_off(m, eval_fn, q_te)
    RESULTS[f'G3:{tag}'] = rec
    extra = ''
    if 'pos_test_on' in rec:
        extra = (f' test on/off {rec["pos_test_on"]:.3f}/{rec["pos_test_off"]:.3f}'
                 ' mm')
    print(f'[G3 {tag}] |Psi| {per_cfg.mean():.3e} absorbable {abs_share:.3f} '
          f'orthogonal {1.0 - abs_share:.3f} (edge {A.shape[1]}p, rank {rank})'
          f'{extra}', flush=True)


def fit_isolated_ur5(exact, xi, M0, q_tr, T_obs2_tr, steps, lr=1e-2):
    m = S2.ScrewCalibRelax(xi, M0, mode='dense')
    m.load_state_dict(exact.state_dict(), strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(0)
    opt = torch.optim.Adam(m.psi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    qt = torch.tensor(q_tr, dtype=DT)
    pt = torch.tensor(T_obs2_tr[:, :3, 3], dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - pt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m


def run_fr3(args):
    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1000).permutation(len(q_np))
    tr, te = perm[:80], perm[80:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
    exact, isolated = G2bc.two_phase('fourier', C, E, q_tr, oq_tr, ot_tr,
                                     args.epochs, args.lr, 0)
    joint = X14.FourierLawArm(C, E, edge_dense=True)
    joint, _ = E34.train_recipe(joint, q_tr, oq_tr, ot_tr, args.epochs,
                                args.lr, 0, clip=0.0, ang_weight=1.0,
                                warm_epochs=0, loss_form='cosine2')
    ev = lambda m, q: GN.eval_pos(m, q, oq_te, ot_te)[0]
    analyze('fr3:joint', poses_fr3, joint, q_tr, ev, q_te)
    analyze('fr3:isolated', poses_fr3, isolated, q_tr, ev, q_te)


def run_ur10(args):
    data = X10.load_ipa(os.path.join(_REPO, 'data'), 'A')
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    Tf_all = X10.fk_np(robot, q_all)
    perm = np.random.default_rng(1000 + 50).permutation(len(q_all))
    tr, te = perm[:80], perm[80:]
    B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], data['goal'][tr], full=True)
    B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                              steps=args.steps, full=True, seed=50)
    T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
    T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
    exact, isolated = G2a.fit_isolated(xi, M0, q_all[tr], T_obs2_tr,
                                       X10.log_se3_np(C), args.steps, 50)
    joint = S8.fit_relaxed(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C),
                           steps=args.steps, seed=50)
    ev = lambda m, q: X10.pose_metrics(X10.apply_sc(m, q), T_obs2_te,
                                       full=True)['pos_mean']
    analyze('ur10:joint', poses_screw, joint, q_all[tr], ev, q_all[te])
    analyze('ur10:isolated', poses_screw, isolated, q_all[tr], ev, q_all[te])


def run_ur5(args):
    data = X11.load_gtpn(X11.DATA)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    Tf_all = X10.fk_np(robot, q_all)
    perm = np.random.default_rng(1000).permutation(len(q_all))
    tr, te = perm[:80], perm[80:]
    B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], None, full=False)
    B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                              steps=args.steps, full=False, seed=0)
    T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
    T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
    exact = X10.train_screwcalib(xi, M0, q_all[tr], T_obs2_tr,
                                 X10.log_se3_np(C), mode='dense',
                                 steps=args.steps, seed=0, full=False)
    joint = S2.fit_relaxed(xi, M0, q_all[tr], T_obs2_tr, X10.log_se3_np(C),
                           steps=args.steps, seed=0)
    isolated = fit_isolated_ur5(exact, xi, M0, q_all[tr], T_obs2_tr, args.steps)
    ev = lambda m, q: X10.pose_metrics(X10.apply_sc(m, q), T_obs2_te,
                                       full=False)['pos_mean']
    analyze('ur5:joint', poses_screw, joint, q_all[tr], ev, q_all[te])
    analyze('ur5:isolated', poses_screw, isolated, q_all[tr], ev, q_all[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default='fr3,ur5,ur10')
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.steps, args.epochs = 200, 300
    t0 = time.time()
    plats = args.platforms.split(',')
    if 'fr3' in plats:
        run_fr3(args)
    if 'ur10' in plats:
        run_ur10(args)
    if 'ur5' in plats:
        run_ur5(args)
    RESULTS['G3:_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=1, epochs=(args.epochs, args.steps))
    print('dump:', out)


if __name__ == '__main__':
    main()
