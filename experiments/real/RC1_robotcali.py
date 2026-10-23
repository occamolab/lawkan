#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
RC1_robotcali.py — RobotCali (ABB IRB 120) draw-wire range calibration.

Data revision (before runs): the x,y,z columns of RobotCali-new.xlsx are the
nominal FK output (our URDF FK matches them to 0.44 mm; fitting does not
improve), while the real measurement is the cable length L (draw-wire):
    L_hat(q) = || p(q) - a ||,
with p from the exact screw node T = M0 * prod_j exp(xi_j (q_j + b_j)) * exp(tool)
and an anchor a. This is the protocol of the original RobotCali papers.

Models:
  baseline : nominal FK + constant tool (6) + anchor (3) = 9p;
  exact    : dense screw (36+6+6) + anchor = 51p (trainable);
  gn       : scipy least-squares of the same 51p class;
  joint    : relaxed MLP psi (14->16->6) + anchor, all trainable;
  isolated : psi only on top of the trained exact edge (G2 protocol).

Registered predictions (plan 2026-09-29, amended 04.10 before the run):
  P-RC1: exact beats baseline by >=1.5x on held-out mean |dL| at N>=80;
  P-RC2: GN ~ exact (parity);
  P-RC3: isolated psi gives no significant gain (law not deformed);
  P-RC4 (boundary): isolated psi wins at N>=160 -> ABB law deformation.

Protocol: splits rng(1000+seed), seeds 0-4, N=40/80/160/320, Adam 3000 steps,
lr 1e-2, metric mean |dL| in mm on held-out poses.

Note: no base gap fit (unlike X10 position calibration). For range measurements
any fixed base transform is absorbable --- rotations preserve the norm and the
translation goes into the free anchor --- so the anchor and the tool carry the
rigid constants. Tool init and Adam follow X10; psi models are seeded before
construction (S2 convention).

Output: repro/json/RC1_robotcali.<ts>.json via repro/dump.
Run:  python RC1_robotcali.py [--quick]
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X10_fk_calib_real as X10          # noqa: E402
import S2_ur5gtpn_relaxed as S2          # noqa: E402
from repro.dump import save_dump         # noqa: E402

DT = torch.float64
RESULTS = {}
DATA = os.environ.get(
    'ROBOTCALI', os.path.join(_REPO, 'data', 'robotcali', 'RobotCali-new.xlsx'))


class _T:
    def __init__(self, A):
        self.A = A


class ABB120:
    """IRB 120 (3.58) FK from irb120_3_58_macro.xacro (MIT, hylander2126/irb120_ros2)."""
    n = 6
    O = [np.array([0, 0, 0]), np.array([0, 0, 0.29]), np.array([0, 0, 0.27]),
         np.array([0, 0, 0.07]), np.array([0.302, 0, 0]), np.array([0.072, 0, 0])]
    ax = ['z', 'y', 'y', 'x', 'y', 'x']

    def _r(self, axis, q):
        c, s = np.cos(q), np.sin(q)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def fkine(self, q):
        T = np.eye(4)
        for i in range(6):
            Ti = np.eye(4)
            Ti[:3, 3] = self.O[i]
            Ti[:3, :3] = self._r(self.ax[i], q[i])
            T = T @ Ti
        return _T(T)


def build_robot():
    robot = ABB120()
    M0 = torch.tensor(robot.fkine(np.zeros(6)).A, dtype=DT)
    xi = torch.zeros(6, 6, dtype=DT)
    h = 1e-6
    Minv = torch.linalg.inv(M0)
    for j in range(6):
        qp, qm = np.zeros(6), np.zeros(6)
        qp[j], qm[j] = h, -h
        Tp = torch.tensor(robot.fkine(qp).A, dtype=DT)
        Tm = torch.tensor(robot.fkine(qm).A, dtype=DT)
        X = Minv @ ((Tp - Tm) / (2 * h))
        xi[j] = torch.cat([torch.stack([X[2, 1], X[0, 2], X[1, 0]]), X[:3, 3]])
    return robot, xi, M0


def load_robotcali(path):
    if not os.path.exists(path):
        print(f"SKIP-DATA: ABB RobotCali xlsx not found at {path}", flush=True)
        print("It is not redistributed (no license); see REPRODUCE.md and tools/check_abb_data.py.",
              flush=True)
        raise SystemExit(0)
    df = pd.read_excel(path)
    q = np.deg2rad(df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values)
    L = df['L'].values / 1000.0
    return q, L


class RangeBaseline(torch.nn.Module):
    """Nominal FK + constant tool offset + anchor (9p)."""

    def __init__(self, p_nom, R_nom, anchor0):
        super().__init__()
        self.register_buffer('p_nom', torch.tensor(p_nom, dtype=DT))
        self.register_buffer('R_nom', torch.tensor(R_nom, dtype=DT))
        self.tool = torch.nn.Parameter(torch.zeros(6, dtype=DT))
        self.anchor = torch.nn.Parameter(torch.tensor(anchor0, dtype=DT))

    def range(self, idx):
        Tt = X10.se3_exp(self.tool.double().unsqueeze(0))[0]
        p = self.p_nom[idx] + (self.R_nom[idx] @ Tt[:3, 3])
        return (p - self.anchor).norm(dim=-1)


class RangeScrew(torch.nn.Module):
    """Screw node (X10.ScrewCalib) + anchor."""

    def __init__(self, xi, M0, anchor0, relax=False):
        super().__init__()
        self.sc = S2.ScrewCalibRelax(xi, M0, mode='dense') if relax \
            else X10.ScrewCalib(xi, M0, mode='dense')
        self.anchor = torch.nn.Parameter(torch.tensor(anchor0, dtype=DT))

    def range(self, q):
        T = self.sc(q)
        return (T[:, :3, 3] - self.anchor).norm(dim=-1)


def anchor_init(p_nom_tr, L_tr):
    sol = least_squares(lambda a: np.linalg.norm(p_nom_tr - a, axis=1) - L_tr,
                        p_nom_tr.mean(0) + np.array([0, 0, -0.5]))
    return sol.x


def fit_adam(model, q_tr, L_tr, steps, lr=1e-2, seed=0, idx=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    Lt = torch.tensor(L_tr, dtype=DT)
    qt = None if idx is not None else torch.tensor(q_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        pred = model.range(qt) if idx is None else model.range(idx)
        loss = 10.0 * ((pred - Lt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return model


def fit_gn(xi, M0, q_tr, L_tr, anchor0, max_nfev=400):
    m = X10.ScrewCalib(xi, M0, mode='dense')
    n = int(xi.shape[0])

    def pack():
        return np.concatenate([m.dxi.detach().numpy().ravel(),
                               m.b.detach().numpy(),
                               m.tool.detach().numpy(), anchor])
    def unpack(x):
        i = 0
        with torch.no_grad():
            m.dxi.copy_(torch.tensor(x[i:i + 6 * n], dtype=torch.float32).view(n, 6))
            i += 6 * n
            m.b.copy_(torch.tensor(x[i:i + n], dtype=torch.float32)); i += n
            m.tool.copy_(torch.tensor(x[i:i + 6], dtype=torch.float32)); i += 6
        return x[i:i + 3]

    anchor = np.array(anchor0)
    qt = torch.tensor(q_tr, dtype=DT)
    Lt = L_tr

    def fun(x):
        a = unpack(x)
        with torch.no_grad():
            T = m(qt)
            p = T[:, :3, 3].numpy().astype(np.float64)
        return np.linalg.norm(p - a, axis=1) - Lt

    x0 = pack()
    sol = least_squares(fun, x0, max_nfev=max_nfev)
    a = unpack(sol.x)
    return m, a


def eval_dL_range(model, q_or_idx, L, idx=None):
    with torch.no_grad():
        if idx is None:
            pred = model.range(torch.tensor(q_or_idx, dtype=DT))
        else:
            pred = model.range(idx)
        r = (pred - torch.tensor(L, dtype=DT)).abs().numpy()
    return float(1000.0 * r.mean())


def eval_dL_screw(m, a, q, L):
    p = X10.apply_sc(m, q)[:, :3, 3]
    return float(1000.0 * np.abs(np.linalg.norm(p - a, axis=1) - L).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--sizes', default='40,80,160,320')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '40', 1, 300
    t0 = time.time()

    robot, xi, M0 = build_robot()
    q_all, L_all = load_robotcali(args.data)
    n_total = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    p_nom = Tf_all[:, :3, 3]
    R_nom = Tf_all[:, :3, :3]
    sizes = [int(s) for s in args.sizes.split(',')]
    print('[RC1] RobotCali ABB IRB 120, range-only (draw-wire)', flush=True)

    res = {(n, k): [] for n in sizes
           for k in ('baseline', 'exact', 'gn', 'joint', 'isolated')}
    res_psi = {(n, k): [] for n in sizes for k in ('joint', 'isolated')}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            a0 = anchor_init(p_nom[tr], L_all[tr])
            base = RangeBaseline(p_nom, R_nom, a0)
            fit_adam(base, None, L_all[tr], args.steps, args.lr, s, idx=torch.tensor(tr))
            res[(N, 'baseline')].append(eval_dL_range(base, None, L_all[te], torch.tensor(te)))

            ex = RangeScrew(xi, M0, a0)
            fit_adam(ex, q_all[tr], L_all[tr], args.steps, args.lr, s)
            res[(N, 'exact')].append(eval_dL_range(ex, q_all[te], L_all[te]))

            gn, a_gn = fit_gn(xi, M0, q_all[tr], L_all[tr], a0)
            res[(N, 'gn')].append(eval_dL_screw(gn, a_gn, q_all[te], L_all[te]))

            torch.manual_seed(s)   # seed psi init (S2 convention: seed before construction)
            joint = RangeScrew(xi, M0, a0, relax=True)
            fit_adam(joint, q_all[tr], L_all[tr], args.steps, args.lr, s)
            res[(N, 'joint')].append(eval_dL_range(joint, q_all[te], L_all[te]))
            with torch.no_grad():
                res_psi[(N, 'joint')].append(float(sum(p.abs().sum() for p in joint.sc.psi.parameters())))

            torch.manual_seed(s)   # seed psi init (S2 convention: seed before construction)
            iso = RangeScrew(xi, M0, a0, relax=True)
            iso.load_state_dict(copy.deepcopy(ex.state_dict()), strict=False)
            for p in iso.parameters():
                p.requires_grad_(False)
            for p in iso.sc.psi.parameters():
                p.requires_grad_(True)
            fit_adam(iso, q_all[tr], L_all[tr], args.steps, args.lr, s)
            res[(N, 'isolated')].append(eval_dL_range(iso, q_all[te], L_all[te]))
            with torch.no_grad():
                res_psi[(N, 'isolated')].append(float(sum(p.abs().sum() for p in iso.sc.psi.parameters())))

            print(f'[RC1 seed {s} N={N}] base {res[(N,"baseline")][-1]:.3f} '
                  f'exact {res[(N,"exact")][-1]:.3f} gn {res[(N,"gn")][-1]:.3f} '
                  f'joint {res[(N,"joint")][-1]:.3f} iso {res[(N,"isolated")][-1]:.3f} mm',
                  flush=True)
        print(f'[RC1] seed {s} done', flush=True)

    for N in sizes:
        for k in ('baseline', 'exact', 'gn', 'joint', 'isolated'):
            a = np.array(res[(N, k)])
            RESULTS[f'RC1:N{N}:{k}'] = dict(
                dL_mm_mean=float(a.mean()), dL_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        if N in [n for n in sizes] and 'joint' in res_psi[(N, 'joint')]:
            for k in ('joint', 'isolated'):
                RESULTS[f'RC1:N{N}:{k}:psi_norm'] = float(np.mean(res_psi[(N, k)]))
        for k, ref in (('exact', 'baseline'), ('gn', 'exact'),
                       ('isolated', 'exact'), ('joint', 'exact')):
            a = np.array(res[(N, k)])
            b = np.array(res[(N, ref)])
            if len(a) >= 2:
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'RC1:paired:N{N}:{k}_vs_{ref}'] = dict(
                    diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    RESULTS['RC1:protocol'] = dict(data=os.path.basename(args.data), metric='mean|dL| mm',
                                   sizes=sizes, seeds=args.seeds, steps=args.steps)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
