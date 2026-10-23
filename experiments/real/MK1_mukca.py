#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MK1_mukca.py — MUKCa socket-consistency calibration (KUKA iiwa14, Panda, FR3-unit).

External mechanical constraint (no tracker): the ball attached to the end
effector must sit in one fixed socket, and the two sockets are 50 mm apart.
Loss on train configurations:
    sum_s Var(p_s) + lam * (||mean(p_0) - mean(p_1)|| - 0.05)^2,
with p_s the predicted ball point (tool is part of the model). Held-out
configurations are scored with the train means (RMS scatter, mm) and the
distance error. Calibration of the KUKA data is 0.4-0.5 mm nominally; the
authors' calibrated KPI is 0.119 mm.

Diagnosis before the full run (KUKA, seed 0): the full dense parameterization
(55p) overfits the consistency objective (test RMS 0.95-2.3 mm on 42 train
configurations), the reduced edge (tool + joint biases, 13p) reaches 0.508 mm
vs 0.768 mm for tool-only, and an isolated psi on top of the reduced edge
reaches 0.193 mm.

Models:
  nominal : tool only (dxi, b frozen, 6p);
  edge_b  : tool + joint biases b (13p);
  dense   : full dense screw (55p, boundary/overfit control);
  iso     : psi only on top of the trained edge_b (G2 protocol);
  joint   : edge_b + psi trained jointly.
Reference: their calibrated KPI (calibrated_urdf/<robot>/kpis.yaml).

Registered predictions (plan 2026-09-29, amended 04.10 before the full run):
  P-MK1: edge_b beats nominal by >=1.3x;
  P-MK2: dense degrades (consistency overfit) - boundary;
  P-MK3: iso psi significantly better than edge_b (composition signal);
  P-MK4: joint psi >= iso (compensation does not help);
  P-MK5 (boundary): if iso psi does not win on Panda/FR3-unit, the KUKA
         signal is platform-specific.

Protocol: per-socket 70/30 split rng(1000+seed), seeds 0-4, Adam 3000 steps,
lr 1e-3, lam=100, metric: test RMS scatter (mm), distance error (mm).

Output: repro/json/MK1_mukca.<ts>.json via repro/dump.
Run:  python MK1_mukca.py [--quick]
"""

import argparse
import copy
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
import S2_ur5gtpn_relaxed as S2          # noqa: E402
import roboticstoolbox as rtb            # noqa: E402
from repro.dump import save_dump
from repro.data_guard import skip_if_missing         # noqa: E402

DT = torch.float64
RESULTS = {}
DATA_ROOT = os.path.join(os.path.dirname(_REPO), 'newdata', 'kinematics_calibration', 'data')
KINOVA_URDF = os.path.join(os.path.dirname(_REPO), 'newdata', 'kinematics_calibration',
                           'urdf', 'gen3lite.urdf')
KPI = {'kuka14': 0.119, 'kinova': 1.731, 'panda_1': np.nan, 'fr3': np.nan}
MODELS = ('nominal', 'edge_b', 'dense', 'iso', 'joint')

PLATFORMS = {
    'kuka14': dict(data='kuka14', view='front', model='LBR'),
    'panda_1': dict(data='panda_1', view='front_center', model='Panda'),
    'fr3': dict(data='fr3', view='front', model='Panda'),
    'kinova': dict(data='kinova', view='front', urdf=KINOVA_URDF),
    'kuka14_left_back': dict(data='kuka14', view='left_back', model='LBR'),
    'kuka14_raised_left': dict(data='kuka14', view='raised_left', model='LBR'),
}


def robot_tensors(robot):
    n = robot.n
    M0 = torch.tensor(robot.fkine(np.zeros(n)).A, dtype=DT)
    xi = torch.zeros(n, 6, dtype=DT)
    h = 1e-6
    Minv = torch.linalg.inv(M0)
    for j in range(n):
        qp, qm = np.zeros(n), np.zeros(n)
        qp[j], qm[j] = h, -h
        Tp = torch.tensor(robot.fkine(qp).A, dtype=DT)
        Tm = torch.tensor(robot.fkine(qm).A, dtype=DT)
        X = Minv @ ((Tp - Tm) / (2 * h))
        xi[j] = torch.cat([torch.stack([X[2, 1], X[0, 2], X[1, 0]]), X[:3, 3]])
    return xi, M0


def load_sockets(platform):
    cfg = PLATFORMS[platform]
    base = os.path.join(DATA_ROOT, cfg.get('data', platform), cfg['view'])
    q0 = np.loadtxt(os.path.join(base, 'hole_0.csv'), delimiter=',')
    q1 = np.loadtxt(os.path.join(base, 'hole_1.csv'), delimiter=',')
    return q0, q1


def platform_tensors(cfg):
    """(xi, M0) for a platform: stock rtb model, or a URDF chain when 'urdf' is set."""
    if 'urdf' in cfg:
        from MK1b_mukca_nominal import urdf_tensors
        return urdf_tensors(cfg['urdf'])
    robot = getattr(rtb.models.URDF, cfg['model'])()
    return robot_tensors(robot)


def socket_loss(m, q0_t, q1_t, lam=100.0):
    p0 = m(q0_t)[:, :3, 3]
    p1 = m(q1_t)[:, :3, 3]
    c0, c1 = p0.mean(0), p1.mean(0)
    var = ((p0 - c0) ** 2).sum(1).mean() + ((p1 - c1) ** 2).sum(1).mean()
    dist = ((c0 - c1).norm() - 0.05) ** 2
    return var + lam * dist


def fit(m, q0_tr, q1_tr, steps, lr, seed):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    q0t = torch.tensor(q0_tr, dtype=DT)
    q1t = torch.tensor(q1_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        loss = socket_loss(m, q0t, q1t)
        loss.backward()
        opt.step()
        sched.step()
    return m


@torch.no_grad()
def evaluate(m, q0_tr, q1_tr, q0_te, q1_te):
    c0 = m(torch.tensor(q0_tr, dtype=DT))[:, :3, 3].mean(0)
    c1 = m(torch.tensor(q1_tr, dtype=DT))[:, :3, 3].mean(0)
    e0 = m(torch.tensor(q0_te, dtype=DT))[:, :3, 3] - c0
    e1 = m(torch.tensor(q1_te, dtype=DT))[:, :3, 3] - c1
    rms = float(np.sqrt(1000 ** 2 * 0.5 * (float((e0 ** 2).sum(1).mean())
                                            + float((e1 ** 2).sum(1).mean()))))
    dist = float(abs(float((c0 - c1).norm()) - 0.05) * 1000)
    return rms, dist


def psi_norm(m):
    with torch.no_grad():
        return float(sum(p.abs().sum() for p in m.psi.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default='kuka14,panda_1,fr3')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=100.0)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if skip_if_missing(DATA_ROOT, 'MUKCa data directory'):
        return
    if args.quick:
        args.seeds, args.steps = 1, 300
    t0 = time.time()

    for plat in args.platforms.split(','):
        cfg = PLATFORMS[plat]
        xi, M0 = platform_tensors(cfg)
        q0, q1 = load_sockets(plat)
        assert q0.shape[1] == xi.shape[0], (plat, q0.shape, xi.shape)
        print(f'[MK1 {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}',
              flush=True)
        rms = {k: [] for k in MODELS}
        dist = {k: [] for k in MODELS}
        pnorm = {k: [] for k in ('iso', 'joint')}
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

            models = {}
            nom = X10.ScrewCalib(xi, M0, mode='dense')
            for p in (nom.dxi, nom.b):
                p.requires_grad_(False)
            models['nominal'] = nom
            eb = X10.ScrewCalib(xi, M0, mode='dense')
            eb.dxi.requires_grad_(False)
            models['edge_b'] = eb
            dn = X10.ScrewCalib(xi, M0, mode='dense')
            models['dense'] = dn

            for k in ('nominal', 'edge_b', 'dense'):
                fit(models[k], q0_tr, q1_tr, args.steps, args.lr, s)
                r, d = evaluate(models[k], q0_tr, q1_tr, q0_te, q1_te)
                rms[k].append(r)
                dist[k].append(d)

            torch.manual_seed(s)   # seed psi init (S2 convention: seed before construction)
            iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
            iso.load_state_dict(copy.deepcopy(models['edge_b'].state_dict()),
                                strict=False)
            for p in iso.parameters():
                p.requires_grad_(False)
            for p in iso.psi.parameters():
                p.requires_grad_(True)
            fit(iso, q0_tr, q1_tr, args.steps, args.lr, s)
            r, d = evaluate(iso, q0_tr, q1_tr, q0_te, q1_te)
            rms['iso'].append(r)
            dist['iso'].append(d)
            pnorm['iso'].append(psi_norm(iso))

            torch.manual_seed(s)   # seed psi init (S2 convention: seed before construction)
            jt = S2.ScrewCalibRelax(xi, M0, mode='dense')
            fit(jt, q0_tr, q1_tr, args.steps, args.lr, s)
            r, d = evaluate(jt, q0_tr, q1_tr, q0_te, q1_te)
            rms['joint'].append(r)
            dist['joint'].append(d)
            pnorm['joint'].append(psi_norm(jt))

            print(f'[MK1 {plat} seed {s}] nom {rms["nominal"][-1]:.3f} '
                  f'edge_b {rms["edge_b"][-1]:.3f} dense {rms["dense"][-1]:.3f} '
                  f'iso {rms["iso"][-1]:.3f} joint {rms["joint"][-1]:.3f} mm',
                  flush=True)
        for k in MODELS:
            RESULTS[f'MK1:{plat}:{k}:rms_mm'] = dict(
                mean=float(np.mean(rms[k])), std=float(np.std(rms[k])),
                per_seed=[float(x) for x in rms[k]])
            RESULTS[f'MK1:{plat}:{k}:dist_mm'] = dict(
                mean=float(np.mean(dist[k])), std=float(np.std(dist[k])),
                per_seed=[float(x) for x in dist[k]])
        for k in ('iso', 'joint'):
            RESULTS[f'MK1:{plat}:{k}:psi_norm'] = float(np.mean(pnorm[k]))
        for k, ref in (('edge_b', 'nominal'), ('dense', 'edge_b'), ('iso', 'edge_b'),
                       ('joint', 'edge_b'), ('iso', 'joint')):
            a = np.array(rms[k]); b = np.array(rms[ref])
            if len(a) >= 2:
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'MK1:{plat}:paired:{k}_vs_{ref}'] = dict(
                    diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        RESULTS[f'MK1:{plat}:their_kpi_mm'] = float(KPI.get(plat, np.nan))

    RESULTS['MK1:protocol'] = dict(seeds=args.seeds, steps=args.steps, lr=args.lr,
                                   lam=args.lam, split='70/30 per socket')
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
