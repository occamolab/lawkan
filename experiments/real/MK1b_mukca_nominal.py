#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MK1b_mukca_nominal.py — equivalence check: the MUKCa nominal URDF vs the stock
LBR base used in MK1 (kuka14/front).

The MUKCa nominal URDF `urdf/iiwa14.urdf` and `rtb.models.URDF.LBR` are the same
iiwa14 chain: the relative transform T_rtb^-1 T_urdf is constant to 2e-10 with
identity rotation and a 40.81 mm z-offset (the fixed ball_joint shift).  The free
tool in our model absorbs this offset, so the optimization problem is identical;
the smoke run reproduces MK1 to three decimals (nom 0.768, edge_b 0.508), and the
full registered run P-MK1b was cancelled as redundant.

Reference (their own KPI formula on kuka14/front): nominal URDF std_dev_fk
0.715 mm / dist 0.125 mm; their full 48-parameter calibration 0.118 mm / 0.0003.
Cross-view (calibrated on front): left_back 0.145 mm, raised_left 0.465 mm.

Run:  python MK1b_mukca_nominal.py [--quick]
"""

import argparse
import copy
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X10_fk_calib_real as X10                      # noqa: E402
import S2_ur5gtpn_relaxed as S2                      # noqa: E402
from MK1_mukca import socket_loss, fit, evaluate, psi_norm, MODELS, KPI  # noqa: E402
from repro.dump import save_dump                     # noqa: E402

DT = torch.float64
RESULTS = {}
_ROOT = os.path.join(os.path.dirname(_REPO), 'newdata', 'kinematics_calibration')
URDF = os.path.join(_ROOT, 'urdf', 'iiwa14.urdf')
DATA = os.path.join(_ROOT, 'data', 'kuka14', 'front')


def _origin(el):
    xyz = np.array([float(v) for v in (el.get('xyz', '0 0 0')).split()])
    rpy = np.array([float(v) for v in (el.get('rpy', '0 0 0')).split()])
    return xyz, rpy


def _chain(path, base='base_link', end='ball_link'):
    joints = {}
    for j in ET.parse(path).getroot().findall('joint'):
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        ax = j.find('axis')
        axis = None if ax is None else np.array([float(v) for v in ax.get('xyz').split()])
        joints[child] = (parent, j.get('type'), _origin(j.find('origin')), axis)
    seq, cur = [], end
    while cur != base:
        if cur not in joints:
            break
        parent, jtype, orig, axis = joints[cur]
        seq.append((jtype, orig, axis))
        cur = parent
    return seq[::-1]


def _fk(seq, q):
    T = np.eye(4)
    qi = 0
    for jtype, (xyz, rpy), axis in seq:
        r, p, w = rpy
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(w), -np.sin(w), 0], [np.sin(w), np.cos(w), 0], [0, 0, 1]])
        Tj = np.eye(4)
        Tj[:3, :3] = Rz @ Ry @ Rx
        Tj[:3, 3] = xyz
        T = T @ Tj
        if jtype in ('revolute', 'continuous'):
            th = q[qi]
            qi += 1
            x, y, z = axis
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            Tj = np.eye(4)
            Tj[:3, :3] = R
            T = T @ Tj
    return T


def urdf_tensors(path):
    seq = _chain(path)
    n = sum(1 for jtype, _, _ in seq if jtype in ('revolute', 'continuous'))
    M0 = _fk(seq, np.zeros(n))
    xi = np.zeros((n, 6))
    h = 1e-6
    Minv = np.linalg.inv(M0)
    for j in range(n):
        qp, qm = np.zeros(n), np.zeros(n)
        qp[j], qm[j] = h, -h
        X = Minv @ ((_fk(seq, qp) - _fk(seq, qm)) / (2 * h))
        xi[j] = [X[2, 1], X[0, 2], X[1, 0], X[0, 3], X[1, 3], X[2, 3]]
    return torch.tensor(xi, dtype=DT), torch.tensor(M0, dtype=DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=100.0)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 1, 300

    xi, M0 = urdf_tensors(URDF)
    q0 = np.loadtxt(os.path.join(DATA, 'hole_0.csv'), delimiter=',')
    q1 = np.loadtxt(os.path.join(DATA, 'hole_1.csv'), delimiter=',')
    assert q0.shape[1] == xi.shape[0], (q0.shape, xi.shape)
    print(f'[MK1b kuka14] base nominal URDF iiwa14, dof {xi.shape[0]}, '
          f'sockets {q0.shape[0]}/{q1.shape[0]}', flush=True)

    rms = {k: [] for k in MODELS}
    dist = {k: [] for k in MODELS}
    pnorm = {k: [] for k in ('iso', 'joint')}
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

        nom = X10.ScrewCalib(xi, M0, mode='dense')
        for p in (nom.dxi, nom.b):
            p.requires_grad_(False)
        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        dn = X10.ScrewCalib(xi, M0, mode='dense')
        models = {'nominal': nom, 'edge_b': eb, 'dense': dn}

        for k in ('nominal', 'edge_b', 'dense'):
            fit(models[k], q0_tr, q1_tr, args.steps, args.lr, s)
            r, d = evaluate(models[k], q0_tr, q1_tr, q0_te, q1_te)
            rms[k].append(r)
            dist[k].append(d)

        torch.manual_seed(s)
        iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
        iso.load_state_dict(copy.deepcopy(models['edge_b'].state_dict()), strict=False)
        for p in iso.parameters():
            p.requires_grad_(False)
        for p in iso.psi.parameters():
            p.requires_grad_(True)
        fit(iso, q0_tr, q1_tr, args.steps, args.lr, s)
        r, d = evaluate(iso, q0_tr, q1_tr, q0_te, q1_te)
        rms['iso'].append(r)
        dist['iso'].append(d)
        pnorm['iso'].append(psi_norm(iso))

        torch.manual_seed(s)
        jt = S2.ScrewCalibRelax(xi, M0, mode='dense')
        fit(jt, q0_tr, q1_tr, args.steps, args.lr, s)
        r, d = evaluate(jt, q0_tr, q1_tr, q0_te, q1_te)
        rms['joint'].append(r)
        dist['joint'].append(d)
        pnorm['joint'].append(psi_norm(jt))

        print(f'[MK1b kuka14 seed {s}] nom {rms["nominal"][-1]:.3f} '
              f'edge_b {rms["edge_b"][-1]:.3f} dense {rms["dense"][-1]:.3f} '
              f'iso {rms["iso"][-1]:.3f} joint {rms["joint"][-1]:.3f} mm',
              flush=True)

    for k in MODELS:
        RESULTS[f'MK1b:kuka14:{k}:rms_mm'] = dict(
            mean=float(np.mean(rms[k])), std=float(np.std(rms[k])),
            per_seed=[float(x) for x in rms[k]])
        RESULTS[f'MK1b:kuka14:{k}:dist_mm'] = dict(
            mean=float(np.mean(dist[k])), std=float(np.std(dist[k])),
            per_seed=[float(x) for x in dist[k]])
    for k in ('iso', 'joint'):
        RESULTS[f'MK1b:kuka14:{k}:psi_norm'] = float(np.mean(pnorm[k]))
    for k, ref in (('edge_b', 'nominal'), ('dense', 'edge_b'), ('iso', 'edge_b'),
                   ('joint', 'edge_b'), ('iso', 'joint')):
        a, b = np.array(rms[k]), np.array(rms[ref])
        if len(a) >= 2:
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'MK1b:kuka14:paired:{k}_vs_{ref}'] = dict(
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    RESULTS['MK1b:kuka14:their_kpi_mm'] = float(KPI['kuka14'])
    RESULTS['MK1b:kuka14:nominal_urdf_kpi_mm'] = 0.715
    RESULTS['MK1b:protocol'] = dict(base='urdf/iiwa14.urdf (their nominal, ball_link)',
                                    seeds=args.seeds, steps=args.steps, lr=args.lr,
                                    lam=args.lam, split='70/30 per socket')
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
