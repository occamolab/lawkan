#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MK1c_shuffle_control.py — shuffled-label falsifier for MK1 (P-MK6).

The MUKCa socket-consistency metric can in principle be gamed by a flexible
composition correction that collapses configurations into two clusters 50 mm
apart, without any physical law deformation.  This control destroys the physical
structure: all configurations of both sockets are concatenated, permuted with
rng(2000) and split into two pseudo-sockets of equal size.  The per-pseudo-socket
70/30 split, models (edge_b, iso), optimizer and metric are otherwise identical
to MK1 (seed 0, 3000 steps, lr 1e-3, lam=100, cosine schedule).

Prediction P-MK6: under shuffled labels the isolated psi must NOT improve over
edge_b (expectation: parity or harm).  If iso still improves by >=2x, the
consistency metric is collapse-gameable and the MK1 iso advantage must be
qualified as a metric artifact, not evidence of law deformation.

Bases: stock rtb LBR / Panda (verified equivalent to the MUKCa nominal URDF up
to a constant tool offset, see MK1b).  Train RMS is recorded next to test RMS to
quantify collapse/overfit.

Output: repro/json/MK1c_shuffle_control.<ts>.json.
Run:  python MK1c_shuffle_control.py [--quick]
"""

import argparse
import copy
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

import X10_fk_calib_real as X10                                  # noqa: E402
import S2_ur5gtpn_relaxed as S2                                  # noqa: E402
from MK1_mukca import (socket_loss, fit, evaluate, psi_norm,     # noqa: E402
                       robot_tensors, load_sockets)
from repro.dump import save_dump                                 # noqa: E402
import roboticstoolbox as rtb                                    # noqa: E402

DT = torch.float64
RESULTS = {}
PLATFORMS = {
    'kuka14': dict(view='front', model='LBR'),
    'panda_1': dict(view='front_center', model='Panda'),
}


@torch.no_grad()
def train_rms(m, q0_tr, q1_tr):
    p0 = m(torch.tensor(q0_tr, dtype=DT))[:, :3, 3]
    p1 = m(torch.tensor(q1_tr, dtype=DT))[:, :3, 3]
    c0, c1 = p0.mean(0), p1.mean(0)
    e0 = ((p0 - c0) ** 2).sum(1).mean()
    e1 = ((p1 - c1) ** 2).sum(1).mean()
    return float(np.sqrt(1000 ** 2 * 0.5 * (float(e0) + float(e1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default='kuka14,panda_1')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=100.0)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.platforms, args.steps = 'kuka14', 300
    t0 = time.time()

    for plat in args.platforms.split(','):
        cfg = PLATFORMS[plat]
        robot = getattr(rtb.models.URDF, cfg['model'])()
        xi, M0 = robot_tensors(robot)
        q0, q1 = load_sockets(plat)
        q = np.vstack([q0, q1])
        shuffle_rng = np.random.default_rng(2000)
        perm = shuffle_rng.permutation(len(q))
        half = len(q) // 2
        q0s, q1s = q[perm[:half]], q[perm[half:2 * half]]
        print(f'[MK1c {plat}] shuffled pseudo-sockets {len(q0s)}/{len(q1s)} '
              f'from {len(q0)}+{len(q1)} configs', flush=True)

        rng = np.random.default_rng(1000)          # same per-socket split as MK1 seed 0
        p0, p1 = rng.permutation(len(q0s)), rng.permutation(len(q1s))
        n0, n1 = int(0.7 * len(q0s)), int(0.7 * len(q1s))
        q0_tr, q0_te = q0s[p0[:n0]], q0s[p0[n0:]]
        q1_tr, q1_te = q1s[p1[:n1]], q1s[p1[n1:]]

        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        fit(eb, q0_tr, q1_tr, args.steps, args.lr, 0)
        eb_te, _ = evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)
        eb_tr = train_rms(eb, q0_tr, q1_tr)

        torch.manual_seed(0)
        iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
        iso.load_state_dict(copy.deepcopy(eb.state_dict()), strict=False)
        for p in iso.parameters():
            p.requires_grad_(False)
        for p in iso.psi.parameters():
            p.requires_grad_(True)
        fit(iso, q0_tr, q1_tr, args.steps, args.lr, 0)
        iso_te, _ = evaluate(iso, q0_tr, q1_tr, q0_te, q1_te)
        iso_tr = train_rms(iso, q0_tr, q1_tr)

        print(f'[MK1c {plat} seed 0] edge_b test {eb_te:.3f} (train {eb_tr:.3f}) | '
              f'iso test {iso_te:.3f} (train {iso_tr:.3f}) mm  psi_norm {psi_norm(iso):.1f}',
              flush=True)
        RESULTS[f'MK1c:{plat}:edge_b:test_rms_mm'] = float(eb_te)
        RESULTS[f'MK1c:{plat}:edge_b:train_rms_mm'] = float(eb_tr)
        RESULTS[f'MK1c:{plat}:iso:test_rms_mm'] = float(iso_te)
        RESULTS[f'MK1c:{plat}:iso:train_rms_mm'] = float(iso_tr)
        RESULTS[f'MK1c:{plat}:iso:psi_norm'] = psi_norm(iso)
        RESULTS[f'MK1c:{plat}:gain_te'] = float(eb_te / max(iso_te, 1e-9))

    RESULTS['MK1c:protocol'] = dict(
        shuffle='rng(2000) permutation of concatenated sockets, equal halves',
        split='70/30 per pseudo-socket, rng(1000)', seeds=1,
        steps=args.steps, lr=args.lr, lam=args.lam)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args, epochs=args.steps)
    print(f'done in {time.time()-t0:.0f}s; dump:', out)


if __name__ == '__main__':
    main()
