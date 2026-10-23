#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
G3b_orthogonality_twoslot.py — the G3 orthogonality decomposition of the FR3
correction under the FINAL two-slot head (registered before run).

G3 measured the correction-vs-edge orthogonality for the Fourier psi on FR3
(orthogonal share substantial for the isolated variant; joint is absorbable).
Here the same decomposition (psi field Psi_i and output Jacobian A over the edge
parameters, absorbable share = 1 - ||Psi - A lstsq||^2 / ||Psi||^2) is recomputed
for the two-slot psi, so the number quoted in the paper matches the final head.

Registered prediction (plan): the two-slot reproduces G3 qualitatively --- the
isolated variant keeps a substantial orthogonal share (>= 0.5) and the joint
variant is largely absorbable (>= 0.9); no revision of the orthogonality story.

Output: repro/json/G3b_orthogonality_twoslot.<ts>.json via repro/dump.
Run:  python G3b_orthogonality_twoslot.py [--quick]
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

import X7_robot_calibration as X7        # noqa: E402
import E34_x7_recipe as E34              # noqa: E402
import X7_gn_ls as GN                    # noqa: E402
import X24_fr3_gate as X24               # noqa: E402
import X33_fr3_twoslot as X33            # noqa: E402
import X34_fr3_kernel as X34             # noqa: E402
import G3_orthogonality as G3            # noqa: E402
from repro.dump import save_dump         # noqa: E402

DT = torch.float64
RESULTS = {}


def run_fr3_twoslot(args):
    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1000).permutation(len(q_np))
    tr, te = perm[:args.N], perm[args.N:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]

    exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, 0)
    feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
    from scipy.cluster.vq import kmeans2
    cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=0)
    centers = torch.tensor(cen, dtype=DT)
    dc = torch.cdist(centers, centers)
    bw = float(torch.median(dc[dc > 0]))

    iso = X33.make_two_slot(exact, centers, bw)
    X33.train(iso, q_tr, oq_tr, ot_tr, args.epochs, args.lr, 0,
              {'global', 'kernel'}, l2_global=0.0)

    joint = X33.make_two_slot(exact, centers, bw)
    for p in joint.parameters():
        p.requires_grad_(True)
    joint, _ = E34.train_recipe(joint, q_tr, oq_tr, ot_tr, args.epochs, args.lr, 0,
                                clip=0.0, ang_weight=1.0, warm_epochs=0,
                                loss_form='cosine2')

    ev = lambda m, q: GN.eval_pos(m, q, oq_te, ot_te)[0]
    G3.RESULTS = RESULTS
    G3.analyze('fr3t:joint', G3.poses_fr3, joint, q_tr, ev, q_te)
    G3.analyze('fr3t:isolated', G3.poses_fr3, iso, q_tr, ev, q_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--N', type=int, default=80)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.N, args.epochs = 20, 300
    t0 = time.time()
    run_fr3_twoslot(args)
    RESULTS['G3b:_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=1, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
