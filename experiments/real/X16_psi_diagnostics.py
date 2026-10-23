#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X16_psi_diagnostics.py — what does the learned law look like?

Trains the relaxed models on the real FR3 at N=80 (representative seed 0, plus
seeds 0-4 for the channel-importance spread) and dumps everything needed to
inspect psi:

  * Fourier-psi: per-channel weight norms grouped as
      A-angles (theta, ax, ay, az  x {1, sin, cos}), A-translation,
      B-angles, B-translation; plus the univariate curves g_c(x) on a grid
      for the top channels (norm over the 6 output components);
  * input-group ablation on the held-out set: zeroing A-angles / A-trans /
      B-angles / B-trans features at test time (which group carries psi?);
  * MLP-psi (X13): partial dependence of the correction norm along theta_B
      and along the flange position of A.

Output: repro/json/X16_psi_diagnostics.<ts>.json via repro/dump.
Run:  python X16_psi_diagnostics.py [--quick]
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

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X13_learned_node as X13              # noqa: E402
import X14_fourier_law as X14               # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def train_model_isolated(cls, C, E, q_tr, oq_tr, ot_tr, epochs, seed, **kw):
    """Two-phase: exact edge (psi frozen), then psi only (G2 protocol)."""
    m = cls(C, E, **kw)
    for p in m.psi.parameters():
        p.requires_grad_(False)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, 5e-3, seed, clip=0.0,
                            ang_weight=1.0, warm_epochs=0, loss_form='cosine2')
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, 5e-3, seed, clip=0.0,
                            ang_weight=1.0, warm_epochs=0, loss_form='cosine2')
    return m


def train_model(cls, C, E, q_tr, oq_tr, ot_tr, epochs, seed, **kw):
    torch.manual_seed(seed)
    m = cls(C, E, **kw)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, 5e-3, seed, clip=0.0,
                            ang_weight=1.0, warm_epochs=0, loss_form='cosine2')
    return m


def fourier_channel_norms(model):
    """W (30,6) -> per-channel norms (30,) and grouped sums (4,)."""
    with torch.no_grad():
        w = model.psi.out.weight.detach().numpy()  # (6, 30)
    norms = np.abs(w).sum(0)                       # 30 channels: 15 per pose
    # per pose: [th(1,sin,cos), ax(3), ay(3), az(3), tx, ty, tz]
    groups = np.array([
        norms[0:12].sum(),      # A angles
        norms[12:15].sum(),     # A translation
        norms[15:27].sum(),     # B angles
        norms[27:30].sum(),     # B translation
    ])
    return norms, groups


def fourier_curves(model, n_grid=121):
    """Univariate curves g_c(x) for the angle channels: value = ||W_c(x)||_2."""
    with torch.no_grad():
        w = model.psi.out.weight.detach().numpy().T   # (30, 6)
    wa = w[np.r_[0:12, 15:27]].reshape(8, 3, 6)   # 8 angle channels x (1,sin,cos) x 6
    x = np.linspace(-np.pi, np.pi, n_grid)
    basis = np.stack([np.ones_like(x), np.sin(x), np.cos(x)], -1)    # (G,3)
    curves = np.einsum('g b, c b o -> c g o', basis, wa)             # (8,G,6)
    norms = np.linalg.norm(curves, axis=-1)                          # (8,G)
    return x, curves, norms


def ablate_forward(model, q, group, B=100):
    """Zero out one feature group inside FourierPsi at test time."""
    orig = model.psi.forward

    def patched(A, Bt):
        feats = []
        for p, tag in ((A, 'A'), (Bt, 'B')):
            th, ax, ay, az, ts = X14.chart_feats(p)
            for x in (th, ax, ay, az):
                feats += [torch.ones_like(x), torch.sin(x), torch.cos(x)]
            feats += [ts[..., 0:1], ts[..., 1:2], ts[..., 2:3]]
        f = torch.cat(feats, -1)
        idx = {'A_angles': list(range(0, 12)), 'A_trans': list(range(12, 15)),
               'B_angles': list(range(15, 27)), 'B_trans': list(range(27, 30))}
        if group is not None:
            f = f.clone()
            f[..., idx[group]] = 0.0
        return 0.1 * model.psi.out(f)

    model.psi.forward = patched
    try:
        with torch.no_grad():
            pred = model(q)
    finally:
        model.psi.forward = orig
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--N', type=int, default=80)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--isolated', action='store_true',
                    help='train psi on the frozen exact edge (G2 protocol)')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 300
    train_fn = train_model_isolated if args.isolated else train_model
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)

    groups_all = []
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        tr, te = perm[:args.N], perm[args.N:]
        m = train_fn(X14.FourierLawArm, C, E, q_all[tr], oq_all[tr], ot_all[tr],
                     args.epochs, s, edge_dense=True)
        norms, groups = fourier_channel_norms(m)
        groups_all.append(groups)
        RESULTS[f'X16:groups:seed{s}'] = [float(x) for x in groups]
        RESULTS[f'X16:channels:seed{s}'] = [float(x) for x in norms]
        if s == 0:
            RESULTS['X16:fourier_W:seed0'] = [
                [float(v) for v in row] for row in
                m.psi.out.weight.detach().numpy()]
            RESULTS['X16:fourier_b:seed0'] = [
                float(v) for v in m.psi.out.bias.detach().numpy()]
            x, curves, cn = fourier_curves(m)
            top = np.argsort(-cn.mean(1))[:3]
            RESULTS['X16:curves_x'] = [float(v) for v in x]
            RESULTS['X16:curves_top_idx'] = [int(t) for t in top]
            RESULTS['X16:curves_top_norms'] = [
                [float(v) for v in cn[t]] for t in top]
            # ablations on the held-out set
            err = {}
            for g in (None, 'A_angles', 'A_trans', 'B_angles', 'B_trans'):
                pred = ablate_forward(m, q_all[te], g)
                err[g or 'none'] = float(GN.eval_pos(
                    _Wrap(pred), q_all[te], oq_all[te], ot_all[te])[0])
            RESULTS['X16:ablation:seed0'] = err
        print(f'[X16] fourier seed {s} groups ' +
              ' '.join(f'{v:.2f}' for v in groups), flush=True)

    groups_all = np.array(groups_all)
    RESULTS['X16:groups_mean'] = [float(v) for v in groups_all.mean(0)]
    RESULTS['X16:groups_std'] = [float(v) for v in groups_all.std(0)]

    # MLP-psi partial dependence (seed 0)
    perm = np.random.default_rng(1000).permutation(n_total)
    tr, te = perm[:args.N], perm[args.N:]
    m = train_fn(X13.LearnedNodeArm, C, E, q_all[tr], oq_all[tr], ot_all[tr],
                 args.epochs, 0, edge_dense=True)
    RESULTS['X16:mlp_psi:seed0'] = [
        [float(v) for v in p.detach().numpy().ravel()] for p in m.psi.parameters()]
    with torch.no_grad():
        pose = (E34.quat_exp(m.base_r).expand(50, 4), m.base_t.expand(50, 3))
        th = torch.linspace(0.0, np.pi, 50, dtype=DT)
        rq = E34.quat_exp(torch.cat([torch.zeros(50, 2, dtype=DT),
                                     th.unsqueeze(-1)], -1))
        zt = torch.zeros(50, 3, dtype=DT)
        pd_B = []
        for k in range(50):
            rqk = rq[k].unsqueeze(0).expand(50, 4)
            f = torch.cat([X13.canon_q(pose[0]), pose[1],
                           X13.canon_q(rqk), zt], -1)
            pd_B.append(float(m.psi(f)[0].norm()))
        RESULTS['X16:pd_B_theta'] = pd_B
        RESULTS['X16:pd_B_x'] = [float(v) for v in th]
    print('[X16] done', flush=True)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('wall_s', round(time.time() - t0, 1))
    print('dump ->', p)


class _Wrap(torch.nn.Module):
    """Wrap a precomputed (quat, t) tuple as a callable model for eval_pos."""

    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, q):
        return self.pred


if __name__ == '__main__':
    main()
