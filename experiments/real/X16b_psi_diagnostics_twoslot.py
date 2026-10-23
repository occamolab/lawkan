#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X16b_psi_diagnostics_twoslot.py — X16 diagnostics of the FR3 correction under the
FINAL two-slot head (registered before run).

Recomputes the X16 diagnostics for the two-slot psi:
  * channel-group weight norms of the global slot (same {1,sin,cos} grouping as the
    Fourier head), and the kernel weight norm;
  * univariate curves g_c(x) of the global slot;
  * input-group ablation on the held-out set (zero A-angles / A-trans / B-angles /
    B-trans inside the two-slot feature vector).
Head-agnostic parts (ablation, partial dependence) are the same idea as X16.

Registered prediction: the two-slot localizes like the Fourier realization --- the
deviation is carried by the pair features on both sides, dominated by the B side;
no revision of the X16 reading.

Output: repro/json/X16b_psi_diagnostics_twoslot.<ts>.json via repro/dump.
Run:  python X16b_psi_diagnostics_twoslot.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X24_fr3_gate as X24                  # noqa: E402
import X33_fr3_twoslot as X33               # noqa: E402
import X34_fr3_kernel as X34                # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}


def group_norms(w6x30):
    """(6,30) -> per-channel norms (30,) and grouped sums (4,)."""
    n = np.abs(w6x30).sum(0)
    return n, np.array([n[0:12].sum(), n[12:15].sum(), n[15:27].sum(), n[27:30].sum()])


def curves(w6x30, n_grid=121):
    """Univariate g_c(x) of the global slot over the 8 angle channels."""
    w = w6x30.T                                  # (30,6)
    wa = w[np.r_[0:12, 15:27]].reshape(8, 3, 6)
    x = np.linspace(-np.pi, np.pi, n_grid)
    basis = np.stack([np.ones_like(x), np.sin(x), np.cos(x)], -1)
    c = np.einsum('gb,cbo->cgo', basis, wa)
    return x, np.linalg.norm(c, axis=-1)


def ablate(model, q, group):
    """Zero a feature group inside the two-slot forward (globalslot + kernel)."""
    orig = model.psi.forward
    idx = {'A_angles': list(range(0, 12)), 'A_trans': list(range(12, 15)),
           'B_angles': list(range(15, 27)), 'B_trans': list(range(27, 30))}

    def patched(A, B):
        f = X34.raw_feats(A, B).reshape(-1, 30)
        if group is not None:
            f = f.clone()
            f[:, idx[group]] = 0.0
        out = model.psi.global_lin(f)
        d2 = torch.cdist(f, model.psi.centers) ** 2
        out = out + torch.exp(-d2 / (2 * model.psi.bw ** 2)) @ model.psi.Wk
        return 0.1 * out

    model.psi.forward = patched
    try:
        with torch.no_grad():
            pred = model(q)
    finally:
        model.psi.forward = orig
    return pred


class _Wrap(torch.nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred

    def forward(self, q):
        return self.pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--N', type=int, default=80)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--centers', type=int, default=20)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 300
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)

    groups_all, knorm_all = [], []
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        tr, te = perm[:args.N], perm[args.N:]
        q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
        exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
        feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
        cen, _ = kmeans2(feats.numpy(), args.centers, minit='++', seed=s)
        centers = torch.tensor(cen, dtype=DT)
        dc = torch.cdist(centers, centers)
        bw = float(torch.median(dc[dc > 0]))
        m = X33.make_two_slot(exact, centers, bw)
        X33.train(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s,
                  {'global', 'kernel'}, l2_global=0.0)

        g_w = m.psi.global_lin.weight.detach().numpy()
        norms, groups = group_norms(g_w)
        kn = float(m.psi.Wk.detach().norm())
        groups_all.append(groups)
        knorm_all.append(kn)
        RESULTS[f'X16b:channels:seed{s}'] = [float(v) for v in norms]
        RESULTS[f'X16b:groups:seed{s}'] = [float(v) for v in groups]
        RESULTS[f'X16b:kernel_norm:seed{s}'] = kn
        if s == 0:
            x, cn = curves(g_w)
            top = np.argsort(-cn.mean(1))[:3]
            RESULTS['X16b:curves_x'] = [float(v) for v in x]
            RESULTS['X16b:curves_top_idx'] = [int(t) for t in top]
            RESULTS['X16b:curves_top_norms'] = [[float(v) for v in cn[t]] for t in top]
            err = {}
            for g in (None, 'A_angles', 'A_trans', 'B_angles', 'B_trans'):
                pred = ablate(m, q_all[te], g)
                err[g or 'none'] = float(GN.eval_pos(
                    _Wrap(pred), q_all[te], oq_all[te], ot_all[te])[0])
            RESULTS['X16b:ablation:seed0'] = err
        print(f'[X16b seed {s}] groups ' + ' '.join(f'{v:.2f}' for v in groups) +
              f' | kernel {kn:.3f}', flush=True)

    groups_all = np.array(groups_all)
    RESULTS['X16b:groups_mean'] = [float(v) for v in groups_all.mean(0)]
    RESULTS['X16b:groups_std'] = [float(v) for v in groups_all.std(0)]
    RESULTS['X16b:kernel_norm_mean'] = float(np.mean(knorm_all))
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
