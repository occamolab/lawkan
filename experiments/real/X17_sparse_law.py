#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X17_sparse_law.py — sparse psi in the node, by analogy with the sparse edge.

The full Fourier psi is a linear readout over 30 basis-expanded chart inputs
(186 parameters; X14). X16 ranked the input atoms by their weight mass
(frozen before this run, from repro/json/X16_psi_diagnostics.20260930-121021):

  thB, thA, axB, ayB, azA, ayA, azB, txA, tzA, txB, axA, tyB, tzB, tyA

where an angle atom is its {1, sin, cos} triple and a translation atom is a
single input. This experiment keeps only the top-k atoms and retrains, the
same protocol as X14 (FR3, X7 splits rng(1000+seed), E34.train_recipe,
4000 epochs, cosine2, clip=0), with the relaxed+dense model only.

Masks:
  top4: 12 inputs, 78 parameters;  top6: 18 inputs, 114;  top8: 22 inputs, 138.
References: exact dense / GN (bench dump), full Fourier psi and MLP psi (X14/X13
dumps, same splits/seeds); paired t-tests vs exact dense and vs full Fourier.

Registered predictions (plan 2026-09-29, before the run):
  P-SP1: top8 and top6 keep the win at N>=60 (relaxed < exact dense, paired);
  P-SP2: at N=20 the harm (relaxed - exact) shrinks as k decreases
         (top6/top4 smaller than the full Fourier psi, paired vs X14 reruns);
  P-SP3 (boundary): top4 loses the win at N>=60 (no better than exact dense).

Output: repro/json/X17_sparse_law.<ts>.json via repro/dump.
Run:  python X17_sparse_law.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X14_fourier_law as X14               # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}

ATOMS = {
    'thA': [0, 1, 2], 'axA': [3, 4, 5], 'ayA': [6, 7, 8], 'azA': [9, 10, 11],
    'txA': [12], 'tyA': [13], 'tzA': [14],
    'thB': [15, 16, 17], 'axB': [18, 19, 20], 'ayB': [21, 22, 23],
    'azB': [24, 25, 26], 'txB': [27], 'tyB': [28], 'tzB': [29],
}
RANK = ['thB', 'thA', 'axB', 'ayB', 'azA', 'ayA', 'azB',
        'txA', 'tzA', 'txB', 'axA', 'tyB', 'tzB', 'tyA']
MASKS = {'top4': RANK[:4], 'top6': RANK[:6], 'top7': RANK[:7],
         'top8': RANK[:8]}


def keep_indices(atoms):
    idx = []
    for a in atoms:
        idx += ATOMS[a]
    return sorted(idx)


class SparsePsi(nn.Module):
    """Fourier readout with a fixed input mask (kept atoms only), zero-init."""

    def __init__(self, keep):
        super().__init__()
        self.out = nn.Linear(30, 6).double()
        m = torch.zeros(30)
        m[keep] = 1.0
        self.register_buffer('mask', m)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, A, B):
        feats = []
        for p in (A, B):
            th, ax, ay, az, ts = X14.chart_feats(p)
            for x in (th, ax, ay, az):
                feats += [torch.ones_like(x), torch.sin(x), torch.cos(x)]
            feats += [ts[..., 0:1], ts[..., 1:2], ts[..., 2:3]]
        raw = torch.cat(feats, -1)
        return 0.1 * self.out(raw * self.mask)

    def n_params(self):
        return int(self.mask.sum().item()) * 6 + 6


class SparseLawArm(X14.FourierLawArm):
    def __init__(self, C, E, atoms, edge_dense=True):
        super().__init__(C, E, edge_dense=edge_dense)
        self.psi = SparsePsi(keep_indices(atoms))

    def n_params(self):
        p = 12
        if self.edge_dense:
            p += 6 * self.n
        p += self.psi.n_params()
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,60,80')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.epochs = '20', 1, 300
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = [int(s) for s in args.sizes.split(',')]
    ref = X14.load_ref()
    x14 = X14.load_json('X14_fourier_law.*.json')
    for n in sizes:
        ref[n]['full_fourier'] = np.array(
            x14[f'X14:n{n}:relaxed_fourier']['per_seed'])

    params = {tag: SparseLawArm(C, E, atoms).n_params()
              for tag, atoms in MASKS.items()}
    print('[X17] mask params:', params, flush=True)
    for tag, atoms in MASKS.items():
        RESULTS[f'X17:mask:{tag}'] = atoms

    res = {(n, tag): [] for n in sizes for tag in MASKS}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            line = []
            for tag, atoms in MASKS.items():
                m = SparseLawArm(C, E, atoms, edge_dense=True)
                m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                        args.lr, s, clip=0.0, ang_weight=1.0,
                                        warm_epochs=0, loss_form='cosine2')
                res[(n_tr, tag)].append(GN.eval_pos(m, q_te, oq_te, ot_te))
                line.append(f'{tag} {res[(n_tr, tag)][-1][0]:.3f}')
            print(f'[X17 seed {s} n={n_tr}] ' + ' | '.join(line), flush=True)
        print(f'[X17] seed {s} done', flush=True)

    for n_tr in sizes:
        for tag in MASKS:
            a = np.array([v[0] for v in res[(n_tr, tag)]])
            r = np.array([v[1] for v in res[(n_tr, tag)]])
            RESULTS[f'X17:n{n_tr}:{tag}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rot_deg_mean=float(r.mean()), params=params[tag],
                per_seed=[float(x) for x in a])
            for ref_name, rr in (('dense', ref[n_tr]['ours']),
                                 ('gn', ref[n_tr]['gn']),
                                 ('full_fourier', ref[n_tr]['full_fourier']),
                                 ('mlp_psi', ref[n_tr]['relaxed_dense'])):
                if len(a) != len(rr) or len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr[:len(a)])
                RESULTS[f'X17:paired:n{n_tr}:{tag}_vs_{ref_name}'] = dict(
                    diff=float(a.mean() - rr[:len(a)].mean()),
                    t=float(t), p=float(p))
    RESULTS['X17:params'] = params
    RESULTS['X17:rank_frozen'] = RANK
    RESULTS['X17:refs'] = {f'n{n}:{k}': [float(x) for x in v]
                           for n in sizes for k, v in ref[n].items()}
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
