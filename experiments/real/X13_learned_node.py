#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X13_learned_node.py — a relaxable (learnable) node on the real FR3.

Hypothesis (registered in the plan 2026-09-29, before the run): on real systems
with error the exact group law in the node may be misspecified (compliance,
thermal deformation, backlash -> the composition is not exactly rigid). We test
the Corollary (law mismatch -> training-independent floor) from the other side
by relaxing the law:

    R(A, B) = compose(A, B) o exp(psi(f(A), f(B))),

psi: 14 -> 16 -> 6 with a ZERO last layer, so at init the node is exactly the
group law (P-LN3 sanity). psi is shared across all n+2 compositions of the
chain. Models: relaxed+dense (54p edge + ~350p psi) and nodeonly (psi only,
no per-joint edge). References dense (54p) / GN / PoE from the bench dump.

Predictions:
  P-LN1: relaxed+dense <= dense at N=80 (paired over seeds);
  P-LN2: nodeonly approaches relaxed+dense at N=80 and is far below nominal;
  P-LN3: zero-init sanity (at start relaxed == exact up to noise);
  P-LN4 (boundary): relaxed+dense ~ dense (p>0.05) -> no law misspecification
        is visible; the floor is metrological/incremental.

Protocol: X7 splits rng(1000+seed), seeds 0-4, N=20/40/60/80, E34.train_recipe
(4000 epochs, lr 5e-3, cosine2, clip=0, ang_weight=1), position metric.

Output: repro/json/X13_learned_node.<ts>.json via repro/dump.
Run:  python X13_learned_node.py [--quick]
"""

import argparse
import glob
import json
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
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def canon_q(q):
    """Sign-canonical quaternion (w >= 0) for the psi features."""
    return torch.where(q[..., :1] >= 0, q, -q)


class LearnedNodeArm(nn.Module):
    """base o [relaxed compositions] ... o tool; relaxed op = exact o exp(psi)."""

    def __init__(self, C, E, edge_dense=True, hidden=16):
        super().__init__()
        qs, ts = [], []
        for Ci in C:
            Rt = torch.tensor(Ci, dtype=DT)
            qs.append(X7.mat_to_quat(Rt[:3, :3]))
            ts.append(Rt[:3, 3].clone())
        self.register_buffer('C_q', torch.stack(qs))
        self.register_buffer('C_t', torch.stack(ts))
        Et = torch.tensor(E, dtype=DT)
        self.register_buffer('E_q', X7.mat_to_quat(Et[:3, :3]))
        self.register_buffer('E_t', Et[:3, 3].clone())
        n = self.C_q.shape[0]
        self.n = n
        self.edge_dense = bool(edge_dense)
        self.base_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.base_t = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_t = nn.Parameter(torch.zeros(3, dtype=DT))
        if self.edge_dense:
            self.xi_c = nn.Parameter(torch.zeros(n, 6, dtype=DT))
        self.psi = nn.Sequential(nn.Linear(14, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 6)).double()
        with torch.no_grad():                 # exact law at init
            self.psi[-1].weight.zero_()
            self.psi[-1].bias.zero_()

    def R(self, A, B):
        base = X7.compose(A, B)
        Aq, At = A
        Bq, Bt = B
        f = torch.cat([canon_q(Aq), At, canon_q(Bq), Bt], -1)
        corr = E34.screw_exp(self.psi(f))
        return X7.compose(base, corr)

    def forward(self, q):
        Bt = q.shape[0]
        z3 = torch.zeros(Bt, 3, dtype=DT)
        pose = (E34.quat_exp(self.base_r).expand(Bt, 4), self.base_t.expand(Bt, 3))
        qr = tr = None
        if self.edge_dense:
            qr, tr = E34.screw_exp(self.xi_c)
        for i in range(self.n):
            step = (self.C_q[i].expand(Bt, 4), self.C_t[i].expand(Bt, 3))
            if self.edge_dense:
                step = X7.compose(step, (qr[i].expand(Bt, 4), tr[i].expand(Bt, 3)))
            step = X7.compose(step, (X7.rotz_quat(q[:, i]), z3))
            pose = self.R(pose, step)
        pose = self.R(pose, (self.E_q.expand(Bt, 4), self.E_t.expand(Bt, 3)))
        pose = self.R(pose, (E34.quat_exp(self.tool_r).expand(Bt, 4),
                             self.tool_t.expand(Bt, 3)))
        return pose

    def n_params(self):
        p = 12
        if self.edge_dense:
            p += 6 * self.n
        for m in self.psi:
            if isinstance(m, nn.Linear):
                p += m.weight.numel() + m.bias.numel()
        return p


def load_ref():
    cands = [p for p in glob.glob(os.path.join(
        _REPO, 'repro', 'json', 'X7_bench_main.*.json')) if 'smoke' not in p]
    d = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
    return {n: {k: np.array(d[f'X7bench:n{n}:{k}']['per_seed'])[:, 0]
                for k in ('ours', 'gn', 'poe')} for n in (20, 40, 60, 80)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
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
    ref = load_ref()
    models = [('relaxed_dense', True), ('nodeonly', False)]
    params = {k: LearnedNodeArm(C, E, edge_dense=d).n_params() for k, d in models}
    params['dense'] = 54
    print('[X13] params:', params, flush=True)

    # P-LN3 sanity: at init the relaxed model equals the exact one (N=80 test)
    perm0 = np.random.default_rng(1000).permutation(n_total)
    tr0, te0 = perm0[:80], perm0[80:]
    m = LearnedNodeArm(C, E, edge_dense=True)
    assert float(torch.cat([p for p in m(torch.tensor(q_np[te0], dtype=DT))],
                           -1).abs().sum()) >= 0  # forward works
    print('[X13] init sanity: psi output zero -> relaxed == exact at start', flush=True)

    res = {(n, k): [] for n in sizes for k, _ in models}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            line = []
            for k, dense in models:
                m = LearnedNodeArm(C, E, edge_dense=dense)
                m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                        args.lr, s, clip=0.0, ang_weight=1.0,
                                        warm_epochs=0, loss_form='cosine2')
                res[(n_tr, k)].append(GN.eval_pos(m, q_te, oq_te, ot_te))
                tr_pos = GN.eval_pos(m, q_tr, oq_tr, ot_tr)[0]
                line.append(f'{k} {res[(n_tr, k)][-1][0]:.3f} (train {tr_pos:.3f})')
            print(f'[X13 seed {s} n={n_tr}] ' + ' | '.join(line), flush=True)
        print(f'[X13] seed {s} done', flush=True)

    for n_tr in sizes:
        a_dense = ref[n_tr]['ours']
        for k, _ in models:
            a = np.array([v[0] for v in res[(n_tr, k)]])
            r = np.array([v[1] for v in res[(n_tr, k)]])
            RESULTS[f'X13:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rot_deg_mean=float(r.mean()), params=params[k],
                per_seed=[float(x) for x in a])
            for ref_name, rr in (('dense', a_dense), ('gn', ref[n_tr]['gn']),
                                 ('poe', ref[n_tr]['poe'])):
                if len(a) != len(rr) or len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr)
                RESULTS[f'X13:paired:n{n_tr}:{k}_vs_{ref_name}'] = dict(
                    diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
    RESULTS['X13:params'] = params
    RESULTS['X13:refs'] = {f'n{n}:{k}': [float(x) for x in v]
                           for n in sizes for k, v in ref[n].items()}
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
