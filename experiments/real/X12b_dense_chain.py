#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X12b_dense_chain.py — the faithful reading of the chain error law: PER-JOINT
sources plus an upstream pass-through (the error of joint j depends on the
joint itself, its position and external factors; the error of joint j+1 depends
on the same PLUS the error of joint j).

    eps_j = c_j + A phi(theta_j) + C eps_{j-1},   eps_0 = 0,
    phi = {1, sin theta_j, cos theta_j},

with c_j free per joint (n x 6, the dense model), A shared (6x3, optional),
and the pass-through C in {0, scalar beta, diag, full}. Init: c=0, A small
random, C=0. This family NESTS the dense const model (A=0, C=0), so the test
is whether the chain coupling adds anything at small N.

Models: dense (54p, reference from the bench), dc_s (55p), dc_d (60p),
dc_f (90p), dcA_s (73p). References GN/PoE from the X7_bench_main dump.

Registered predictions (plan 2026-09-29, amendment 30.09):
  P-R6: dc_s <= dense at N=20 (paired over seeds); coupling helps at matched budget;
  P-R7: the learned pass-through does not collapse to zero;
  P-R8: dcA_s <= dense at N=20 and within noise at N=80 (per-joint source +
        shared position law + coupling);
  P-R9 (boundary): if no variant beats dense at N=20, the chain coupling does
        not help on this data.

Protocol: X7 splits rng(1000+seed), seeds 0-4, N=20/40/60/80, E34.train_recipe
(4000 epochs, lr 5e-3, cosine2, clip=0, ang_weight=1), position metric.

Output: repro/json/X12b_dense_chain.<ts>.json via repro/dump.
Run:  python X12b_dense_chain.py [--quick]
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


class DenseChainArm(nn.Module):
    def __init__(self, C, E, coupling='none', use_A=False):
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
        self.coupling = coupling
        self.use_A = bool(use_A)
        assert coupling in ('none', 'scalar', 'diag', 'full')
        self.c = nn.Parameter(torch.zeros(n, 6, dtype=DT))
        if self.use_A:
            self.A = nn.Parameter(torch.randn(6, 3, dtype=DT) * 0.01)
        if coupling == 'scalar':
            self.beta = nn.Parameter(torch.zeros(1, dtype=DT))
        elif coupling == 'diag':
            self.beta = nn.Parameter(torch.zeros(6, dtype=DT))
        elif coupling == 'full':
            self.Cm = nn.Parameter(torch.zeros(6, 6, dtype=DT))
        self.base_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.base_t = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_t = nn.Parameter(torch.zeros(3, dtype=DT))

    def errors(self, th):
        B = th.shape[0]
        s = self.c.unsqueeze(0).expand(B, -1, -1).clone()
        if self.use_A:
            phi = torch.stack([torch.ones_like(th), torch.sin(th),
                               torch.cos(th)], -1)
            s = s + phi @ self.A.t()
        e = torch.zeros(B, self.n, 6, dtype=DT)
        prev = torch.zeros(B, 6, dtype=DT)
        for j in range(self.n):
            u = s[:, j]
            if self.coupling == 'scalar':
                u = u + self.beta[0] * prev
            elif self.coupling == 'diag':
                u = u + self.beta * prev
            elif self.coupling == 'full':
                u = u + prev @ self.Cm.t()
            prev = u
            e[:, j] = u
        return e

    def forward(self, q):
        B = q.shape[0]
        xi = self.errors(q)
        qr, tr = E34.screw_exp(xi)
        z3 = torch.zeros(B, 3, dtype=DT)
        pose = (E34.quat_exp(self.base_r).expand(B, 4), self.base_t.expand(B, 3))
        for i in range(self.n):
            pose = X7.compose(pose, (self.C_q[i].expand(B, 4), self.C_t[i].expand(B, 3)))
            pose = X7.compose(pose, (qr[:, i], tr[:, i]))
            pose = X7.compose(pose, (X7.rotz_quat(q[:, i]), z3))
        pose = X7.compose(pose, (self.E_q.expand(B, 4), self.E_t.expand(B, 3)))
        pose = X7.compose(pose, (E34.quat_exp(self.tool_r).expand(B, 4),
                                 self.tool_t.expand(B, 3)))
        return pose

    def n_params(self):
        p = 12 + 6 * self.n
        if self.use_A:
            p += 18
        p += {'none': 0, 'scalar': 1, 'diag': 6, 'full': 36}[self.coupling]
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
    models = [('dc_s', 'scalar', False), ('dc_d', 'diag', False),
              ('dc_f', 'full', False), ('dcA_s', 'scalar', True)]
    params = {k: DenseChainArm(C, E, coupling=c, use_A=a).n_params()
              for k, c, a in models}
    params['dense'] = 54
    print('[X12b] params:', params, flush=True)

    res = {(n, k): [] for n in sizes for k, _, _ in models}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            line = []
            for k, cpl, use_a in models:
                m = DenseChainArm(C, E, coupling=cpl, use_A=use_a)
                m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                        args.lr, s, clip=0.0, ang_weight=1.0,
                                        warm_epochs=0, loss_form='cosine2')
                res[(n_tr, k)].append(GN.eval_pos(m, q_te, oq_te, ot_te))
                line.append(f'{k} {res[(n_tr, k)][-1][0]:.3f}')
            print(f'[X12b seed {s} n={n_tr}] ' + ' | '.join(line), flush=True)
        print(f'[X12b] seed {s} done', flush=True)

    for n_tr in sizes:
        a_dense = ref[n_tr]['ours']
        for k, _, _ in models:
            a = np.array([v[0] for v in res[(n_tr, k)]])
            r = np.array([v[1] for v in res[(n_tr, k)]])
            RESULTS[f'X12b:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rot_deg_mean=float(r.mean()), params=params[k],
                per_seed=[float(x) for x in a])
            for ref_name, rr in (('dense', a_dense), ('gn', ref[n_tr]['gn']),
                                 ('poe', ref[n_tr]['poe'])):
                if len(a) != len(rr) or len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr)
                RESULTS[f'X12b:paired:n{n_tr}:{k}_vs_{ref_name}'] = dict(
                    diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
    RESULTS['X12b:params'] = params
    RESULTS['X12b:refs'] = {f'n{n}:{k}': [float(x) for x in v]
                            for n in sizes for k, v in ref[n].items()}
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
