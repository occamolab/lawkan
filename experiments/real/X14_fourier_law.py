#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X14_fourier_law.py — Fourier-psi (micro-KAN) in the node, real FR3.

Same placement as X13 (R(A,B) = compose(A,B) o exp(psi(f(A),f(B))), shared psi,
zero-initialized output), but psi is a KAN-style additive Fourier readout over
the paper's chart features instead of a generic MLP:

  per pose: (theta, alpha, beta) from the quaternion chart + t / 0.5 m
  basis   : {1, sin x, cos x} per angle; translations linear
  readout : 30 -> 6, last layer zero-init    ->  psi = 186 parameters

Models: relaxed_fourier + dense edge (240p) and nodeonly_fourier (198p).
References: X13 MLP-psi per-seed (same splits/seeds), dense/GN/PoE from the
bench dump.

Registered predictions (plan 2026-09-29, before the run):
  P-F1: at N=80 Fourier-psi ~ MLP-psi (|diff| < 0.1 mm, paired);
  P-F2: at N=20 Fourier-psi beats MLP-psi (less freedom: 150 vs 342 p);
  P-F3: interpretability: theta / axis-of-B features dominate;
  P-F4 (boundary): if Fourier loses at both N=80 and N=20, psi stays MLP.
Decision point after X14: KAN vs MLP for psi (and the LawKAN naming note).

Protocol: identical to X13 (X7 splits rng(1000+seed), 5 seeds, N=20..80,
E34.train_recipe, 4000 epochs, cosine2 w=1, clip=0).

Output: repro/json/X14_fourier_law.<ts>.json via repro/dump.
Run:  python X14_fourier_law.py [--quick]
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


def chart_feats(p, t_scale=0.5, eps=0.01):
    """(quat, t) -> (theta, axis_x, axis_y, axis_z, t/t_scale).

    Axis components with a regularized norm (v / sqrt(|v|^2 + eps^2)): smooth
    everywhere, equal to the unit axis away from v=0, and free of the atan2(0,0)
    singularity that NaN-ed the gradients at init (alpha/beta removed; the
    regularized components carry the same information)."""
    q, t = p
    q = torch.where(q[..., :1] >= 0, q, -q)
    w, v = q[..., :1], q[..., 1:]
    vn = v.norm(dim=-1, keepdim=True)
    theta = 2.0 * torch.atan2(vn, w.clamp_min(1e-12))
    axis = v / torch.sqrt(vn ** 2 + eps ** 2)
    return theta, axis[..., 0:1], axis[..., 1:2], axis[..., 2:3], t / t_scale


class FourierPsi(nn.Module):
    """{1, sin, cos} per angle channel + linear translations -> 6, zero last layer."""

    def __init__(self):
        super().__init__()
        self.out = nn.Linear(30, 6).double()
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, A, B):
        feats = []
        for p in (A, B):
            th, ax, ay, az, ts = chart_feats(p)
            for x in (th, ax, ay, az):
                feats += [torch.ones_like(x), torch.sin(x), torch.cos(x)]
            feats += [ts[..., 0:1], ts[..., 1:2], ts[..., 2:3]]
        return 0.1 * self.out(torch.cat(feats, -1))  # small-correction gain (cf. MLPResidual)


class FourierLawArm(nn.Module):
    def __init__(self, C, E, edge_dense=True):
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
        self.psi = FourierPsi()

    def R(self, A, B):
        base = X7.compose(A, B)
        corr = E34.screw_exp(self.psi(A, B))
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
        p += self.psi.out.weight.numel() + self.psi.out.bias.numel()
        return p


def load_json(pattern, keys=None):
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', pattern))
             if 'smoke' not in p]
    d = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
    return d


def load_ref():
    d = load_json('X7_bench_main.*.json')
    ref = {n: {k: np.array(d[f'X7bench:n{n}:{k}']['per_seed'])[:, 0]
               for k in ('ours', 'gn', 'poe')} for n in (20, 40, 60, 80)}
    x13 = load_json('X13_learned_node.*.json')
    for n in (20, 40, 60, 80):
        for k in ('relaxed_dense', 'nodeonly'):
            ref[n][k] = np.array(x13[f'X13:n{n}:{k}']['per_seed'])
    return ref


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
    models = [('relaxed_fourier', True), ('nodeonly_fourier', False)]
    params = {k: FourierLawArm(C, E, edge_dense=d).n_params() for k, d in models}
    params['psi'] = 186
    params['mlp_psi_x13'] = 342
    print('[X14] params:', params, flush=True)

    res = {(n, k): [] for n in sizes for k, _ in models}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            line = []
            for k, dense in models:
                m = FourierLawArm(C, E, edge_dense=dense)
                m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                        args.lr, s, clip=0.0, ang_weight=1.0,
                                        warm_epochs=0, loss_form='cosine2')
                res[(n_tr, k)].append(GN.eval_pos(m, q_te, oq_te, ot_te))
                tr_pos = GN.eval_pos(m, q_tr, oq_tr, ot_tr)[0]
                line.append(f'{k} {res[(n_tr, k)][-1][0]:.3f} (train {tr_pos:.3f})')
                if k == 'relaxed_fourier':
                    with torch.no_grad():
                        w = m.psi.out.weight.detach().numpy()
                    RESULTS[f'X14:readout:n{n_tr}:seed{s}'] = [
                        float(x) for x in np.abs(w).sum(0)]
            print(f'[X14 seed {s} n={n_tr}] ' + ' | '.join(line), flush=True)
        print(f'[X14] seed {s} done', flush=True)

    for n_tr in sizes:
        for k, _ in models:
            a = np.array([v[0] for v in res[(n_tr, k)]])
            r = np.array([v[1] for v in res[(n_tr, k)]])
            RESULTS[f'X14:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rot_deg_mean=float(r.mean()), params=params[k],
                per_seed=[float(x) for x in a])
            for ref_name, rr in (('dense', ref[n_tr]['ours']),
                                 ('gn', ref[n_tr]['gn']),
                                 ('mlp_psi', ref[n_tr]['relaxed_dense']),
                                 ('nodeonly_mlp', ref[n_tr]['nodeonly'])):
                if len(a) != len(rr) or len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr)
                RESULTS[f'X14:paired:n{n_tr}:{k}_vs_{ref_name}'] = dict(
                    diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
    RESULTS['X14:params'] = params
    RESULTS['X14:refs'] = {f'n{n}:{k}': [float(x) for x in v]
                           for n in sizes for k, v in ref[n].items()}
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
