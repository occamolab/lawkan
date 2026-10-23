#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X19_tau_psi.py — P-TAU: a physics-informed correction on the FR3.

Instead of the agnostic pair-based psi (186/342 parameters), the correction is a
per-joint gravity deflection
    Delta q_j(q) = alpha_j * tau_j(q) + h(tau_j(q)),
where tau(q) is the nominal gravity torque (RNEA with the standard Panda inertial
parameters; DH.Panda is kinematically identical to the URDF chain used by X7),
computed deterministically from the encoders.  h is one shared scalar MLP with a
zero-initialized output, alpha the per-joint gains: ~56 parameters, zero at init.

Protocol: identical to the FR3 isolated-psi runs (G2bc): X7 splits rng(1000+seed),
sizes 20/40/60/80, seeds 0-4, epochs 4000, lr 5e-3; the exact dense edge is trained
first and frozen, then only psi trains.  Generic references are read from the
G2bc random-split dump (G2f:fourier:*).

Registered predictions (plan 2026-10-05):
  P-TAU1: physics ~ generic (Fourier) at N in {60, 80}, TOST 10%;
  P-TAU2: physics better than generic at N=20;
  P-TAU3: the response h(tau) is odd and monotone;
  P-TAU4: gain magnitude ordering across joints (qualitative).

Run:  python X19_tau_psi.py [--quick]
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import X13_learned_node as X13              # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
from repro.dump import save_dump            # noqa: E402
from repro.canon import load as load_canon  # noqa: E402

DT = torch.float64
RESULTS = {}


def gravity_torque(q_row):
    """Nominal gravity torque (N m) for one configuration; cached."""
    key = tuple(np.round(np.asarray(q_row, dtype=np.float64), 9))
    if key not in _TAU_CACHE:
        import roboticstoolbox as rtb
        pd = _PANDA_DH[0]
        _TAU_CACHE[key] = np.asarray(
            pd.rne(np.asarray(key), np.zeros(7), np.zeros(7)),
            dtype=np.float64).copy()
    return _TAU_CACHE[key]


_TAU_CACHE = {}
_PANDA_DH = [None]


def tau_tensor(q_t):
    rows = q_t.detach().cpu().numpy()
    return torch.tensor(np.stack([gravity_torque(r) for r in rows]), dtype=DT)


class TauPsi(nn.Module):
    """Per-joint gravity deflection: Delta q_j = alpha_j tau_j + h(tau_j)."""

    def __init__(self, n, hidden=16):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n, dtype=DT))
        self.h = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1)).double()
        with torch.no_grad():
            self.h[-1].weight.zero_()
            self.h[-1].bias.zero_()

    def delta(self, tau):
        return self.alpha.unsqueeze(0) * tau + self.h(tau.unsqueeze(-1)).squeeze(-1)


class TauArm(X13.LearnedNodeArm):
    """X13 chain with the agnostic psi replaced by the tau-based correction."""

    def __init__(self, C, E, hidden=16):
        super().__init__(C, E, edge_dense=True, hidden=hidden)
        self.psi = TauPsi(self.n, hidden=hidden)

    def R(self, A, B):
        # the correction lives in the joint angles (gravity deflection),
        # the composition itself stays exact
        return X7.compose(A, B)

    def forward(self, q):
        Bt = q.shape[0]
        dq = self.psi.delta(tau_tensor(q))
        qq = q + dq
        z3 = torch.zeros(Bt, 3, dtype=DT)
        pose = (E34.quat_exp(self.base_r).expand(Bt, 4), self.base_t.expand(Bt, 3))
        qr = tr = None
        if self.edge_dense:
            qr, tr = E34.screw_exp(self.xi_c)
        for i in range(self.n):
            step = (self.C_q[i].expand(Bt, 4), self.C_t[i].expand(Bt, 3))
            if self.edge_dense:
                step = X7.compose(step, (qr[i].expand(Bt, 4), tr[i].expand(Bt, 3)))
            step = X7.compose(step, (X7.rotz_quat(qq[:, i]), z3))
            pose = self.R(pose, step)
        pose = self.R(pose, (self.E_q.expand(Bt, 4), self.E_t.expand(Bt, 3)))
        pose = self.R(pose, (E34.quat_exp(self.tool_r).expand(Bt, 4),
                             self.tool_t.expand(Bt, 3)))
        return pose


def two_phase_tau(C, E, q_tr, oq_tr, ot_tr, epochs, lr, seed):
    m = TauArm(C, E)
    for p in m.psi.parameters():
        p.requires_grad_(False)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    exact = copy.deepcopy(m)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    return exact, m


def generic_ref(n):
    d = load_canon('g2bc_fig')
    return np.array(d[f'G2f:fourier:n{n}:isolated']['per_seed'])[:5]


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
        args.sizes, args.seeds, args.epochs = '60', 1, 300
    sizes = [int(s) for s in args.sizes.split(',')]
    t0 = time.time()

    import roboticstoolbox as rtb
    import G2bc_isolated_psi as G2
    _PANDA_DH[0] = rtb.models.DH.Panda()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    tau_all = tau_tensor(q_all)
    print(f'[X19] n={n_total}, tau range per joint: '
          f'{[round(float(tau_all[:, j].abs().max()), 2) for j in range(tau_all.shape[1])]}',
          flush=True)

    res = {(n, k): [] for n in sizes for k in ('exact_own', 'physics')}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            ex, ph = two_phase_tau(C, E, q_all[tr], oq_all[tr], ot_all[tr],
                                   args.epochs, args.lr, s)
            r_ex = G2.GN.eval_pos(ex, q_all[te], oq_all[te], ot_all[te])
            r_ph = G2.GN.eval_pos(ph, q_all[te], oq_all[te], ot_all[te])
            res[(n_tr, 'exact_own')].append(r_ex)
            res[(n_tr, 'physics')].append(r_ph)
            print(f'[X19 seed {s} n={n_tr}] exact {r_ex[0]:.3f} physics {r_ph[0]:.3f}',
                  flush=True)
        print(f'[X19] seed {s} done', flush=True)

    from scipy import stats
    for n_tr in sizes:
        a = np.array([v[0] for v in res[(n_tr, 'physics')]])
        b = np.array([v[0] for v in res[(n_tr, 'exact_own')]])
        g = generic_ref(n_tr)[:len(a)]
        RESULTS[f'X19:n{n_tr}:exact_own'] = dict(mean=float(b.mean()), std=float(b.std()),
                                                 per_seed=[float(x) for x in b])
        RESULTS[f'X19:n{n_tr}:physics'] = dict(mean=float(a.mean()), std=float(a.std()),
                                               per_seed=[float(x) for x in a])
        RESULTS[f'X19:n{n_tr}:generic'] = dict(mean=float(g.mean()), std=float(g.std()),
                                               per_seed=[float(x) for x in g])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'X19:paired:n{n_tr}:physics_vs_exact'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        t, p = stats.ttest_rel(a, g)
        RESULTS[f'X19:paired:n{n_tr}:physics_vs_generic'] = dict(
            diff=float(a.mean() - g.mean()), t=float(t), p=float(p))
        tol = 0.10 * g.mean()
        se = (a - g).std(ddof=1) / np.sqrt(len(a))
        half = stats.t.ppf(0.95, len(a) - 1) * se
        RESULTS[f'X19:TOST10:n{n_tr}'] = dict(
            margin=float(tol), ci_lo=float((a - g).mean() - half),
            ci_hi=float((a - g).mean() + half),
            pass_=bool(abs((a - g).mean()) + half < tol))
        print(f'[X19 n={n_tr}] exact {b.mean():.3f} physics {a.mean():.3f} '
              f'generic {g.mean():.3f} | phys-generic {a.mean()-g.mean():+.3f} '
              f'(TOST {"pass" if RESULTS[f"X19:TOST10:n{n_tr}"]["pass_"] else "fail"})',
              flush=True)

    # form check P-TAU3 + gains P-TAU4 on the last trained model
    with torch.no_grad():
        tg = torch.linspace(-15, 15, 121, dtype=DT)
        y = ph.psi.h(tg.unsqueeze(-1)).squeeze(-1) + ph.psi.alpha[0] * tg
        odd = float((y + y.flip(0)).abs().mean() / y.abs().mean().clamp_min(1e-12))
        dy = y[1:] - y[:-1]
        mono = float((dy >= -1e-6).float().mean())
        RESULTS['X19:form:h_oddness'] = odd
        RESULTS['X19:form:h_monotone_frac'] = mono
        RESULTS['X19:gains:alpha'] = ph.psi.alpha.detach().cpu().numpy().tolist()
        print(f'[X19] form: oddness {odd:.3f}, monotone frac {mono:.2f}; '
              f'alpha {np.round(RESULTS["X19:gains:alpha"], 3).tolist()}', flush=True)

    RESULTS['X19:protocol'] = dict(sizes=sizes, seeds=args.seeds, epochs=args.epochs,
                                   lr=args.lr)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'done {time.time()-t0:.0f}s; dump: {out}')


if __name__ == '__main__':
    main()
