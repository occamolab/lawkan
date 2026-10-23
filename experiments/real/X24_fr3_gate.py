#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X24_fr3_gate.py — does the validation-gated checkpoint (the Kinova remedy, X23)
change the FR3 isolated law correction? (registered before run)

On Kinova the isolated psi collapses on some seeds because the socket objective
has a degenerate point-collapse minimum (X23). FR3 uses a position+orientation
objective and its isolated correction is stable across seeds, so the gate is
expected to be neutral at N>=60 and to help only the N=20 overfit. We test this
on the canonical FR3 protocol (X7 splits rng(1000+seed), seeds 0-4,
N=20/40/60/80, Fourier psi, isolated on the frozen exact dense edge).

Registered predictions (plan 2026-10-13, before the run):
  P-FRG1: at N>=60 the gated correction is at parity with the base correction
          (paired, no significant difference);
  P-FRG2: at N=20 the gate shrinks the overfit (base-vs-exact harm) and the
          across-seed std;
  P-FRG3 (boundary): if the gate improves N>=60 significantly, it becomes the
          preferred protocol on FR3 too; if it degrades N>=60, it is not.
  Falsifier: gate worse than base at N>=60 (paired p<0.05) -> do not adopt.

Output: repro/json/X24_fr3_gate.<ts>.json via repro/dump.
Run:  python X24_fr3_gate.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X14_fourier_law as X14               # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def build_fourier(C, E, seed):
    torch.manual_seed(seed)
    return X14.FourierLawArm(C, E, edge_dense=True)


def phase1_exact(C, E, q_tr, oq_tr, ot_tr, epochs, lr, seed):
    m = build_fourier(C, E, seed)
    for p in m.psi.parameters():
        p.requires_grad_(False)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    return m


def phase2_base(exact, q_tr, oq_tr, ot_tr, epochs, lr, seed):
    m = copy.deepcopy(exact)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    return m


def phase2_gated(exact, q_tr, oq_tr, ot_tr, epochs, lr, seed, val_frac=0.2,
                 every=25):
    """Validation-gated phase 2: 80/20 split of the *train* poses, keep the
    best val-position state (no test leakage)."""
    rng = np.random.default_rng(500 + seed)
    n = len(q_tr)
    perm = rng.permutation(n)
    k = int((1 - val_frac) * n)
    tr, va = perm[:k], perm[k:]
    m = copy.deepcopy(exact)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    qtr, oqtr, ottr = q_tr[tr], oq_tr[tr], ot_tr[tr]
    best = (1e9, None, -1)
    for ep in range(epochs):
        opt.zero_grad()
        q_pred, t_pred = m(qtr)
        dp = (t_pred - ottr).norm(dim=-1)
        dot = (q_pred * oqtr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % every == 0 or ep == epochs - 1:
            v = GN.eval_pos(m, q_tr[va], oq_tr[va], ot_tr[va])[0]
            if v < best[0]:
                best = (v, copy.deepcopy(m.state_dict()), ep)
    if best[1] is not None:
        m.load_state_dict(best[1])
    return m, best[2]


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

    keys = ('exact', 'base', 'gated')
    res = {n: {k: [] for k in keys} for n in sizes}
    ckpt = {n: [] for n in sizes}

    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            ex = phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            base = phase2_base(ex, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            gat, step = phase2_gated(ex, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            res[n_tr]['exact'].append(GN.eval_pos(ex, q_te, oq_te, ot_te)[0])
            res[n_tr]['base'].append(GN.eval_pos(base, q_te, oq_te, ot_te)[0])
            res[n_tr]['gated'].append(GN.eval_pos(gat, q_te, oq_te, ot_te)[0])
            ckpt[n_tr].append(int(step))
            print(f'[X24 seed {s} n={n_tr}] exact {res[n_tr]["exact"][-1]:.3f} '
                  f'base {res[n_tr]["base"][-1]:.3f} gated {res[n_tr]["gated"][-1]:.3f} '
                  f'(ckpt {step})', flush=True)
        print(f'[X24] seed {s} done', flush=True)

    for n_tr in sizes:
        for k in keys:
            a = np.array(res[n_tr][k])
            RESULTS[f'X24:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        for k in ('base', 'gated'):
            a = np.array(res[n_tr][k]); b = np.array(res[n_tr]['exact'])
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'X24:paired:n{n_tr}:{k}_vs_exact'] = dict(
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        a = np.array(res[n_tr]['gated']); b = np.array(res[n_tr]['base'])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'X24:paired:n{n_tr}:gated_vs_base'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    RESULTS['X24:ckpt_step'] = {str(n): ckpt[n] for n in sizes}
    RESULTS['X24:protocol'] = dict(seeds=args.seeds, epochs=args.epochs, lr=args.lr,
                                   split='X7 rng(1000+seed)', psi='fourier',
                                   val='80/20 of train poses', sizes=sizes)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
