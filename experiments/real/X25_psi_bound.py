#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X25_psi_bound.py — why does the learned law correction carry seed instability,
and is it architecturally fixable? (registered before run)

Hypothesis: psi is an unbounded linear readout applied as exp(psi_gain * psi(f));
exp is periodic in the rotational part, so a large psi makes the correction
non-identifiable (many psi give the same SE(3) element) and lets the optimizer
drift/ collapse. If so, bounding the correction magnitude removes the Kinova
collapse architecturally, without a validation gate.

Diagnostics: per-seed correction magnitude (mean ||psi(f)||), and its relation to
the collapse.

Variants (Kinova socket protocol, MK1: X7-style 70/30 split rng(1000+seed),
seeds 0-9, 3000 steps, lr 1e-3, lam 100):
  base  : stock isolated psi (S2.ScrewCalibRelax);
  bound : psi output squashed per component by wmax*tanh(./wmax) (removes the
          wrap-around / unbounded drift);
  reg   : stock psi + L2 penalty lambda*||psi_params||^2.

Registered predictions (plan 2026-10-13, before the run):
  P-B1: base collapses on 2/10 seeds (seeds 8,9; reproduces MK1) and the
        collapsed seeds have a larger correction magnitude than the healthy ones;
  P-B2: 'chart' removes the collapse (0 collapsed) without degrading the healthy
        seeds (paired vs base on seeds 0-7: no significant difference);
  P-B3 (boundary): if 'chart' does not remove the collapse or raises the floor on
        healthy seeds, the instability is not (only) the unbounded magnitude.
  Falsifier: 'chart' worse than base on healthy seeds (paired p<0.05) -> do not
        adopt the bound.

Output: repro/json/X25_psi_bound.<ts>.json via repro/dump.
Run:  python X25_psi_bound.py [--quick]
"""

import argparse
import copy
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

import MK1_mukca as MK              # noqa: E402
import S2_ur5gtpn_relaxed as S2     # noqa: E402
import X10_fk_calib_real as X10     # noqa: E402
from repro.dump import save_dump    # noqa: E402

DT = torch.float64
RESULTS = {}
PLAT = 'kinova'


class RotationBoundedPsi(nn.Module):
    """Wrap a psi head so the *rotation* part of its 6-vector output has norm
    bounded by wmax < pi (exp is then injective -> a non-periodic correction
    chart), while the translation part stays linear. Zero maps to zero."""
    def __init__(self, inner, wmax=1.5):
        super().__init__()
        self.inner = inner
        self.wmax = float(wmax)

    def forward(self, x):
        y = self.inner(x)
        w, v = y[..., :3], y[..., 3:]
        n = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.tanh(n / self.wmax) * self.wmax / n
        return torch.cat([w * scale, v], dim=-1)


def build_iso(xi, M0, edge_state, seed, bound=None):
    torch.manual_seed(seed)
    iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
    iso.load_state_dict(copy.deepcopy(edge_state), strict=False)
    if bound is not None:
        iso.psi = RotationBoundedPsi(iso.psi, wmax=bound)
    for p in iso.parameters():
        p.requires_grad_(False)
    for p in iso.psi.parameters():
        p.requires_grad_(True)
    return iso


def fit(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    q0t = torch.tensor(q0_tr, dtype=DT)
    q1t = torch.tensor(q1_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        loss = MK.socket_loss(m, q0t, q1t)
        if l2:
            loss = loss + l2 * sum((p ** 2).sum() for p in m.psi.parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


@torch.no_grad()
def psi_effect(m, edge_model, q0, q1):
    """Mean displacement (mm) that psi moves the predicted sockets relative to
    the frozen exact edge --- the correction's measured effect magnitude."""
    q0t = torch.tensor(q0, dtype=DT)
    q1t = torch.tensor(q1, dtype=DT)
    p0 = m(q0t)[:, :3, 3] - edge_model(q0t)[:, :3, 3]
    p1 = m(q1t)[:, :3, 3] - edge_model(q1t)[:, :3, 3]
    return float(0.5 * (p0.norm(dim=-1).mean() + p1.norm(dim=-1).mean())) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed-list', default='0,1,2,8,9')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wmax', type=float, default=1.5)
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 2, 300
    t0 = time.time()

    xi, M0 = MK.platform_tensors(MK.PLATFORMS[PLAT])
    q0, q1 = MK.load_sockets(PLAT)
    print(f'[X25 {PLAT}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}',
          flush=True)

    variants = ('base', 'chart')
    rms = {v: [] for v in ('edge_b',) + variants}
    pnorm = {v: [] for v in variants}

    seed_list = [int(x) for x in args.seed_list.split(',')] if args.seed_list else list(range(args.seeds))
    for s in seed_list:
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        MK.fit(eb, q0_tr, q1_tr, args.steps, args.lr, s)
        rms['edge_b'].append(MK.evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)[0])
        edge_state = copy.deepcopy(eb.state_dict())
        for p in eb.parameters():
            p.requires_grad_(False)

        for v in variants:
            bound = args.wmax if v == 'chart' else None
            l2 = args.l2 if v == 'reg' else 0.0
            m = build_iso(xi, M0, edge_state, s, bound=bound)
            fit(m, q0_tr, q1_tr, args.steps, args.lr, s, l2=l2)
            rms[v].append(MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)[0])
            pnorm[v].append(psi_effect(m, eb, q0_te, q1_te))

        print(f'[X25 seed {s}] edge_b {rms["edge_b"][-1]:.3f} | '
              f'base {rms["base"][-1]:.3f} (effect {pnorm["base"][-1]:.2f} mm) '
              f'chart {rms["chart"][-1]:.3f}', flush=True)

    for k, a in rms.items():
        a = np.array(a)
        RESULTS[f'X25:{k}:rms_mm'] = dict(mean=float(a.mean()), std=float(a.std()),
                                          per_seed=[float(x) for x in a])
    for v in variants:
        RESULTS[f'X25:{v}:psi_effect_mm'] = [float(x) for x in pnorm[v]]
    edge = np.array(rms['edge_b'])
    for v in variants:
        a = np.array(rms[v])
        t, p = stats.ttest_rel(a, edge)
        RESULTS[f'X25:paired:{v}_vs_edge_b'] = dict(
            diff=float(a.mean() - edge.mean()), t=float(t), p=float(p),
            n_collapsed=int((a > edge).sum()))
    for v in ('chart',):
        a = np.array(rms[v]); b = np.array(rms['base'])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'X25:paired:{v}_vs_base'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
    # diagnostic: correction magnitude on collapsed vs healthy seeds (base)
    b = np.array(rms['base']); pn = np.array(pnorm['base'])
    coll = b > edge
    RESULTS['X25:diag:psi_norm_collapsed_mean'] = float(pn[coll].mean()) if coll.any() else None
    RESULTS['X25:diag:psi_norm_healthy_mean'] = float(pn[~coll].mean()) if (~coll).any() else None
    RESULTS['X25:protocol'] = dict(platform=PLAT, seeds=str(seed_list), steps=args.steps,
                                   lr=args.lr, wmax=args.wmax, l2=args.l2,
                                   collapse='iso_rms > edge_b_rms')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
