#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X26_fullbound.py — bound ALL components of psi (not only rotation) + a
rotation/translation decomposition of the collapse (registered before run).

X25 (seed 8) refuted the rotation-only chart: the crash is an unbounded *linear*
head extrapolating past the training support of f(A),f(B), not the exp winding.
So the bound must cover all six se(3) components. Variants (Kinova, MK1 protocol):
  base    : stock isolated psi (unbounded);
  tanh    : effective correction bounded per component, eff_i -> b_i*tanh(eff_i/b_i),
            eff = psi_gain*psi(f); registered thresholds b_rot = 0.02 rad (~1.1 deg),
            b_trans = 0.005 m (5 mm); physical: these manipulators' non-rigid
            deformations are mm / sub-degree, while the collapse effect is learned;
  tanh_l2 : tanh + L2 on psi weights, lambda = 1e-2.

Diagnostic (base): split the psi effect into rotation and translation shares by
zeroing omega vs zeroing v.

Registered predictions (plan 2026-10-13, before run):
  P-FB1: tanh removes the collapse (0 collapsed) without degrading healthy seeds;
  P-FB2: tanh_l2 ~ tanh (L2 not needed);
  P-FB3 (boundary): neither fixes -> not a psi-magnitude problem (-> X23 gate).
  Falsifier: tanh worse than base on healthy seeds (paired p<0.05).
  Diagnostic: "translation carries ~X% of the psi effect at the collapse".

Output: repro/json/X26_fullbound.<ts>.json.
Run:  python X26_fullbound.py [--quick]
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
GAIN = 0.1


class FullBounded(nn.Module):
    """Wrap psi so the *effective* correction eff = gain*psi(f) is bounded per
    component: eff_i -> b_i*tanh(eff_i/b_i); returns eff/gain. Records, over the
    last forward calls, max|eff|/b per component (saturation) and max|eff|."""
    def __init__(self, inner, b):
        super().__init__()
        self.inner = inner
        self.register_buffer('b', torch.tensor(b, dtype=DT))
        self.register_buffer('sat', torch.zeros(6, dtype=DT))
        self.register_buffer('effmax', torch.zeros(6, dtype=DT))

    def reset_stats(self):
        self.sat.zero_()
        self.effmax.zero_()

    def forward(self, f):
        eff = GAIN * self.inner(f)
        with torch.no_grad():
            self.sat.copy_(torch.maximum(self.sat, (eff.abs() / self.b).amax(0)))
            self.effmax.copy_(torch.maximum(self.effmax, eff.abs().amax(0)))
        return (self.b / GAIN) * torch.tanh(eff / self.b)


class Recorder(nn.Module):
    """Passthrough that records max|eff| per component (for the base model)."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.register_buffer('effmax', torch.zeros(6, dtype=DT))

    def reset_stats(self):
        self.effmax.zero_()

    def forward(self, f):
        eff = GAIN * self.inner(f)
        with torch.no_grad():
            self.effmax.copy_(torch.maximum(self.effmax, eff.abs().amax(0)))
        return self.inner(f)


class Masked(nn.Module):
    """Zero selected components of psi's output (for the decomposition)."""
    def __init__(self, inner, mask):
        super().__init__()
        self.inner = inner
        self.register_buffer('mask', torch.tensor(mask, dtype=DT))

    def forward(self, f):
        return self.inner(f) * self.mask


def build_iso(xi, M0, edge_state, seed, wrap=None):
    torch.manual_seed(seed)
    iso = S2.ScrewCalibRelax(xi, M0, mode='dense')
    iso.load_state_dict(copy.deepcopy(edge_state), strict=False)
    if wrap is not None:
        iso.psi = wrap(iso.psi)
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
def disp(m, q):
    return m(torch.tensor(q, dtype=DT))[:, :3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-list', default='0,1,2,8,9')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--b-rot', type=float, default=0.05)
    ap.add_argument('--b-trans', type=float, default=0.01)
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seed_list, args.steps = '0,8', 300
    t0 = time.time()

    xi, M0 = MK.platform_tensors(MK.PLATFORMS[PLAT])
    q0, q1 = MK.load_sockets(PLAT)
    seeds = [int(x) for x in args.seed_list.split(',')]
    b = [args.b_rot] * 3 + [args.b_trans] * 3

    variants = ('base', 'tanh', 'tanh_l2')
    rms = {v: [] for v in ('edge_b',) + variants}
    share = {v: [] for v in ('rot', 'trans')}
    allstats = {v: [] for v in variants}

    for s in seeds:
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

        # train variants
        m_base = None
        sstats = {}
        for v, wrap, l2 in (
                ('base', lambda inner: Recorder(inner), 0.0),
                ('tanh', lambda inner: FullBounded(inner, b), 0.0),
                ('tanh_l2', lambda inner: FullBounded(inner, b), args.l2)):
            m = build_iso(xi, M0, edge_state, s, wrap=wrap)
            m.psi.reset_stats()
            fit(m, q0_tr, q1_tr, args.steps, args.lr, s, l2=l2)
            m.psi.reset_stats()
            rms[v].append(MK.evaluate(m, q0_tr, q1_tr, q0_te, q1_te)[0])
            with torch.no_grad():
                m(torch.tensor(q0_te, dtype=DT))
                m(torch.tensor(q1_te, dtype=DT))
            effmax = m.psi.effmax.cpu().numpy()
            sstats[v] = dict(eff_rot_max=float(effmax[:3].max()),
                            eff_trans_max=float(effmax[3:].max()))
            if v == 'base':
                m_base = m
            if hasattr(m.psi, 'sat'):
                sstats[v]['sat_max'] = float(m.psi.sat.cpu().numpy().max())

        # decomposition on the base model: zero omega (keep v) vs zero v (keep omega)
        m_wonly = copy.deepcopy(m_base)
        m_wonly.psi = Masked(m_wonly.psi, [1, 1, 1, 0, 0, 0])
        m_vonly = copy.deepcopy(m_base)
        m_vonly.psi = Masked(m_vonly.psi, [0, 0, 0, 1, 1, 1])
        with torch.no_grad():
            d_full0 = disp(m_base, q0_te) - disp(eb, q0_te)
            d_full1 = disp(m_base, q1_te) - disp(eb, q1_te)
            dw0 = disp(m_wonly, q0_te) - disp(eb, q0_te)
            dw1 = disp(m_wonly, q1_te) - disp(eb, q1_te)
            dv0 = disp(m_vonly, q0_te) - disp(eb, q0_te)
            dv1 = disp(m_vonly, q1_te) - disp(eb, q1_te)
            r_rot = float(0.5 * ((d_full0 - dw0).norm(dim=-1).mean()
                                 + (d_full1 - dw1).norm(dim=-1).mean()))
            r_v = float(0.5 * ((d_full0 - dv0).norm(dim=-1).mean()
                               + (d_full1 - dv1).norm(dim=-1).mean()))
        share['rot'].append(r_rot)
        share['trans'].append(r_v)
        tot = r_rot + r_v + 1e-12
        for v in variants:
            allstats[v].append(sstats[v])
        print(f'[X26 seed {s}] edge_b {rms["edge_b"][-1]:.3f} | '
              f'base {rms["base"][-1]:.3f} tanh {rms["tanh"][-1]:.3f} '
              f'tanh_l2 {rms["tanh_l2"][-1]:.3f} | share trans '
              f'{100 * r_v / tot:.0f}% | base eff_trans_max '
              f'{sstats["base"]["eff_trans_max"]*1000:.2f}mm eff_rot_max '
              f'{sstats["base"]["eff_rot_max"]*1000:.2f}mrad | tanh sat_max '
              f'{sstats["tanh"].get("sat_max", float("nan")):.2f}', flush=True)

    for k, a in rms.items():
        a = np.array(a)
        RESULTS[f'X26:{k}:rms_mm'] = dict(mean=float(a.mean()), std=float(a.std()),
                                          per_seed=[float(x) for x in a])
    edge = np.array(rms['edge_b'])
    for v in variants:
        a = np.array(rms[v])
        t, p = stats.ttest_rel(a, edge)
        RESULTS[f'X26:paired:{v}_vs_edge_b'] = dict(
            diff=float(a.mean() - edge.mean()), t=float(t), p=float(p),
            n_collapsed=int((a > edge).sum()))
    for v in ('tanh', 'tanh_l2'):
        a = np.array(rms[v]); bb = np.array(rms['base'])
        t, p = stats.ttest_rel(a, bb)
        RESULTS[f'X26:paired:{v}_vs_base'] = dict(
            diff=float(a.mean() - bb.mean()), t=float(t), p=float(p))
    tot = np.array(share['trans']) + np.array(share['rot'])
    RESULTS['X26:share_trans'] = dict(
        per_seed=[float(x) for x in (np.array(share['trans']) / tot)],
        mean=float(np.mean(np.array(share['trans']) / tot)))
    RESULTS['X26:stats'] = {v: allstats[v] for v in variants}
    RESULTS['X26:protocol'] = dict(platform=PLAT, seeds=seeds, steps=args.steps,
                                   lr=args.lr, b_rot=args.b_rot, b_trans=args.b_trans,
                                   l2=args.l2, gain=GAIN,
                                   collapse='iso_rms > edge_b_rms')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
