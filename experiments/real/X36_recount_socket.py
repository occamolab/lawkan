#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X36_recount_socket.py — recount the Table 4 isolated-psi rows (socket MK1 + ABB RC1)
under the final LOCAL KERNEL head (X32). For a small, fixed configuration set the
appendix recommends the kernel head alone (it has no large-N regime to add a global
trend); this script replaces the old S2-MLP psi cell and reports both heads with the
same seeds/splits, so R-1 is compared like-for-like inside the run.

Registered (plan 2026-10-14, R-1): every recounted psi-cell <= the old head's cell
(the kernel removes the Kinova seed collapse; elsewhere it matches or improves).
Boundary P-B5 follow-up: cross-seed spread does not worsen.

Output: repro/json/X36_recount_socket.<ts>.json.
Run:  python X36_recount_socket.py [--quick]
"""

import argparse
import copy
import math
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

import X10_fk_calib_real as X10             # noqa: E402
import S2_ur5gtpn_relaxed as S2             # noqa: E402
import MK1_mukca as MK                       # noqa: E402
import RC1_robotcali as RC1                  # noqa: E402
import X32_kernel as X32                     # noqa: E402
from repro.dump import save_dump             # noqa: E402

torch.set_num_threads(int(os.environ.get('X36_THREADS', '2')))

DT = torch.float64
RESULTS = {}
DOSE_OVERRIDE = None
SOCKET = ['kuka14', 'panda_1', 'fr3', 'kinova']


def collect_feats(psi_mod, run_fn):
    """Capture the psi input features from one forward pass."""
    fs = []

    def hook(module, inp, out):
        fs.append(inp[0].detach().cpu())

    h = psi_mod.register_forward_hook(hook)
    with torch.no_grad():
        run_fn()
    h.remove()
    return torch.cat(fs, 0)


def centers_bw(feats, k, seed):
    from scipy.cluster.vq import kmeans2
    cen, _ = kmeans2(feats.numpy(), k, minit='++', seed=seed)
    centers = torch.tensor(cen, dtype=DT)
    dc = torch.cdist(centers, centers)
    bw = float(torch.median(dc[dc > 0])) if (dc > 0).any() else 1.0
    return centers, bw


class TwoSlotPsiSocket(nn.Module):
    """The socket two-slot head = the platform's global head + a local kernel.

    On the sockets the global head is the same small MLP as the baseline (14->16->6,
    zero-initialized last layer), so the two-slot reduces to the platform head at init
    (plus a zero kernel); the local kernel adds a data-local term. Both zero-init.
    """

    def __init__(self, dim, centers, bw, hidden=16):
        super().__init__()
        self.global_mlp = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, 6)).double()
        with torch.no_grad():
            self.global_mlp[-1].weight.zero_()
            self.global_mlp[-1].bias.zero_()
        self.register_buffer('centers', centers.clone())
        self.bw = float(bw)
        self.Wk = nn.Parameter(torch.zeros(centers.shape[0], 6, dtype=DT))

    def global_parameters(self):
        return list(self.global_mlp.parameters())

    def forward(self, f):
        f = f.reshape(-1, f.shape[-1])
        out = self.global_mlp(f)
        d2 = torch.cdist(f, self.centers) ** 2
        out = out + torch.exp(-d2 / (2 * self.bw ** 2)) @ self.Wk
        return out


def dose_for(n_tr):
    """L2 dose on the global slot by training size (X26c table); --dose-override replaces it."""
    if DOSE_OVERRIDE is not None:
        return DOSE_OVERRIDE
    for thr, d in ((20, 3e-3), (40, 3e-4), (60, 1e-4)):
        if n_tr <= thr:
            return d
    return 0.0


def slot_off_eval(m, q0_tr, q1_tr, q0_te, q1_te, which='kernel'):
    """Probe C: held-out RMS with one slot switched off (kernel or global)."""
    m2 = copy.deepcopy(m)
    if which == 'kernel':
        m2.psi.Wk.data.zero_()
    else:
        with torch.no_grad():
            m2.psi.global_mlp[-1].weight.zero_()
            m2.psi.global_mlp[-1].bias.zero_()
    return MK.evaluate(m2, q0_tr, q1_tr, q0_te, q1_te)[0]


def build_twoslot_boxed(xi, M0, edge_state, seed, centers, bw):
    torch.manual_seed(seed)
    m = S2.ScrewCalibRelax(xi, M0, mode='dense')
    m.load_state_dict(copy.deepcopy(edge_state), strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi = TwoSlotPsiSocket(14, centers, bw)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    return m


def fit_twoslot_socket(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0):
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
            loss = loss + l2 * sum((q ** 2).sum() for q in m.psi.global_parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def fit_twoslot_abb(m, q_tr, L_tr, steps, lr, seed, l2=0.0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    Lt = torch.tensor(L_tr, dtype=DT)
    qt = torch.tensor(q_tr, dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        loss = 10.0 * ((m.range(qt) - Lt) ** 2).mean()
        if l2:
            loss = loss + l2 * sum((q ** 2).sum() for q in m.sc.psi.global_parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def recount_socket(plat, seeds, steps, lr, k, quick, heads=('kernel', 'twoslot'), seq=False):
    xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
    q0, q1 = MK.load_sockets(plat)
    res = {v: [] for v in ('edge_b', 'mlp', 'kernel', 'twoslot', 'koff', 'goff')}
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        MK.fit(eb, q0_tr, q1_tr, steps, lr, s)
        res['edge_b'].append(MK.evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)[0])
        edge_state = copy.deepcopy(eb.state_dict())

        probe = X32.build_iso(xi, M0, edge_state, s)
        feats = collect_feats(probe.psi, lambda: probe(torch.tensor(
            np.concatenate([q0_tr, q1_tr], 0), dtype=DT)))
        assert feats.shape[1] == 14, (plat, 'feature dim', feats.shape)
        centers, bw = centers_bw(feats, k, s)

        mlp = X32.build_iso(xi, M0, edge_state, s)
        X32.fit(mlp, q0_tr, q1_tr, steps, lr, s)
        res['mlp'].append(MK.evaluate(mlp, q0_tr, q1_tr, q0_te, q1_te)[0])

        if 'kernel' in heads:
            kr = X32.build_iso(xi, M0, edge_state, s, kernel=(centers, bw))
            X32.fit(kr, q0_tr, q1_tr, steps, lr, s, l2=0.0)
            res['kernel'].append(MK.evaluate(kr, q0_tr, q1_tr, q0_te, q1_te)[0])
        else:
            res['kernel'].append(float('nan'))

        if 'twoslot' in heads:
            ts = build_twoslot_boxed(xi, M0, edge_state, s, centers, bw)
            if seq:
                # sequential-on-residual: train the kernel first, then the global
                # slot on its residual (avoids the joint global-slot degradation).
                for p in ts.parameters():
                    p.requires_grad_(False)
                ts.psi.Wk.requires_grad_(True)
                fit_twoslot_socket(ts, q0_tr, q1_tr, steps, lr, s, l2=0.0)
                ts.psi.Wk.requires_grad_(False)
                for p in ts.psi.global_parameters():
                    p.requires_grad_(True)
                fit_twoslot_socket(ts, q0_tr, q1_tr, steps, lr, s, l2=0.0)
            else:
                fit_twoslot_socket(ts, q0_tr, q1_tr, steps, lr, s, l2=dose_for(int(n0)))
            res['twoslot'].append(MK.evaluate(ts, q0_tr, q1_tr, q0_te, q1_te)[0])
            res['koff'].append(slot_off_eval(ts, q0_tr, q1_tr, q0_te, q1_te, 'kernel'))
            res['goff'].append(slot_off_eval(ts, q0_tr, q1_tr, q0_te, q1_te, 'global'))
        else:
            res['twoslot'].append(float('nan'))
            res['koff'].append(float('nan'))
            res['goff'].append(float('nan'))

        print(f'[X36 {plat} seed {s}] edge_b {res["edge_b"][-1]:.3f} '
              f'mlp {res["mlp"][-1]:.3f} kernel {res["kernel"][-1]:.3f} '
              f'twoslot {res["twoslot"][-1]:.3f} | koff {res["koff"][-1]:.3f} '
              f'goff {res["goff"][-1]:.3f} mm (bw {bw:.3f}, n0 {n0})', flush=True)

    for v in res:
        a = np.array(res[v])
        RESULTS[f'X36:{plat}:{v}:rms_mm'] = dict(
            mean=float(a.mean()), std=float(a.std()),
            per_seed=[float(x) for x in a])
    for v in ('kernel', 'twoslot'):
        a, b = np.array(res[v]), np.array(res['mlp'])
        if np.isfinite(a).all():
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'X36:{plat}:paired:{v}_vs_mlp'] = dict(
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        RESULTS[f'X36:{plat}:n_collapse_{v}'] = int((np.array(res[v]) > np.array(res['edge_b'])).sum())
    RESULTS[f'X36:{plat}:n_collapse_mlp'] = int((np.array(res['mlp']) > np.array(res['edge_b'])).sum())


def recount_abb(seeds, sizes, steps, lr, k, data, quick, heads=('kernel', 'twoslot')):
    robot, xi, M0 = RC1.build_robot()
    q_all, L_all = RC1.load_robotcali(data)
    Tf_all = X10.fk_np(robot, q_all)
    p_nom, R_nom = Tf_all[:, :3, 3], Tf_all[:, :3, :3]
    res = {(N, v): [] for N in sizes for v in ('exact', 'mlp', 'kernel', 'twoslot')}
    for s in seeds:
        perm = np.random.default_rng(1000 + s).permutation(len(q_all))
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            a0 = RC1.anchor_init(p_nom[tr], L_all[tr])

            ex = RC1.RangeScrew(xi, M0, a0)
            RC1.fit_adam(ex, q_all[tr], L_all[tr], steps, lr, s)
            res[(N, 'exact')].append(RC1.eval_dL_range(ex, q_all[te], L_all[te]))
            ex_state = copy.deepcopy(ex.state_dict())

            probe = RC1.RangeScrew(xi, M0, a0, relax=True)
            probe.load_state_dict(copy.deepcopy(ex_state), strict=False)
            for p in probe.parameters():
                p.requires_grad_(False)
            feats = collect_feats(probe.sc.psi, lambda: probe.range(torch.tensor(q_all[tr], dtype=DT)))
            centers, bw = centers_bw(feats, k, s)

            torch.manual_seed(s)
            mlp = RC1.RangeScrew(xi, M0, a0, relax=True)
            mlp.load_state_dict(copy.deepcopy(ex_state), strict=False)
            for p in mlp.parameters():
                p.requires_grad_(False)
            for p in mlp.sc.psi.parameters():
                p.requires_grad_(True)
            RC1.fit_adam(mlp, q_all[tr], L_all[tr], steps, lr, s)
            res[(N, 'mlp')].append(RC1.eval_dL_range(mlp, q_all[te], L_all[te]))

            if 'kernel' in heads:
                torch.manual_seed(s)
                kr = RC1.RangeScrew(xi, M0, a0, relax=True)
                kr.load_state_dict(copy.deepcopy(ex_state), strict=False)
                for p in kr.parameters():
                    p.requires_grad_(False)
                kr.sc.psi = X32.KernelPsi(centers, bw)
                for p in kr.sc.psi.parameters():
                    p.requires_grad_(True)
                RC1.fit_adam(kr, q_all[tr], L_all[tr], steps, lr, s)
                res[(N, 'kernel')].append(RC1.eval_dL_range(kr, q_all[te], L_all[te]))
            else:
                res[(N, 'kernel')].append(float('nan'))

            if 'twoslot' in heads:
                ts = RC1.RangeScrew(xi, M0, a0, relax=True)
                ts.load_state_dict(copy.deepcopy(ex_state), strict=False)
                for p in ts.parameters():
                    p.requires_grad_(False)
                ts.sc.psi = TwoSlotPsiSocket(14, centers, bw)
                for p in ts.sc.psi.parameters():
                    p.requires_grad_(True)
                fit_twoslot_abb(ts, q_all[tr], L_all[tr], steps, lr, s, l2=dose_for(N))
                res[(N, 'twoslot')].append(RC1.eval_dL_range(ts, q_all[te], L_all[te]))
            else:
                res[(N, 'twoslot')].append(float('nan'))

            print(f'[X36 abb seed {s} N={N}] exact {res[(N,"exact")][-1]:.3f} '
                  f'mlp {res[(N,"mlp")][-1]:.3f} kernel {res[(N,"kernel")][-1]:.3f} '
                  f'twoslot {res[(N,"twoslot")][-1]:.3f} mm (bw {bw:.3f})', flush=True)

    for N in sizes:
        for v in ('exact', 'mlp', 'kernel', 'twoslot'):
            a = np.array(res[(N, v)])
            RESULTS[f'X36:abb:N{N}:{v}:dL_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                per_seed=[float(x) for x in a])
        for v in ('kernel', 'twoslot'):
            a, b = np.array(res[(N, v)]), np.array(res[(N, 'mlp')])
            if np.isfinite(a).all():
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'X36:abb:N{N}:paired:{v}_vs_mlp'] = dict(
                    diff=float(a.mean() - b.mean()), t=float(t), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default=','.join(SOCKET))
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'robotcali', 'RobotCali-new.xlsx'))
    ap.add_argument('--seed-list', default='0,1,2,8,9')
    ap.add_argument('--heads', default='kernel,twoslot',
                    help='variants to run alongside the MLP reference: kernel,twoslot')
    ap.add_argument('--dose-override', type=float, default=float('nan'),
                    help='fixed L2 dose on the global slot (NaN = X26c size table; 0 = no dose)')
    ap.add_argument('--seq', action='store_true',
                    help='train the two slots sequentially (kernel then global on residual)')
    ap.add_argument('--socket-seeds', default='',
                    help='per-platform seeds for kuka14/panda_1/kinova (default: seed-list)')
    ap.add_argument('--fr3-seeds', default='',
                    help='seeds for the fr3-unit socket (default: seed-list)')
    ap.add_argument('--abb-seeds', default='',
                    help='seeds for ABB (default: seed-list)')
    ap.add_argument('--abb-sizes', default='80,160,320')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--abb-lr', type=float, default=1e-2)
    ap.add_argument('--centers', type=int, default=20)
    ap.add_argument('--skip-abb', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    global DOSE_OVERRIDE
    if not math.isnan(args.dose_override):
        DOSE_OVERRIDE = args.dose_override
    seeds = [int(x) for x in args.seed_list.split(',')]
    if args.quick:
        args.platforms = 'kinova'
        seeds, args.steps, args.abb_sizes = [0, 8], 300, '80'

    def _parse(s):
        return [int(x) for x in s.split(',') if s.strip()]

    heads = tuple(h for h in args.heads.split(',') if h)
    t0 = time.time()
    for plat in args.platforms.split(','):
        s = seeds
        if plat == 'fr3' and args.fr3_seeds:
            s = _parse(args.fr3_seeds)
        elif plat in ('kuka14', 'panda_1', 'kinova') and args.socket_seeds:
            s = _parse(args.socket_seeds)
        recount_socket(plat, s, args.steps, args.lr, args.centers, args.quick, heads=heads, seq=args.seq)
    abb_seeds = _parse(args.abb_seeds) if args.abb_seeds else seeds
    if not args.skip_abb and os.path.exists(args.data):
        recount_abb(abb_seeds, [int(x) for x in args.abb_sizes.split(',')],
                    args.steps, args.abb_lr, args.centers, args.data, args.quick, heads=heads)
    else:
        print('[X36] ABB skipped (no data or --skip-abb)', flush=True)

    RESULTS['X36:protocol'] = dict(
        platforms=args.platforms, seeds=seeds, socket_seeds=args.socket_seeds,
        fr3_seeds=args.fr3_seeds, abb_seeds=args.abb_seeds or args.seed_list,
        heads=list(heads), steps=args.steps, centers=args.centers,
        head='MLP (reference) / local kernel (X32) / two-slot, isolated on the frozen edge_b/exact',
        abb_sizes=args.abb_sizes)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
