#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X46_convergence.py — convergence controls: (A1) MLP budget/size on FR3, (B1) ABB
borderline cells at 4x budget, (B2) Kinova socket collapse at 4x budget.

Motivation: the MLP opponent is trained a fixed 4000 epochs (undertrained?); why
such a large MLP; and the borderline crossplat cells may be budget-limited.
Registered in the plan (2026-10-17) before the run.
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


def _find_repo(start):
    d = start
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, 'repro')):
        d = os.path.dirname(d)
    return d


_REPO = _find_repo(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_DATA = os.path.join(_REPO, 'data', 'calib_data.pickle')
if not os.path.exists(_DATA):
    _DATA = os.path.join(_HERE, 'data', 'calib_data.pickle')

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X14_fourier_law as X14               # noqa: E402
import X24_fr3_gate as X24                  # noqa: E402
import X33_fr3_twoslot as X33               # noqa: E402
import X34_fr3_kernel as X34                # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import RC1_robotcali as RC1                 # noqa: E402
import X36_recount_socket as X36            # noqa: E402
import X32_kernel as X32                    # noqa: E402
import MK1_mukca as MK                      # noqa: E402
import X10_fk_calib_real as X10             # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}

CKPTS = (100, 250, 500, 1000, 2000, 4000, 8000)
CK = CKPTS


def train_ckpt(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr, seed,
               freeze_all=False, slots=(), l2_global=0.0, schedule_epochs=None):
    if freeze_all:
        for p in m.parameters():
            p.requires_grad_(False)
        if 'kernel' in slots:
            m.psi.Wk.requires_grad_(True)
        if 'global' in slots:
            m.psi.global_lin.weight.requires_grad_(True)
            m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(schedule_epochs or epochs), eta_min=0.0)
    out = {}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if l2_global:
            loss = loss + l2_global * (m.psi.global_lin.weight ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
        if ep in CK:
            out[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
    return out


T0 = time.time()


def _progress(tag, data=None):
    import json as _j
    d = dict(data if data is not None else RESULTS)
    if not any(k.startswith(('A1:', 'X46:')) for k in d):
        return
    d['_progress'] = dict(tag=tag, wall_s=time.time() - T0)
    _j.dump(d, open(os.path.join(_REPO, 'repro', 'json', 'X46_progress.json'), 'w'),
            indent=1)


def run_A1(sizes, seeds, base_epochs, base_lr, epochs, out, schedule_epochs=None):
    from scipy.cluster.vq import kmeans2
    q_np, obs_np = X7.load_philnad_dataset(os.path.join(_HERE, 'data', 'calib_data.pickle'))
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    DOSE = {20: 3e-3, 40: 3e-4, 60: 1e-4, 80: 0.0}
    import json as _json
    _canon = _json.load(open(os.path.join(_REPO, 'repro', 'canonical.json')))
    x22 = _json.loads(open(os.path.join(_REPO, 'repro', 'json',
                       _canon['x22']['file'])).read())
    cfg = {N: x22[f'X22:cfg:n{N}']['selected'] for N in sizes}
    models = ('mlp', 'two_slot')
    res = {(N, k): {ep: [] for ep in CK} for N in sizes for k in models}
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, base_epochs, base_lr, s)
            feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
            cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=s)
            centers = torch.tensor(cen, dtype=DT)
            dat = torch.cdist(centers, centers)
            bw = float(torch.median(dat[dat > 0]))
            ts = X33.make_two_slot(exact, centers, bw)
            r_ts = train_ckpt(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs,
                              base_lr, s, freeze_all=True,
                              slots=('global', 'kernel'), l2_global=DOSE.get(N, 0.0),
                              schedule_epochs=schedule_epochs)
            h_sel, lr_sel = cfg[N][s]
            torch.manual_seed(s)
            mlp = X7.MLPResidual(C, E, hidden=int(h_sel))
            r_mlp = train_ckpt(mlp, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs,
                               float(lr_sel), s, schedule_epochs=schedule_epochs)
            for ep in CK:
                res[(N, 'two_slot')][ep].append(r_ts[ep])
                res[(N, 'mlp')][ep].append(r_mlp[ep])
            last = CK[-1]
            print(f'[A1 seed {s} N={N}] ts {r_ts[last]:.3f} '
                  f'mlp(h={int(h_sel)},lr={lr_sel}) {r_mlp[last]:.3f} | ts '
                  + ' '.join(f'{e}:{r_ts[e]:.3f}' for e in CK)
                  + ' | mlp ' + ' '.join(f'{e}:{r_mlp[e]:.3f}' for e in CK),
                  flush=True)
            _progress(f'A1 seed {s} N={N}',
                      {f'A1:N{N}:{k}:ep{ep}': res[(N, k)][ep]
                       for k in models for ep in CK})
    for N in sizes:
        for k in models:
            for ep in CK:
                a = np.array(res[(N, k)][ep])
                RESULTS[f'X46:A1:N{N}:{k}:ep{ep}'] = dict(
                    pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                    per_seed=[float(v) for v in a])


def run_B1(sizes, seeds, steps, lr, out):
    robot, xi, M0 = RC1.build_robot()
    q_all, L_all = RC1.load_robotcali(RC1.DATA)
    Tf = RC1.X10.fk_np(robot, q_all)
    p_nom = Tf[:, :3, 3]
    res = {(N, k): [] for N in sizes for k in ('exact', 'psi')}
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(q_all))
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            a0 = RC1.anchor_init(p_nom[tr], L_all[tr])
            ex = RC1.RangeScrew(xi, M0, a0)
            RC1.fit_adam(ex, q_all[tr], L_all[tr], steps, lr, s)
            res[(N, 'exact')].append(RC1.eval_dL_range(ex, q_all[te], L_all[te]))
            st = copy.deepcopy(ex.state_dict())
            ps = RC1.RangeScrew(xi, M0, a0, relax=True)
            ps.load_state_dict(st, strict=False)
            for p in ps.parameters():
                p.requires_grad_(False)
            for p in ps.sc.psi.parameters():
                p.requires_grad_(True)
            RC1.fit_adam(ps, q_all[tr], L_all[tr], steps, lr, s)
            res[(N, 'psi')].append(RC1.eval_dL_range(ps, q_all[te], L_all[te]))
            print(f'[B1 seed {s} N={N}] exact {res[(N,"exact")][-1]:.4f} '
                  f'psi {res[(N,"psi")][-1]:.4f}', flush=True)
            _progress(f'B1 seed {s} N={N}')
    for N in sizes:
        for k in ('exact', 'psi'):
            a = np.array(res[(N, k)])
            RESULTS[f'X46:B1:N{N}:{k}'] = dict(
                dL_mm_mean=float(a.mean()), dL_mm_std=float(a.std()),
                per_seed=[float(v) for v in a])


def run_B2(seeds, edge_steps, edge_lr, steps, lr):
    xi, M0 = MK.platform_tensors(MK.PLATFORMS['kinova'])
    q0, q1 = MK.load_sockets('kinova')
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
        n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
        q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
        q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]
        eb = X10.ScrewCalib(xi, M0, mode='dense')
        eb.dxi.requires_grad_(False)
        MK.fit(eb, q0_tr, q1_tr, edge_steps, edge_lr, s)
        st = copy.deepcopy(eb.state_dict())
        mlp = X32.build_iso(xi, M0, st, s)
        X32.fit(mlp, q0_tr, q1_tr, steps, lr, s)
        v = MK.evaluate(mlp, q0_tr, q1_tr, q0_te, q1_te)[0]
        vals.append(v)
        print(f'[B2 seed {s}] global-head {v:.4f}', flush=True)
        _progress(f'B2 seed {s}')
    RESULTS['X46:B2:kinova:global'] = dict(
        rms_mm_mean=float(np.mean(vals)), rms_mm_std=float(np.std(vals)),
        per_seed=[float(v) for v in vals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='20,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--base-epochs', type=int, default=4000)
    ap.add_argument('--base-lr', type=float, default=5e-3)
    ap.add_argument('--epochs', type=int, default=8000)
    ap.add_argument('--schedule-epochs', type=int, default=0,
                    help='cosine T_max (0 = same as --epochs)')
    ap.add_argument('--ckpt-step', type=int, default=0,
                    help='fine checkpoint grid step (0 = the default CKPTS)')
    ap.add_argument('--a1-only', action='store_true')
    ap.add_argument('--skip-a1', action='store_true')
    ap.add_argument('--abb-steps', type=int, default=12000)
    ap.add_argument('--abb-sizes', default='80,320')
    ap.add_argument('--abb-seeds', type=int, default=10)
    ap.add_argument('--sock-steps', type=int, default=12000)
    ap.add_argument('--sock-seeds', type=int, default=10)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.epochs = '20', 1, 300
        args.abb_steps, args.abb_sizes, args.abb_seeds = 500, '80', 1
        args.sock_steps, args.sock_seeds = 500, 1
    global CK
    if args.ckpt_step:
        CK = list(range(args.ckpt_step, args.epochs + 1, args.ckpt_step))
    else:
        CK = sorted(set([c for c in CKPTS if c <= args.epochs] + [args.epochs]))
    t0 = time.time()
    sizes = [int(s) for s in args.sizes.split(',')]
    sched = args.schedule_epochs or args.epochs
    if not args.skip_a1:
        run_A1(sizes, args.seeds, args.base_epochs, args.base_lr, args.epochs, None,
               schedule_epochs=sched)
        print('[A1 done]', flush=True)
    if not args.a1_only:
        run_B1([int(s) for s in args.abb_sizes.split(',')], args.abb_seeds,
               args.abb_steps, 1e-2, None)
        print('[B1 done]', flush=True)
        run_B2(args.sock_seeds, 3000, 1e-3, args.sock_steps, 1e-3)
        print('[B2 done]', flush=True)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
