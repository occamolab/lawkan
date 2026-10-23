#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
G2bc_isolated_psi.py — isolated psi on FR3 and on the synthetic S1 controls
(frozen exact edge; two-phase training; Fourier and MLP psi variants).

Motivation: the joint training of edge + psi degenerates when the platform has
no real law deformation (S8/UR10, MLP psi: edge drifts, psi compensates; test
1.024 vs exact 0.434 mm; psi off -> 11.7 mm). Isolating the correction fixes it:
fit the exact edge first, freeze it, then train only the zero-initialized psi.

Phase 1: exact dense edge (psi frozen), E34.train_recipe with the model's own
protocol. Phase 2: psi only (everything else frozen), same epochs/lr/seed.

Psi variants:
  fourier  X14.FourierLawArm (186p readout over 30 basis-expanded inputs),
  mlp      X13.LearnedNodeArm (14->16->6, 342p).
FR3 protocol: philnad, X7 splits rng(1000+seed); fourier seeds 0-4 x
N=20/40/60/80 (as X14); mlp pilot seeds 0-2 x N=20/60/80. S1: exact-law and
deformed-law data, seeds 0-2, N=20/80/160, both psi.

Registered predictions (plan 2026-09-29, before the run):
  P-G2b1: FR3, N>=60: isolated beats its own exact edge (paired, p<0.05) for
          the Fourier psi;
  P-G2b2: FR3, N=20: no significant harm (paired vs own exact, p>0.05);
  P-G2c1: S1 exact-law: isolated ~ own exact (no significant harm);
  P-G2c2: S1 deformed-law: isolated recovers (beats own exact) at N>=80;
  P-G2e:  the MLP psi shows the same isolation property (FR3 win kept at N>=60,
          no small-N harm; S1 behaviour as above);
  P-G2d (boundary): if isolation kills the FR3 win or fails to remove the
          small-N/degenerate behavior, the protocol needs a gate instead.

Output: repro/json/G2bc_isolated_psi.<ts>.json via repro/dump.
Run:  python G2bc_isolated_psi.py [--part fr3|s1|both] [--quick]
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
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X10_fk_calib_real as X10             # noqa: E402
import X11_ur5_gtpn_sweep as X11            # noqa: E402
import X13_learned_node as X13              # noqa: E402
import X14_fourier_law as X14               # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import S1_synthetic_law as S1               # noqa: E402
import S2_ur5gtpn_relaxed as S2             # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def make_model(psi, C, E):
    if psi == 'fourier':
        return X14.FourierLawArm(C, E, edge_dense=True)
    return X13.LearnedNodeArm(C, E, edge_dense=True)


def two_phase(psi, C, E, q_tr, oq_tr, ot_tr, epochs, lr, seed):
    """Return (exact_edge_model, isolated_psi_model)."""
    m = make_model(psi, C, E)
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


def psi_norm(m):
    with torch.no_grad():
        return float(sum(p.detach().abs().sum().item()
                         for p in m.psi.parameters()))


def run_fr3(args, sizes, seeds, psi):
    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    ref = X14.load_ref()
    x14 = X14.load_json('X14_fourier_law.*.json')
    x13 = X14.load_json('X13_learned_node.*.json')
    for n in sizes:
        ref[n]['joint_fourier'] = np.array(x14[f'X14:n{n}:relaxed_fourier']['per_seed'])
        ref[n]['joint_mlp'] = np.array(x13[f'X13:n{n}:relaxed_dense']['per_seed'])
    res = {(n, k): [] for n in sizes for k in ('exact_own', 'isolated')}
    par = {(n, k): [] for n in sizes for k in ('psi_w_exact', 'psi_w_iso')}
    tag = 'G2f' if getattr(args, 'fr3_split', 'random') == 'random' else 'G2fC'
    if tag == 'G2fC':
        from scipy.cluster.vq import kmeans2
        qk = q_np - q_np.mean(0)
        qk = qk / (qk.std(0) + 1e-9)
    for s in range(seeds):
        if tag == 'G2f':
            perm = np.random.default_rng(1000 + s).permutation(n_total)
            splits = {n_tr: (perm[:n_tr], perm[n_tr:]) for n_tr in sizes}
        else:
            _, labels = kmeans2(qk, 4, minit='++', seed=1000 + s)
            cs = [np.where(labels == c)[0] for c in range(4)]
            splits = {}
            for n_tr in sizes:
                cbest = min(range(4),
                            key=lambda c: abs(len(cs[c]) - (n_total - n_tr)))
                te = cs[cbest]
                pool = np.concatenate([cs[c] for c in range(4) if c != cbest])
                rng = np.random.default_rng(2000 + s)
                splits[n_tr] = (rng.permutation(pool)[:n_tr], te)
        for n_tr in sizes:
            tr, te = splits[n_tr]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            ex, iso = two_phase(psi, C, E, q_tr, oq_tr, ot_tr, args.epochs,
                                args.lr, s)
            res[(n_tr, 'exact_own')].append(GN.eval_pos(ex, q_te, oq_te, ot_te))
            res[(n_tr, 'isolated')].append(GN.eval_pos(iso, q_te, oq_te, ot_te))
            par[(n_tr, 'psi_w_exact')].append(psi_norm(ex))
            par[(n_tr, 'psi_w_iso')].append(psi_norm(iso))
            print(f'[G2-fr3-{psi} seed {s} n={n_tr}] exact '
                  f'{res[(n_tr, "exact_own")][-1][0]:.3f} isolated '
                  f'{res[(n_tr, "isolated")][-1][0]:.3f}', flush=True)
        print(f'[G2-fr3-{psi}] seed {s} done', flush=True)
    for n_tr in sizes:
        for k in ('exact_own', 'isolated'):
            a = np.array([v[0] for v in res[(n_tr, k)]])
            r = np.array([v[1] for v in res[(n_tr, k)]])
            RESULTS[f'{tag}:{psi}:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rot_deg_mean=float(r.mean()), per_seed=[float(x) for x in a])
        a = np.array([v[0] for v in res[(n_tr, 'isolated')]])
        b = np.array([v[0] for v in res[(n_tr, 'exact_own')]])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'{tag}:{psi}:paired:n{n_tr}:isolated_vs_exact_own'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        for ref_name in ('ours', 'gn', 'joint_fourier', 'joint_mlp',
                         'relaxed_dense'):
            rr = ref[n_tr][ref_name]
            if len(a) != len(rr):
                continue
            t, p = stats.ttest_rel(a, rr)
            RESULTS[f'{tag}:{psi}:paired:n{n_tr}:isolated_vs_{ref_name}'] = dict(
                diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
        RESULTS[f'{tag}:{psi}:psi_weight:n{n_tr}'] = dict(
            exact=float(np.mean(par[(n_tr, 'psi_w_exact')])),
            isolated=float(np.mean(par[(n_tr, 'psi_w_iso')])))


def fit_isolated_ur5(exact, xi, M0, q_tr, T_obs2_tr, steps, seed, lr=1e-2):
    m = S2.ScrewCalibRelax(xi, M0, mode='dense')
    m.load_state_dict(exact.state_dict(), strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(100 + seed)
    opt = torch.optim.Adam(m.psi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    qt = torch.tensor(q_tr, dtype=DT)
    pt = torch.tensor(T_obs2_tr[:, :3, 3], dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - pt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m


def run_ur5(args, sizes, seeds):
    data = X11.load_gtpn(X11.DATA)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    Tf_all = X10.fk_np(robot, q_all)
    def _latest(pattern):
        cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', pattern))
                 if 'smoke' not in p]
        return json.load(open(sorted(cands, key=os.path.getmtime)[-1])) if cands else None

    x11 = _latest('X11_ur5_gtpn_sweep.*.json')
    s2 = _latest('S2_ur5gtpn_relaxed.*.json')
    res = {(n, k): [] for n in sizes for k in ('exact_own', 'isolated')}
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(q_all))
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], None, full=False)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=False, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            ex = X10.train_screwcalib(xi, M0, q_all[tr], T_obs2_tr,
                                      X10.log_se3_np(C), mode='dense',
                                      steps=args.steps, seed=s, full=False)
            iso = fit_isolated_ur5(ex, xi, M0, q_all[tr], T_obs2_tr,
                                   args.steps, s)
            res[(n_tr, 'exact_own')].append(X10.pose_metrics(
                X10.apply_sc(ex, q_all[te]), T_obs2_te, full=False)['pos_mean'])
            res[(n_tr, 'isolated')].append(X10.pose_metrics(
                X10.apply_sc(iso, q_all[te]), T_obs2_te, full=False)['pos_mean'])
            joint = s2[f'S2:n{n_tr}:relaxed']['per_seed'][s] if s2 else float('nan')
            exact_ref = x11[f'X11:n{n_tr}:ours']['per_seed'][s]
            print(f'[G2-ur5 seed {s} n={n_tr}] exact {res[(n_tr, "exact_own")][-1]:.3f}'
                  f' isolated {res[(n_tr, "isolated")][-1]:.3f} joint {joint:.3f}'
                  f' x11ours {exact_ref:.3f}', flush=True)
        print(f'[G2-ur5] seed {s} done', flush=True)
    for n_tr in sizes:
        for k in ('exact_own', 'isolated'):
            a = np.array(res[(n_tr, k)])
            RESULTS[f'G2u5:n{n_tr}:{k}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
        a = np.array(res[(n_tr, 'isolated')])
        b = np.array(res[(n_tr, 'exact_own')])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'G2u5:paired:n{n_tr}:isolated_vs_exact_own'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))
        if s2 is not None:
            jr = np.array(s2[f'S2:n{n_tr}:relaxed']['per_seed'])
            if len(jr) == len(a) and len(a) >= 2:
                t, p = stats.ttest_rel(a, jr)
                RESULTS[f'G2u5:paired:n{n_tr}:isolated_vs_joint'] = dict(
                    diff=float(a.mean() - jr.mean()), t=float(t), p=float(p))


def run_s1(args, sizes, seeds, psi):
    C, E = X7.extract_nominal_chain()
    for tag, deform in (('exact', False), ('deformed', True)):
        data = [S1.gen_dataset(C, E, 500, 7000 + s, deform=deform)
                for s in range(seeds)]
        res = {(n, k): [] for n in sizes for k in ('exact_own', 'isolated')}
        for s in range(seeds):
            q, oq, ot = data[s]
            perm = np.random.default_rng(1000 + s).permutation(len(q))
            for n_tr in sizes:
                tr, te = perm[:n_tr], perm[n_tr:]
                q_tr, oq_tr, ot_tr = q[tr], oq[tr], ot[tr]
                q_te, oq_te, ot_te = q[te], oq[te], ot[te]
                ex, iso = two_phase(psi, C, E, q_tr, oq_tr, ot_tr, args.epochs,
                                    args.lr, s)
                res[(n_tr, 'exact_own')].append(GN.eval_pos(ex, q_te, oq_te, ot_te))
                res[(n_tr, 'isolated')].append(GN.eval_pos(iso, q_te, oq_te, ot_te))
                print(f'[G2-s1-{tag}-{psi} seed {s} n={n_tr}] exact '
                      f'{res[(n_tr, "exact_own")][-1][0]:.3f} isolated '
                      f'{res[(n_tr, "isolated")][-1][0]:.3f}', flush=True)
            print(f'[G2-s1-{tag}-{psi}] seed {s} done', flush=True)
        for n_tr in sizes:
            for k in ('exact_own', 'isolated'):
                a = np.array([v[0] for v in res[(n_tr, k)]])
                RESULTS[f'G2s:{psi}:{tag}:n{n_tr}:{k}'] = dict(
                    pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                    per_seed=[float(x) for x in a])
            a = np.array([v[0] for v in res[(n_tr, 'isolated')]])
            b = np.array([v[0] for v in res[(n_tr, 'exact_own')]])
            if len(a) >= 2:
                t, p = stats.ttest_rel(a, b)
                RESULTS[f'G2s:{psi}:{tag}:paired:n{n_tr}:isolated_vs_exact_own'] = \
                    dict(diff=float(a.mean() - b.mean()), t=float(t), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['fr3', 'ur5', 's1', 'all'])
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes_fr3', default='20,40,60,80')
    ap.add_argument('--seeds_fr3', type=int, default=5)
    ap.add_argument('--sizes_m', default='20,60,80')
    ap.add_argument('--seeds_m', type=int, default=3)
    ap.add_argument('--sizes_s1', default='20,80,160')
    ap.add_argument('--seeds_s1', type=int, default=3)
    ap.add_argument('--sizes_ur5', default='20,40,60,80')
    ap.add_argument('--seeds_ur5', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--steps_ur5', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--fr3-split', choices=['random', 'cluster'], default='random')
    ap.add_argument('--fr3-psi', default='fourier,mlp', help='which psi variants to run on FR3')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes_fr3, args.seeds_fr3 = '20', 1
        args.sizes_m, args.seeds_m = '20', 1
        args.sizes_s1, args.seeds_s1 = '20', 1
        args.sizes_ur5, args.seeds_ur5 = '20', 1
        args.steps_ur5 = 200
        args.epochs = 300
    t0 = time.time()
    print('[G2bc] isolated psi (frozen exact edge), fourier + mlp', flush=True)
    if args.part in ('fr3', 'all'):
        if 'fourier' in args.fr3_psi.split(','):
            run_fr3(args, [int(s) for s in args.sizes_fr3.split(',')], args.seeds_fr3,
                    'fourier')
        if 'mlp' in args.fr3_psi.split(','):
            run_fr3(args, [int(s) for s in args.sizes_m.split(',')], args.seeds_m,
                    'mlp')
    if args.part in ('ur5', 'all'):
        args.steps = args.steps_ur5
        run_ur5(args, [int(s) for s in args.sizes_ur5.split(',')],
                args.seeds_ur5)
        args.steps = 2000
    if args.part in ('s1', 'all'):
        for psi in ('fourier', 'mlp'):
            run_s1(args, [int(s) for s in args.sizes_s1.split(',')],
                   args.seeds_s1, psi)
    RESULTS['G2bc:_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=(args.seeds_fr3, args.seeds_m, args.seeds_s1,
                           args.seeds_ur5),
                    epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
