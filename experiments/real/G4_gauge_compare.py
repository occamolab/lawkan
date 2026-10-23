#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
G4_gauge_compare.py — head-to-head: isolation vs sparse vs orthogonality (FR3).

All variants share ONE frozen exact edge (phase 1, psi frozen) per (seed, N):
  iso        full Fourier psi (30 inputs), ordinary psi-only phase;
  iso_top6   sparse psi (X17/X16 ranking), psi-only;
  iso_top8   sparse psi, psi-only;
  iso_orth   psi-only with an absorbable-share penalty: rho = ||P_A D||^2/||D||^2,
             loss = loss * (1 + lam*rho), lam = 1, where D is the psi effect field
             in the linearized pose embedding (t, vee(R-R^T)/2) and P_A is the
             projector on the span of edge-tangent fields (finite differences at
             the exact edge, train configurations).

Metrics: held-out position error (mm), |psi| weight norm, rho.
References: dense/GN (bench), joint Fourier (X14 dump), sparse-joint (X17 dump).

Registered predictions (plan 2026-09-29, before the run):
  P-G4a: iso keeps the FR3 win at N>=60 (paired vs own exact);
  P-G4b: iso_top6/iso_top8 keep the win too (some gap to full psi allowed);
  P-G4c: iso_orth keeps the win and has the smallest rho (<=0.1 at N=80);
  P-G4d (boundary): if iso_orth loses the win, part of the deformation is
         edge-absorbable and plain isolation is the better trade-off.

Output: repro/json/G4_gauge_compare.<ts>.json via repro/dump.
Run:  python G4_gauge_compare.py [--quick]
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
import X14_fourier_law as X14               # noqa: E402
import X17_sparse_law as X17                # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}

VARIANTS = ('iso', 'iso_top6', 'iso_top8', 'iso_orth')


def quat_to_mat_torch(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def emb_np(T):
    """Linearized pose embedding: (t, vee((R - R^T)/2)) -> (N, 6) numpy."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    skew = R - R.transpose(0, 2, 1)
    w = 0.5 * np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], -1)
    return np.concatenate([t, w], -1)


def emb_torch(qp, tp):
    R = quat_to_mat_torch(qp)
    skew = R - R.transpose(1, 2)
    w = 0.5 * torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], -1)
    return torch.cat([tp, w], -1)


def poses_fr3(m, q):
    with torch.no_grad():
        qp, tp = m(q)
    qp, tp = qp.numpy(), tp.numpy()
    T = np.tile(np.eye(4), (len(qp), 1, 1))
    T[:, :3, :3] = quat_to_mat_torch(torch.tensor(qp)).numpy()
    T[:, :3, 3] = tp
    return T


def edge_jacobian_emb(m, q, eps=1e-6):
    T0 = poses_fr3(m, q)
    e0 = emb_np(T0).reshape(-1)
    cols = []
    for name, p in m.named_parameters():
        if name.startswith('psi'):
            continue
        flat = p.view(-1)
        for idx in range(flat.numel()):
            with torch.no_grad():
                flat[idx] += eps
            Tp = poses_fr3(m, q)
            with torch.no_grad():
                flat[idx] -= eps
            cols.append((emb_np(Tp).reshape(-1) - e0) / eps)
    return np.stack(cols, axis=1)


def psi_effect_embed(m, q, emb_off):
    T_on = poses_fr3(m, q)
    sd = {k: v.clone() for k, v in m.psi.state_dict().items()}
    with torch.no_grad():
        for p in m.psi.parameters():
            p.zero_()
    T_off = poses_fr3(m, q)
    m.psi.load_state_dict(sd)
    D = emb_np(T_on).reshape(-1) - emb_np(T_off).reshape(-1)
    return D


def rho_of(D, Pproj):
    a = Pproj @ D
    denom = float(D @ D)
    return float(a @ a) / denom if denom > 0 else 0.0


def psi_norm(m):
    with torch.no_grad():
        return float(sum(p.detach().abs().sum().item()
                         for p in m.psi.parameters()))


def train_orth(m, q_tr, oq_tr, ot_tr, emb_off, Pproj, epochs, lr, seed,
               lam=1.0, ang_weight=1.0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.psi.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    emb_off_t = torch.tensor(emb_off, dtype=DT)
    P = torch.tensor(Pproj, dtype=DT)
    for _ in range(epochs):
        opt.zero_grad()
        qp, tp = m(q_tr)
        dp = (tp - ot_tr).norm(dim=-1)
        dot = (qp * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + ang_weight * 2.0 * \
            (1.0 - dot.clamp(max=1.0 - 1e-7)).mean()
        D = emb_torch(qp, tp).reshape(-1) - emb_off_t
        absorb = P @ D
        rho = (absorb @ absorb) / ((D @ D) + 1e-12)
        total = loss * (1.0 + lam * rho)
        total.backward()
        opt.step()
        sched.step()
    return m


def load_json(pattern, required=True):
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', pattern))
             if 'smoke' not in p]
    if not cands:
        if required:
            raise FileNotFoundError('no dump for %s' % pattern)
        return None
    return json.load(open(sorted(cands, key=os.path.getmtime)[-1]))


def make_variant(tag, C, E, exact_sd):
    if tag == 'iso_top6':
        m = X17.SparseLawArm(C, E, X17.MASKS['top6'], edge_dense=True)
    elif tag == 'iso_top8':
        m = X17.SparseLawArm(C, E, X17.MASKS['top8'], edge_dense=True)
    else:
        m = X14.FourierLawArm(C, E, edge_dense=True)
    m.load_state_dict(exact_sd, strict=False)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,60,80')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--lam', type=float, default=1.0)
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
    ref = X14.load_ref()
    x14 = load_json('X14_fourier_law.*.json')
    x17 = load_json('X17_sparse_law.*.json', required=False)
    for n in sizes:
        ref[n]['joint_fourier'] = np.array(
            x14[f'X14:n{n}:relaxed_fourier']['per_seed'])
        if x17 is not None:
            for tag, key in (('top6', 'sparse_joint_top6'),
                             ('top8', 'sparse_joint_top8')):
                k = f'X17:n{n}:{tag}'
                if k in x17:
                    ref[n][key] = np.array(x17[k]['per_seed'])

    res = {(n, v): [] for n in sizes for v in VARIANTS}
    rho = {(n, v): [] for n in sizes for v in VARIANTS}
    ex_res = {n: [] for n in sizes}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]

            exact = X14.FourierLawArm(C, E, edge_dense=True)
            for p in exact.psi.parameters():
                p.requires_grad_(False)
            exact, _ = E34.train_recipe(exact, q_tr, oq_tr, ot_tr, args.epochs,
                                        args.lr, s, clip=0.0, ang_weight=1.0,
                                        warm_epochs=0, loss_form='cosine2')
            ex_res[n_tr].append(GN.eval_pos(exact, q_te, oq_te, ot_te)[0])

            A = edge_jacobian_emb(exact, q_tr)
            Pproj = A @ np.linalg.pinv(A)
            emb_off = emb_np(poses_fr3(exact, q_tr)).reshape(-1)

            line = []
            for tag in VARIANTS:
                m = make_variant(tag, C, E, copy.deepcopy(exact.state_dict()))
                if tag == 'iso_orth':
                    train_orth(m, q_tr, oq_tr, ot_tr, emb_off, Pproj,
                               args.epochs, args.lr, s, lam=args.lam)
                else:
                    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                            args.lr, s, clip=0.0, ang_weight=1.0,
                                            warm_epochs=0, loss_form='cosine2')
                res[(n_tr, tag)].append(GN.eval_pos(m, q_te, oq_te, ot_te)[0])
                D = psi_effect_embed(m, q_tr, emb_off)
                rho[(n_tr, tag)].append(rho_of(D, Pproj))
                line.append(f'{tag} {res[(n_tr, tag)][-1]:.3f} '
                            f'(rho {rho[(n_tr, tag)][-1]:.2f})')
            print(f'[G4 seed {s} n={n_tr}] exact {ex_res[n_tr][-1]:.3f} | '
                  + ' | '.join(line), flush=True)
        print(f'[G4] seed {s} done', flush=True)

    for n_tr in sizes:
        a_ex = np.array(ex_res[n_tr])
        RESULTS[f'G4:n{n_tr}:exact_own'] = dict(
            pos_mm_mean=float(a_ex.mean()), per_seed=[float(x) for x in a_ex])
        for tag in VARIANTS:
            a = np.array(res[(n_tr, tag)])
            r = np.array(rho[(n_tr, tag)])
            RESULTS[f'G4:n{n_tr}:{tag}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                rho_mean=float(r.mean()), params=None,
                per_seed=[float(x) for x in a],
                rho_per_seed=[float(x) for x in r])
            t, p = stats.ttest_rel(a, a_ex)
            RESULTS[f'G4:paired:n{n_tr}:{tag}_vs_exact_own'] = dict(
                diff=float(a.mean() - a_ex.mean()), t=float(t), p=float(p))
            for ref_name in ('ours', 'gn', 'joint_fourier',
                             'sparse_joint_top6', 'sparse_joint_top8'):
                if ref_name not in ref[n_tr]:
                    continue
                rr = ref[n_tr][ref_name]
                if len(a) != len(rr):
                    rr = rr[:len(a)]
                if len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr)
                RESULTS[f'G4:paired:n{n_tr}:{tag}_vs_{ref_name}'] = dict(
                    diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
        for tag in VARIANTS:
            m = {'iso': X14.FourierLawArm(C, E, edge_dense=True),
                 'iso_top6': X17.SparseLawArm(C, E, X17.MASKS['top6']),
                 'iso_top8': X17.SparseLawArm(C, E, X17.MASKS['top8']),
                 'iso_orth': X14.FourierLawArm(C, E, edge_dense=True)}[tag]
            RESULTS[f'G4:n{n_tr}:{tag}']['params'] = m.n_params()
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump:', p)


if __name__ == '__main__':
    main()
