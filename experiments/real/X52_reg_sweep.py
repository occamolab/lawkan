#!/usr/bin/env python3
"""X52 -- why the two-slot spikes: a regularization sweep on the A1 setup.

Same FR3 task as X46 A1 (N=20/60/80, 5 seeds, 2500 epochs, checkpoint
every 10), varying one stabilizer at a time on the two-slot:
  base      -- lr 5e-3, cosine, no clip, no warmup (the X46 recipe)
  clip      -- + gradient-norm clipping 1.0
  warmup    -- + 200-epoch linear warmup before the cosine
  lr1e3     -- lr 1e-3 (same schedule)
Output: repro/json/X52_reg_sweep.<ts>.json
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import X7_robot_calibration as X7          # noqa: E402
import X24_fr3_gate as X24           # noqa: E402
import X33_fr3_twoslot as X33               # noqa: E402
import X34_fr3_kernel as X34                # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}
CK = list(range(10, 2501, 10))


def train_variant(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr,
                  seed, clip=0.0, warmup=0, l2=0.0):
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi.Wk.requires_grad_(True)
    m.psi.global_lin.weight.requires_grad_(True)
    m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    out = {}
    for ep in range(1, epochs + 1):
        if warmup and ep <= warmup:
            for g in opt.param_groups:
                g["lr"] = lr * ep / warmup
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if l2:
            loss = loss + l2 * (m.psi.global_lin.weight ** 2).sum()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if not (warmup and ep <= warmup):
            sched.step()
        if ep in CK:
            out[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='20,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=2500)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.epochs = '20', 1, 300
    t0 = time.time()
    q_np, obs_np = X7.load_philnad_dataset(os.path.join(_HERE, 'data', 'calib_data.pickle'))
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = [int(s) for s in args.sizes.split(',')]
    variants = dict(base=dict(), clip=dict(clip=1.0), warmup=dict(warmup=200),
                    lr1e3=dict(lr_scale=0.2), l2x10=dict(dose_x=10.0),
                    all=dict(clip=1.0, warmup=200, lr_scale=0.2))
    res = {(v, N): [] for v in variants for N in sizes}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, s)
            feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
            from scipy.cluster.vq import kmeans2
            cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=s)
            centers = torch.tensor(cen, dtype=DT)
            bw = float(torch.median(torch.cdist(centers, centers)[torch.cdist(centers, centers) > 0]))
            # only the base + key variants to keep the run bounded
            for v in ('base', 'clip', 'warmup', 'lr1e3'):
                ts = X33.make_two_slot(exact, centers, bw)
                kw = dict(variants[v])
                lr = 5e-3 * kw.pop('lr_scale', 1.0)
                dose = {20: 3e-3, 60: 1e-4, 80: 0.0}[N] * kw.pop('dose_x', 1.0)
                r = train_variant(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                                  args.epochs, lr, s, l2=dose, **kw)
                # spikiness = max - final over the trajectory
                vals = np.array([r[e] for e in CK if e in r])
                res[(v, N)].append((float(vals.max()), float(vals[-1]),
                                    float(np.mean(vals > 1.5 * vals[-1]))))
            print(f'[X52 seed {s} N={N}] ' + ' '.join(
                f'{v}: max {res[(v,N)][-1][0]:.2f} fin {res[(v,N)][-1][1]:.2f}'
                for v in ('base', 'clip', 'warmup', 'lr1e3')), flush=True)
    for (v, N), a in res.items():
        a = np.array(a, dtype=float).reshape(-1, 3)
        RESULTS[f'X52:{v}:N{N}'] = dict(
            spike=float(a[:, 0].mean() - a[:, 1].mean()),
            max_mean=float(a[:, 0].mean()), final_mean=float(a[:, 1].mean()),
            frac_above=float(a[:, 2].mean()), per_seed=a.tolist())
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds,
                  extra_meta={'note': 'X52: regularization sweep vs the two-slot spikes'})
    print('dump ->', p)


if __name__ == '__main__':
    main()
