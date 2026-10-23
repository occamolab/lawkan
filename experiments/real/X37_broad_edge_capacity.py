#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X37_broad_edge_capacity.py — does the capacity lever that stabilizes the psi
correction also fix the seed-bimodality of the DENSE IMU edge? (registered before run)

The X5 real-IMU dense edge (27 p) is seed-bimodal at long horizons (one seed
diverges to ~73 deg at 5 s); the pruned 18-p analytic-support edge is stable.
X26/X26a/X32 showed that capacity (an L2 dose) and locality (a kernel) stabilize
the learned correction; here we apply the capacity lever (weight decay on the
edge) to the IMU edge, at the X5 protocol (trial-level split, 800/2000 windows).

Registered prediction P-D: a weight-decay dose removes the outlier seed (the
per-seed spread collapses to the pruned-edge level) WITHOUT improving the mean
--- the floor is the measurement noise, not the learned law. I.e. the IMU
failure is a data/noise boundary, not a method boundary.

Output: repro/json/X37_broad_edge_capacity.<ts>.json via repro/dump.
Run:  python X37_broad_edge_capacity.py --data_dir ../broad/data_hdf5 [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X5_broad_real_imu as X5               # noqa: E402
from repro.dump import save_dump             # noqa: E402

RESULTS = {}
DT = torch.float64


def train_eval_wd(model, Xtr, Ytr, Btr, Xte, Yte, Bte, epochs, lr, seed, wd):
    """X5.train_eval with an L2 (weight-decay) dose on the edge."""
    torch.manual_seed(seed)
    model = model.to(X5.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr, Btr)
        d = (pred * Ytr).sum(-1).abs().clamp(max=1.0 - 1e-7)
        loss = (2.0 * (1.0 - d)).mean()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = model(Xte, Bte)
    return X5.geo_deg_q(pred, Yte).mean().item()


def spread(v):
    a = np.array(v)
    return dict(mean=float(a.mean()), std=float(a.std()),
                max=float(a.max()), min=float(a.min()),
                spread=float(a.max() - a.min()))


def run_block(name, W, stride, wds, seeds, data_dir, groups, epochs):
    X, Y, B, fs, T = X5.make_dataset(data_dir, W=W, stride=stride, groups=groups)
    tr, te = X5.trial_split(T, 800, 2000)
    Xtr, Ytr, Btr = X[tr], Y[tr], B[tr]
    Xte, Yte, Bte = X[te], Y[te], B[te]
    dt = 1.0 / fs
    rows = [('dense', lambda: X5.QuatIMU(W, K=1, dt=dt, bias_dim=3), wds),
            ('sparse', lambda: X5.QuatIMU(W, K=1, dt=dt, bias_dim=3, sparse=True), [0.0])]
    for label, ctor, wdl in rows:
        for wd in wdl:
            errs = []
            for s in seeds:
                torch.manual_seed(100 + s)
                m = ctor()
                errs.append(train_eval_wd(m, Xtr, Ytr, Btr, Xte, Yte, Bte,
                                          epochs=epochs, lr=3e-3, seed=100 + s, wd=wd))
            key = f'X37:{name}:{label}:wd{wd:g}'
            RESULTS[key] = dict(per_seed=[float(v) for v in errs], wd=wd, **spread(errs))
            print(f'[X37 {name}] {label} wd={wd:g} -> ' +
                  ' '.join(f'{v:.1f}' for v in errs) +
                  f'  (mean {np.mean(errs):.1f}, spread {max(errs)-min(errs):.1f})', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get(
        'BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--groups', default='disturbed')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--wds', default='0,3e-4,1e-3,3e-3')
    ap.add_argument('--only', default='b-1s,c-5s',
                    help='comma list of b-1s,c-5s (a-short is not informative here)')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs, args.only = 1, 300, 'b-1s'
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    seeds = list(range(args.seeds))
    wds = [float(w) for w in args.wds.split(',')]
    only = set(args.only.split(','))
    t0 = time.time()

    if 'b-1s' in only:
        run_block('b-1s', W=286, stride=286 // 4, wds=wds, seeds=seeds,
                  data_dir=args.data_dir, groups=groups, epochs=args.epochs)
    if 'c-5s' in only:
        run_block('c-5s', W=1430, stride=1430 // 10, wds=wds, seeds=seeds,
                  data_dir=args.data_dir, groups=groups, epochs=args.epochs)

    RESULTS['X37:protocol'] = dict(
        blocks=sorted(only), seed_count=len(seeds), epochs=args.epochs, wds=wds,
        groups=groups, head='dense 27p IMU edge + L2 weight-decay dose vs sparse 18p',
        note='P-D: dose removes the outlier seed; mean not improved (noise floor)')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=len(seeds), epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
