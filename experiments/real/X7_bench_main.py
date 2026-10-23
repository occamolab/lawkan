#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X7_bench_main.py — Stage 2: the fixed protocol on the real FR3 arm against the
competitors, on the same splits (seeds 0-4, sizes 20/40/60/80).

Protocol (after Stage-1 fixes): learned models use the bounded loss
2(1-cos) with weight w (default 1.0), no grad clipping (X7-style), 4000 epochs;
ours = E34.ScrewArm const (exp, 54p); MLP = X7.MLPResidual; classical =
Gauss-Newton least squares of the same model class on position residuals
(X7_gn_ls.fit); PoE-LS per-seed from the archived X7 dump (deterministic fit,
unaffected by the fixes).

Sections:
  main   : ours / MLP / GN / PoE-LS at n=20..80, seeds 0-4, paired stats
  sweep  : ours and MLP at n=80 with cosine2 w in {0, 0.25, 1.0} (seeds 0-2)
Partial dumps after each section.

Output: repro/json/X7_bench_main.<ts>.json via repro/dump.
Run:  python X7_bench_main.py            # both sections
      python X7_bench_main.py --quick    # 1 seed, 200 epochs, n=80 only
"""

import argparse
import glob
import json
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

import X7_robot_calibration as X7       # noqa: E402
import E34_x7_recipe as E34              # noqa: E402
import X7_gn_ls as GN                    # noqa: E402  (fit, eval_pos)
from repro.dump import save_dump         # noqa: E402
from scipy import stats                  # noqa: E402

DT = E34.DT
RESULTS = {}


def ours_model(C, E):
    return E34.ScrewArm(C, E, mode='const')


def mlp_model(C, E):
    return X7.MLPResidual(C, E)


def train(m, q_tr, oq_tr, ot_tr, epochs, lr, seed, w):
    return E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed, clip=0.0,
                            ang_weight=w, warm_epochs=0, loss_form='cosine2')


def load_poe_archive():
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', 'x7.*.json'))
             if 'smoke' not in p]
    if not cands:
        return {}
    p = sorted(cands, key=os.path.getmtime)[-1]
    d = json.load(open(p))
    out = {}
    for n in (20, 40, 60, 80):
        key = f'X7:n{n}:PoE-LS'
        if key in d:
            out[n] = np.array(d[key]['per_seed'], dtype=float)   # [[pos,rot],...]
    print(f'[X7bench] PoE-LS archive: {os.path.basename(p)}', flush=True)
    return out


def reg_cell(key, arr, params=None, extra=None):
    arr = np.asarray(arr, dtype=float)
    e = dict(pos_mm_mean=float(arr[:, 0].mean()), pos_mm_std=float(arr[:, 0].std()),
             rot_deg_mean=float(arr[:, 1].mean()), rot_deg_std=float(arr[:, 1].std()),
             per_seed=[[float(a), float(b)] for a, b in arr])
    if params is not None:
        e['params'] = int(params)
    if extra:
        e.update(extra)
    RESULTS[key] = e
    print(f'{key:28s} pos {e["pos_mm_mean"]:.3f}±{e["pos_mm_std"]:.3f} '
          f'rot {e["rot_deg_mean"]:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--sweep-seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.sweep_seeds, args.sizes, args.epochs = 1, 1, '80', 200
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = [int(s) for s in args.sizes.split(',')]
    poe = load_poe_archive()

    # ---------------- main table ----------------
    cells = {(n, m): [] for n in sizes for m in ('ours', 'mlp', 'gn')}
    for seed in range(args.seeds):
        perm = np.random.default_rng(1000 + seed).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]

            m = ours_model(C, E)
            m, gmax = train(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, 1.0)
            cells[(n_tr, 'ours')].append(GN.eval_pos(m, q_te, oq_te, ot_te))

            m = mlp_model(C, E)
            m, _ = train(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, 1.0)
            cells[(n_tr, 'mlp')].append(GN.eval_pos(m, q_te, oq_te, ot_te))

            mg, _ = GN.fit(C, E, q_tr, oq_tr, ot_tr, bal=False)
            cells[(n_tr, 'gn')].append(GN.eval_pos(mg, q_te, oq_te, ot_te))
            print(f'[main seed {seed} n={n_tr}] ours {cells[(n_tr,"ours")][-1][0]:.3f} '
                  f'mlp {cells[(n_tr,"mlp")][-1][0]:.3f} '
                  f'gn {cells[(n_tr,"gn")][-1][0]:.3f} | gradmax {gmax:.2f}', flush=True)

    for n_tr in sizes:
        for m in ('ours', 'mlp', 'gn'):
            reg_cell(f'X7bench:n{n_tr}:{m}', cells[(n_tr, m)],
                     params={'ours': 54, 'mlp': None, 'gn': 54}[m])
        if n_tr in poe:
            reg_cell(f'X7bench:n{n_tr}:poe', poe[n_tr][:, :2], params=0)
        # paired vs ours
        a = np.array(cells[(n_tr, 'ours')])
        for ref, arr in (('mlp', np.array(cells[(n_tr, 'mlp')])),
                         ('gn', np.array(cells[(n_tr, 'gn')])),
                         ('poe', poe.get(n_tr, np.zeros((0, 2)))[:, :2])):
            if len(arr) != len(a):
                continue
            t, p = stats.ttest_rel(a[:, 0], arr[:, 0])
            RESULTS[f'X7bench:paired:n{n_tr}:ours_vs_{ref}'] = dict(
                n=int(len(a)), diff_pos=float(a[:, 0].mean() - arr[:, 0].mean()),
                t=float(t), p=float(p))
    save_dump(RESULTS, __file__, quick=args.quick, seeds=args.seeds,
              epochs=args.epochs)

    # ---------------- loss sweep at n=80 ----------------
    sw = {}
    for seed in range(args.sweep_seeds):
        perm = np.random.default_rng(1000 + seed).permutation(n_total)
        n_tr = 80
        tr, te = perm[:n_tr], perm[n_tr:]
        q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
        q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
        for w in (0.0, 0.25, 1.0):
            for kind, builder in (('ours', ours_model), ('mlp', mlp_model)):
                m = builder(C, E)
                m, _ = train(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, w)
                sw.setdefault((kind, w), []).append(
                    GN.eval_pos(m, q_te, oq_te, ot_te))
                print(f'[sweep seed {seed}] {kind} w={w}: '
                      f'{sw[(kind, w)][-1][0]:.3f}', flush=True)
    for (kind, w), arr in sw.items():
        reg_cell(f'X7bench:sweep:n80:{kind}:w{w}', arr)
    RESULTS['_meta'] = dict(wall_s=time.time() - t0, sizes=sizes,
                            seeds=args.seeds, sweep_seeds=args.sweep_seeds,
                            epochs=args.epochs, loss_form='cosine2', clip=0.0)
    p = save_dump(RESULTS, __file__, quick=args.quick, seeds=args.seeds,
                  epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
