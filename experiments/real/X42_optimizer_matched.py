#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X42_optimizer_matched.py — the reference model class (base + per-joint SE(3)
corrections + tool, 54p) fitted with OUR optimizer + OUR loss, on the FR3 splits.

Motivation. Table 6 (armcal) compares our LawKAN rows (Adam + cosine, bounded
loss dp^2 + 2(1-cos(theta/2))) against the classical `reference` rows solved by
other optimizers (robotkinecal LS, scipy LM). That comparison is confounded: a
number difference can come from the optimizer/loss, not the model. This probe
removes the confound: it fits the SAME class as the GN reference with the SAME
Adam + cosine schedule and the SAME bounded loss as our rows.

Models:
  ref_adam    E34.ScrewArm(C,E,mode='const'), Adam+cosine, loss dp^2+2(1-dot)
  ours_exact  X24.phase1_exact (our exact node; same class, same optimizer/loss)

Paired vs gn_bal and gn_pos (X7_gn_ls, same splits).

Output: repro/json/X42_optimizer_matched.<ts>.json via repro/dump.
Run:  python X42_optimizer_matched.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
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
import X24_fr3_gate as X24                  # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def train_ref_adam(m, q_tr, oq_tr, ot_tr, epochs, lr, seed):
    """The reference class under OUR optimizer + OUR loss (E34.train_recipe cosine2)."""
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    return m


def load_gn():
    import glob
    import json
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json',
                                               'X7_gn_ls.*.json'))
             if 'smoke' not in p]
    d = json.load(open(sorted(cands, key=os.path.getmtime)[-1]))
    return {n: {k: np.array(d[f'X7gn:n{n}:{k}']['per_seed'])[:, 0]
                for k in ('gn_pos', 'gn_bal')} for n in (20, 40, 60, 80)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.epochs = 1, 300
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(
        os.path.join(_HERE, 'data', 'calib_data.pickle'))
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = (20, 40, 60, 80)

    res = {(s, k): [] for s in sizes for k in ('ref_adam', 'ours_exact')}
    for seed in range(args.seeds):
        perm = np.random.default_rng(1000 + seed).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]

            m = E34.ScrewArm(C, E, mode='const')
            m = train_ref_adam(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed)
            res[(n_tr, 'ref_adam')].append(GN.eval_pos(m, q_te, oq_te, ot_te)[0])

            me = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed)
            res[(n_tr, 'ours_exact')].append(GN.eval_pos(me, q_te, oq_te, ot_te)[0])

            print(f'[X42 seed {seed} n={n_tr}] ref_adam {res[(n_tr, "ref_adam")][-1]:.3f} '
                  f'ours_exact {res[(n_tr, "ours_exact")][-1]:.3f}', flush=True)

    gn = load_gn()
    for s in sizes:
        for k in ('ref_adam', 'ours_exact'):
            a = np.array(res[(s, k)])
            RESULTS[f'X42:n{s}:{k}'] = dict(pos_mm_mean=float(a.mean()),
                                           pos_mm_std=float(a.std()),
                                           per_seed=[float(v) for v in a])
        ra = np.array(res[(s, 'ref_adam')])
        oe = np.array(res[(s, 'ours_exact')])
        for name, b in (('ours_exact', oe), ('gn_bal', gn[s]['gn_bal']),
                        ('gn_pos', gn[s]['gn_pos'])):
            b = b[:len(ra)]
            if len(b) < 2:
                continue
            t, p = stats.ttest_rel(ra, b)
            RESULTS[f'X42:paired:n{s}:ref_adam_vs_{name}'] = dict(
                diff=float((ra - b).mean()), t=float(t), p=float(p))
        print(f'n={s}: ref_adam {ra.mean():.3f} ours_exact {oe.mean():.3f} '
              f'gn_bal {gn[s]["gn_bal"].mean():.3f} gn_pos {gn[s]["gn_pos"].mean():.3f}')

    RESULTS['_meta'] = dict(wall_s=time.time() - t0,
                            model='E34.ScrewArm const (54p) / X24 exact',
                            optimizer='Adam + cosine, loss dp^2 + 2(1-dot)',
                            seeds=args.seeds, sizes=list(sizes))
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds)
    print('dump ->', p)


if __name__ == '__main__':
    main()
