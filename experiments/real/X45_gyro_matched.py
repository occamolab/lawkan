#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X45_gyro_matched.py — tab:gyrocal under a matched optimizer: the learned (M,b)
and the classical (M,b) share the SAME geodesic loss; here BOTH are solved with
the SAME LM (scipy least_squares) as well, to remove the Adam-vs-LM confound.

Protocol identical to X6 (undisturbed BROAD, 70/30 window split, geodesic error).
Reported per block: raw / bias-only / classical LM / learned Adam / learned LM.

Output: repro/json/X45_gyro_matched.<ts>.json via repro/dump.
Run:  python X45_gyro_matched.py --data_dir /path/to/broad/data_hdf5 [--quick]
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

import X6_broad_calibration as X6               # noqa: E402
from repro.dump import save_dump
from repro.data_guard import skip_if_missing                # noqa: E402

RESULTS = {}


def fit_lm_on_learned(Xtr, Ytr, dt, max_nfev=200):
    """LM on the SAME CalibratedIntegrator (12p): a matched-optimizer stand-in."""
    import scipy.optimize as opt
    X = torch.as_tensor(np.asarray(Xtr), dtype=torch.float64)
    Y = torch.as_tensor(np.asarray(Ytr), dtype=torch.float64)
    m = X6.CalibratedIntegrator(dt).double()

    def resid(x):
        with torch.no_grad():
            m.bias.copy_(torch.tensor(x[9:12], dtype=torch.float64))
            m.M.copy_(torch.tensor(x[:9], dtype=torch.float64).reshape(3, 3))
            q = m(X)
        d = (q * Y).sum(-1).abs().clamp(max=1 - 1e-7)
        return (2.0 * torch.acos(d)).numpy()

    x0 = np.concatenate([np.eye(3).ravel(), np.zeros(3)])
    sol = opt.least_squares(resid, x0, method='trf', max_nfev=max_nfev)
    with torch.no_grad():
        m.M.copy_(torch.tensor(sol.x[:9], dtype=torch.float64).reshape(3, 3))
        m.bias.copy_(torch.tensor(sol.x[9:12], dtype=torch.float64))
    return m


def run(name, data_dir, W, stride, groups, seeds, epochs, lr, quick):
    X, Y, tidx, names, fs = X6.assemble(data_dir, W, stride, groups)
    dt = 1.0 / fs
    g = np.random.default_rng(12345)
    perm = g.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    q_raw = X6.integrate(Xte, dt)
    RESULTS[f'X45:{name}:raw'] = float(X6.geo_deg(q_raw, Yte).mean())

    files = X6.list_broad_files(data_dir, groups)
    tb = []
    for path, gr in files:
        tri = X6.load_broad_trial(path)
        tb.append(X6.estimate_bias(tri['gyr'], tri['move'], margin_s=0.5, fs=tri['fs']))
    tb = torch.from_numpy(np.stack(tb)).float()
    b_te = tb[torch.as_tensor(tidx[te])]
    q_bias = X6.integrate(Xte - b_te.unsqueeze(1), dt)
    RESULTS[f'X45:{name}:bias'] = float(X6.geo_deg(q_bias, Yte).mean())

    p_lm = X6.lm_fit(Xtr.cpu().numpy(), Ytr.cpu().numpy(), dt)
    M_lm = torch.tensor(p_lm[:9], dtype=torch.float64).reshape(3, 3)
    b_lm = torch.tensor(p_lm[9:12], dtype=torch.float64)
    q_lm = X6.integrate(torch.einsum('ij,bsj->bsi', M_lm, Xte.to(torch.float64) - b_lm), dt)
    RESULTS[f'X45:{name}:classical_lm'] = float(X6.geo_deg(q_lm, Yte).mean())

    m_lm = fit_lm_on_learned(Xtr.cpu().numpy(), Ytr.cpu().numpy(), dt)
    with torch.no_grad():
        q_olm = X6.geo_deg(m_lm(Xte), Yte)
    RESULTS[f'X45:{name}:learned_lm'] = float(q_olm.mean())

    adams = []
    for s in seeds:
        torch.manual_seed(100 + s)
        model = X6.CalibratedIntegrator(dt)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = (X6.geo_rad(model(Xtr), Ytr) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            adams.append(float(X6.geo_deg(model(Xte), Yte).mean()))
    RESULTS[f'X45:{name}:learned_adam_mean'] = float(np.mean(adams))
    RESULTS[f'X45:{name}:learned_adam_per_seed'] = [float(v) for v in adams]
    t, p = stats.ttest_1samp(adams, RESULTS[f'X45:{name}:classical_lm'])
    RESULTS[f'X45:{name}:adam_vs_lm'] = dict(diff=float(np.mean(adams) - RESULTS[f'X45:{name}:classical_lm']),
                                            p=float(p))
    print(f'[{name}] raw {RESULTS[f"X45:{name}:raw"]:.4f} bias {RESULTS[f"X45:{name}:bias"]:.4f} '
          f'LM {RESULTS[f"X45:{name}:classical_lm"]:.4f} learned_LM '
          f'{RESULTS[f"X45:{name}:learned_lm"]:.4f} learned_Adam {np.mean(adams):.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--groups', default='undisturbed')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    if args.quick:
        args.seeds, args.epochs = 1, 200
    t0 = time.time()
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    blocks = [('a-short', dict(W=20, stride=20)),
              ('b-035s', dict(W=100, stride=100)),
              ('c-1s', dict(W=286, stride=286))]
    if args.quick:
        blocks = blocks[-1:]
    for name, kw in blocks:
        run(name, args.data_dir, groups=groups, seeds=tuple(range(args.seeds)),
            epochs=args.epochs, lr=args.lr, quick=args.quick, **kw)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds,
                  extra_meta={'note': 'P-X45: learned (M,b) and classical (M,b) under the SAME LM',
                              'loss': 'geodesic (shared)'})
    print('dump ->', p)


if __name__ == '__main__':
    main()
