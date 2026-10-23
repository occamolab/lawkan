#!/usr/bin/env python3
"""X56 -- X53-kgrid: re-grid the kernel on q14 (bw x k), seed 3/N=60."""
import argparse, copy, os, sys
os.environ['FEAT'] = 'quat'
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA = os.path.join(_HERE, 'data', 'calib_data.pickle')
if not os.path.exists(_DATA):
    _DATA = os.path.join(_REPO, 'data', 'calib_data.pickle')
import X7_robot_calibration as X7, X24_fr3_gate as X24, X33_fr3_twoslot as X33
import X34_fr3_kernel as X34, X7_gn_ls as GN, X54_pole_check as X54
from scipy.cluster.vq import kmeans2
from repro.dump import save_dump

DT = X54.DT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    if a.quick:
        a.epochs = 200   # comparable to base (plateau ~200-250)
    q_np, obs_np = X7.load_philnad_dataset(_DATA)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1000 + a.seed).permutation(len(q_np))
    tr, te = perm[:a.N], perm[a.N:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
    exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, a.seed)
    feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
    base_bw = None
    RES = {'cells': {}}
    ks = (20,) if a.quick else (8, 12, 20, 33)
    for k in ks:
        cen, _ = kmeans2(feats.numpy(), k, minit='++', seed=a.seed)
        centers = torch.tensor(cen, dtype=DT)
        d = torch.cdist(centers, centers); md = float(torch.median(d[d > 0]))
        fs = ((1.0, 2.0, 4.0) if a.quick else (0.5, 1.0, 2.0, 4.0))
        for f in fs:
            ts = X33.make_two_slot(exact, centers, md * f)
            r, ns, thk = X54.train_ckpt(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                                        a.epochs, 5e-3, a.seed)
            vals = np.array([r[e] for e in sorted(r)])
            print(f'k={k:2d} bw={f:.1f}x  max {vals.max():6.2f}  final {vals[-1]:.3f}',
                  flush=True)
            RES['cells'][f'k{k}_bw{f}'] = {'k': int(k), 'bw_factor': float(f),
                                           'max': float(vals.max()), 'final': float(vals[-1]),
                                           'trace': {int(e): float(r[e]) for e in sorted(r)}}
    save_dump(RES, __file__, quick=bool(a.quick), args=a, seeds=1, epochs=a.epochs)

if __name__ == '__main__':
    main()
