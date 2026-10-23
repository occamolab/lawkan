#!/usr/bin/env python3
"""X57 -- remaining cells: global-only q14, k=n (NW), w>=0 canonicalization."""
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
import X34_fr3_kernel as X34, X54_pole_check as X54
from scipy.cluster.vq import kmeans2
from repro.dump import save_dump

DT = X54.DT
QF_ORIG = X34.raw_feats

def qfeats_canon(A, B):
    f = QF_ORIG(A, B)
    # canonicalize each operand's quaternion half (w>=0)
    q = f[:, :4]
    s = torch.sign(q[:, :1] + 1e-30)
    q = q * s
    return torch.cat([q, f[:, 4:]], -1)

def cell(exact, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, centers, bw, seed, epochs,
         kernel=True):
    ts = X33.make_two_slot(exact, centers, bw)
    ts.psi.use_kernel = kernel
    r, ns, thk = X54.train_ckpt(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                                epochs, 5e-3, seed)
    vals = np.array([r[e] for e in sorted(r)])
    return vals.max(), vals[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=200)
    a = ap.parse_args()
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
    n_tr = len(q_tr)
    specs = [('quat14', 20, True, False), ('global-only', 20, False, False),
             ('k=n', n_tr, True, False), ('canon', 20, True, True)]
    RES = {'cells': {}}
    for name, k, kernel, canon in specs:
        X34.raw_feats = qfeats_canon if canon else QF_ORIG
        cen, _ = kmeans2(feats.numpy(), k, minit='++', seed=a.seed)
        centers = torch.tensor(cen, dtype=DT)
        d = torch.cdist(centers, centers); md = float(torch.median(d[d > 0]))
        mx, fin = cell(exact, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, centers, md,
                       a.seed, a.epochs, kernel=kernel)
        print(f'{name:12s} k={k:2d}  max {mx:6.2f}  final {fin:.3f}', flush=True)
        RES['cells'][name] = {'k': int(k), 'kernel': bool(kernel), 'canon': bool(canon),
                              'max': float(mx), 'final': float(fin)}
    save_dump(RES, __file__, args=a, seeds=1, epochs=a.epochs)

if __name__ == '__main__':
    main()
