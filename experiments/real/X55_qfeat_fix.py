#!/usr/bin/env python3
"""X55 -- X53-fix smoke: quaternion pair features remove the operand branch cut.

Same broken case as X54 (FR3 arm, seed 3), base lr 5e-3, small grid. The
only change: `raw_feats` emits the operand quaternions (w,x,y,z) + the
translation (3) instead of the axis-angle [1,sin,cos] chart -- so the
features are invariant under the log branch (theta,a)->(2pi-theta,-a).
Prediction (registered X53-fix): the residual spike disappears; final not worse.
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA = os.path.join(_HERE, 'data', 'calib_data.pickle')
if not os.path.exists(_DATA):
    _DATA = os.path.join(_REPO, 'data', 'calib_data.pickle')

import X7_robot_calibration as X7          # noqa: E402
import X24_fr3_gate as X24                  # noqa: E402
import X33_fr3_twoslot as X33               # noqa: E402
import X34_fr3_kernel as X34                # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X54_pole_check as X54                # noqa: E402  (train_ckpt)
from repro.dump import save_dump            # noqa: E402


def qfeats(A, B):
    fs = []
    for p in (A, B):
        q, t = p
        fs += [q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4],
               t[..., 0:1], t[..., 1:2], t[..., 2:3]]
    return torch.cat(fs, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=5e-3)
    args = ap.parse_args()
    X34.raw_feats = qfeats          # the fix
    X33.raw_feats = qfeats          # the two-slot head uses its own symbol
    X34.QF, X34.HY = True, False    # match the 14-column features

    def cfeats(m, q):
        fs = []
        def hook(mod, inp, out):
            A, B = inp
            fs.append(qfeats(A, B).reshape(-1, 14).detach().cpu())
        h = m.psi.register_forward_hook(hook)
        with torch.no_grad():
            m(torch.tensor(q, dtype=X54.DT))
        h.remove()
        return torch.cat(fs, 0)

    X34.collect_feats = cfeats
    q_np, obs_np = X7.load_philnad_dataset(_DATA)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=X54.DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=X54.DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=X54.DT)
    perm = np.random.default_rng(1000 + args.seed).permutation(len(q_np))
    tr, te = perm[:args.N], perm[args.N:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
    exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, args.seed)
    feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
    from scipy.cluster.vq import kmeans2
    cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=args.seed)
    centers = torch.tensor(cen, dtype=X54.DT)
    d = torch.cdist(centers, centers)
    bw = float(torch.median(d[d > 0]))
    ts = X33.make_two_slot(exact, centers, bw)
    r, ns, thk = X54.train_ckpt(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                                args.epochs, args.lr, args.seed)
    eps = sorted(r)
    vals = np.array([r[e] for e in eps])
    hot = int((vals > 1.5 * vals[-1]).sum())
    print(f'X53-fix (quaternion feats) N={args.N} seed={args.seed}: '
          f'max {vals.max():.2f}  final {vals[-1]:.2f}  '
          f'epochs>1.5x {hot}/{len(eps)}')
    save_dump({'X53-fix': {'max': float(vals.max()), 'final': float(vals[-1]),
                           'epochs_gt_1p5x': hot,
                           'trace': {int(e): float(r[e]) for e in eps}}},
              __file__, args=args, seeds=1, epochs=args.epochs)


if __name__ == '__main__':
    main()
