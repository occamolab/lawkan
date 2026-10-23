#!/usr/bin/env python3
"""X54 -- is the two-slot spike a chart singularity (pole)? Treatment smoke.

The broken case: X46 A1, FR3 arm, seed 3, N=60 (spike to 8.9 mm at ~500
epochs). Variants, same exact edge / data / seeds:
  base     -- the X46 recipe (lr 5e-3, cosine, no bound)
  eps      -- chart axis regularized harder (eps 0.01 -> 0.1 in chart_feats)
  bounded  -- psi parameters clamped after each step to +/-bound (treatment, not output tanh)
  lr1e3    -- lr 1e-3 (the X52 fix)
Metric: max error over the checkpoints, final error, and the epochs where the
error exceeds 1.5x the final value.
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
import X14_fourier_law as X14               # noqa: E402
from repro.dump import save_dump            # noqa: E402

_ORIG_CF = X14.chart_feats
TH = {'lo': float('nan'), 'hi': float('nan')}


def _cf(p, *a, **k):
    out = _ORIG_CF(p, *a, **k)
    th = out[0] if isinstance(out, (tuple, list)) else out[..., 0]
    with torch.no_grad():
        TH['lo'] = float(th.min())
        TH['hi'] = float(th.max())
    return out


X14.chart_feats = _cf
import X7_gn_ls as GN                       # noqa: E402

DT = torch.float64
CK = list(range(10, 1001, 10))


def train_ckpt(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr, seed,
               clip=0.0, bound=0.0):
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
    ns = {}
    THK = {}
    rec = {'v': float('nan')}
    def _hook(mod, inp, out):
        rec['v'] = float(out.detach().norm(dim=-1).max())
        f = inp[0][0].detach()
        TH['lo'] = float(f[..., 0].min())
        TH['hi'] = float(f[..., 0].max())
    m.psi.register_forward_hook(_hook)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()
        if bound:
            with torch.no_grad():
                for p in params:
                    p.clamp_(min=-bound, max=bound)
        if ep in CK:
            out[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
            ns[ep] = rec['v']
            THK[ep] = (TH['lo'], TH['hi'])
    return out, ns, THK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=5e-3)
    args = ap.parse_args()
    q_np, obs_np = X7.load_philnad_dataset(_DATA)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1000 + args.seed).permutation(len(q_np))
    tr, te = perm[:args.N], perm[args.N:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
    exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, args.seed)
    feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
    from scipy.cluster.vq import kmeans2
    cen, _ = kmeans2(feats.numpy(), 20, minit='++', seed=args.seed)
    centers = torch.tensor(cen, dtype=DT)
    d = torch.cdist(centers, centers)
    bw = float(torch.median(d[d > 0]))
    exact_err = GN.eval_pos(exact, q_te, oq_te, ot_te)[0]
    print(f'exact error: {exact_err:.3f} mm')
    RES = {'exact_mm': float(exact_err), 'results': {}}
    for name, kw in (('base', {}), ('clip1', dict(clip=1.0)),
                     ('bound0.2', dict(bound=0.2)), ('lr1e3', {})):
        lr = 1e-3 if name == 'lr1e3' else args.lr
        ts = X33.make_two_slot(exact, centers, bw)
        r, ns, thk = train_ckpt(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                                args.epochs, lr, args.seed, **kw)
        eps_k = [e for e in CK if e in r]
        vals = np.array([r[e] for e in eps_k])
        nv = np.array([ns[e] for e in eps_k])
        hot = int((vals > 1.5 * vals[-1]).sum())
        epmax = eps_k[int(np.argmax(vals))]
        print(f'{name:9s} max {vals.max():7.2f}  final {vals[-1]:6.2f}  '
              f'epochs>1.5x {hot} | ||w|| min {nv.min():.3f} med {np.median(nv):.3f} '
              f'max {nv.max():.3f}  at err-max {ns[epmax]:.3f} | '
              f'theta_op at err-max {thk[epmax][0]:.2f} .. {thk[epmax][1]:.2f}', flush=True)
        RES['results'][name] = {
            'max': float(vals.max()), 'final': float(vals[-1]),
            'epochs_gt_1p5x': int(hot), 'err_epoch': int(epmax),
            'w_min': float(nv.min()), 'w_med': float(np.median(nv)), 'w_max': float(nv.max()),
            'theta_op_at_err': [float(thk[epmax][0]), float(thk[epmax][1])],
            'trace': {int(e): float(r[e]) for e in eps_k},
            'norm_trace': {int(e): float(ns[e]) for e in eps_k}}
    save_dump(RES, __file__, quick=bool(getattr(args, 'quick', False)), args=args,
              seeds=1, epochs=args.epochs)


if __name__ == '__main__':
    main()
