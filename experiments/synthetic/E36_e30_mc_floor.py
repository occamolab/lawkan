# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# QuatKAN - E36: Monte-Carlo Bayes floor for the full-SO(3) readout (E30)
#
# E30's caption quotes only a target-noise-only lower bound (5.48 deg). This
# computes the E10/E33-style floor WITH input noise for the same task:
# per-x posterior over the composed rotation from noisy phase draws (importance
# weights prod sin(beta)), fitted as
#   (a) extrinsic (sign-invariant) quaternion mean: top eigenvector of
#       sum_i w_i q_i q_i^T;
#   (b) intrinsic minimizer of the reported metric 2*acos|<p,Y>| via projected
#       Adam;
# both evaluated against the fresh noisy target Y of the E30 task (rnoise=0.03).
# Seeds 100-104 (the E30 test draws), N=4000, M=256.
# Journal: E36 (correction, registered before the run 2026-09-28).
#
# Run: python E36_e30_mc_floor.py [--quick]
# Output: repro/json/E36_e30_mc_floor.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import os
import time

import numpy as np
import torch

from E30_full_so3 import (make_compn_data, quat_mul, expmap, quat_geo_deg,
                          device)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

E30_DUMP = load_canon('e30')
NOISE = 0.03
ANG2 = (0.7, 1.1)


def posterior_rotations(X, M, noise=NOISE, seed=777):
    """Draw per-x posterior rotations from noisy phase perturbations of X."""
    N = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    draws = X.unsqueeze(1) + noise*torch.randn(N, M, 4, generator=g).to(device)
    al = draws[..., 0::2]
    be = draws[..., 1::2].clamp(1e-4, np.pi-1e-4)
    w = torch.sin(be).prod(-1)
    w = w/w.sum(1, keepdim=True)
    axes = torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al),
                        torch.cos(be)], dim=-1)
    q = quat_mul(expmap((ANG2[1]/2)*axes[:, :, 1]),
                 expmap((ANG2[0]/2)*axes[:, :, 0]))
    return q/(q.norm(dim=-1, keepdim=True) + 1e-12), w


def extrinsic_mean(q, w):
    E = torch.einsum('nm,nmc,nmd->ncd', w, q, q)
    E = 0.5*(E + E.transpose(-1, -2))
    _, V = torch.linalg.eigh(E)
    return V[:, :, -1]


def intrinsic_mean(q, w, p0, steps=200, lr=0.05):
    p = p0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pn = p/(p.norm(dim=-1, keepdim=True) + 1e-12)
        loss = (w*quat_geo_deg(pn.unsqueeze(1), q)).sum(1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            p /= (p.norm(dim=-1, keepdim=True) + 1e-12)
    return p.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--N', type=int, default=4000)
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()
    N, M, ns = args.N, args.M, args.seeds
    if args.quick:
        N, M, ns = 200, 32, 1

    ref = E30_DUMP
    lawkan = ref['E30:QuatKAN-SU(2)']['mean']
    lie = ref['E30:Lie head h=32']['mean']
    print(f"E36 | MC SO(3) floor | N={N} M={M} seeds={ns} | "
          f"LawKAN={lawkan:.3f} Lie={lie:.3f}", flush=True)

    t0 = time.time()
    ext, intr, bound = [], [], []
    for s in range(ns):
        seed = 100 + s
        X, Y = make_compn_data(N, seed=seed)
        _, Yclean = make_compn_data(N, seed=seed, rnoise=0.0)
        q, w = posterior_rotations(X, M, seed=777 + s)
        p_ext = extrinsic_mean(q, w)
        ext.append(float(quat_geo_deg(p_ext, Y).mean()))
        p_int = intrinsic_mean(q, w, p_ext)
        intr.append(float(quat_geo_deg(p_int, Y).mean()))
        bound.append(float(quat_geo_deg(Yclean, Y).mean()))
        print(f"[seed {seed}] ext {ext[-1]:6.3f} | int {intr[-1]:6.3f} | "
              f"target-only {bound[-1]:6.3f}", flush=True)

    results = {
        'E36:floor_extrinsic': dict(mean=float(np.mean(ext)), std=float(np.std(ext)),
                                    per_seed=ext),
        'E36:floor_intrinsic': dict(mean=float(np.mean(intr)), std=float(np.std(intr)),
                                    per_seed=intr),
        'E36:target_noise_only': dict(mean=float(np.mean(bound)),
                                      per_seed=bound),
    }
    floor = results['E36:floor_intrinsic']['mean']
    results['E36:note'] = dict(
        text=f"floor {floor:.3f} deg; LawKAN {lawkan:.3f} (+{lawkan-floor:.3f}); "
             f"Lie {lie:.3f} (+{lie-floor:.3f})")
    print(' ', results['E36:note']['text'], flush=True)

    results['E36:P(E36)'] = dict(
        text=f"{'PASS' if floor >= results['E36:target_noise_only']['mean'] - 1e-6 else 'FAIL'}: "
             f"floor {floor:.3f} >= target-only "
             f"{results['E36:target_noise_only']['mean']:.3f}")
    results['E36:falsified'] = dict(
        text=f"{'FALSIFIED' if floor > lawkan else 'ok'}: floor {floor:.3f} vs LawKAN {lawkan:.3f}")
    for k in ['E36:P(E36)', 'E36:falsified']:
        print(' ', k, results[k]['text'], flush=True)

    results['E36:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              extra_meta=dict(N=N, M=M, task='E30 full-SO(3) readout',
                              ref_dump='e30'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
