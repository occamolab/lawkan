# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E33: Bayes floor under the REPORTED metric (Tier 2, reviewer #2)
#
# E10/E10b computed the floor as the minimizer of the training loss 1-cos
# (weighted mean of posterior directions), while the paper reports arccos
# (geodesic angle). Recompute the floor as the minimizer of the reported
# metric (weighted intrinsic/Karcher mean on S^2) on the SAME Monte-Carlo
# posterior draws, and report both values side by side.
# Regimes: comp n=2 (angles 0.7/1.1), comp n=3, random hidden angles (P16b).
# Pre-registration: E33 in the experiment journal (2026-09-27).
#
# Run: python E33_floor_reported_metric.py [--quick]
# Output: repro/json/E33_floor_reported_metric.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import math
import time

import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANG2 = (0.7, 1.1)
ANG3 = (0.7, 1.1, 0.9)
PROBE = torch.tensor([0., 0., 1.], device=device)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def axis_angle_to_quat(axis, angle):
    half = angle.unsqueeze(-1)/2
    return torch.cat([torch.cos(half), torch.sin(half)*axis], dim=-1)


def rotate(q, v):
    u, w = q[..., 1:], q[..., :1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)


def vec_to_angles(v):
    return torch.stack([torch.atan2(v[..., 1], v[..., 0]),
                        torch.acos(torch.clamp(v[..., 2], -1+1e-6, 1-1e-6))], dim=-1)


def angles_to_vec(a):
    al, be = a[..., 0], a[..., 1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al),
                        torch.cos(be)], dim=-1)


def geo_deg(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(cos))


def make_compn_data(N, n=2, noise=0.03, seed=1, angles=None):
    """Verbatim from run_main_experiments.py (E1 data generator)."""
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N, 3, generator=g)
        a = a/a.norm(dim=-1, keepdim=True)
        axes.append(a)
    if angles == 'random':
        ths = [0.2 + 2.3*torch.rand(N, generator=g) for _ in range(n)]
    elif angles is None:
        ths = [torch.full((N,), 0.7 + 0.2*i) for i in range(n)]
    else:
        ths = [torch.full((N,), float(angles[i])) for i in range(n)]
    prod = axis_angle_to_quat(axes[-1], ths[-1])
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[i], ths[i]))
    v = rotate(prod, torch.tensor([0., 0., 1.]).expand(N, 3))
    v = v/v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N, 2*n, generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N, 2, generator=g)
    return X.to(device), Y.to(device)


def posterior_draws(N, n, angles, noise, M, seed, random_angles=False):
    """Posterior directions v (N,M,3) and weights w (N,M); mirrors E10/E10b."""
    X, Y = make_compn_data(N, n=n, noise=noise, seed=seed, angles=angles)
    g = torch.Generator().manual_seed(seed + 777)
    draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
    al = draws[..., 0::2]
    be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
    w = torch.sin(be).prod(-1)
    w = w/w.sum(1, keepdim=True)
    axes = torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al),
                        torch.cos(be)], dim=-1)
    if random_angles:
        th = (0.2 + 2.3*torch.rand(N, M, n, generator=g)).to(device)
    else:
        th = torch.stack([torch.full((N, M), float(angles[i]), device=device)
                          for i in range(n)], dim=-1)
    prod = axis_angle_to_quat(axes[:, :, -1], th[:, :, -1])
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i], th[:, :, i]))
    v = rotate(prod, PROBE.expand(N, M, 3))
    v = v/v.norm(dim=-1, keepdim=True)
    return v, w, Y


def extrinsic_floor(v, w, Y):
    """E10 baseline: minimizer of 1-cos = weighted mean of directions."""
    m = (v*w.unsqueeze(-1)).sum(1)
    m = m/(m.norm(dim=-1, keepdim=True) + 1e-8)
    return geo_deg(vec_to_angles(m), Y).mean().item()


def intrinsic_floor(v, w, Y, iters=20):
    """Karcher (weighted intrinsic) mean on S^2, evaluated against the fresh noisy target Y
    (same estimand as the extrinsic E10 floor; fixed 2026-09-27: the first version wrongly
    evaluated against the posterior draws v, which understated the floor)."""
    m = (v*w.unsqueeze(-1)).sum(1)
    m = m/(m.norm(dim=-1, keepdim=True) + 1e-8)
    for _ in range(iters):
        c = (v*m.unsqueeze(1)).sum(-1).clamp(-1+1e-7, 1-1e-7)
        ang = torch.acos(c)
        logv = v - c.unsqueeze(-1)*m.unsqueeze(1)
        logv = logv/logv.norm(dim=-1, keepdim=True).clamp_min(1e-12)*ang.unsqueeze(-1)
        step = (w.unsqueeze(-1)*logv).sum(1)
        m = m + step
        m = m/(m.norm(dim=-1, keepdim=True) + 1e-8)
    return geo_deg(vec_to_angles(m), Y).mean().item()


def posterior_spread(v, w, iters=20):
    """Diagnostic: weighted mean angle between the posterior draws and their Karcher mean
    (the in-sample quantity the first E33 version reported as 'intrinsic floor')."""
    m = (v*w.unsqueeze(-1)).sum(1)
    m = m/(m.norm(dim=-1, keepdim=True) + 1e-8)
    for _ in range(iters):
        c = (v*m.unsqueeze(1)).sum(-1).clamp(-1+1e-7, 1-1e-7)
        ang = torch.acos(c)
        logv = v - c.unsqueeze(-1)*m.unsqueeze(1)
        logv = logv/logv.norm(dim=-1, keepdim=True).clamp_min(1e-12)*ang.unsqueeze(-1)
        step = (w.unsqueeze(-1)*logv).sum(1)
        m = m + step
        m = m/(m.norm(dim=-1, keepdim=True) + 1e-8)
    c = (v*m.unsqueeze(1)).sum(-1).clamp(-1+1e-7, 1-1e-7)
    return torch.rad2deg((w*torch.acos(c)).sum(1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--iters', type=int, default=20)
    args = ap.parse_args()
    N, M = (200, 64) if args.quick else (2000, 256)
    Mrnd = 512 if args.quick else 2048

    print(f"Device: {device} | E33 floor under reported metric | N={N} M={M}", flush=True)
    t0 = time.time()
    results = {}
    regimes = [
        ('n2', dict(n=2, angles=ANG2, M=M, random_angles=False)),
        ('n3', dict(n=3, angles=ANG3, M=M, random_angles=False)),
        ('random (P16b)', dict(n=2, angles='random', M=Mrnd, random_angles=True)),
    ]
    for name, kw in regimes:
        v, w, Y = posterior_draws(N, seed=100, noise=0.03, **kw)
        ext = extrinsic_floor(v, w, Y)
        intr = intrinsic_floor(v, w, Y, iters=args.iters)
        spr = posterior_spread(v, w, iters=args.iters)
        results[f'E33:{name}:extrinsic'] = dict(mean=float(ext))
        results[f'E33:{name}:intrinsic'] = dict(mean=float(intr))
        results[f'E33:{name}:delta'] = dict(mean=float(intr - ext))
        results[f'E33:{name}:posterior_spread'] = dict(mean=float(spr))
        print(f"[{name:14s}] extrinsic {ext:8.3f} deg | intrinsic {intr:8.3f} deg | "
              f"delta {intr-ext:+.4f} | spread {spr:.3f}", flush=True)

    old_n2 = 2.697
    ok = results['E33:n2:intrinsic']['mean'] <= old_n2 + 0.05
    results['E33:P(E33.1)'] = dict(
        text=f"{'PASS' if ok else 'FAIL'}: intrinsic n2 "
             f"{results['E33:n2:intrinsic']['mean']:.3f} <= old {old_n2} + 0.05")
    print(f"[E33.1] {results['E33:P(E33.1)']['text']}", flush=True)
    results['E33:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args,
              extra_meta=dict(N=N, M=M, Mrnd=Mrnd))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
