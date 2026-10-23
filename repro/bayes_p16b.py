#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayes floor for the P16b random-angle task (hidden, non-input angles).

Same Monte-Carlo posterior-mean methodology as E10 in run_main_experiments.py,
extended to marginalise over the hidden random angles theta ~ U[0.2, 2.5].
Also prints the fixed-angle E10 floors as a sanity check (n=2 must be ~2.697).
"""
import json, math, os, torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE = torch.tensor([0., 0., 1.], device=device)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1); w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

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
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)

def geo_deg(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(cos))

def make_compn_data(N, n=2, noise=0.03, seed=1, angles=None):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N, 3, generator=g); a = a/a.norm(dim=-1, keepdim=True); axes.append(a)
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

def floor_fixed(n, angles, N=2000, noise=0.03, M=256, seed=100):
    X, Y = make_compn_data(N, n=n, noise=noise, seed=seed, angles=angles)
    g = torch.Generator().manual_seed(seed + 777)
    draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
    al = draws[..., 0::2]; be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
    w = torch.sin(be).prod(-1); w = w/w.sum(1, keepdim=True)
    axes = torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)
    prod = axis_angle_to_quat(axes[:, :, -1], torch.full((N, M), float(angles[-1]), device=device))
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i],
                        torch.full((N, M), float(angles[i]), device=device)))
    v = rotate(prod, PROBE.expand(N, M, 3)); v = v/v.norm(dim=-1, keepdim=True)
    mean_dir = (v*w.unsqueeze(-1)).sum(1); mean_dir = mean_dir/(mean_dir.norm(dim=-1, keepdim=True)+1e-8)
    return geo_deg(vec_to_angles(mean_dir), Y).mean().item()

def floor_random(n=2, N=2000, noise=0.03, M=2048, seed=100, lo=0.2, hi=2.5):
    X, Y = make_compn_data(N, n=n, noise=noise, seed=seed, angles='random')
    g = torch.Generator().manual_seed(seed + 777)
    draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
    al = draws[..., 0::2]; be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
    w = torch.sin(be).prod(-1); w = w/w.sum(1, keepdim=True)
    axes = torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)   # (N,M,n,3)
    th = (lo + (hi-lo)*torch.rand(N, M, n, generator=g)).to(device)            # hidden angles, per draw
    prod = axis_angle_to_quat(axes[:, :, -1], th[:, :, -1])
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i], th[:, :, i]))
    v = rotate(prod, PROBE.expand(N, M, 3)); v = v/v.norm(dim=-1, keepdim=True)
    mean_dir = (v*w.unsqueeze(-1)).sum(1); mean_dir = mean_dir/(mean_dir.norm(dim=-1, keepdim=True)+1e-8)
    return geo_deg(vec_to_angles(mean_dir), Y).mean().item()

results = {
    'E10:n2 (sanity, expect ~2.697)': round(floor_fixed(2, (0.7, 1.1)), 3),
    'E10:n3 (sanity, expect ~3.057)': round(floor_fixed(3, (0.7, 1.1, 0.9)), 3),
    'E10b:P16b random hidden, M=256': round(floor_random(M=256), 3),
    'E10b:P16b random hidden, M=2048': round(floor_random(M=2048), 3),
}
print("===== SUMMARY-JSON =====")
print(json.dumps(results, indent=2))
# portable write next to this script (the harness also extracts the SUMMARY-JSON)
try:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json', 'bayes_p16b.json')
    json.dump(results, open(out_path, 'w'), indent=2)
except Exception as e:
    print("write skipped:", e)
