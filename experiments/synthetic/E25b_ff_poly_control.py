# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — E25b: FEED-FORWARD polynomial control (no state-conditioning)
#
# Pre-registered in the experiment journal (commit 8400133),
# held-out seeds 30-34, committed BEFORE this file existed / this run.
#
# Purpose: close the ablation gap in the rigid-body section. The only
# feed-forward control so far is Fourier K=1 (E17), which differs from the
# state-conditioned model on TWO axes at once (edge form AND unrolling).
# E25b isolates the edge form: a polynomial edge in w0 (linear/quad/cubic,
# parameter counts identical to E25: 12/39/120) with a SINGLE exp node and no
# state, no unroll.
#
# Task and protocol are verbatim from E25_state_rigid.py:
#   Euler equations, I=diag(1,2,3), w0 components uniform in magnitude on
#   [0.5,1.5], 20 steps, dt=0.05; input = w0 only; target = final orientation.
#   800 train / 4000 test, held-out seeds 30-34, Adam, 3000 epochs, grad clip
#   1.0, model init torch.manual_seed(100+seed); lr linear/quad 1e-2, cubic 1e-3.
#
# Run: python E25b_ff_poly_control.py
# Output: repro/json/e25b.20260926-ffpoly.json
# ============================================================
import torch, torch.nn as nn, time, json
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
import E25_state_rigid as E   # reuse poly_features/rotate/vec_to_angles/PROBE/geo_deg/geo_loss/make_rigid_data

device = E.device
RESULTS = {}

def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,4) for v in vals]}")

class FFStateKAN(nn.Module):
    """One-shot: edge = polynomial(w0) -> v in R^3; node = ONE exp. No state, no unroll."""
    def __init__(self, mode='quad'):
        super().__init__(); self.mode = mode
        d = {'linear': 4, 'quad': 13, 'wcubic': 40}[mode]
        self.W = nn.Parameter(torch.zeros(d, 3).normal_(std=0.01))
    def forward(self, w0):
        v = E.poly_features(w0, self.mode) @ self.W          # (B,3)
        nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q = torch.cat([torch.cos(nv), torch.sin(nv)*v/nv], -1)      # single node (exp)
        p = E.rotate(q, E.PROBE.expand(w0.shape[0], 3))
        return E.vec_to_angles(p/(p.norm(dim=-1, keepdim=True)+1e-6))

class FFFixed20(nn.Module):
    """Optional depth-isolation control: 20 compositions, step edge sees
    [poly(w0), one-hot(s)]; NO evolving state. Params = (13+20)*3 = 99."""
    def __init__(self, steps=20):
        super().__init__(); self.steps = steps
        self.W = nn.Parameter(torch.zeros(13 + steps, 3).normal_(std=0.01))
    def forward(self, w0):
        B = w0.shape[0]
        base = E.poly_features(w0, 'quad')
        q = None
        for s in range(self.steps):
            oh = torch.zeros(B, self.steps, device=w0.device); oh[:, s] = 1.0
            v = torch.cat([base, oh], -1) @ self.W
            nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            dq = torch.cat([torch.cos(nv), torch.sin(nv)*v/nv], -1)
            q = dq if q is None else E.quat_mul(q, dq)
        p = E.rotate(q, E.PROBE.expand(B, 3))
        return E.vec_to_angles(p/(p.norm(dim=-1, keepdim=True)+1e-6))

def train_eval(ctor, lr, seed, epochs=3000):
    torch.manual_seed(100 + seed)
    Xtr, Ytr = E.make_rigid_data(800, seed=seed)
    Xte, Yte = E.make_rigid_data(4000, seed=seed + 100)
    m = ctor().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    diverged = False
    for ep in range(epochs):
        opt.zero_grad(); loss = E.geo_loss(m(Xtr), Ytr); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss):
            diverged = True; break
    with torch.no_grad():
        err = E.geo_deg(m(Xte), Yte).mean().item()
    return (float('nan') if diverged else err), sum(p.numel() for p in m.parameters())

def main():
    t0 = time.time()
    configs = [
        ('FF-linear', lambda: FFStateKAN('linear'), 1e-2),
        ('FF-quad',   lambda: FFStateKAN('quad'),   1e-2),   # CRITICAL (P-E25b.1)
        ('FF-cubic',  lambda: FFStateKAN('wcubic'), 1e-3),
        ('FF-quad-fixed20', lambda: FFFixed20(20),  1e-2),   # optional, exploratory
    ]
    for name, ctor, lr in configs:
        vals = []; ps = 0
        for seed in range(30, 35):
            e, ps = train_eval(ctor, lr, seed)
            vals.append(e)
        reg(f'E25b:{name}', vals, params=ps, lr=lr, held_out_seeds=[30,31,32,33,34])
    RESULTS['E25b:wall_s'] = time.time() - t0
    save_dump(RESULTS, __file__, seeds=5, seed_start=30, epochs=3000)
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
