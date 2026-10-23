#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X20_fric_synth.py — P-FRIC-synth: backlash/hysteresis, a known violator with a
predicted sign.

Synthetic data on the FR3 nominal chain (X7): for each pose q_true, two matched
approaches are simulated, q_meas = q_true + s * beta, where s in {-1,+1}^7 is the
sign of the last motion of each joint and beta the (fixed) backlash deflection.
A constrained correction delta_j = gamma_j s_j + c_j (14 parameters, zero at
init) is trained on train poses; the exact law composes; nothing else is learned.

Registered predictions (plan 2026-10-05):
  P-FR1: sign signature — the matched-pair difference of the nominal residuals
         aligns with the predicted direction (fraction >= 0.9, permutation
         p < 0.01);
  P-FR2: the constrained correction removes >= 80% of the hysteresis floor and
         is inert on the no-friction control (TOST 10%);
  P-FR3: shuffled-s control — no signature, no gain.

Output: repro/json/X20_fric_synth.<ts>.json
Run: python X20_fric_synth.py [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = torch.float64
RESULTS = {}


class ChainPose(torch.nn.Module):
    """FK on the X7 nominal chain to the end-effector point (no corrections)."""

    def __init__(self, C, E):
        super().__init__()
        qs, ts = [], []
        for Ci in C:
            Rt = torch.tensor(Ci, dtype=DT)
            qs.append(X7.mat_to_quat(Rt[:3, :3]))
            ts.append(Rt[:3, 3].clone())
        self.register_buffer('C_q', torch.stack(qs))
        self.register_buffer('C_t', torch.stack(ts))
        Et = torch.tensor(E, dtype=DT)
        self.register_buffer('E_q', X7.mat_to_quat(Et[:3, :3]))
        self.register_buffer('E_t', Et[:3, 3].clone())
        self.n = self.C_q.shape[0]

    def forward(self, q):
        B = q.shape[0]
        z3 = torch.zeros(B, 3, dtype=DT)
        pose = (E34.quat_exp(torch.zeros(3, dtype=DT)).expand(B, 4), z3)
        for i in range(self.n):
            step = X7.compose((self.C_q[i].expand(B, 4), self.C_t[i].expand(B, 3)),
                              (X7.rotz_quat(q[:, i]), z3))
            pose = X7.compose(pose, step)
        pose = X7.compose(pose, (self.E_q.expand(B, 4), self.E_t.expand(B, 3)))
        return pose[1]


def make_data(rng, n_poses, beta, noise_mm):
    q_true = rng.uniform(-1.8, 1.8, size=(n_poses, 7))
    sA = rng.choice([-1.0, 1.0], size=(n_poses, 7))
    sB = -sA
    qA = q_true + sA * beta[None, :]
    qB = q_true + sB * beta[None, :]
    return (torch.tensor(q_true, dtype=DT), torch.tensor(sA, dtype=DT),
            torch.tensor(qA, dtype=DT), torch.tensor(qB, dtype=DT), noise_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--poses', type=int, default=600)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=1e-2)
    ap.add_argument('--beta-deg', type=float, default=0.3)
    ap.add_argument('--noise-mm', type=float, default=0.05)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.poses, args.steps = 120, 500
    t0 = time.time()

    C, E = X7.extract_nominal_chain()
    fk = ChainPose(C, E)
    rng = np.random.default_rng(0)
    beta = torch.full((7,), np.deg2rad(args.beta_deg), dtype=DT)

    q_true, sA, qA, qB, noise = make_data(rng, args.poses, beta.numpy(), args.noise_mm)
    sB = -sA
    ntr = int(0.7 * args.poses)
    tr = slice(0, ntr)
    te = slice(ntr, args.poses)

    # measurements: FK(q_meas) + noise
    def meas(q_meas):
        return fk(q_meas) + torch.tensor(rng.normal(0, args.noise_mm / 1000.0,
                                                    size=(len(q_meas), 3)), dtype=DT)

    pA, pB = meas(qA), meas(qB)          # physical endpoints (deflected)
    # encoders read q_true; the rigid model sees only q_true
    nA, nB = fk(q_true), fk(q_true)
    floor_tr = float((nA[tr] - pA[tr]).norm(dim=-1).mean() * 1000.0)  # mm
    floor_te = float((nA[te] - pA[te]).norm(dim=-1).mean() * 1000.0)

    # P-FR1 sign signature on matched pairs (test poses)
    d_meas = pA[te] - pB[te]                       # measured pair difference
    d_pred = (fk(q_true[te] + sA[te] * beta) - fk(q_true[te] - sA[te] * beta))  # predicted direction
    dots = (d_meas * d_pred).sum(-1)
    frac_pos = float((dots > 0).float().mean())
    # permutation: sign flips of predicted directions
    Bp = 2000
    null = []
    for _ in range(Bp):
        fl = torch.tensor(rng.choice([-1.0, 1.0], size=len(dots)), dtype=DT)
        null.append(float(((dots * fl) > 0).float().mean()))
    p_perm = float(np.mean(np.array(null) >= frac_pos))
    RESULTS['X20:sign_frac'] = frac_pos
    RESULTS['X20:sign_perm_p'] = p_perm
    print(f'[X20] P-FR1 sign fraction {frac_pos:.3f}, permutation p={p_perm:.4f}', flush=True)

    # constrained correction: predicted endpoint = fk(q_true + gamma*s + c)
    def fit_correction(q_in_tr, s_tr, p_tr, steps, lr, shuffle_s=False):
        gamma = torch.zeros(7, dtype=DT, requires_grad=True)
        c = torch.zeros(7, dtype=DT, requires_grad=True)
        opt = torch.optim.Adam([gamma, c], lr=lr)
        s_use = s_tr[torch.randperm(len(s_tr))] if shuffle_s else s_tr
        for it in range(steps):
            opt.zero_grad()
            qc = q_in_tr + (gamma.unsqueeze(0) * s_use + c.unsqueeze(0))
            loss = ((fk(qc) - p_tr) ** 2).sum(-1).mean()
            loss.backward()
            opt.step()
        return gamma.detach(), c.detach()

    gA = torch.cat([q_true[tr], q_true[tr]])
    sAll = torch.cat([sA[tr], sB[tr]])
    pAll = torch.cat([pA[tr], pB[tr]])
    gamma, cc = fit_correction(gA, sAll, pAll, args.steps, args.lr)
    with torch.no_grad():
        qc = q_true[te] + (gamma.unsqueeze(0) * sA[te] + cc.unsqueeze(0))
        err_corr = float((fk(qc) - pA[te]).norm(dim=-1).mean() * 1000.0)
    gain = 100.0 * (1 - err_corr / max(floor_te, 1e-12))
    RESULTS['X20:floor_mm'] = floor_te
    RESULTS['X20:corrected_mm'] = err_corr
    RESULTS['X20:gain_pct'] = gain
    RESULTS['X20:gamma'] = gamma.tolist()
    print(f'[X20] P-FR2 floor {floor_te:.4f} corrected {err_corr:.4f} mm '
          f'gain {gain:.1f}%', flush=True)

    # P-FR3 shuffled-s control
    gS, cS = fit_correction(gA, sAll, pAll, args.steps, args.lr, shuffle_s=True)
    with torch.no_grad():
        qc = q_true[te] + (gS.unsqueeze(0) * sA[te] + cS.unsqueeze(0))
        err_sh = float((fk(qc) - pA[te]).norm(dim=-1).mean() * 1000.0)
    gain_sh = 100.0 * (1 - err_sh / max(floor_te, 1e-12))
    RESULTS['X20:shuffled_gain_pct'] = gain_sh
    print(f'[X20] P-FR3 shuffled-s gain {gain_sh:.1f}%', flush=True)

    # no-friction control (beta = 0)
    qF = q_true
    pF = meas(qF)
    gam0, c0 = fit_correction(torch.cat([qF[tr], qF[tr]]),
                              torch.cat([sA[tr], sB[tr]]),
                              torch.cat([pF[tr], pF[tr]]), args.steps, args.lr)
    with torch.no_grad():
        qc = qF[te] + (gam0.unsqueeze(0) * sA[te] + c0.unsqueeze(0))
        err_fric0 = float((fk(qc) - pF[te]).norm(dim=-1).mean() * 1000.0)
    floor0 = float((fk(qF[te]) - pF[te]).norm(dim=-1).mean() * 1000.0)
    RESULTS['X20:nofriction_floor_mm'] = floor0
    RESULTS['X20:nofriction_corrected_mm'] = err_fric0
    RESULTS['X20:nofriction_effect_mm'] = err_fric0 - floor0
    print(f'[X20] P-FR2 control (no friction): floor {floor0:.4f} corrected '
          f'{err_fric0:.4f} mm (effect {err_fric0-floor0:+.4f})', flush=True)

    RESULTS['X20:protocol'] = dict(poses=args.poses, steps=args.steps,
                                   beta_deg=args.beta_deg, noise_mm=args.noise_mm)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'done {time.time()-t0:.0f}s; dump: {out}')


if __name__ == '__main__':
    main()
