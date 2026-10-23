#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
E34_x7_recipe.py — the Section-3/6.4 recipe on REAL robot data (learnable PoE)
=============================================================================

Companion to X7_robot_calibration.py. X7's paper model is *constant* per-joint
SE(3) corrections (63 raw parameters, constant edges). This script is the
planned E34 (audit B6, insight #5): it puts the RECIPE (theta-DEPENDENT edge)
on the same real Franka calibration, and adds E21-style automatic pruning of
the per-joint harmonic dictionary.

Recipe edge (per joint j), primary model ScrewArm (SE(3) edge), nested form:
    xi_j(theta) = xi_c_j + theta_j * d_j + Fourier_j(theta_j)  in R^6
    T_j = exp(xi_j(theta)) ∘ Rz(theta_j)                 (SE(3), exact screw)
    T = base ∘ Π_i (C_i ∘ T_j) ∘ E ∘ tool                (quaternion composition)
with xi_c the constant per-joint correction (the 'matrix' model X7 fits),
d the first-order theta term and
    Fourier_j(theta) = Σ_{k=1..K} [ c_{j,k} cos(k theta_j) + s_{j,k} sin(k theta_j) ] in R^6.
mode='const'  -> xi_c only (matrix analogue); mode='linear' -> xi_c + theta*d;
mode='fourier' -> full (constant + first order + harmonics) [ours, P56-P59].
The DC term is the fix for the earlier total-screw form (theta*xi0 + F), which
could not express a constant joint offset and carried a span floor.
The rotation-only edge QuatKANArm (v_j in R^3) is kept for provenance (--with-rot).

Pruning (E21 style): train the full K dictionary, rank harmonics per joint by
group norm, keep the top-m per joint, fix the mask and retrain (curve over m).

Registered predictions: P56–P59 (see the journal, pre-registered before a run).
Run after the X10b-v3/X10c queue (machine shared); smoke first with --quick.

Run (when authorised):
    python E34_x7_recipe.py                 # full: 5 seeds x {20,40,60,80}
    python E34_x7_recipe.py --quick         # 1 seed, n=80 smoke
    python E34_x7_recipe.py --prune         # + E21-style pruning curve
    python E34_x7_recipe.py --prune --K 2   # richer discovery curve (K harmonics)
    python E34_x7_recipe.py --stage prune --K 2  # only the curve (quick first look)
    python E34_x7_recipe.py --skip-poe      # skip the robotkinecal PoE-LS

Output: repro/json/E34_x7_recipe.<ts>.json   (no fixed filenames), via repro/dump.
  Keys:
    E34:n<n>:<method>     per-seed test & train pos/rot + gap (all methods)
    E34:paired:n<n>       paired t-tests of LawKANArm vs references, TOST vs PoE-LS
    E34:prune:n<n>:keep<k>  pruning curve, per-seed test/train, order per seed
                          (--prune-seeds N controls the number of splits, 0=--seeds)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7           # noqa: E402  (reuse data + protocol)
from repro.dump import save_dump            # noqa: E402

RESULTS = {}
DT = torch.float64


def paired_t(a, b):
    """Paired t-test across shared seeds (a - b); positive diff = a worse."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(n=int(len(d)), diff=float(d.mean()) if len(d) else None)
    if len(d) < 2:
        return out
    if np.allclose(d, 0.0):
        out.update(t=0.0, p=1.0)
        return out
    t, p = stats.ttest_rel(a, b)
    out.update(t=float(t), p=float(p), sd=float(d.std(ddof=1)))
    return out


def tost_parity(a, b, margin_frac=0.10):
    """TOST equivalence of a vs classical baseline b at a margin of margin_frac*mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    ref = margin_frac * abs(float(np.mean(b)))
    if len(d) < 2 or ref == 0.0:
        return dict(margin=ref, p=None, pass_tost=None)
    p_lo = stats.ttest_1samp(d, -ref, alternative='greater').pvalue
    p_hi = stats.ttest_1samp(d, ref, alternative='less').pvalue
    p = float(max(p_lo, p_hi))
    return dict(margin=float(ref), p=p, pass_tost=bool(p < 0.05))


def axis_from_ab(ab):
    """(n,2) free angles -> unit axis (sin b cos a, sin b sin a, cos b)."""
    al, be = ab[:, 0], ab[:, 1]
    return torch.stack([torch.sin(be) * torch.cos(al),
                        torch.sin(be) * torch.sin(al),
                        torch.cos(be)], dim=-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def active_params(model):
    """Effective parameter count under the pruning mask: masked harmonics get
    zero gradient and stay at their zero init, so they are not free parameters."""
    n = n_params(model)
    if hasattr(model, 'mask') and hasattr(model, 'fc'):
        n -= int((~model.mask).sum().item()) * model.fc.shape[-1] * 2
    return n


def train_recipe(model, q_tr, oq_tr, ot_tr, epochs, lr, seed, clip=1.0,
                 ang_weight=1.0, warm_epochs=0, loss_form='angle2'):
    """Adam + cosine decay, WITH gradient clipping (mirrors E21/E31d).

    X7's own train_model has no clip; here we clip at 1.0 because the
    theta-dependent Fourier edge has more parameters, we want the pruning
    curve comparable to E31d, and we want a signal of any gradient growth.
    Loss is |dp|^2 [m^2] + ang_weight * |ang|^2 [rad^2]; --ang-weight < 1
    rebalances position against orientation (--ang-weight 0.25 ~ L=0.5 m).
    Returns (model, max_grad_norm_seen)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gmax = 0.0
    warm, saved_mask = 0, None
    if warm_epochs and hasattr(model, 'mask'):
        warm = max(0, min(int(warm_epochs), int(epochs) - 1))
        saved_mask = model.mask.clone()
        model.mask.zero_()
        print(f'    [warm start] harmonics masked for {warm}/{epochs} epochs')
    for ep in range(epochs):
        if saved_mask is not None and ep == warm:
            model.mask.copy_(saved_mask)
        opt.zero_grad()
        q_pred, t_pred = model(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        if loss_form == 'cosine2':
            loss = (dp ** 2).mean() + ang_weight * 2.0 *                 (1.0 - dot.clamp(max=1.0 - 1e-7)).mean()
        else:
            ang = 2.0 * torch.acos(dot.clamp(max=1.0))
            loss = (dp ** 2).mean() + ang_weight * (ang ** 2).mean()
        loss.backward()
        if clip and clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # returns pre-clip norm
        else:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        gmax = max(gmax, float(gn))
        opt.step()
        sched.step()
    return model, gmax


@torch.no_grad()
def edge_stats(model, q):
    """Per-joint rotation-vector diagnostics: |v_j| and the Fourier part."""
    v = model.joint_vectors(q)                                  # (B,7,3)
    a = model._unit(model.axis)
    lin = q.unsqueeze(-1) * a.unsqueeze(0)                      # (B,7,3)
    four = v - lin
    vn = v.norm(dim=-1)                                         # (B,7)
    fn = four.norm(dim=-1)
    return dict(v_mean=float(vn.mean()), v_max=float(vn.max()),
                v_per_joint=[float(x) for x in vn.mean(0)],
                four_mean=float(fn.mean()), four_max=float(fn.max()))


@torch.no_grad()
def edge_stats_screw(model, q):
    """Per-joint SE(3)-edge diagnostics: |xi| total, the constant, linear and
    Fourier parts, and the split into rotation (|omega|) and translation (|nu|)."""
    xi = model.joint_xi(q)                                      # (B,7,6)
    if getattr(model, 'theta_b', False):
        g = (q + model.b.unsqueeze(0)).unsqueeze(-1)
        core = g * (model.xi_nom + model.dxi).unsqueeze(0)
        c = torch.zeros_like(xi)
    else:
        core = model.xi_c.unsqueeze(0).expand_as(xi)
        c = core
    four = xi - core
    dn = torch.zeros_like(xi[..., 0])
    if getattr(model, 'canon', False):
        if getattr(model, 'canon_style', 'norm') == 'angle':
            a = axis_from_ab(model.d_ab)
        else:
            a = model.d_axis / model.d_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        screw = torch.cat([a, model.d_nu], dim=-1)
        lin = q.unsqueeze(-1) * model.d_lam.view(1, -1, 1) * screw.unsqueeze(0)
        four = four - lin
        dn = lin.norm(dim=-1)
    elif hasattr(model, 'd'):
        lin = q.unsqueeze(-1) * model.d.unsqueeze(0)
        four = four - lin
        dn = lin.norm(dim=-1)
    cn = c.norm(dim=-1)
    xn = xi.norm(dim=-1)
    fn = four.norm(dim=-1)
    return dict(v_mean=float(xn.mean()), v_max=float(xn.max()),
                v_per_joint=[float(x) for x in xn.mean(0)],
                four_mean=float(fn.mean()), four_max=float(fn.max()),
                four_per_joint=[float(x) for x in fn.mean(0)],
                const_mean=float(cn.mean()), lin_mean=float(dn.mean()),
                rot_max=float(xi[..., :3].norm(dim=-1).max()),
                trans_max=float(xi[..., 3:].norm(dim=-1).max()))


# ---------------------------------------------------------------------------
# Recipe model: theta-dependent per-joint edge + base/tool
# ---------------------------------------------------------------------------

def quat_exp(v):
    """Rotation vector (…,3) -> unit quaternion (…,4), w-first (retraction, no norm divide)."""
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat([torch.cos(n / 2), torch.sin(n / 2) * v / n], dim=-1)


def hat3(w):
    B = w.shape[:-1]
    W = torch.zeros(*B, 3, 3, dtype=w.dtype, device=w.device)
    W[..., 0, 1] = -w[..., 2]; W[..., 0, 2] = w[..., 1]
    W[..., 1, 0] = w[..., 2];  W[..., 1, 2] = -w[..., 0]
    W[..., 2, 0] = -w[..., 1]; W[..., 2, 1] = w[..., 0]
    return W


def screw_exp(xi):
    """SE(3) exponential: xi (…,6) = (omega, nu) -> ((…,4) quat w-first, (…,3) translation).

    Same V-matrix convention as X10_fk_calib_real.se3_exp; the rotation part is
    exactly quat_exp(omega), so with xi = [0,0,1,0,0,0]*theta this is Rz(theta).
    """
    w, nu = xi[..., :3], xi[..., 3:]
    th = w.norm(dim=-1, keepdim=True)
    th_s = th.clamp_min(1e-8)
    W = hat3(w)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(W.shape)
    b = ((1.0 - torch.cos(th_s)) / th_s ** 2).unsqueeze(-1)
    c = ((th_s - torch.sin(th_s)) / th_s ** 3).unsqueeze(-1)
    V = I + b * W + c * (W @ W)
    t = (V @ nu.unsqueeze(-1)).squeeze(-1)
    return quat_exp(w), t


class QuatKANArm(nn.Module):
    """Sec-3 recipe on the real arm: v_j = theta_j a_j + Fourier_j(theta_j),
    q_j = exp(v_j), Hamilton composition, with base + tool SE(3).

    mode:
      'fourier' : full harmonic dictionary (K >= 1)          [ours]
      'linear'  : Fourier = 0 (learned-axis PoE)             [K=0]
      'const'   : Fourier -> a single constant vector/joint  [P59 diagnostic]
    """

    def __init__(self, C, E, K=1, mode='fourier'):
        super().__init__()
        assert mode in ('fourier', 'linear', 'const')
        self.K, self.mode = K, mode
        qs, ts = [], []
        for Ci in C:
            Rt = torch.tensor(Ci, dtype=DT)
            qs.append(X7.mat_to_quat(Rt[:3, :3]))
            ts.append(Rt[:3, 3].clone())
        self.register_buffer('C_q', torch.stack(qs))       # (7,4)
        self.register_buffer('C_t', torch.stack(ts))       # (7,3)
        Et = torch.tensor(E, dtype=DT)
        self.register_buffer('E_q', X7.mat_to_quat(Et[:3, :3]))
        self.register_buffer('E_t', Et[:3, 3].clone())
        n = self.C_q.shape[0]
        self.n = n
        # base + tool, as SE(3) via exp(rotvec) (retraction; no q/||q|| normalization)
        self.base_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.base_t = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_t = nn.Parameter(torch.zeros(3, dtype=DT))
        # learned per-joint axis (init z), unit-normalised in the forward pass
        raw = torch.zeros(n, 3, dtype=DT)
        raw[:, 2] = 1.0
        self.axis = nn.Parameter(raw)
        # harmonic coefficients (cos / sin), zero-init (fourier mode only)
        if mode == 'fourier':
            self.fc = nn.Parameter(torch.zeros(n, K, 3, dtype=DT))
            self.fs = nn.Parameter(torch.zeros(n, K, 3, dtype=DT))
            self.register_buffer('mask', torch.ones(n, K, dtype=torch.bool))
        elif mode == 'const':
            self.c_const = nn.Parameter(torch.zeros(n, 3, dtype=DT))   # const offset

    def _unit(self, v):
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def joint_vectors(self, th):
        """th (B,7) -> v (B,7,3)."""
        B = th.shape[0]
        a = self._unit(self.axis)                          # (7,3)
        v = th.unsqueeze(-1) * a.unsqueeze(0)              # (B,7,3)
        if self.mode == 'fourier':
            masked = self.fc * self.mask.unsqueeze(-1)
            maskeds = self.fs * self.mask.unsqueeze(-1)    # (7,K,3)
            for k in range(1, self.K + 1):
                ck = torch.cos(k * th).unsqueeze(-1)       # (B,7,1)
                sk = torch.sin(k * th).unsqueeze(-1)
                v = v + masked[:, k - 1].unsqueeze(0) * ck \
                      + maskeds[:, k - 1].unsqueeze(0) * sk
        elif self.mode == 'const':
            v = v + self.c_const.unsqueeze(0)
        return v

    def forward(self, q):
        B = q.shape[0]
        v = self.joint_vectors(q)                          # (B,7,3)
        nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        qj = torch.cat([torch.cos(nv / 2), torch.sin(nv / 2) * v / nv], -1)  # (B,7,4)
        pose = (quat_exp(self.base_r).expand(B, 4), self.base_t.expand(B, 3))
        z = torch.zeros(B, 3, dtype=DT, device=q.device)
        for i in range(self.n):
            pose = X7.compose(pose, (self.C_q[i].expand(B, 4), self.C_t[i].expand(B, 3)))
            pose = X7.compose(pose, (qj[:, i, :], z))
        pose = X7.compose(pose, (self.E_q.expand(B, 4), self.E_t.expand(B, 3)))
        pose = X7.compose(pose, (quat_exp(self.tool_r).expand(B, 4),
                                 self.tool_t.expand(B, 3)))
        return pose

    # --- pruning helpers (E21 style) ---
    def harmonic_norms(self):
        """(7,K) group norm of each harmonic (cos & sin, all 3 components)."""
        return torch.sqrt((self.fc ** 2).sum(-1) + (self.fs ** 2).sum(-1))

    def set_mask_keep_top(self, keep):
        """Keep the top-`keep` harmonics per joint (by harmonic_norms), mask the rest."""
        norms = self.harmonic_norms().detach()
        order = norms.argsort(dim=-1, descending=True)
        m = torch.zeros_like(norms, dtype=torch.bool)
        for j in range(m.shape[0]):
            m[j, order[j, :keep]] = True
        self.mask.copy_(m)
        return m


class ScrewArm(nn.Module):
    """Sec-3 recipe on the real arm, NESTED residual form: per joint

        xi_j(theta) = xi_c_j + theta_j * d_j + Fourier_j(theta_j)   in R^6,
        T_j = exp(xi_j(theta)) ∘ Rz(theta_j),
        T = base ∘ Π_i (C_i ∘ T_j) ∘ E ∘ tool.

    Nesting is the point of the design:
      xi_c = d = F = 0          -> X7.NominalFK exactly (nominal FK);
      xi_c only                 -> constant per-joint SE(3) correction (the
                                   'matrix' model X7 fits; exp(xi_c) covers
                                   all of SE(3), so this nests corr ∘ Rz);
      xi_c + theta*d + F(theta) -> constant + first-order + harmonic
                                   theta-dependence [ours, P56-P59].

    A DC term is present by construction: {theta, cos k theta, sin k theta}
    alone cannot express a joint offset / constant axis shift, so the earlier
    total-screw form (theta*xi0 + F) was NOT a superset of the matrix model
    and carried a training-independent span floor (Proposition 1).

    mode:
      'const'   : xi = xi_c                         (matrix analogue: F=0, d=0)
      'linear'  : xi = xi_c + theta*d                (first-order)
      'fourier' : xi = xi_c + theta*d + F(theta)     [ours]
    The rotation-only QuatKANArm edge is kept for provenance.

    canon=True replaces the free linear term theta*d by a canonical screw,
    xi_lin = lam_j * theta * (a_j, nu_j) with |a_j| = 1 and a scalar gain
    lam_j (init 0): the joint increment is a valid unit screw direction by
    construction, so the optimizer cannot stretch its magnitude to absorb
    offsets or gauge. This is to the linear term what the exp-map is to the
    q/||q|| parameterization. Exact for a revolute joint with offset:
    exp((theta+delta) xi_a) = exp(delta xi_a) o exp(theta xi_a).

    theta_b=True selects the co-author's screw form (NP_master 1.1):
    T_j = exp( (theta_j + b_j) * (xi_nom_j + dxi_j) + F_j(theta) ), with the
    nominal z-screw xi_nom, learned twist corrections dxi (6/joint), joint
    zeros b (1/joint) and the same Fourier residual F on top; F=0 and
    dxi=0 gives the nominal FK. The learned model is the constant-edge
    ScrewCalib of X10 with a theta-edge added; theta_b is complementary to
    canon (mutually exclusive).
    """

    def __init__(self, C, E, K=1, mode='fourier', canon=False, theta_b=False,
                 canon_style='norm'):
        super().__init__()
        assert mode in ('fourier', 'linear', 'const')
        assert not (canon and theta_b)
        self.K, self.mode, self.canon = K, mode, bool(canon)
        self.theta_b = bool(theta_b)
        self.canon_style = str(canon_style)
        qs, ts = [], []
        for Ci in C:
            Rt = torch.tensor(Ci, dtype=DT)
            qs.append(X7.mat_to_quat(Rt[:3, :3]))
            ts.append(Rt[:3, 3].clone())
        self.register_buffer('C_q', torch.stack(qs))       # (n,4)
        self.register_buffer('C_t', torch.stack(ts))       # (n,3)
        Et = torch.tensor(E, dtype=DT)
        self.register_buffer('E_q', X7.mat_to_quat(Et[:3, :3]))
        self.register_buffer('E_t', Et[:3, 3].clone())
        n = self.C_q.shape[0]
        self.n = n
        self.base_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.base_t = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_r = nn.Parameter(torch.zeros(3, dtype=DT))
        self.tool_t = nn.Parameter(torch.zeros(3, dtype=DT))
        if not self.theta_b:
            self.xi_c = nn.Parameter(torch.zeros(n, 6, dtype=DT))
        if self.theta_b:
            xn = torch.zeros(n, 6, dtype=DT)
            xn[:, 2] = 1.0
            self.register_buffer('xi_nom', xn)
            self.dxi = nn.Parameter(torch.zeros(n, 6, dtype=DT))
            self.b = nn.Parameter(torch.zeros(n, dtype=DT))
        if mode in ('linear', 'fourier') and not self.theta_b:
            if self.canon:
                self.d_nu = nn.Parameter(torch.zeros(n, 3, dtype=DT))
                self.d_lam = nn.Parameter(torch.zeros(n))  # gain, init 0
                if self.canon_style == 'angle':
                    # axis from two free angles; no normalization anywhere
                    self.d_ab = nn.Parameter(torch.zeros(n, 2, dtype=DT))
                else:
                    ax = torch.zeros(n, 3, dtype=DT)
                    ax[:, 2] = 1.0
                    self.d_axis = nn.Parameter(ax)      # normalized in forward
            else:
                self.d = nn.Parameter(torch.zeros(n, 6, dtype=DT))
        if mode == 'fourier':
            self.fc = nn.Parameter(torch.zeros(n, K, 6, dtype=DT))
            self.fs = nn.Parameter(torch.zeros(n, K, 6, dtype=DT))
            self.register_buffer('mask', torch.ones(n, K, dtype=torch.bool))

    def joint_xi(self, th):
        """th (B,n) -> xi (B,n,6)."""
        if self.theta_b:
            xi_eff = self.xi_nom + self.dxi                    # (n,6)
            g = (th + self.b.unsqueeze(0)).unsqueeze(-1)       # (B,n,1)
            v = g * xi_eff.unsqueeze(0)
        else:
            v = self.xi_c.unsqueeze(0).expand(th.shape[0], -1, -1)
        if self.theta_b:
            pass
        elif self.mode in ('linear', 'fourier'):
            if self.canon:
                if self.canon_style == 'angle':
                    a = axis_from_ab(self.d_ab)
                else:
                    a = self.d_axis / self.d_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                screw = torch.cat([a, self.d_nu], dim=-1)       # (n,6)
                gain = self.d_lam.view(1, -1, 1)                # (1,n,1)
                v = v + th.unsqueeze(-1) * gain * screw.unsqueeze(0)
            else:
                v = v + th.unsqueeze(-1) * self.d.unsqueeze(0)  # (B,n,6)
        if self.mode == 'fourier':
            masked = self.fc * self.mask.unsqueeze(-1)
            maskeds = self.fs * self.mask.unsqueeze(-1)    # (n,K,6)
            for k in range(1, self.K + 1):
                ck = torch.cos(k * th).unsqueeze(-1)       # (B,n,1)
                sk = torch.sin(k * th).unsqueeze(-1)
                v = v + masked[:, k - 1].unsqueeze(0) * ck \
                      + maskeds[:, k - 1].unsqueeze(0) * sk
        return v

    def forward(self, q):
        B = q.shape[0]
        xi = self.joint_xi(q)                              # (B,n,6)
        qr, tr = screw_exp(xi)                             # (B,n,4), (B,n,3)
        z3 = torch.zeros(B, 3, dtype=DT, device=q.device)
        pose = (quat_exp(self.base_r).expand(B, 4), self.base_t.expand(B, 3))
        for i in range(self.n):
            pose = X7.compose(pose, (self.C_q[i].expand(B, 4), self.C_t[i].expand(B, 3)))
            pose = X7.compose(pose, (qr[:, i, :], tr[:, i, :]))
            if not self.theta_b:
                pose = X7.compose(pose, (X7.rotz_quat(q[:, i]), z3))
        pose = X7.compose(pose, (self.E_q.expand(B, 4), self.E_t.expand(B, 3)))
        pose = X7.compose(pose, (quat_exp(self.tool_r).expand(B, 4),
                                 self.tool_t.expand(B, 3)))
        return pose

    # --- pruning helpers (E21 style) ---
    def harmonic_norms(self):
        """(n,K) group norm of each harmonic over all 6 components."""
        return torch.sqrt((self.fc ** 2).sum(-1) + (self.fs ** 2).sum(-1))

    def set_mask_keep_top(self, keep):
        norms = self.harmonic_norms().detach()
        order = norms.argsort(dim=-1, descending=True)
        m = torch.zeros_like(norms, dtype=torch.bool)
        for j in range(m.shape[0]):
            m[j, order[j, :keep]] = True
        self.mask.copy_(m)
        return m


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=50)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--ang-weight', type=float, default=1.0,
                    help='weight of the squared rotation term (default 1.0)')
    ap.add_argument('--clip', type=float, default=1.0,
                    help='global gradient-norm clip (default 1.0; 0 = off)')
    ap.add_argument('--K', type=int, default=1, help='Fourier harmonics per joint')
    ap.add_argument('--prune', action='store_true', help='E21-style pruning curve')
    ap.add_argument('--prune-seeds', type=int, default=0,
                    help='splits for the pruning curve (0 = use --seeds)')
    ap.add_argument('--prune-mode', choices=['rank', 'random', 'val'], default='rank',
                    help='pruning-curve mask source: magnitude ranking, random masks, '
                         'or inner-split validation selection')
    ap.add_argument('--val-frac', type=float, default=0.25,
                    help='validation fraction of the train set for --prune-mode val')
    ap.add_argument('--warm-epochs', type=int, default=0,
                    help='warm start: train with all harmonics masked for this many '
                         'epochs, then release them (optimizer state kept)')
    ap.add_argument('--edge-canon', action='store_true',
                    help='canonical linear term: lam_j * theta * (unit axis, moment) '
                         'instead of a free 6-vector theta*d')
    ap.add_argument('--loss-form', choices=['angle2', 'cosine2'], default='angle2',
                    help='angle2 = legacy acos^2 loss; cosine2 = 2(1-cos) (bounded grad)')
    ap.add_argument('--canon-style', choices=['norm', 'angle'], default='norm',
                    help='canonical axis: normalize a raw vector, or two free angles')
    ap.add_argument('--edge-coauthor', action='store_true',
                    help="co-author screw form: exp((theta+b_j)*(xi_nom+dxi_j) + F) "
                         "(X10 ScrewCalib + theta-edge)")
    ap.add_argument('--stage', choices=['all', 'main', 'prune'], default='all',
                    help='all: main + prune; main: skip prune; prune: only the curve')
    ap.add_argument('--with-rot', action='store_true',
                    help='also train the rotation-only QuatKANArm edge (provenance)')
    ap.add_argument('--skip-poe', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.edge_canon and args.edge_coauthor:
        ap.error('--edge-canon and --edge-coauthor are mutually exclusive')

    if args.quick:
        args.seeds, args.sizes, args.epochs = 1, '80', 300

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    n_total = len(q_np)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    print(f'[E34] dataset: {n_total} poses x {q_np.shape[1]} joints; K={args.K}')

    sizes = [int(s) for s in args.sizes.split(',')]
    seed_ids = range(args.seed_start, args.seed_start + args.seeds)
    methods = ['Nominal', 'BaseFit', 'PoE-LS', 'MLP-residual', 'PerJointConst',
               'LawKANArm-const', 'LawKANArm-linear', 'LawKANArm']
    if args.with_rot:
        methods.append('QuatKANArm-rot')
    results = {(s, m): [] for s in sizes for m in methods}
    gmax_log, edge_log = [], []
    params = {'PerJointConst': n_params(X7.NominalFK(C, E, fit_base=True,
                                                     fit_joints=True, fit_tool=True)),
              'LawKANArm-const': n_params(ScrewArm(C, E, mode='const')),
              'LawKANArm-linear': n_params(ScrewArm(C, E, mode='linear', canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)),
              'LawKANArm': n_params(ScrewArm(C, E, K=args.K, canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)),
              'QuatKANArm-rot': n_params(QuatKANArm(C, E, K=args.K))}
    print('[E34] params:', params)

    run_main = args.stage in ('all', 'main')
    s_max = max(sizes)
    dense_models = {}
    for si, seed in enumerate(seed_ids):
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(n_total)
        for n_tr in (sizes if run_main else []):
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, q_te = q_all[tr], q_all[te]
            oq_tr, ot_tr = oq_all[tr], ot_all[tr]
            oq_te, ot_te = oq_all[te], ot_all[te]

            te_tr = {}
            nom = X7.NominalFK(C, E)
            mm, dg = X7.eval_model(nom, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(nom, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'Nominal')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['Nominal'] = (float(mm.mean()), float(mt.mean()))

            m = X7.NominalFK(C, E, fit_base=True)
            m, _ = train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
            mm, dg = X7.eval_model(m, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(m, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'BaseFit')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['BaseFit'] = (float(mm.mean()), float(mt.mean()))

            if not args.skip_poe:
                from spatialmath import SE3
                q_tr_list = [q_np[i].tolist() for i in tr]
                obs_tr = [SE3(obs_np[i]) for i in tr]
                mm_a, dg_a = X7.run_poe_ls(q_tr_list, obs_tr,
                                           [q_np[i] for i in te],
                                           [obs_np[i] for i in te], C, E)
                results[(n_tr, 'PoE-LS')].append((mm_a.mean(), dg_a.mean(), None, None))

            torch.manual_seed(seed)
            m = X7.MLPResidual(C, E)
            m, _ = train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
            mm, dg = X7.eval_model(m, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(m, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'MLP-residual')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['MLP-residual'] = (float(mm.mean()), float(mt.mean()))

            torch.manual_seed(seed)
            m = X7.NominalFK(C, E, fit_base=True, fit_joints=True, fit_tool=True)
            m, _ = train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
            mm, dg = X7.eval_model(m, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(m, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'PerJointConst')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['PerJointConst'] = (float(mm.mean()), float(mt.mean()))

            torch.manual_seed(seed)
            m = ScrewArm(C, E, K=args.K, mode='fourier', canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)
            m, gmax = train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, warm_epochs=args.warm_epochs, loss_form=args.loss_form)
            gmax_log.append(gmax)
            st = edge_stats_screw(m, q_tr)
            edge_log.append(st)
            mm, dg = X7.eval_model(m, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(m, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'LawKANArm')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['LawKANArm'] = (float(mm.mean()), float(mt.mean()))
            if n_tr == s_max:
                dense_models[seed] = (m, gmax)
            print(f'    [LawKANArm SE3] max|grad|={gmax:.2f} | |xi| mean {st["v_mean"]:.3f} '
                  f'max {st["v_max"]:.3f} | |Fourier| mean {st["four_mean"]:.4f} '
                  f'| |omega|max {st["rot_max"]:.3f} |nu|max {st["trans_max"]:.4f}')

            torch.manual_seed(seed)
            mcst = ScrewArm(C, E, mode='const')
            mcst, _ = train_recipe(mcst, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
            mm, dg = X7.eval_model(mcst, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(mcst, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'LawKANArm-const')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['LawKANArm-const'] = (float(mm.mean()), float(mt.mean()))

            torch.manual_seed(seed)
            mlin = ScrewArm(C, E, K=args.K, mode='linear', canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)
            mlin, _ = train_recipe(mlin, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
            mm, dg = X7.eval_model(mlin, q_te, oq_te, ot_te)
            mt, dt = X7.eval_model(mlin, q_tr, oq_tr, ot_tr)
            results[(n_tr, 'LawKANArm-linear')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
            te_tr['LawKANArm-linear'] = (float(mm.mean()), float(mt.mean()))

            if args.with_rot:
                torch.manual_seed(seed)
                mr = QuatKANArm(C, E, K=args.K, mode='fourier')
                mr, _ = train_recipe(mr, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
                mm, dg = X7.eval_model(mr, q_te, oq_te, ot_te)
                mt, dt = X7.eval_model(mr, q_tr, oq_tr, ot_tr)
                results[(n_tr, 'QuatKANArm-rot')].append((mm.mean(), dg.mean(), mt.mean(), dt.mean()))
                te_tr['QuatKANArm-rot'] = (float(mm.mean()), float(mt.mean()))

            line = ' | '.join(f'{mth} '
                              f'{np.mean([r[0] for r in results[(n_tr, mth)]]):.3f}mm'
                              for mth in methods if results[(n_tr, mth)])
            print(f'[seed {seed} n={n_tr:3d}] {line}')
            gap = ' | '.join(f'{mth} {te_tr[mth][0]:.3f}/{te_tr[mth][1]:.3f}'
                             for mth in ('PerJointConst', 'LawKANArm-const',
                                         'LawKANArm-linear', 'LawKANArm',
                                         'MLP-residual')
                             if mth in te_tr)
            print(f'    test/train mm: {gap}')

    # --- per-seed registration: test & train, per method and size ---
    def cell(rows, idx):
        vals = [float(r[idx]) for r in rows if r[idx] is not None]
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                    per_seed=vals, n=len(vals))

    for s in sizes:
        for mth in methods:
            rows = results[(s, mth)]
            if not rows:
                continue
            entry = dict(n_seeds=len(rows),
                         test_pos_mm=cell(rows, 0),
                         test_rot_deg=cell(rows, 1))
            if any(r[2] is not None for r in rows):
                entry.update(
                    train_pos_mm=cell(rows, 2),
                    train_rot_deg=cell(rows, 3),
                    train_test_gap_mm=float(cell(rows, 0)['mean'] -
                                            cell(rows, 2)['mean']))
            RESULTS[f'E34:n{s}:{mth}'] = entry

    # --- paired statistics on shared splits (LawKANArm vs references) ---
    def fmt(v):
        return 'n/a' if v is None else f'{v:+.4f}'

    refs = [mth for mth in ('PerJointConst', 'LawKANArm-const',
                             'LawKANArm-linear', 'MLP-residual', 'PoE-LS')
            if any(results[(s0, mth)] for s0 in sizes)]
    for s in sizes:
        a = results[(s, 'LawKANArm')]
        if not a:
            continue
        a_pos, a_rot = [r[0] for r in a], [r[1] for r in a]
        cell_p = {}
        for ref in refs:
            b = results[(s, ref)]
            if not b:
                continue
            sub = dict(pos=paired_t(a_pos, [r[0] for r in b]),
                       rot=paired_t(a_rot, [r[1] for r in b]))
            if ref == 'PoE-LS':
                sub['pos_tost'] = tost_parity(a_pos, [r[0] for r in b])
                sub['rot_tost'] = tost_parity(a_rot, [r[1] for r in b])
            cell_p[ref] = sub
        lin = results[(s, 'LawKANArm-linear')]
        const = results[(s, 'PerJointConst')]
        if lin and const:
            cell_p['linear_vs_PerJointConst'] = dict(
                pos=paired_t([r[0] for r in lin], [r[0] for r in const]),
                rot=paired_t([r[1] for r in lin], [r[1] for r in const]))
        RESULTS[f'E34:paired:n{s}'] = cell_p
        txt = ' | '.join(
            f'{ref} Δpos {fmt(cell_p[ref]["pos"].get("diff"))} '
            f'p={cell_p[ref]["pos"].get("p")}'
            for ref in cell_p if ref in refs)
        print(f'[paired n={s}] LawKANArm vs {txt}')

    # --- registered-prediction verdicts (P56–P58) ---
    def mv(s, mth):
        r = results[(s, mth)]
        return np.mean([x[0] for x in r]), np.mean([x[1] for x in r])

    for s in sizes:
        if not results[(s, 'LawKANArm')]:
            continue
        qk_p, qk_r = mv(s, 'LawKANArm')
        lin_p, lin_r = mv(s, 'LawKANArm-linear')
        pc_p, _ = mv(s, 'PerJointConst')
        ml_p, ml_r = mv(s, 'MLP-residual')
        entry = dict(P56_const_pass=bool(qk_p <= pc_p),
                     P57_mlp_pos_pass=bool(qk_p <= ml_p),
                     P57_mlp_rot_pass=bool(qk_r <= ml_r),
                     LawKANArm_mm=float(qk_p), LawKANArm_rot_deg=float(qk_r),
                     LawKANArm_linear_mm=float(lin_p),
                     LawKANArm_linear_rot_deg=float(lin_r),
                     PerJointConst_mm=float(pc_p), MLP_mm=float(ml_p))
        if results[(s, 'PoE-LS')]:
            pl_p, pl_r = mv(s, 'PoE-LS')
            entry.update(P58_pos_ratio=float(qk_p / pl_p),
                         P58_rot_ratio=float(qk_r / pl_r),
                         P58_pos_ratio_linear=float(lin_p / pl_p),
                         PoE_LS_mm=float(pl_p))
        RESULTS[f'E34:verdicts:n{s}'] = entry
        print(f'[verdict n={s}] P56 {entry["P56_const_pass"]} '
              f'({qk_p:.3f} vs const {pc_p:.3f} mm) | '
              f'P57 {entry["P57_mlp_pos_pass"]} ({qk_p:.3f} vs MLP {ml_p:.3f}) | '
              f'linear {lin_p:.3f} mm')

    # --- E21-style pruning curve (P59), repeated across prune-seeds ---
    if (args.prune or args.stage == 'prune') and args.K >= 1:
        s = max(sizes)
        n_pr = args.prune_seeds or args.seeds
        pr_seeds = list(range(args.seed_start, args.seed_start + n_pr))
        keep_range = list(range(args.K, -1, -1))
        agg = {k: dict(te_pos=[], te_rot=[], tr_pos=[], tr_rot=[], pos_std=[],
                       grad=[], np=None, val_pos=[]) for k in keep_range}
        orders, selected = {}, {}
        for seed in pr_seeds:
            rng = np.random.default_rng(1000 + seed)
            perm = rng.permutation(n_total)
            tr, te = perm[:s], perm[s:]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            q_val = oq_val = ot_val = None
            if args.prune_mode == 'val':
                n_val = max(1, int(round(args.val_frac * s)))
                perm2 = np.random.default_rng(5000 + seed).permutation(s)
                val, itr = tr[perm2[:n_val]], tr[perm2[n_val:]]
                q_tr, oq_tr, ot_tr = q_all[itr], oq_all[itr], ot_all[itr]
                q_val, oq_val, ot_val = q_all[val], oq_all[val], ot_all[val]
            else:
                q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            # 1) mask source: magnitude ranking, random, or val-mode ranking
            if args.prune_mode == 'random':
                gfull, order = float('nan'), None
                print(f'[E34:prune] seed {seed} random masks (same size as keep)')
            else:
                if args.prune_mode == 'rank' and seed in dense_models:
                    full, gfull = dense_models[seed]
                else:
                    torch.manual_seed(seed)
                    full = ScrewArm(C, E, K=args.K, mode='fourier', canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)
                    full, gfull = train_recipe(full, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, loss_form=args.loss_form)
                norms = full.harmonic_norms().detach()
                order = norms.argsort(dim=-1, descending=True)
                orders[str(seed)] = order.tolist()
                print(f'[E34:prune] seed {seed} full max|grad|={gfull:.2f}; '
                      f'per-joint top harmonic order: {order.tolist()}')
            # 2) retrain FROM SCRATCH for each budget (mask fixed), same seed (E21/E31d)
            seed_sel = {}
            for keep in keep_range:
                torch.manual_seed(seed)
                m = ScrewArm(C, E, K=args.K, mode='fourier', canon=args.edge_canon, theta_b=args.edge_coauthor, canon_style=args.canon_style)
                mask = torch.zeros((m.n, args.K), dtype=torch.bool)
                if args.prune_mode == 'random':
                    rng_m = np.random.default_rng(9000 + 101 * seed + keep)
                    for j in range(mask.shape[0]):
                        if keep > 0:
                            mask[j, list(rng_m.choice(args.K, size=keep, replace=False))] = True
                else:
                    for j in range(mask.shape[0]):
                        mask[j, order[j, :keep]] = True
                m.mask.copy_(mask)
                m, gm = train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, clip=args.clip, ang_weight=args.ang_weight, warm_epochs=args.warm_epochs, loss_form=args.loss_form)
                mm, dg = X7.eval_model(m, q_te, oq_te, ot_te)
                mt, dt = X7.eval_model(m, q_tr, oq_tr, ot_tr)
                a = agg[keep]
                a['te_pos'].append(float(mm.mean()))
                a['te_rot'].append(float(dg.mean()))
                a['tr_pos'].append(float(mt.mean()))
                a['tr_rot'].append(float(dt.mean()))
                a['pos_std'].append(float(mm.std()))
                a['grad'].append(float(gm))
                a['np'] = active_params(m)
                if args.prune_mode == 'val':
                    mv, _ = X7.eval_model(m, q_val, oq_val, ot_val)
                    a['val_pos'].append(float(mv.mean()))
                    seed_sel[keep] = float(mv.mean())
                print(f'    seed {seed} keep={keep} p={a["np"]} '
                      f'test {mm.mean():.3f}mm train {mt.mean():.3f}mm'
                      + (f' val {a["val_pos"][-1]:.3f}mm' if args.prune_mode == 'val' else ''))
            if args.prune_mode == 'val' and seed_sel:
                selected[str(seed)] = min(seed_sel, key=seed_sel.get)
        for keep in keep_range:
            a = agg[keep]
            entry = dict(
                mean=float(np.mean(a['te_pos'])), std=float(np.std(a['te_pos'])),
                per_seed=[float(x) for x in a['te_pos']],
                test_rot_deg=dict(mean=float(np.mean(a['te_rot'])),
                                  std=float(np.std(a['te_rot'])),
                                  per_seed=[float(x) for x in a['te_rot']]),
                train_pos_mm=dict(mean=float(np.mean(a['tr_pos'])),
                                  std=float(np.std(a['tr_pos'])),
                                  per_seed=[float(x) for x in a['tr_pos']]),
                train_rot_deg=dict(mean=float(np.mean(a['tr_rot'])),
                                   std=float(np.std(a['tr_rot'])),
                                   per_seed=[float(x) for x in a['tr_rot']]),
                test_train_gap_mm=float(np.mean(a['te_pos']) - np.mean(a['tr_pos'])),
                keep=keep, n_params=a['np'],
                pos_std=float(np.mean(a['pos_std'])),
                grad_max=float(max(a['grad'])),
                n_prune_seeds=len(pr_seeds), prune_mode=args.prune_mode,
                order_per_seed=orders)
            if args.prune_mode == 'val' and a['val_pos']:
                entry['val_pos_mm'] = dict(mean=float(np.mean(a['val_pos'])),
                                           std=float(np.std(a['val_pos'])),
                                           per_seed=[float(x) for x in a['val_pos']])
            RESULTS[f'E34:prune:n{s}:keep{keep}'] = entry
        if args.prune_mode == 'val' and selected:
            sel_test = [agg[selected[str(seed)]]['te_pos'][i] for i, seed in enumerate(pr_seeds)]
            sel_train = [agg[selected[str(seed)]]['tr_pos'][i] for i, seed in enumerate(pr_seeds)]
            sel_keep = [int(selected[str(seed)]) for seed in pr_seeds]
            RESULTS[f'E34:prune-selected:n{s}'] = dict(
                keep_per_seed=sel_keep,
                test_pos_mm=dict(mean=float(np.mean(sel_test)), std=float(np.std(sel_test)),
                                 per_seed=[float(x) for x in sel_test]),
                train_pos_mm=dict(mean=float(np.mean(sel_train)), std=float(np.std(sel_train)),
                                  per_seed=[float(x) for x in sel_train]))
            print(f'[E34:prune-val] selected keep per seed {sel_keep}; '
                  f'test {np.mean(sel_test):.3f}mm')

    RESULTS['_meta_extra'] = dict(
        params=params, K=args.K, mode='fourier', seed_start=args.seed_start,
        clip=args.clip, ang_weight=args.ang_weight, warm_epochs=args.warm_epochs,
        edge_canon=args.edge_canon, canon_style=args.canon_style, edge_coauthor=args.edge_coauthor,
        edge_form='exp(xi_c + theta*d + F) ∘ Rz (nested residual)',
        grad_max_max=float(max(gmax_log)) if gmax_log else None,
        grad_max_median=float(np.median(gmax_log)) if gmax_log else None,
        edge_v_mean=float(np.mean([s['v_mean'] for s in edge_log])) if edge_log else None,
        edge_v_max=float(max([s['v_max'] for s in edge_log])) if edge_log else None,
        edge_const_mean=float(np.mean([s['const_mean'] for s in edge_log])) if edge_log else None,
        edge_lin_mean=float(np.mean([s['lin_mean'] for s in edge_log])) if edge_log else None,
        edge_four_mean=float(np.mean([s['four_mean'] for s in edge_log])) if edge_log else None)
    print(f"[E34] max|grad| max {RESULTS['_meta_extra']['grad_max_max']}, "
          f"edge |v| mean {RESULTS['_meta_extra']['edge_v_mean']}")
    dump_path = save_dump(RESULTS, __file__, quick=args.quick, seeds=args.seeds,
                          epochs=args.epochs, seed_start=args.seed_start, args=args)
    print(f'[E34] dump -> {dump_path}')
    print('\n===== SUMMARY-JSON =====')
    import json
    print(json.dumps(RESULTS, indent=1, default=float))


if __name__ == '__main__':
    main()
