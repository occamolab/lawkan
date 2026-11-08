#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X7_robot_calibration.py — Real-robot kinematic calibration benchmark
====================================================================

Companion probe to the LawKAN paper. X5 (BROAD gyro-only
dead-reckoning) established a boundary: when the analytic group law IS the
right computation, learned models have nothing to add. X7 tests the opposite
regime on real data: forward-kinematics (FK) calibration of a real 7-DOF
robot arm, where the computation is rotation composition but the kinematic
parameters contain an unknown defect — the setting our synthetic arm
benchmark (Section 6.2) was designed to mimic.

Dataset
-------
PhilNad/robot-arm-kinematic-calibration (GitHub, MIT):
real Franka Research 3, 114 poses. Joint encoders + end-effector poses
observed by an Intel RealSense D405 camera mounted on the flange and
overlooking a printed calibration board. The pickle ships in the repo:
    https://raw.githubusercontent.com/PhilNad/robot-arm-kinematic-calibration/main/Examples/calib_data.pickle
Observed flange poses are reconstructed exactly as in Examples/FrankaReal.py:
    obs = X_B_W @ X_C_B @ inv(X_C_T)
with X_C_B = camera pose relative to the board (from the images),
X_C_T = camera relative to TCP (hand-measured constants of the 3D-printed
mount, copied verbatim from FrankaReal.py), X_B_W = board pose relative to
the robot base (hand-measured, copied verbatim).

Sanity anchor (this script, split-independent):
    nominal FK (panda_link8) vs observations:
    position error ~10.7 mm mean / ~1.9 mm std (pose-dependent part),
    rotation error ~2.15 deg mean / ~0.5 deg std.
The pose-dependent std is the calibratable defect; the constant mean is
dominated by the hand-measured board placement and is absorbed by a base
transform in every fitted method.

Registered predictions (BEFORE running — measured-study protocol)
-----------------------------------------------------------------
P32 QuatKANArm (structured corrections — base + per-joint quaternion
    corrections + tool, composed by Hamilton products; more expressive than
    PoE's minimal 4-parameters-per-joint model: 63 parameters vs 34
    identifiable) achieves <= MLP-residual test error at every training-set
    size (data-efficiency of the right inductive bias).
P33 At <=40 training poses QuatKANArm test error is within +10% of the
    classical PoE least-squares calibration (robotkinecal); at >=60 poses
    PoE-LS may win — either outcome is reported.
P34 All calibrated models beat Nominal+BaseFit beyond the constant offset,
    i.e. they reduce the pose-dependent residual (~1.9 mm / ~0.5 deg), not
    just the hand-measurement offset.

Honest caveats
--------------
* 114 poses total — tiny. We use repeated random splits (5 seeds) with
  identical splits across methods (paired comparison).
* X_C_T / X_B_W are hand-measured; their constant error aliases into a LEFT
  base transform. The learned models carry that base; PoE-LS is now given the
  same constant left base fit (best_left_base on train), so the comparison is
  symmetric w.r.t. the board placement. (Without it, the learned models held an
  unfair advantage — audit F8.)
* PoE-LS (robotkinecal) is the actual classical baseline used by the
  dataset authors, with a minimal parametrization — not a strawman.

Dependencies: numpy, torch, roboticstoolbox-python, robotkinecal
(pip install roboticstoolbox-python robotkinecal). CPU-only, ~25-35 min
for the full 5-seed × 4-size protocol (use --seeds 1 --sizes 80 for a
~1-min smoke run).

License: AGPL-3.0-or-later (see repository LICENSE).
"""

import argparse
import json
import os
import pickle
import urllib.request

import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo(start):
    d = start
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, 'repro')):
        d = os.path.dirname(d)
    return d


_REPO = _find_repo(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_URL = ("https://raw.githubusercontent.com/PhilNad/robot-arm-kinematic-"
            "calibration/main/Examples/calib_data.pickle")

# ---------------------------------------------------------------------------
# Dataset loading / observation reconstruction (mirrors Examples/FrankaReal.py)
# ---------------------------------------------------------------------------

def _mk(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _Rz_deg(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_philnad_dataset(pickle_path):
    """Download (if needed) and reconstruct observed flange poses.

    Returns
    -------
    q : (114, 7) float64 joint configurations
    obs : (114, 4, 4) float64 observed flange SE(3) poses in base frame
    """
    if not os.path.exists(pickle_path):
        os.makedirs(os.path.dirname(os.path.abspath(pickle_path)), exist_ok=True)
        print(f"[X7] downloading dataset -> {pickle_path}")
        urllib.request.urlretrieve(DATA_URL, pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    q = np.asarray(data["joint_positions"], dtype=np.float64)
    cam = np.stack([np.asarray(m, dtype=np.float64)
                    for m in data["camera_poses"]])

    # Camera relative to TCP (constants copied verbatim from FrankaReal.py)
    X_C_T = _mk(_Rz_deg(90.0), [94.5 / 1000, -9.0 / 1000, 40.55 / 1000])
    # Board relative to robot base (world); copied verbatim from FrankaReal.py
    board_centre = ((82.25 + 4 * 25 + 3 + 250) / 1000,
                    -(4.5 * 25 + 3 + 250) / 1000)
    top_left = (board_centre[0] + 0.5 / 2 - 0.025,
                board_centre[1] - 0.5 / 2 + 0.025)
    X_B_W = _mk(np.eye(3), [top_left[0], top_left[1], -9.7 / 1000])

    obs = np.stack([X_B_W @ cam[m] @ np.linalg.inv(X_C_T)
                    for m in range(len(cam))])
    return q, obs


# ---------------------------------------------------------------------------
# Nominal Panda kinematics (constants extracted from rtb URDF model)
# ---------------------------------------------------------------------------

def extract_nominal_chain():
    """Per-joint constant SE(3) C_i with T_i(q_i) = C_i @ Rz(q_i),
    plus the fixed EE offset E (panda_link8). Verified against
    rtb fkine to machine precision."""
    import roboticstoolbox as rtb
    from spatialmath import SE3
    panda = rtb.models.URDF.Panda()
    C, E = [], np.eye(4)
    for l in panda.links:
        if l.isjoint:
            C.append(np.asarray(SE3(l.ets[0].A()).A, dtype=np.float64))
        else:
            E = E @ np.asarray(SE3(l.ets[0].A()).A, dtype=np.float64) \
                if len(l.ets) > 0 else E
    return C, E


# ---------------------------------------------------------------------------
# Quaternion helpers (Hamilton convention, w-first) — torch, differentiable
# ---------------------------------------------------------------------------

def mat_to_quat(R):
    """(…,3,3) rotation matrix -> (…,4) unit quaternion, w-first.
    Shepperd's method: pick the largest of the four squared components, so the
    formula stays stable near trace(R) = -1 (the single w-branch is not)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    s = lambda t: 2.0 * torch.sqrt(t.clamp_min(1e-12))
    s0, s1, s2, s3 = s(1 + tr), s(1 + m00 - m11 - m22), s(1 + m11 - m00 - m22), s(1 + m22 - m00 - m11)
    c0 = tr > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    w = torch.where(c0, s0 / 4, torch.where(c1, (m21 - m12) / s1,
              torch.where(c2, (m02 - m20) / s2, (m10 - m01) / s3)))
    x = torch.where(c0, (m21 - m12) / s0, torch.where(c1, s1 / 4,
              torch.where(c2, (m01 + m10) / s2, (m02 + m20) / s3)))
    y = torch.where(c0, (m02 - m20) / s0, torch.where(c1, (m01 + m10) / s1,
              torch.where(c2, s2 / 4, (m12 + m21) / s3)))
    z = torch.where(c0, (m10 - m01) / s0, torch.where(c1, (m02 + m20) / s1,
              torch.where(c2, (m12 + m21) / s2, s3 / 4)))
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def quat_rot(q, v):
    """Rotate vector(s) v (…,3) by quaternion(s) q (…,4)."""
    qv = q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def rotz_quat(theta):
    """Quaternion of Rz(theta); theta (…,) -> (…,4)."""
    h = theta / 2.0
    return torch.stack([torch.cos(h), torch.zeros_like(h),
                        torch.zeros_like(h), torch.sin(h)], dim=-1)


def compose(p1, p2):
    """SE(3) composition of (q,t) pairs: (q1,t1) ∘ (q2,t2)."""
    q1, t1 = p1
    q2, t2 = p2
    return quat_mul(q1, q2), t1 + quat_rot(q1, t2)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class NominalFK(nn.Module):
    """T(q) = base ∘ Π_i (C_i ∘ corr_i ∘ Rz(q_i)) ∘ E.

    corr_i and base are identity unless parameters are registered by
    subclass flags. Quaternion composition throughout (Hamilton product).
    """

    def __init__(self, C, E, fit_base=False, fit_joints=False,
                 fit_tool=False):
        super().__init__()
        qs, ts = [], []
        for Ci in C:
            Rt = torch.tensor(Ci, dtype=torch.float64)
            qs.append(mat_to_quat(Rt[:3, :3]))
            ts.append(Rt[:3, 3].clone())
        self.register_buffer("C_q", torch.stack(qs))      # (7,4)
        self.register_buffer("C_t", torch.stack(ts))      # (7,3)
        Et = torch.tensor(E, dtype=torch.float64)
        self.register_buffer("E_q", mat_to_quat(Et[:3, :3]))
        self.register_buffer("E_t", Et[:3, 3].clone())
        # learnable corrections (raw params; quaternions normalized in fwd)
        if fit_base:
            self.base_q = nn.Parameter(torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            self.base_t = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        else:
            self.base_q = None
        if fit_joints:
            dq = torch.zeros(7, 4, dtype=torch.float64)
            dq[:, 0] = 1.0
            self.j_q = nn.Parameter(dq)
            self.j_t = nn.Parameter(torch.zeros(7, 3, dtype=torch.float64))
        else:
            self.j_q = None
        if fit_tool:
            self.tool_q = nn.Parameter(torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
            self.tool_t = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        else:
            self.tool_q = None

    def _norm(self, q):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def fk(self, q):
        """q (B,7) -> ((B,4) quat, (B,3) pos)."""
        B = q.shape[0]
        if self.base_q is not None:
            pose = (self._norm(self.base_q).expand(B, 4),
                    self.base_t.expand(B, 3))
        else:
            pose = (torch.tensor([1.0, 0.0, 0.0, 0.0],
                                 dtype=torch.float64).expand(B, 4),
                    torch.zeros(3, dtype=torch.float64).expand(B, 3))
        for i in range(7):
            pose = compose(pose, (self.C_q[i].expand(B, 4),
                                  self.C_t[i].expand(B, 3)))
            if self.j_q is not None:
                pose = compose(pose, (self._norm(self.j_q[i]).expand(B, 4),
                                      self.j_t[i].expand(B, 3)))
            pose = compose(pose, (rotz_quat(q[:, i]),
                                  torch.zeros(B, 3, dtype=torch.float64,
                                              device=q.device)))
        pose = compose(pose, (self.E_q.expand(B, 4), self.E_t.expand(B, 3)))
        if self.tool_q is not None:
            pose = compose(pose, (self._norm(self.tool_q).expand(B, 4),
                                  self.tool_t.expand(B, 3)))
        return pose

    def forward(self, q):
        return self.fk(q)


class MLPResidual(nn.Module):
    """Nominal FK + learned base + MLP(q) end-effector residual.

    T_pred = base ∘ T_nom(q) ∘ exp([w(q), v(q)]) — the residual acts in the
    end-effector frame, the standard residual-calibration placement.
    """

    def __init__(self, C, E, hidden=64):
        super().__init__()
        self.nom = NominalFK(C, E, fit_base=True, fit_joints=False)
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 6)).double()
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, q):
        (qn, tn) = self.nom(q)
        delta = self.net(q) * 0.1  # small-residual scaling (net is float64)
        w, v = delta[:, :3], delta[:, 3:]
        ang = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dq = torch.cat([torch.cos(ang / 2),
                        torch.sin(ang / 2) * w / ang], dim=-1)
        return quat_mul(qn, dq), tn + quat_rot(qn, v)


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------

def pose_error(pred, obs_q, obs_t):
    """pred=(q,t); returns per-sample (pos_m, angle_rad)."""
    q, t = pred
    dp = (t - obs_t).norm(dim=-1)
    dot = (q * obs_q).sum(-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)
    return dp, ang


def train_model(model, q_tr, oq_tr, ot_tr, epochs, lr, seed, ang_weight=1.0):
    """Adam + cosine decay to zero. The cosine schedule is load-bearing:
    with constant lr=1e-2 some seeds diverge on the (base, per-joint, tool)
    parametrization (mixed quaternion/translation scales); decaying lr to 0
    makes all seeds converge to the same optimum (verified: 5/5 seeds,
    1.434±0.094 mm at n=60 vs 5.118±4.995 mm without the schedule)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        opt.zero_grad()
        dp, ang = pose_error(model(q_tr), oq_tr, ot_tr)
        loss = (dp ** 2).mean() + ang_weight * (ang ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def eval_model(model, q_te, oq_te, ot_te):
    dp, ang = pose_error(model(q_te), oq_te, ot_te)
    return dp.numpy() * 1000.0, np.degrees(ang.numpy())  # mm, deg


# ---------------------------------------------------------------------------
# Classical baseline: PoE least-squares calibration (robotkinecal)
# ---------------------------------------------------------------------------

def _mat_to_quat_np(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S; x = (R[2,1]-R[1,2])/S; y = (R[0,2]-R[2,0])/S; z = (R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0]-R[1,1]-R[2,2]) * 2.0
        w = (R[2,1]-R[1,2])/S; x = 0.25*S; y = (R[0,1]+R[1,0])/S; z = (R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1]-R[0,0]-R[2,2]) * 2.0
        w = (R[0,2]-R[2,0])/S; x = (R[0,1]+R[1,0])/S; y = 0.25*S; z = (R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1.0 + R[2,2]-R[0,0]-R[1,1]) * 2.0
        w = (R[1,0]-R[0,1])/S; x = (R[0,2]+R[2,0])/S; y = (R[1,2]+R[2,1])/S; z = 0.25*S
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)

def _quat_to_mat_np(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def best_left_base(pred_Ts, obs_Ts):
    """Optimal constant LEFT transform B minimizing || B @ pred_i - obs_i ||:
    the same base degree of freedom the learned models fit. B_i = obs_i @ pred_i^-1
    is nearly constant (the hand-measured board placement error); average in SE(3)."""
    Bs = [O @ np.linalg.inv(T) for T, O in zip(pred_Ts, obs_Ts)]
    t = np.mean([B[:3, 3] for B in Bs], axis=0)
    qs = [_mat_to_quat_np(B[:3, :3]) for B in Bs]
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[0]) < 0:
            qs[i] = -qs[i]
    q = np.mean(qs, axis=0); q = q / np.linalg.norm(q)
    B = np.eye(4); B[:3, :3] = _quat_to_mat_np(q); B[:3, 3] = t
    return B

def run_poe_ls(q_tr_list, obs_tr_se3, q_te, obs_te, C, E, fit_base=True):
    """Calibrate on train, evaluate on test. Returns (mm_array, deg_array).
    With fit_base=True a constant LEFT base transform is fitted on the train
    predictions (rigid alignment), matching the base degree of freedom that the
    learned models (BaseFit / MLP-residual / QuatKANArm) also carry — so the
    comparison is symmetric w.r.t. the hand-measured board placement."""
    import roboticstoolbox as rtb
    from spatialmath import SE3
    from robotkinecal import SerialRobotKineCal
    panda = rtb.models.URDF.Panda()
    ee_idx = [i for i, l in enumerate(panda.links)
              if l.name == 'panda_link8'][0]
    cal = SerialRobotKineCal(panda, 'panda_link8', verbose=False)
    cal.set_observations(q_tr_list, obs_tr_se3)
    cal.solve(max_iterations=200)
    if fit_base:
        Ttr = [np.asarray(cal.forward_kinematics(ee_idx, np.asarray(q)))
               for q in q_tr_list]
        obs_tr_np = [np.asarray(o) for o in obs_tr_se3]
        B = best_left_base(Ttr, obs_tr_np)
    else:
        B = np.eye(4)
    pe, re = [], []
    for i in range(len(q_te)):
        T = B @ np.asarray(cal.forward_kinematics(ee_idx, np.asarray(q_te[i])))
        D = np.linalg.inv(T) @ obs_te[i]
        pe.append(np.linalg.norm(D[:3, 3]) * 1000.0)
        re.append(np.degrees(np.arccos(
            np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))
    return np.asarray(pe), np.asarray(re)


# ---------------------------------------------------------------------------
# Main protocol: paired repeated splits, train-size curve
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "calib_data.pickle"))
    ap.add_argument("--sizes", default="20,40,60,80")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--ang-weight", type=float, default=1.0,
                    help="weight of the squared rotation term (position metric: 0.25)")
    ap.add_argument("--skip-poe", action="store_true",
                    help="skip the robotkinecal PoE-LS baseline")
    args = ap.parse_args()

    q_np, obs_np = load_philnad_dataset(args.data)
    n_total = len(q_np)
    print(f"[X7] dataset: {n_total} poses, {q_np.shape[1]} joints")

    C, E = extract_nominal_chain()

    q_all = torch.tensor(q_np, dtype=torch.float64)
    oq_all = mat_to_quat(torch.tensor(obs_np[:, :3, :3],
                                      dtype=torch.float64))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=torch.float64)

    # sanity anchor: nominal FK vs observations (no fitting at all)
    nom = NominalFK(C, E)
    mm, dg = eval_model(nom, q_all, oq_all, ot_all)
    print(f"[X7] sanity — Nominal zero-shot: pos {mm.mean():.2f}±{mm.std():.2f} mm, "
          f"rot {dg.mean():.3f}±{dg.std():.3f} deg (expect ~10.7/~1.9 mm, ~2.15/~0.5 deg)")

    sizes = [int(s) for s in args.sizes.split(",")]
    methods = ["Nominal", "BaseFit", "PoE-LS", "MLP-residual", "QuatKANArm"]
    results = {(s, m): [] for s in sizes for m in methods}

    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, q_te = q_all[tr], q_all[te]
            oq_tr, ot_tr = oq_all[tr], ot_all[tr]
            oq_te, ot_te = oq_all[te], ot_all[te]

            # --- Nominal (no fit) ---
            mm, dg = eval_model(NominalFK(C, E), q_te, oq_te, ot_te)
            results[(n_tr, "Nominal")].append((mm.mean(), dg.mean()))

            # --- BaseFit (learned constant base SE(3) only) ---
            m = NominalFK(C, E, fit_base=True)
            train_model(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, ang_weight=args.ang_weight)
            mm, dg = eval_model(m, q_te, oq_te, ot_te)
            results[(n_tr, "BaseFit")].append((mm.mean(), dg.mean()))

            # --- PoE-LS (classical, robotkinecal) ---
            if not args.skip_poe:
                from spatialmath import SE3
                q_tr_list = [q_np[i].tolist() for i in tr]
                obs_tr = [SE3(obs_np[i]) for i in tr]
                obs_te_np = [obs_np[i] for i in te]
                mm_a, dg_a = run_poe_ls(q_tr_list, obs_tr,
                                        [q_np[i] for i in te], obs_te_np, C, E)
                results[(n_tr, "PoE-LS")].append((mm_a.mean(), dg_a.mean()))

            # --- MLP-residual ---
            torch.manual_seed(seed)   # B3 fix: seed init before constructing the model
            m = MLPResidual(C, E)
            train_model(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, ang_weight=args.ang_weight)
            mm, dg = eval_model(m, q_te, oq_te, ot_te)
            results[(n_tr, "MLP-residual")].append((mm.mean(), dg.mean()))

            # --- LawKANArm (structured: base + per-joint quaternion
            #     corrections + tool — same identifiable set as PoE-LS) ---
            m = NominalFK(C, E, fit_base=True, fit_joints=True,
                          fit_tool=True)
            train_model(m, q_tr, oq_tr, ot_tr, args.epochs, args.lr, seed, ang_weight=args.ang_weight)
            mm, dg = eval_model(m, q_te, oq_te, ot_te)
            results[(n_tr, "QuatKANArm")].append((mm.mean(), dg.mean()))

            line = " | ".join(
                f"{mth} {np.mean([r[0] for r in results[(n_tr, mth)]]):.3f}mm"
                for mth in methods if results[(n_tr, mth)])
            print(f"[seed {seed} n={n_tr:3d}] {line}")

    print("\n=== X7 final (mean ± std over seeds; test poses) ===")
    hdr = f"{'train n':>8s} " + " ".join(f"{m:>22s}" for m in methods)
    print(hdr)
    summary = {}
    for n_tr in sizes:
        cells = []
        for mth in methods:
            r = results[(n_tr, mth)]
            if not r:
                cells.append(f"{'—':>22s}")
                continue
            mm_v = np.array([x[0] for x in r])
            dg_v = np.array([x[1] for x in r])
            cells.append(f"{mm_v.mean():6.3f}±{mm_v.std():.3f}mm "
                         f"{dg_v.mean():4.2f}°")
            summary[f"X7:n{n_tr}:{mth}"] = dict(
                pos_mm_mean=float(mm_v.mean()), pos_mm_std=float(mm_v.std()),
                rot_deg_mean=float(dg_v.mean()), rot_deg_std=float(dg_v.std()),
                per_seed=[[float(a), float(b)] for a, b in r])
        print(f"{n_tr:>8d} " + " ".join(f"{c:>22s}" for c in cells))

    # registered-prediction verdicts
    print("\n=== verdicts vs registered predictions ===")
    for n_tr in sizes:
        def mv(mth):
            r = results[(n_tr, mth)]
            return (np.mean([x[0] for x in r]), np.mean([x[1] for x in r]))
        qk, ml = mv("QuatKANArm"), mv("MLP-residual")
        print(f"n={n_tr:3d} P32 (QuatKANArm <= MLP): pos "
              f"{'PASS' if qk[0] <= ml[0] else 'FAIL'} "
              f"({qk[0]:.3f} vs {ml[0]:.3f} mm), rot "
              f"{'PASS' if qk[1] <= ml[1] else 'FAIL'} "
              f"({qk[1]:.3f} vs {ml[1]:.3f} deg)", end="")
        if results[(n_tr, "PoE-LS")]:
            pl = mv("PoE-LS")
            print(f" | P33 (vs PoE-LS): pos ratio {qk[0]/pl[0]:.2f}, "
                  f"rot ratio {qk[1]/pl[1]:.2f}", end="")
        bf = mv("BaseFit")
        print(f" | P34 (vs BaseFit): pos {bf[0]/max(qk[0],1e-9):.2f}x better")
        entry = dict(P32_pos_pass=bool(qk[0] <= ml[0]),
                     P32_rot_pass=bool(qk[1] <= ml[1]),
                     QuatKANArm_mm=float(qk[0]), MLP_mm=float(ml[0]),
                     QuatKANArm_deg=float(qk[1]), MLP_deg=float(ml[1]),
                     P34_basefit_x_better=float(bf[0] / max(qk[0], 1e-9)))
        if results[(n_tr, "PoE-LS")]:
            entry["PoE_LS_mm"] = float(pl[0])
            entry["PoE_LS_deg"] = float(pl[1])
            entry["P33_pos_ratio"] = float(qk[0] / pl[0])
            entry["P33_rot_ratio"] = float(qk[1] / pl[1])
        summary[f"X7:verdicts:n{n_tr}"] = entry

    summary["_meta"] = dict(dataset=str(args.data), n_poses=int(n_total),
                            seeds=int(args.seeds), sizes=sizes,
                            epochs=int(args.epochs),
                            skip_poe=bool(args.skip_poe), version="1.0")
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
