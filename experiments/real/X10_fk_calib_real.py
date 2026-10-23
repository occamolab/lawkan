# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - X10: FK calibration on three real robot platforms (replication)
#
# Origin: exploratory experiment 1.1 of the co-author branch NP_master
# (fk_calib_real.py + ScrewCalib from IK1_robot_fk.py). This file is the
# self-contained port used for the pre-registered replication P48.
#
# Model (no neural network): exact SE(3) product-of-exponentials composition
# with learnable corrections on the joint twists,
#     T(q) = M0 . exp(xi_1 (q_1+b_1)) ... exp(xi_n (q_n+b_n)) . exp(tool),
#     xi_j = xi_nom,j + dxi_j  (dense: 6/joint; sparse: 3/joint rotation-only)
#     + zero offsets b_j + tool (6).  55 parameters for 7 axes, 48 for 6.
#
# Fair protocol (same for every platform):
#   nominal FK   : rtb URDF chain at q (0 fitted parameters)
#   nominal+gap  : + constant base<->tracker B and tool C, fitted on TRAIN
#                  (12 numbers; absorbs the mount/tracker offset)
#   ScrewCalib   : the model above + fixed B; tool initialised at the gap-fit C;
#                  Adam, cosine LR, full batch, 2000 steps, train only.
#   train subsets 20/40/80 poses, test = held-out poses; seeds default 0-4
#   (the pre-registered replication uses --seed_start 50).
#
# Datasets (MIT; not committed, download into --data_dir):
#   FR3  : PhilNad robot-arm-kinematic-calibration (114 poses, camera)
#          data/calib_data.pickle (auto-downloaded if missing)
#   UR10 : christianlandgraf/ipa-robot-accuracy
#          data/ipa/Series_A/measurements_grid.csv (laser tracker, 6D)
#   UR5  : pauljac7700/serial_robots_datasets
#          data/pauljac/UR5/3D_UR5_uncalibrated_{grid,random}_cleaned.csv
#          (laser tracker, position only)
#
# Run:  python X10_fk_calib_real.py --seed_start 50
#       python X10_fk_calib_real.py --check          # data sanity only
#       python X10_fk_calib_real.py --quick          # smoke (1 seed, N=20)
# JSON: repro/json/x10.<YYYYMMDD>-fkcalib3.json
# ============================================================
import argparse, json, math, os, pickle, time, urllib.request
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
import pandas as pd
import torch, torch.nn as nn
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation

_threads = int(os.environ.get('LAWKAN_THREADS', '0') or 0)
torch.set_num_threads(_threads if _threads > 0 else max(1, min(8, os.cpu_count() or 1)))

DATA_URL = ("https://raw.githubusercontent.com/PhilNad/robot-arm-kinematic-"
            "calibration/main/Examples/calib_data.pickle")


# ---------------------------------------------------------------------------
# Exact SE(3) composition node and learnable screw corrections
# ---------------------------------------------------------------------------
def hat3(w):
    B = w.shape[:-1]
    W = torch.zeros(*B, 3, 3, dtype=w.dtype, device=w.device)
    W[..., 0, 1] = -w[..., 2]; W[..., 0, 2] = w[..., 1]
    W[..., 1, 0] = w[..., 2];  W[..., 1, 2] = -w[..., 0]
    W[..., 2, 0] = -w[..., 1]; W[..., 2, 1] = w[..., 0]
    return W


def se3_exp(xi):
    """xi: (...,6) = (omega, v) -> (...,4,4), with series handling near zero."""
    w, v = xi[..., :3], xi[..., 3:]
    th = w.norm(dim=-1, keepdim=True)
    th_s = th.clamp_min(1e-8)
    W = hat3(w)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(W.shape)
    a = (torch.sin(th_s) / th_s).unsqueeze(-1)
    b = ((1 - torch.cos(th_s)) / th_s ** 2).unsqueeze(-1)
    c = ((th_s - torch.sin(th_s)) / th_s ** 3).unsqueeze(-1)
    R = I + a * W + b * (W @ W)
    V = I + b * W + c * (W @ W)
    p = (V @ v.unsqueeze(-1)).squeeze(-1)
    T = torch.zeros(*xi.shape[:-1], 4, 4, dtype=xi.dtype, device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def fk_body(xi, M0, q, b=None, tool=None):
    """T = M0 prod_j exp(xi_j (q_j + b_j)) exp(tool); xi (n,6), q (B,n)."""
    if b is not None:
        q = q + b
    B, n = q.shape
    T = M0.unsqueeze(0).expand(B, 4, 4).clone()
    for j in range(n):
        T = T @ se3_exp(xi[j] * q[:, j].unsqueeze(-1))
    if tool is not None:
        T = T @ se3_exp(tool.unsqueeze(0))
    return T


class ScrewCalib(nn.Module):
    """Exact SE(3) composition + learnable joint twist corrections.
    mode='fourier' adds a per-joint Fourier theta-edge F_j(theta) on top of the
    (theta_j + b_j) screw, i.e. the constant-edge calibrator plus theta-edge.
    mode='canon' replaces the free twist correction by a canonical screw:
    omega_j = gain_j * unit(a_j) with a scalar gain (init 1) and a free moment
    nu_j (init 0), so the joint twist is a unit-screw direction by construction;
    init equals the nominal z-screw (gain 1, a = z, nu = 0, b = 0)."""
    def __init__(self, xi, M0, mode='dense', K=0, canon_style='norm'):
        super().__init__()
        self.register_buffer('xi0', xi.float())
        self.register_buffer('M0', M0.float())
        n = xi.shape[0]
        self.mode = mode
        assert mode in ('dense', 'sparse', 'fourier', 'canon')
        self.K = int(K)
        self.canon_style = str(canon_style)
        if mode == 'sparse':
            self.dw = nn.Parameter(torch.zeros(n, 3))
        elif mode == 'canon':
            ax = xi[:, :3].clone()
            gn = ax.norm(dim=-1).clamp_min(1e-12)
            self.gain = nn.Parameter(gn)
            self.nu = nn.Parameter(xi[:, 3:].clone())
            if self.canon_style == 'angle':
                a0 = ax / gn.unsqueeze(-1)
                ab = torch.stack([torch.atan2(a0[:, 1], a0[:, 0]),
                                  torch.acos(a0[:, 2].clamp(-1 + 1e-7, 1 - 1e-7))],
                                 -1)
                self.axi_ab = nn.Parameter(ab)
            else:
                self.axi = nn.Parameter(ax / gn.unsqueeze(-1))
        else:
            self.dxi = nn.Parameter(torch.zeros(n, 6))
        self.b = nn.Parameter(torch.zeros(n))
        self.tool = nn.Parameter(torch.zeros(6))
        if mode == 'fourier':
            self.fc = nn.Parameter(torch.zeros(n, self.K, 6))
            self.fs = nn.Parameter(torch.zeros(n, self.K, 6))

    def twists(self):
        if self.mode == 'sparse':
            return torch.cat([self.xi0[:, :3] + self.dw, self.xi0[:, 3:]], -1)
        if self.mode == 'canon':
            if self.canon_style == 'angle':
                al, be = self.axi_ab[:, 0], self.axi_ab[:, 1]
                a = torch.stack([torch.sin(be) * torch.cos(al),
                                 torch.sin(be) * torch.sin(al),
                                 torch.cos(be)], -1)
            else:
                a = self.axi / self.axi.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return torch.cat([self.gain.unsqueeze(-1) * a, self.nu], -1)
        return self.xi0 + self.dxi

    def forward(self, q):
        if self.mode != 'fourier':
            return fk_body(self.twists().double(), self.M0.double(), q.double(),
                           b=self.b.double(), tool=self.tool.double())
        qd = q.double()
        g = qd + self.b.double().unsqueeze(0)                # (B,n)
        v = g.unsqueeze(-1) * self.twists().double().unsqueeze(0)
        for k in range(1, self.K + 1):
            v = v + self.fc[:, k-1].double().unsqueeze(0) * torch.cos(k * qd).unsqueeze(-1) \
                  + self.fs[:, k-1].double().unsqueeze(0) * torch.sin(k * qd).unsqueeze(-1)
        B, n = qd.shape
        T = self.M0.double().unsqueeze(0).expand(B, 4, 4).clone()
        for j in range(n):
            T = T @ se3_exp(v[:, j, :])
        T = T @ se3_exp(self.tool.double().unsqueeze(0))
        return T

    def n_params(self):
        n = self.xi0.shape[0]
        extra = 12 * n * self.K if self.mode == 'fourier' else 0
        if self.mode == 'canon':
            return 8 * n + 6 + extra
        return (n * (3 if self.mode == 'sparse' else 6)) + n + 6 + extra


# ---------------------------------------------------------------------------
# Numpy SE(3) helpers for the gap fit (12-parameter baseline)
# ---------------------------------------------------------------------------
def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def log_se3_np(T):
    R, p = T[:3, :3], T[:3, 3]
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-10:
        return np.zeros(6)
    w = ang / (2 * np.sin(ang)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    V_inv = (np.eye(3) - 0.5 * W
             + (1 / ang ** 2) * (1 - ang * np.sin(ang) / (2 * (1 - np.cos(ang)))) * (W @ W))
    return np.concatenate([w, V_inv @ p])


def avg_se3(Rs, ts):
    quats = Rotation.from_matrix(Rs).as_quat()
    ref = quats[0].copy()
    quats[quats @ ref < 0] *= -1
    M = quats.T @ quats
    _, V = np.linalg.eigh(M)
    return Rotation.from_quat(V[:, -1]).as_matrix(), ts.mean(0)


def kabsch_t(A, B):
    Am, Bm = A - A.mean(0), B - B.mean(0)
    U, _, Vt = np.linalg.svd(Am.T @ Bm)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, B.mean(0) - R @ A.mean(0)


def pose_metrics(T_pred, T_obs, full=True):
    dp = np.linalg.norm(T_pred[:, :3, 3] - T_obs[:, :3, 3], axis=1) * 1000.0
    out = dict(pos_mean=float(dp.mean()), pos_median=float(np.median(dp)),
               pos_max=float(dp.max()), pos_p90=float(np.percentile(dp, 90)))
    if full:
        tr = np.trace(T_pred[:, :3, :3].transpose(0, 2, 1) @ T_obs[:, :3, :3],
                      axis1=1, axis2=2)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        out.update(rot_mean=float(ang.mean()), rot_median=float(np.median(ang)),
                   rot_max=float(ang.max()))
    return out


# ---------------------------------------------------------------------------
# Nominal FK tensors from roboticstoolbox (screws by central differences)
# ---------------------------------------------------------------------------
def get_robot_tensors(name):
    robot = {'panda': rtb.models.URDF.Panda,
             'ur10': rtb.models.URDF.UR10,
             'ur5': rtb.models.URDF.UR5}[name]()
    n = robot.n
    M0 = torch.tensor(robot.fkine(np.zeros(n)).A, dtype=torch.float64)
    xi = torch.zeros(n, 6, dtype=torch.float64)
    h = 1e-6
    Minv = torch.linalg.inv(M0)
    for j in range(n):
        qp, qm = np.zeros(n), np.zeros(n)
        qp[j], qm[j] = h, -h
        Tp = torch.tensor(robot.fkine(qp).A, dtype=torch.float64)
        Tm = torch.tensor(robot.fkine(qm).A, dtype=torch.float64)
        X = Minv @ ((Tp - Tm) / (2 * h))
        xi[j] = torch.cat([torch.stack([X[2, 1], X[0, 2], X[1, 0]]), X[:3, 3]])
    return robot, xi, M0


def fk_np(robot, q):
    return np.array([robot.fkine(qq).A for qq in q], dtype=np.float64)


# ---------------------------------------------------------------------------
# Gap fit: T_pred = B @ FK_nom(q) @ C ~ T_obs, fitted on TRAIN only (LBFGS)
# ---------------------------------------------------------------------------
def fit_gaps_torch(Tf_tr, T_obs_tr, B0, C0, steps=3000, full=True, seed=0):
    torch.manual_seed(seed)
    pB = torch.tensor(log_se3_np(B0), dtype=torch.float64, requires_grad=True)
    pC = torch.tensor(log_se3_np(C0), dtype=torch.float64, requires_grad=True)
    Tft = torch.tensor(Tf_tr, dtype=torch.float64)
    Tmt = torch.tensor(T_obs_tr, dtype=torch.float64)
    opt = torch.optim.LBFGS([pB, pC], lr=1.0, max_iter=max(200, min(steps, 1000)),
                            history_size=20, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        B = se3_exp(pB.unsqueeze(0))[0]
        C = se3_exp(pC.unsqueeze(0))[0]
        Tp = B @ Tft @ C
        loss = 10.0 * ((Tp[:, :3, 3] - Tmt[:, :3, 3]) ** 2).sum(-1).mean()
        if full:
            tr = (Tp[:, :3, :3].transpose(-1, -2) @ Tmt[:, :3, :3]).diagonal(
                dim1=-2, dim2=-1).sum(-1)
            loss = loss + (1.0 - ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        B = se3_exp(pB.unsqueeze(0))[0].numpy()
        C = se3_exp(pC.unsqueeze(0))[0].numpy()
    return B, C


def gap_init(Tf_tr, T_obs_tr, goal_tr=None, full=True):
    if goal_tr is not None:
        Ci = np.linalg.inv(Tf_tr) @ goal_tr
        C0 = _T(*avg_se3(Ci[:, :3, :3], Ci[:, :3, 3]))
    else:
        C0 = np.eye(4)
    Bi = T_obs_tr @ np.linalg.inv(Tf_tr @ C0)
    if full:
        B0 = _T(*avg_se3(Bi[:, :3, :3], Bi[:, :3, 3]))
    else:
        R, t = kabsch_t((Tf_tr @ C0)[:, :3, 3], T_obs_tr[:, :3, 3])
        B0 = _T(R, t)
    return B0, C0


def train_screwcalib(xi, M0, q_tr, T_tr, tool_init, mode='dense',
                     steps=2000, seed=0, full=True, K=0, canon_style='norm'):
    torch.manual_seed(100 + seed)
    model = ScrewCalib(xi, M0, mode, K=K, canon_style=canon_style)
    with torch.no_grad():
        model.tool.copy_(torch.tensor(tool_init, dtype=model.tool.dtype))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.0)
    qt = torch.tensor(q_tr, dtype=torch.float64)
    Tt = torch.tensor(T_tr, dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        Tp = model(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - Tt[:, :3, 3]) ** 2).sum(-1).mean()
        if full:
            tr = (Tp[:, :3, :3].transpose(-1, -2) @ Tt[:, :3, :3]).diagonal(
                dim1=-2, dim2=-1).sum(-1)
            loss = loss + (1.0 - ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def apply_sc(model, q):
    return model(torch.tensor(q, dtype=torch.float64)).numpy()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_philnad(data_dir):
    path = os.path.join(data_dir, 'calib_data.pickle')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        print(f'[X10] downloading FR3 dataset -> {path}')
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    q = np.asarray(data['joint_positions'], dtype=np.float64)
    cam = np.stack([np.asarray(m, dtype=np.float64) for m in data['camera_poses']])
    c, s = np.cos(np.radians(90.0)), np.sin(np.radians(90.0))
    X_C_T = _T(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]),
               [94.5 / 1000, -9.0 / 1000, 40.55 / 1000])
    bc = ((82.25 + 4 * 25 + 3 + 250) / 1000, -(4.5 * 25 + 3 + 250) / 1000)
    tl = (bc[0] + 0.5 / 2 - 0.025, bc[1] - 0.5 / 2 + 0.025)
    X_B_W = _T(np.eye(3), [tl[0], tl[1], -9.7 / 1000])
    T_obs = np.stack([X_B_W @ cam[m] @ np.linalg.inv(X_C_T) for m in range(len(cam))])
    return dict(name='philnad', robot='panda', q=q, T=T_obs, full=True, goal=None,
                note='FR3; obs reconstructed with X7 hand-measured camera/board constants')


_IPA_FILES = {'A': 'Series_A/measurements_grid.csv'}


def load_ipa(data_dir, series='A'):
    df = pd.read_csv(os.path.join(data_dir, 'ipa', _IPA_FILES[series]),
                     sep=';', decimal=',')
    q = df[['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']].values.astype(np.float64)
    T = np.tile(np.eye(4), (len(df), 1, 1))
    T[:, :3, :3] = Rotation.from_euler(
        'XYZ', df[['measurement_rx', 'measurement_ry', 'measurement_rz']].values,
        degrees=True).as_matrix()
    T[:, :3, 3] = df[['measurement_x', 'measurement_y', 'measurement_z']].values / 1000.0
    goal = np.tile(np.eye(4), (len(df), 1, 1))
    goal[:, :3, :3] = Rotation.from_euler(
        'xyz', df[['Goal roll', 'Goal pitch', 'Goal yaw']].values).as_matrix()
    goal[:, :3, 3] = df[['Goal x', 'Goal y', 'Goal z']].values
    return dict(name='ipa_%s' % series, robot='ur10', q=q, T=T, full=True, goal=goal,
                note='laser tracker T-Mac; Goal pose used only for physical tool init')


def load_pauljac(data_dir, robot_kind='UR5'):
    base = os.path.join(data_dir, 'pauljac', robot_kind)
    g = pd.read_csv(os.path.join(base, '3D_%s_uncalibrated_grid_cleaned.csv' % robot_kind))
    r = pd.read_csv(os.path.join(base, '3D_%s_uncalibrated_random_cleaned.csv' % robot_kind))

    def q_of(df):
        return np.deg2rad(df.filter(like='joint').values.astype(np.float64))

    def p_of(df):
        return (df[['x_t', 'y_t', 'z_t']].values
                + df[['x_dif', 'y_dif', 'z_dif']].values) / 1000.0

    def T_of(p):
        T = np.tile(np.eye(4), (len(p), 1, 1))
        T[:, :3, 3] = p
        return T

    return dict(name='pauljac_%s' % robot_kind,
                robot='ur5' if robot_kind == 'UR5' else None,
                q_grid=q_of(g), T_grid=T_of(p_of(g)),
                q_rand=q_of(r), T_rand=T_of(p_of(r)), full=False, goal=None,
                note='position only (x_t + x_dif); 20 official random test poses')


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
def summarise(per_seed_metrics):
    out = {}
    for k in per_seed_metrics[0].keys():
        vals = [m[k] for m in per_seed_metrics]
        out[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                      per_seed=[float(v) for v in vals])
    return out


def evaluate_split(q_te, T_te, Tf_te, B, C, sc_models, full):
    out = {}
    if full:
        out['nominal_raw'] = pose_metrics(Tf_te, T_te, full=True)
    out['nominal_gap'] = pose_metrics(B @ Tf_te @ C, T_te, full=full)
    T_obs2 = np.array([np.linalg.inv(B) @ t for t in T_te])
    for mode, model in sc_models.items():
        out['screw_%s' % mode] = pose_metrics(apply_sc(model, q_te), T_obs2, full=full)
    return out


def run_dataset(data, sizes, seeds, steps, modes, seed_start, results, K=0,
                canon_style='norm', split='random'):
    robot, xi, M0 = get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = fk_np(robot, q_all)
    full = data['full']
    per_seed = {}
    raw = pose_metrics(Tf_all, T_all, full=full)
    if split == 'cluster':
        from scipy.cluster.vq import kmeans2
    for s in range(seed_start, seed_start + seeds):
        per_seed['seed%d' % s] = {}
        if split == 'random':
            perm = np.random.default_rng(1000 + s).permutation(n)
        else:
            qk = q_all - q_all.mean(0)
            qk = qk / (qk.std(0) + 1e-9)
            _, labels = kmeans2(qk, 4, minit='++', seed=1000 + s)
        for N in sizes:
            if N >= n:
                continue
            if split == 'random':
                tr, te = perm[:N], perm[N:]
            else:
                cs = [np.where(labels == c)[0] for c in range(4)]
                cbest = min(range(4), key=lambda c: abs(len(cs[c]) - (n - N)))
                te = cs[cbest]
                pool = np.concatenate([cs[c] for c in range(4) if c != cbest])
                rng = np.random.default_rng(2000 + s)
                tr = rng.permutation(pool)[:N]
            goal_tr = data['goal'][tr] if data['goal'] is not None else None
            B0, C0 = gap_init(Tf_all[tr], T_all[tr], goal_tr, full=full)
            B, C = fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                  steps=steps, full=full, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            models = {}
            for mode in modes:
                models[mode] = train_screwcalib(
                    xi, M0, q_all[tr], T_obs2_tr, log_se3_np(C), mode=mode,
                    steps=steps, seed=s, full=full, K=K, canon_style=canon_style)
            rec = evaluate_split(q_all[te], T_all[te], Tf_all[te], B, C, models, full)
            for k, v in rec.items():
                v['n_test'] = int(len(te))
            per_seed['seed%d' % s]['N%d' % N] = rec
            print('   seed %d N=%d: ' % (s, N) + ' | '.join(
                '%s %.3f mm' % (k, v['pos_mean']) for k, v in rec.items()
                if k != 'nominal_raw'), flush=True)
        print('  seed %d done' % s, flush=True)
    agg = {}
    for N in sizes:
        recs = [per_seed['seed%d' % s]['N%d' % N] for s in range(seed_start, seed_start + seeds)
                if 'N%d' % N in per_seed['seed%d' % s]]
        if not recs:
            continue
        agg['N%d' % N] = {k: summarise([r[k] for r in recs]) for k in recs[0]}
    results[data['name']] = dict(
        robot=data['robot'], n_poses=int(n), full_pose=bool(full),
        note=data['note'], nominal_raw=raw, aggregate=agg, per_seed=per_seed,
        params={m: ScrewCalib(xi, M0, m, K=K, canon_style=canon_style).n_params()
                for m in modes})


def run_pauljac(data, sizes, seeds, steps, modes, seed_start, results, K=0,
                canon_style='norm'):
    robot, xi, M0 = get_robot_tensors(data['robot'])
    qg, Tg = data['q_grid'], data['T_grid']
    qr, Tr = data['q_rand'], data['T_rand']
    Tf_g = fk_np(robot, qg)
    raw_r = pose_metrics(fk_np(robot, qr), Tr, full=False)
    per_seed = {}
    for s in range(seed_start, seed_start + seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(qg))
        per_seed['seed%d' % s] = {}
        for N in sizes:
            if N >= len(qg):
                continue
            tr, te = perm[:N], perm[N:]
            B0, C0 = gap_init(Tf_g[tr], Tg[tr], None, full=False)
            B, C = fit_gaps_torch(Tf_g[tr], Tg[tr], B0, C0, steps=steps,
                                  full=False, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in Tg[tr]])
            models = {}
            for mode in modes:
                models[mode] = train_screwcalib(xi, M0, qg[tr], T_obs2_tr,
                                                log_se3_np(C), mode=mode,
                                                steps=steps, seed=s, full=False)
            rec = {}
            rec['nominal_gap_random'] = pose_metrics(B @ fk_np(robot, qr) @ C, Tr, full=False)
            T_obs2_r = np.array([np.linalg.inv(B) @ t for t in Tr])
            rec['nominal_gap_gridhold'] = pose_metrics(B @ Tf_g[te] @ C, Tg[te], full=False)
            T_obs2_g = np.array([np.linalg.inv(B) @ t for t in Tg[te]])
            for mode, model in models.items():
                rec['screw_%s_random' % mode] = pose_metrics(apply_sc(model, qr), T_obs2_r, full=False)
                rec['screw_%s_gridhold' % mode] = pose_metrics(apply_sc(model, qg[te]), T_obs2_g, full=False)
            per_seed['seed%d' % s]['N%d' % N] = rec
            print('   seed %d N=%d: ' % (s, N) + ' | '.join(
                '%s %.3f mm' % (k, v['pos_mean']) for k, v in rec.items()), flush=True)
        print('  seed %d done' % s, flush=True)
    agg = {}
    for N in sizes:
        recs = [per_seed['seed%d' % s]['N%d' % N] for s in range(seed_start, seed_start + seeds)]
        agg['N%d' % N] = {k: summarise([r[k] for r in recs]) for k in recs[0]}
    results[data['name']] = dict(
        robot='ur5', n_poses=int(len(qg)), full_pose=False, note=data['note'],
        nominal_raw_random=raw_r, aggregate=agg, per_seed=per_seed,
        params={m: ScrewCalib(xi, M0, m, K=K, canon_style=canon_style).n_params()
                for m in modes})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--datasets', default='philnad,ipa,pauljac')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=0)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--modes', default='dense,sparse',
                    help="dense | sparse | fourier (dense+theta-edge) | canon "
                         "(unit-screw twist, gain+moment)")
    ap.add_argument('--K', type=int, default=1, help='Fourier harmonics for mode=fourier')
    ap.add_argument('--canon-style', choices=['norm', 'angle'], default='norm',
                    help='canon mode: normalize a vector, or two free angles')
    ap.add_argument('--out', default=None)
    ap.add_argument('--split', choices=['random', 'cluster'], default='random')
    ap.add_argument('--check', action='store_true', help='load datasets only, no training')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.sizes, args.steps = 1, '20', 200
        args.modes, args.datasets = 'dense', 'philnad'
    sizes = [int(s) for s in args.sizes.split(',')]
    modes = tuple(m for m in args.modes.split(',') if m)
    todo = args.datasets.split(',')
    out = args.out

    results = {'_meta': dict(
        experiment='X10 FK calibration on three real platforms (P48 replication)',
        date=time.strftime('%Y-%m-%d %H:%M'), sizes=sizes, seeds=args.seeds,
        seed_start=args.seed_start, steps=args.steps, modes=list(modes),
        split=args.split,
        device='cpu', torch=torch.__version__,
        datasets=dict(
            philnad='PhilNad robot-arm-kinematic-calibration (FR3, 114 poses, MIT)',
            ipa_A='IPA/landgraf ipa-robot-accuracy Series A grid (UR10, laser tracker, MIT)',
            pauljac_UR5='pauljac serial_robots_datasets UR5 grid+random20 (position, MIT)'))}

    if 'philnad' in todo:
        d = load_philnad(args.data_dir)
        print('== PhilNad FR3 ==', flush=True)
        if args.check:
            print('   q', d['q'].shape, 'T', d['T'].shape)
        else:
            run_dataset(d, sizes, args.seeds, args.steps, modes, args.seed_start, results, K=args.K, canon_style=args.canon_style, split=args.split)
    if 'ipa' in todo:
        d = load_ipa(args.data_dir, 'A')
        print('== IPA UR10 (Series A grid) ==', flush=True)
        if args.check:
            print('   q', d['q'].shape, 'T', d['T'].shape)
        else:
            run_dataset(d, sizes, args.seeds, args.steps, modes, args.seed_start, results, K=args.K, canon_style=args.canon_style, split=args.split)
    if 'pauljac' in todo:
        d = load_pauljac(args.data_dir, 'UR5')
        print('== pauljac UR5 (position only) ==', flush=True)
        if args.check:
            print('   q_grid', d['q_grid'].shape, 'q_rand', d['q_rand'].shape)
        else:
            run_pauljac(d, sizes, args.seeds, args.steps, modes, args.seed_start, results, K=args.K, canon_style=args.canon_style)

    if args.check:
        print('CHECK OK (no training)')
        return

    results['_meta']['wall_seconds'] = float(time.time() - t0)
    out = save_dump(results, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, seed_start=args.seed_start,
                    epochs=args.steps, out=args.out)
    print('\n===== SUMMARY-JSON =====')
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
