# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — X5: REAL IMU probe on the BROAD dataset  (v1.1)
# (Section 6.1 on real data; companion to run_extensions.py)
#
# Dataset: BROAD — Berlin Robust Orientation Estimation Assessment Dataset
#   Laidig, Caruso, Cereatti, Seel, "BROAD — A Benchmark for Robust Inertial
#   Orientation Estimation", Data 6(7):72, 2021. doi:10.3390/data6070072
#   License: CC-BY 4.0. Repo: https://github.com/dlaidig/broad
#   HDF5 fields used: imu_gyr (N,3) rad/s, opt_quat (N,4) w-first ENU,
#   movement (N,) bool, attribute sampling_rate = 2000/7 Hz (~286 Hz).
#
# What this probe does (windows, same protocol as E15/X3):
#   W = 20 steps (~0.07 s)  — direct replication of the synthetic §6.1 task:
#                             gyro-only strapdown integration, bias ~0.
#   W = 100 steps (~0.35 s) — intermediate regime.
#   W = 286 steps (1.00 s)  — honest drift column: gyro bias now matters,
#                             models also receive the stationary-segment bias
#                             estimate as 3 extra features (edge input).
#   W = 1430 steps (5.00 s) — same, longer horizon.
#
# IMPORTANT (2026-09-21): on clean gyro-only data the strapdown integrator is
# already near-optimal, so a learned composer has nothing to add — the
# informative setting is the DISTURBED group (structured gyro noise: tapping,
# vibration, magnets, office, mixed), where the integrator carries the noise
# into the error and a learned model can in principle suppress it. Default
# group is therefore 'disturbed'; use --groups undisturbed for the clean arm.
#
# v1.1 BUG FIX (loss and metric; same class of bug as X6 v1.0):
#   v1.0 trained on MSE between ROTVEC targets and evaluated with the
#   rotvec-difference norm. The rotvec chart is discontinuous at angle pi
#   (rotations of pi-eps and pi+eps about the same axis have rotvecs pointing
#   in OPPOSITE directions, differing by ~2*pi), so for 1 s / 5 s windows
#   the loss received spurious ~2pi gradients and the metric was inflated.
#   All v1.0 numbers for the b-1s / c-5s blocks (e.g. "QuatIMU 56.98 deg")
#   are RETRACTED; the short-window conclusions (a/b2/d) were in the
#   small-angle regime where rotvec-MSE ~= geodesic and are unaffected in
#   substance. v1.1: targets stored as quaternions; loss = mean theta^2 and
#   metric = theta with theta = 2*acos(|<q_pred, q_tgt>|).
#   THE BOTTOM LINE of X5 does not depend on this fix: the strapdown
#   integrator baseline is training-free, and its numbers are now reported
#   with the correct metric.
#
# Honest baselines that do NOT need training:
#   RawIntegrator — semi-implicit Euler on raw gyro (the "do nothing" floor);
#   BiasIntegrator — same after subtracting the stationary-segment bias.
#
# Trainable models (3 seeds each, paired t-tests):
#   QuatIMU — shared per-step edge from run_extensions.py, bias-subtracted
#             input; for the 1 s / 5 s windows the edge also gets the bias
#             estimate as a linear DIRECTION correction (scaled by |omega|*dt).
#   MLPFlat h=8 / h=32 — the strongest synthetic opponents, same inputs.
#
# Requirements: torch, numpy, h5py, scipy (p-values only).
# Data: NOT included. Download once (see README_X5 below or header comment):
#   git clone --depth 1 https://github.com/dlaidig/broad.git
#   cd broad && git lfs install && git lfs pull --include="data_hdf5/*" --exclude=""
# Run:
#   python X5_broad_real_imu.py --data_dir /path/to/broad/data_hdf5
# ============================================================
import argparse, math, os, sys, time, json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.data_guard import skip_if_missing  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}")

def reg_text(key, text):
    RESULTS[key] = dict(text=text)
    print(f"[{key}] {text}")

def paired_t(a, b):
    d = np.array(a) - np.array(b); n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float('nan')
    return float(t), n-1, float(p)

# ------------------------------------------------------------
# Shared geometry (identical to run_main_experiments.py / run_extensions.py)
# ------------------------------------------------------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)

def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def rotate(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]

def axis_angle_to_quat(axis, angle):
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    h = (angle / 2).unsqueeze(-1)
    return torch.cat([torch.cos(h), axis * torch.sin(h)], -1)

def quat_to_rotvec(q):
    """Unit quaternion -> rotation vector (axis * angle), angle in (-pi, pi]."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w = q[..., 0].clamp(-1+1e-7, 1-1e-7)
    ang = 2*torch.acos(w)                              # in [0, pi]
    sin_half = torch.sqrt((1 - w*w)).clamp_min(1e-8)
    axis = q[..., 1:] / sin_half.unsqueeze(-1)
    # shortest rotation: if angle > pi, flip axis
    over = ang > math.pi
    ang = torch.where(over, 2*math.pi - ang, ang)
    axis = torch.where(over.unsqueeze(-1), -axis, axis)
    return axis * ang.unsqueeze(-1)

def geo_deg_q(q_pred, q_tgt):
    """True SO(3) geodesic angle in degrees, double-cover safe (v1.1 metric).
    q_pred, q_tgt: (...,4) unit quaternions."""
    d = (q_pred * q_tgt).sum(-1).abs().clamp(max=1.0 - 1e-9)
    return torch.rad2deg(2.0 * torch.acos(d))

def geo_rad_q(q_pred, q_tgt):
    """Same in radians (v1.1 training loss)."""
    d = (q_pred * q_tgt).sum(-1).abs().clamp(max=1.0 - 1e-7)
    return 2.0 * torch.acos(d)

def rotvec_to_quat(rv):
    """Rotation vector (axis*angle) -> unit quaternion (for the MLP output)."""
    ang = rv.norm(dim=-1).clamp_min(1e-8)
    axis = rv / ang.unsqueeze(-1)
    h = (ang / 2).unsqueeze(-1)
    return torch.cat([torch.cos(h), axis * torch.sin(h)], dim=-1)

def geo_deg(a, b):
    """DEPRECATED (v1.0 metric): rotvec-difference norm, NOT the geodesic.
    Discontinuous at angle pi — see the v1.1 header note. Kept only so old
    log-parsing code does not break; new results use geo_deg_q."""
    return (a - b).norm(dim=-1) * 180 / math.pi

# ------------------------------------------------------------
# BROAD parser
# ------------------------------------------------------------
def load_broad_trial(path):
    """Load one BROAD HDF5 trial.

    Returns dict with:
      gyr    (N,3) float32 rad/s
      quat   (N,4) float64 unit quaternion, w-first
      move   (N,)  bool
      fs     float sampling rate, Hz
      name   trial name from the filename
    """
    import h5py
    with h5py.File(path, 'r') as f:
        gyr = np.asarray(f['imu_gyr'][()], dtype=np.float32)
        quat = np.asarray(f['opt_quat'][()], dtype=np.float64)
        move = np.asarray(f['movement'][()]).astype(bool)
        fs = float(np.asarray(f.attrs['sampling_rate']).ravel()[0])
    # defensive normalization of ground truth
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-12)
    name = os.path.splitext(os.path.basename(path))[0]
    return dict(gyr=gyr, quat=quat, move=move, fs=fs, name=name)

def list_broad_files(data_dir, groups=('undisturbed',)):
    """Return [(path, groups-of-trial)] filtered by trials.json groups."""
    with open(os.path.join(data_dir, 'trials.json'), 'r') as fh:
        meta = json.load(fh)
    out = []
    for name, info in meta['trials'].items():
        if all(g in info['groups'] for g in groups):
            p = os.path.join(data_dir, name + '.hdf5')
            if os.path.exists(p):
                out.append((p, info['groups']))
    out.sort()
    return out

# ------------------------------------------------------------
# Window construction
# ------------------------------------------------------------
def estimate_bias(gyr, move, margin_s, fs):
    """Gyroscope bias from the leading stationary segment (movement == False).

    BROAD trials start with a rest phase; the mean of gyro over it is the
    standard offline bias estimate (same convention as the VQF reference).
    Returns (bias (3,), n_rest_samples).
    """
    n_rest = 0
    while n_rest < len(move) and not move[n_rest]:
        n_rest += 1
    n_use = max(n_rest - int(margin_s * fs), 1)
    bias = gyr[:n_use].mean(axis=0)
    return bias, n_rest

def build_windows(trial, W, stride, bias, min_move_frac=0.5):
    """Slice one trial into fixed windows of W consecutive IMU samples.

    Input  X: (nwin, W, 3)  gyro with bias already subtracted
    Target Y: (nwin, 4)     ground-truth relative orientation as a
                            QUATERNION q_rel = conj(q_start) * q_end
                            (v1.1: quaternion, not rotvec — see header)
    Only windows whose movement fraction is >= min_move_frac are kept (we
    evaluate on motion, like the BROAD metrics themselves). Windows whose
    start or end ground-truth quaternion is invalid (NaN — the OMC system
    loses the marker during occlusions — or a non-unit norm) are dropped.
    """
    gyr = trial['gyr'] - bias.astype(np.float32)
    quat = torch.from_numpy(trial['quat']).to(torch.float64)
    move = trial['move']
    N = len(gyr)
    # valid ground-truth mask: finite and unit-norm quaternion
    qn = quat.norm(dim=-1)
    valid = torch.isfinite(quat).all(dim=-1) & ((qn - 1.0).abs() < 1e-2)
    Xs, Ys = [], []
    for s in range(0, N - W + 1, stride):
        e = s + W
        if move[s:e].mean() < min_move_frac:
            continue
        if not (valid[s] and valid[e]):     # occlusion gap at a window edge
            continue
        q_rel = quat_mul(quat_conj(quat[s]), quat[e])          # w-first
        Xs.append(gyr[s:e])
        Ys.append(q_rel.numpy().astype(np.float32))
    if not Xs:
        return np.zeros((0, W, 3), np.float32), np.zeros((0, 4), np.float32)
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)

def make_dataset(data_dir, W, stride, groups=('undisturbed',), verbose=True):
    """Build the pooled dataset for window length W over all matching trials.

    Returns X (n, W*3), Y (n, 4) quaternion targets, B (n, 3) bias-per-window
    (the constant per-trial estimate, repeated — a legitimate input feature),
    plus a list of per-trial window counts.
    """
    files = list_broad_files(data_dir, groups)
    assert files, f"No BROAD files found in {data_dir} for groups {groups}"
    Xs, Ys, Bs, Tids, per_trial = [], [], [], [], []
    fs0 = None
    for ti, (path, gr) in enumerate(files):
        tr = load_broad_trial(path)
        fs0 = tr['fs'] if fs0 is None else fs0
        bias, n_rest = estimate_bias(tr['gyr'], tr['move'], margin_s=0.5, fs=tr['fs'])
        X, Y = build_windows(tr, W=W, stride=stride, bias=bias)
        per_trial.append((tr['name'], len(X), float(np.linalg.norm(bias)), n_rest))
        if len(X):
            Xs.append(X.reshape(len(X), -1))
            Ys.append(Y)
            Bs.append(np.repeat(bias[None, :].astype(np.float32), len(X), axis=0))
            Tids.append(np.full(len(X), ti, dtype=np.int64))
    X = torch.from_numpy(np.concatenate(Xs)).to(device)
    Y = torch.from_numpy(np.concatenate(Ys)).to(device, torch.float64)
    B = torch.from_numpy(np.concatenate(Bs)).to(device)
    T = torch.from_numpy(np.concatenate(Tids)).to(device)
    if verbose:
        print(f"  dataset W={W} ({W/fs0:.3f} s): {len(X)} windows from {len(files)} trials, fs={fs0:.2f} Hz")
        for name, n, bn, nr in per_trial:
            print(f"    {name:45s} windows={n:5d}  |bias|={bn:.5f} rad/s  rest={nr} samples")
    return X, Y, B, fs0, T

def trial_split(T, n_train, n_test, seed=12345):
    """Split windows by TRIAL so no trial appears in both train and test
    (removes the window-overlap leakage of the earlier random split).
    Returns (train_idx, test_idx); caps each pool to n_train/n_test."""
    tids = T.unique().tolist()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(tids), generator=g).tolist()
    n_tr = max(1, int(round(0.7 * len(tids))))
    tr_trials = torch.tensor([tids[i] for i in perm[:n_tr]], device=T.device)
    te_trials = torch.tensor([tids[i] for i in perm[n_tr:]], device=T.device)
    tr_idx = torch.isin(T, tr_trials).nonzero(as_tuple=True)[0]
    te_idx = torch.isin(T, te_trials).nonzero(as_tuple=True)[0]
    def cap(idx, k, s):
        if len(idx) <= k:
            return idx
        gg = torch.Generator().manual_seed(s)
        p = torch.randperm(len(idx), generator=gg)
        return idx[p[:k]]
    return cap(tr_idx, n_train, seed + 1), cap(te_idx, n_test, seed + 2)

# ------------------------------------------------------------
# Non-trainable integrator baselines (the honest "do nothing" floor)
# ------------------------------------------------------------
def strapdown_integrate(omega_seq, dt, q0=None):
    """Semi-implicit Euler strapdown integration of a gyro sequence.
    omega_seq (B, S, 3) rad/s. Returns final unit quaternion (B, 4).
    The per-step increments dq_s depend only on omega_s, so the ascending
    product is evaluated as a log-depth tree of pairwise Hamilton products
    (associativity: identical math and gradients, ~60x faster than the
    sequential loop on 1k-step windows)."""
    B, S, _ = omega_seq.shape
    om = omega_seq.to(device=device, dtype=torch.float64)   # ensure device+dtype
    w = om
    wn = w.norm(dim=-1).clamp_min(1e-8)
    dq = axis_angle_to_quat(w / wn.unsqueeze(-1), wn * dt)   # (B, S, 4)
    if q0 is not None:
        dq = torch.cat([q0.to(device=device, dtype=torch.float64).unsqueeze(1), dq], dim=1)
    while dq.shape[1] > 1:
        if dq.shape[1] % 2 == 1:
            eye = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64, device=device)
            dq = torch.cat([dq, eye.expand(dq.shape[0], 1, 4)], dim=1)
        dq = quat_mul(dq[:, 0::2], dq[:, 1::2])
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return dq[:, 0]

# ------------------------------------------------------------
# Trainable models
# ------------------------------------------------------------
def vec_to_angles(v):
    """Unit vector -> (alpha, beta): alpha = atan2(y, x), beta = acos(z)."""
    x, y, z = v.unbind(-1)
    return torch.stack([torch.atan2(y, x), torch.acos(z.clamp(-1+1e-6, 1-1e-6))], -1)

class QuatIMU(nn.Module):
    """Shared per-step QuatKAN edge, copied from run_extensions.py (do not
    change the architecture): direction of omega via the K=1 Fourier
    phase-pair edge, magnitude via the linear channel v = |w|*dt*dir(w),
    node = ascending Hamilton product. Output is the TOTAL rotation as a
    QUATERNION (v1.1: was rotvec; the target is a quaternion now).
    Optionally takes the per-window bias estimate as an extra input: the
    Fourier direction uses omega only, and the bias enters as a linear
    DIRECTION correction (vdir -= W_bias @ bias), so its effect scales with
    |omega|*dt and vanishes at rest."""
    def __init__(self, steps, K=1, dt=1.0, bias_dim=0, sparse=False, canon=False,
                 canon_style='norm'):
        super().__init__()
        self.steps, self.K, self.dt, self.bias_dim = steps, K, dt, bias_dim
        self.sparse = sparse
        self.canon = bool(canon)
        self.canon_style = str(canon_style)
        self.feat_idx = [1, 5, 8] if sparse else list(range((2*K+1)**2))
        self.w = nn.Parameter(torch.randn(len(self.feat_idx), 3) * 0.05)
        if self.canon:
            self.gain = nn.Parameter(torch.ones(1))
            if self.canon_style == 'angle':
                # expose the angles: alpha/beta offsets predicted from features
                # (init 0 -> exact analytic direction), no normalization at all
                self.wa = nn.Parameter(torch.zeros(len(self.feat_idx), 1))
                self.wb_ang = nn.Parameter(torch.zeros(len(self.feat_idx), 1))
                if bias_dim:
                    self.wbias_a = nn.Parameter(torch.zeros(bias_dim, 1))
                    self.wbias_b = nn.Parameter(torch.zeros(bias_dim, 1))
            else:
                # deterministic init at the analytic direction: unit axis components
                with torch.no_grad():
                    self.w.zero_()
                    pos = ({0: 2, 1: 0, 2: 1} if self.sparse else {1: 2, 5: 0, 8: 1})
                    for row, comp in pos.items():
                        self.w[row, comp] = 1.0
        if bias_dim and not (self.canon and self.canon_style == 'angle'):
            self.wb = nn.Parameter(torch.zeros(bias_dim, 3))
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1,self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x, bias=None):
        B = x.shape[0]
        om = x.reshape(B, self.steps, 3)
        m = om.norm(dim=-1).clamp_min(1e-8)
        a = vec_to_angles(om / m.unsqueeze(-1)).reshape(B*self.steps, 2)
        f = self.group_feats(a)
        f = f[:, self.feat_idx]
        if self.canon and self.canon_style == 'angle':
            da = (f @ self.wa).squeeze(-1)
            db = (f @ self.wb_ang).squeeze(-1)
            if self.bias_dim and bias is not None:
                da = da + (bias @ self.wbias_a).squeeze(-1).unsqueeze(1) \
                    .expand(B, self.steps).reshape(B*self.steps)
                db = db + (bias @ self.wbias_b).squeeze(-1).unsqueeze(1) \
                    .expand(B, self.steps).reshape(B*self.steps)
            al, be = a[:, 0] + da, a[:, 1] + db
            vdir = torch.stack([torch.sin(be)*torch.cos(al),
                                torch.sin(be)*torch.sin(al),
                                torch.cos(be)], -1).reshape(B, self.steps, 3)
            v = (m*self.dt*self.gain).unsqueeze(-1) * vdir
        else:
            vdir = torch.einsum('bf,fc->bc', f, self.w).reshape(B, self.steps, 3)
            if self.bias_dim and bias is not None:
                # subtract a LEARNED linear correction from the bias estimate
                vdir = vdir - torch.einsum('bd,dc->bc', bias, self.wb).unsqueeze(1)
            if self.canon:
                vdir = vdir / (vdir.norm(dim=-1, keepdim=True) + 1e-8)
                v = (m*self.dt*self.gain).unsqueeze(-1) * vdir
            else:
                v = (m*self.dt).unsqueeze(-1) * vdir
        nv = v.norm(dim=-1, keepdim=True)
        q = torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1)
        # log-depth tree composition (same math/gradients as the ascending loop)
        while q.shape[1] > 1:
            if q.shape[1] % 2 == 1:
                eye = torch.tensor([1.0, 0, 0, 0], dtype=q.dtype, device=q.device)
                q = torch.cat([q, eye.expand(q.shape[0], 1, 4)], dim=1)
            q = quat_mul(q[:, 0::2], q[:, 1::2])
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return q[:, 0].to(torch.float64)       # v1.1: quaternion output

class MLPFlat(nn.Module):
    """The synthetic strongest opponent, verbatim from run_extensions.py.
    v1.1: the raw 3-vector output is interpreted as a rotation vector and
    mapped through exp (rotvec_to_quat) so that the geodesic loss is
    well-defined for angles beyond pi."""
    def __init__(self, din, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h),
                                 nn.Tanh(), nn.Linear(h, 3))
    def forward(self, x, bias=None):
        if bias is not None:
            x = torch.cat([x, bias], dim=-1)
        return rotvec_to_quat(self.net(x)).to(torch.float64)

# ------------------------------------------------------------
# One full run of a model on a prepared dataset
# ------------------------------------------------------------
def train_eval(model, Xtr, Ytr, Btr, Xte, Yte, Bte, epochs=3000, lr=3e-3, seed=0,
               cosine=False, loss_form='angle2'):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xtr, Btr)
        if loss_form == 'cosine2':
            d = (pred * Ytr).sum(-1).abs().clamp(max=1.0 - 1e-7)
            loss = 2.0 * (1.0 - d).mean()      # bounded-gradient geodesic
        else:
            # v1.1: true SO(3) geodesic loss theta^2, smooth for all angles
            # (replaces rotvec-MSE, which is discontinuous at pi — see header)
            loss = (geo_rad_q(pred, Ytr)**2).mean()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
    with torch.no_grad():
        pred = model(Xte, Bte)
    return geo_deg_q(pred, Yte).mean().item()

# ------------------------------------------------------------
# Main X5 driver
# ------------------------------------------------------------
def run_setting(name, W, data_dir, stride, seeds=(0,1,2), n_train=800, n_test=2000,
                epochs=3000, use_bias_feature=False, groups=('undisturbed',),
                canon=False, cosine=False, loss_form='angle2',
                canon_style='norm'):
    print(f"\n===== X5:{name}  (W={W}, bias_feature={use_bias_feature}) =====")
    X, Y, B, fs, T = make_dataset(data_dir, W=W, stride=stride, groups=groups)
    n = len(X)
    # trial-level split: no window-level leakage across train/test
    tr, te = trial_split(T, n_train, n_test)
    print(f"  trial split: {len(T[tr].unique())} train trials / {len(T[te].unique())} test trials; "
          f"{len(tr)} train / {len(te)} test windows")
    Xtr, Ytr, Btr = X[tr], Y[tr], B[tr]
    Xte, Yte, Bte = X[te], Y[te], B[te]
    dt = 1.0 / fs

    # --- non-trainable integrator baselines ---
    Wseq = W
    om_te = Xte.reshape(-1, Wseq, 3)
    om_raw_te = om_te + Bte.unsqueeze(1)          # re-add bias for the raw baseline
    q_bias = strapdown_integrate(om_te, dt)       # bias-subtracted already in X
    q_raw  = strapdown_integrate(om_raw_te, dt)
    reg(f'X5:{name}:BiasIntegrator', geo_deg_q(q_bias, Yte).cpu().numpy(), params=0)
    reg(f'X5:{name}:RawIntegrator',  geo_deg_q(q_raw,  Yte).cpu().numpy(), params=0)

    # --- trainable models ---
    din = Wseq*3
    models = [
        ('QuatIMU',   lambda: QuatIMU(Wseq, K=1, dt=dt, bias_dim=3 if use_bias_feature else 0)),
        ('MLP h=8',   lambda: MLPFlat(din + (3 if use_bias_feature else 0), h=8)),
        ('MLP h=32',  lambda: MLPFlat(din + (3 if use_bias_feature else 0), h=32)),
    ]
    if use_bias_feature:
        models.append(('QuatIMU-sparse',
                       lambda: QuatIMU(Wseq, K=1, dt=dt, bias_dim=3, sparse=True)))
    if canon:
        models.append(('QuatIMU-canon',
                       lambda: QuatIMU(Wseq, K=1, dt=dt,
                                       bias_dim=3 if use_bias_feature else 0,
                                       canon=True, canon_style=canon_style)))
    for label, ctor in models:
        errs = []
        for s in seeds:
            torch.manual_seed(100 + s)          # seed model init (was drawn pre-seed)
            m = ctor()
            errs.append(train_eval(m, Xtr, Ytr, Btr if use_bias_feature else None,
                                   Xte, Yte, Bte if use_bias_feature else None,
                                   epochs=epochs, lr=3e-3, seed=100+s, cosine=cosine, loss_form=loss_form))
        npar = sum(p.numel() for p in ctor().parameters())
        reg(f'X5:{name}:{label}', errs, params=npar)

    # paired t-tests vs strongest MLP
    for mname in ['QuatIMU', 'MLP h=8']:
        a = RESULTS[f'X5:{name}:{mname}']['per_seed']
        b = RESULTS[f'X5:{name}:MLP h=32']['per_seed']
        t, df, p = paired_t(a, b)
        reg_text(f'X5:{name}:ttest:{mname.replace(" ","")}',
                 f"{mname} vs MLP h=32: paired t={t:.2f}, df={df}, p={p:.2e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')),
                    help='path to broad/data_hdf5 (must contain trials.json + *.hdf5)')
    ap.add_argument('--groups', default='disturbed',
                    help='comma-separated trials.json groups to include '
                         '(default: disturbed — the setting where learning is needed)')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--loss-form', choices=['angle2', 'cosine2'], default='angle2',
                    help='angle2 = legacy theta^2; cosine2 = 2(1-cos), bounded grad')
    ap.add_argument('--canon-style', choices=['norm', 'angle'], default='norm',
                    help='canonical direction head: normalize a vector, or angles')
    ap.add_argument('--canon', action='store_true',
                    help='canonical direction edge: unit direction + scalar gain, '
                         'analytic init (X5b stability probe)')
    ap.add_argument('--cosine', action='store_true',
                    help='cosine LR schedule in train_eval (X5b)')
    ap.add_argument('--only', default=None,
                    help="comma-separated blocks to run, e.g. --only d-short,d-035s "
                         "(choices: a-short, b-1s, b2-035s, c-5s, d-short, d-035s; "
                         "default: all on --groups)")
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    seeds = tuple(range(args.seeds))
    t0 = time.time()

    only = set(args.only.split(',')) if args.only else None
    def want(name):
        return only is None or name in only

    # X5a: 20 steps ~0.07 s — direct replication of synthetic §6.1, no bias
    # feature (bias contributes ~0.01 deg here; raw vs bias integrator will
    # show this). Same 800/2000 split as the synthetic protocol.
    if want('a-short'):
        run_setting('a-short',  W=20,   data_dir=args.data_dir, stride=20,
                    seeds=seeds, use_bias_feature=False, groups=groups, canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)

    # X5b: 286 steps = 1.00 s — honest horizon, bias estimate as a feature.
    # Overlapping windows (stride=W//4) are needed to reach the 800/2000 split
    # (adjacent windows are correlated — see the c-5s note below).
    if want('b-1s'):
        run_setting('b-1s',     W=286,  data_dir=args.data_dir, stride=286//4,
                    seeds=seeds, use_bias_feature=True,  groups=groups, canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)

    # X5b2: 100 steps ~0.35 s — the intermediate regime. On 0.07 s the task is
    # noise-limited (all models tie at the integrator floor ~0.6 deg); on 1 s
    # the strapdown integrator already solves the task (~2 deg) and the learned
    # composers fail (57-87 deg). 0.35 s sits between: the integrator is
    # starting to drift but the 100-step composition is still learnable.
    if want('b2-035s'):
        run_setting('b2-035s',  W=100,  data_dir=args.data_dir, stride=100//4,
                    seeds=seeds, use_bias_feature=True,  groups=groups, canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)

    # X5c: 1430 steps = 5.00 s — long horizon. Heavily overlapping windows
    # (stride=W//10) are needed to reach the 800/2000 split on the disturbed
    # group (W//4 yields only ~1500 windows); adjacent windows are correlated,
    # so treat the per-seed spread (not the test std) as the uncertainty.
    if want('c-5s'):
        run_setting('c-5s',     W=1430, data_dir=args.data_dir, stride=1430//10,
                    seeds=seeds, use_bias_feature=True,  groups=groups, canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)

    # X5d: disturbed-group shortcut blocks. Run on the disturbed trials, where
    # structured gyro noise gives a learned model something to suppress.
    # d-short mirrors a-short (0.07 s) and d-035s mirrors b2-035s (0.35 s),
    # so the clean/disturbed comparison is apples-to-apples.
    if want('d-short'):
        run_setting('d-short',  W=20,   data_dir=args.data_dir, stride=20,
                    seeds=seeds, use_bias_feature=False, groups=('disturbed',), canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)
    if want('d-035s'):
        run_setting('d-035s',   W=100,  data_dir=args.data_dir, stride=100,
                    seeds=seeds, use_bias_feature=True,  groups=('disturbed',), canon=args.canon, cosine=args.cosine, loss_form=args.loss_form, canon_style=args.canon_style)

    RESULTS['_meta'] = dict(wall_s=time.time()-t0, device=str(device))
    print(f"\nTotal wall time: {time.time()-t0:.1f} s")
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
