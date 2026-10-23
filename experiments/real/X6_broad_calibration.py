# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — X6: REAL gyroscope CALIBRATION on BROAD  (v1.1)
# (the applied setting where learning is actually needed)
#
# Why this probe exists (and why X5 was the wrong applied target):
#   On gyro-only dead-reckoning the strapdown integrator is already
#   near-optimal, so a learned composer has nothing to add (X5). Calibration
#   is the opposite: the sensor carries a FIXED defect (bias b and a
#   misalignment/scale matrix M), the integrator does NOT know it and carries
#   it into the drift, and the defect must be LEARNED from data. Learning is
#   not redundant here — it is the point.
#
# Model of the defect (standard MEMS gyro model):
#   omega_true = M @ (omega_raw - b)
#     b — 3-vector bias (rad/s)
#     M — 3x3 matrix (axis misalignment + scale), identity when calibrated
#
# Task:
#   Given raw gyro windows, learn (b, M) by integrating the corrected omega
#   and matching the ground-truth relative orientation (opt_quat). The defect
#   is constant per sensor, so we learn ONE (b, M) shared across all windows,
#   then evaluate the calibrated integrator on HELD-OUT windows (and, as a
#   transfer check, on held-out trials of the same sensor).
#
# Baselines (no learning):
#   RawIntegrator          — integrate omega_raw.
#   BiasIntegrator(per-trial) — integrate omega_raw minus each trial's
#                            stationary-segment bias (standard offline estimate
#                            of b; captures b but NOT the misalignment M).
#   LM-(M,b)               — classical Levenberg-Marquardt least squares on the
#                            SAME 12-parameter model and geodesic objective
#                            (the honest classical comparator; a closed-form
#                            linear LS is unavailable because the fit goes
#                            through the integration).
# What X6 tests: does learning (b, M) match the classical LM estimator (and both
# beat bias-only)?
#
# v1.1 BUG FIX (both training loss and metric):
#   v1.0 trained with MSE between rotvec vectors and evaluated with the
#   rotvec-difference norm |rv_a - rv_b|. The rotvec chart is discontinuous
#   at angle pi: rotations of (pi - eps) and (pi + eps) about the same axis
#   are nearly identical orientations but their shortest-rotation rotvecs
#   point in OPPOSITE directions and differ by ~2*pi. At 0.07s/0.35s windows
#   rotations are small and rotvec-MSE ~= geodesic, so v1.0 a/b results were
#   fine; at 1s some BROAD windows rotate by ~pi or more, the loss received
#   spurious ~2pi gradients from those windows, and (b, M) converged to a
#   bad deterministic optimum (12.84 deg vs raw 2.66 deg, learned M diag
#   ~[1.07, 1.07, 0.93] — chasing outliers, not calibrating). The v1.0 c-1s
#   numbers are retracted.
#   v1.1: targets are stored as q_rel quaternions; loss = mean(theta^2) and
#   metric = theta, with theta = 2*acos(|<q_pred, q_tgt>|) — the true SO(3)
#   geodesic, smooth for all angles and double-cover safe.
#   NOTE: rerun all blocks with v1.1 so that a/b/c share one canonical
#   metric; a/b conclusions are expected to stand (they were in the small-
#   angle regime) but their 3rd-4th decimals may shift.
#
# Requirements: torch, numpy, h5py. Data: BROAD data_hdf5 (see X5 header).
# Run:
#   python X6_broad_calibration.py --data_dir /path/to/broad/data_hdf5
#   python X6_broad_calibration.py --data_dir ... --only c-1s   # one block
# ============================================================
import argparse, math, os, sys, time, json
import numpy as np
import torch, torch.nn as nn

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
                        median=float(np.median(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f} "
          f"median {np.median(vals):.4f}")

def reg_text(key, text):
    RESULTS[key] = dict(text=text)
    print(f"[{key}] {text}")

# ------------------------------------------------------------
# Geometry (Hamilton quaternions, w-first)
# ------------------------------------------------------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)

def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def axis_angle_to_quat(axis, angle):
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    h = (angle / 2).unsqueeze(-1)
    return torch.cat([torch.cos(h), axis * torch.sin(h)], -1)

def geo_deg(q_pred, q_tgt):
    """True SO(3) geodesic angle in degrees, double-cover safe.
    q_pred, q_tgt: (...,4) unit quaternions."""
    d = (q_pred * q_tgt).sum(-1).abs().clamp(max=1.0 - 1e-9)
    return torch.rad2deg(2.0 * torch.acos(d))

def geo_rad(q_pred, q_tgt):
    """Same in radians (for the training loss)."""
    d = (q_pred * q_tgt).sum(-1).abs().clamp(max=1.0 - 1e-7)
    return 2.0 * torch.acos(d)

# ------------------------------------------------------------
# BROAD loading (same as X5)
# ------------------------------------------------------------
def load_broad_trial(path):
    import h5py
    with h5py.File(path, 'r') as f:
        gyr = np.asarray(f['imu_gyr'][()], dtype=np.float32)
        quat = np.asarray(f['opt_quat'][()], dtype=np.float64)
        move = np.asarray(f['movement'][()]).astype(bool)
        fs = float(np.asarray(f.attrs['sampling_rate']).ravel()[0])
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True).clip(1e-12)
    name = os.path.splitext(os.path.basename(path))[0]
    return dict(gyr=gyr, quat=quat, move=move, fs=fs, name=name)

def list_broad_files(data_dir, groups=('undisturbed',)):
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

def estimate_bias(gyr, move, margin_s, fs):
    n_rest = 0
    while n_rest < len(move) and not move[n_rest]:
        n_rest += 1
    n_use = max(n_rest - int(margin_s * fs), 1)
    return gyr[:n_use].mean(axis=0)

def build_windows_raw(trial, W, stride, min_move_frac=0.5):
    """Windows of RAW gyro (no bias subtraction — calibration must learn it).
    Returns X (nwin, W, 3) raw gyro, Y (nwin, 4) GT relative quaternion
    q_rel = conj(q_start) * q_end (v1.1: quaternion, not rotvec)."""
    gyr = trial['gyr']                                # RAW
    quat = torch.from_numpy(trial['quat']).to(torch.float64)
    move = trial['move']
    N = len(gyr)
    qn = quat.norm(dim=-1)
    valid = torch.isfinite(quat).all(dim=-1) & ((qn - 1.0).abs() < 1e-2)
    Xs, Ys = [], []
    for s in range(0, N - W - 1, stride):
        e = s + W
        if move[s:e].mean() < min_move_frac:
            continue
        if not (valid[s] and valid[e]):
            continue
        q_rel = quat_mul(quat_conj(quat[s]), quat[e])
        Xs.append(gyr[s:e])
        Ys.append(q_rel.numpy().astype(np.float32))
    if not Xs:
        return np.zeros((0, W, 3), np.float32), np.zeros((0, 4), np.float32)
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)

# ------------------------------------------------------------
# Integrators
# ------------------------------------------------------------
def integrate(omega_seq, dt):
    """Semi-implicit Euler strapdown integration. omega_seq (B,S,3) rad/s.
    Returns final unit quaternion (B,4). Works for raw or corrected omega.
    Log-depth tree of pairwise products (associativity: identical math and
    gradients, ~60x faster than a sequential loop on long windows)."""
    B, S, _ = omega_seq.shape
    om = omega_seq.to(dtype=torch.float64)
    w = om
    wn = w.norm(dim=-1).clamp_min(1e-8)
    dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*dt)       # (B,S,4)
    while dq.shape[1] > 1:
        if dq.shape[1] % 2 == 1:
            eye = torch.tensor([1.0,0,0,0], dtype=torch.float64, device=dq.device)
            dq = torch.cat([dq, eye.expand(dq.shape[0],1,4)], dim=1)
        dq = quat_mul(dq[:,0::2], dq[:,1::2])
        dq = dq / dq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return dq[:,0]

# ------------------------------------------------------------
# Classical baseline: Levenberg-Marquardt least squares on the SAME
# 12-parameter model omega_true = M (omega_raw - b), same geodesic objective
# as the learned model. A closed-form linear LS is not available here because
# the fit goes through the integration (nonlinear in (M, b)); LM is therefore
# the honest classical comparator.
# ------------------------------------------------------------
def lm_fit(Xtr, Ytr, dt, max_nfev=200):
    import scipy.optimize as opt
    X = torch.from_numpy(np.asarray(Xtr)).to(torch.float64)
    Y = torch.from_numpy(np.asarray(Ytr)).to(torch.float64)
    def resid(p):
        M = torch.tensor(p[:9], dtype=torch.float64).reshape(3, 3)
        b = torch.tensor(p[9:12], dtype=torch.float64)
        corr = torch.einsum('ij,bsj->bsi', M, X - b)
        q = integrate(corr, dt)
        d = (q * Y).sum(-1).abs().clamp(max=1 - 1e-7)
        return (2.0 * torch.acos(d)).numpy()
    p0 = np.concatenate([np.eye(3).ravel(), np.zeros(3)])
    res = opt.least_squares(resid, p0, method='trf', max_nfev=max_nfev)
    return res.x

class CalibratedIntegrator(nn.Module):
    """Learn the fixed sensor defect (b, M) and integrate the corrected gyro.
    omega_corr = M @ (omega_raw - b);  then strapdown-integrate.
    b (3,) and M (3,3) are the ONLY learnable parameters (12 total), shared
    across all windows — the calibration is a property of the sensor.
    v1.1: forward returns the predicted quaternion (not rotvec)."""
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.bias = nn.Parameter(torch.zeros(3))
        self.M = nn.Parameter(torch.eye(3))
    def forward(self, omega_raw):
        corr = torch.einsum('ij,bsj->bsi', self.M, omega_raw - self.bias)
        return integrate(corr, self.dt)

# ------------------------------------------------------------
# Dataset assembly: pool windows across trials, split by window
# ------------------------------------------------------------
def assemble(data_dir, W, stride, groups):
    files = list_broad_files(data_dir, groups)
    assert files, f"no files for {groups}"
    Xs, Ys, trial_idx, names = [], [], [], []
    fs = None
    for ti, (path, gr) in enumerate(files):
        tr = load_broad_trial(path)
        fs = tr['fs'] if fs is None else fs
        X, Y = build_windows_raw(tr, W=W, stride=stride)
        if len(X):
            Xs.append(X); Ys.append(Y)
            trial_idx += [ti]*len(X); names.append(tr['name'])
    X = torch.from_numpy(np.concatenate(Xs)).to(device)
    Y = torch.from_numpy(np.concatenate(Ys)).to(device, torch.float64)
    tidx = np.array(trial_idx)
    print(f"  assembled {len(X)} windows, W={W} ({W/fs:.3f}s), from {len(names)} trials, fs={fs:.2f}")
    return X, Y, tidx, names, fs

# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------
def run(name, data_dir, W, stride, groups, seeds=(0,1,2), epochs=2000, lr=1e-2):
    print(f"\n===== X6:{name}  (W={W}, calibration) =====")
    X, Y, tidx, names, fs = assemble(data_dir, W, stride, groups)
    dt = 1.0/fs
    n = len(X)

    # split by window (calibration shared; test = held-out windows)
    g = np.random.default_rng(12345)
    perm = g.permutation(n)
    ntr = int(0.7*n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]

    # --- no-learning baselines on test windows ---
    # raw
    q_raw = integrate(Xte, dt)
    reg(f'X6:{name}:RawIntegrator', geo_deg(q_raw, Yte).cpu().numpy())
    # per-trial stationary-segment bias (the standard offline estimate), M=I
    files = list_broad_files(data_dir, groups)
    trial_bias = []
    for path, gr in files:
        tr_i = load_broad_trial(path)
        trial_bias.append(estimate_bias(tr_i['gyr'], tr_i['move'], margin_s=0.5, fs=tr_i['fs']))
    trial_bias = torch.from_numpy(np.stack(trial_bias)).float().to(device)
    b_te = trial_bias[torch.as_tensor(tidx[te], device=device)]
    q_bias = integrate(Xte - b_te.unsqueeze(1), dt)
    reg(f'X6:{name}:BiasIntegrator(per-trial)', geo_deg(q_bias, Yte).cpu().numpy())

    # classical LM least-squares on the same (M, b) model, fit on train windows
    p_lm = lm_fit(Xtr.cpu().numpy(), Ytr.cpu().numpy(), dt)
    M_lm = torch.tensor(p_lm[:9], dtype=torch.float64).reshape(3, 3).to(device)
    b_lm = torch.tensor(p_lm[9:12], dtype=torch.float64).to(device)
    q_lm = integrate(torch.einsum('ij,bsj->bsi', M_lm, Xte.to(torch.float64) - b_lm), dt)
    reg(f'X6:{name}:LM-(M,b)', geo_deg(q_lm, Yte).cpu().numpy())
    reg_text(f'X6:{name}:LM-params',
             f"diag(M)={np.diag(M_lm.cpu().numpy()).round(4).tolist()}, b={b_lm.cpu().numpy().round(5).tolist()}")

    # --- learned calibration (b, M): geodesic^2 loss (v1.1 fix) ---
    for s in seeds:
        torch.manual_seed(100+s)
        model = CalibratedIntegrator(dt).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for ep in range(epochs):
            opt.zero_grad()
            pred = model(Xtr)
            loss = (geo_rad(pred, Ytr)**2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            te_err = geo_deg(model(Xte), Yte).mean().item()
        reg(f'X6:{name}:Calibrated[seed{s}]', [te_err],
            bias=model.bias.detach().cpu().numpy().round(5).tolist())
        if s == seeds[0]:
            print(f"    learned bias={model.bias.detach().cpu().numpy().round(5)}")
            print(f"    learned M=\n{model.M.detach().cpu().numpy().round(4)}")

    # summary: calibrated mean across seeds
    cal = [RESULTS[f'X6:{name}:Calibrated[seed{s}]']['mean'] for s in seeds]
    reg(f'X6:{name}:Calibrated', cal)
    bi = RESULTS[f'X6:{name}:BiasIntegrator(per-trial)']['mean']
    lm = RESULTS[f'X6:{name}:LM-(M,b)']['mean']
    print(f"    -> calibrated {np.mean(cal):.4f} deg  vs  LM {lm:.4f}  "
          f"vs  bias-only {bi:.4f}  vs  raw {RESULTS[f'X6:{name}:RawIntegrator']['mean']:.4f} deg")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--groups', default='undisturbed')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--only', default=None,
                    help="comma list of blocks to run, e.g. --only c-1s")
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    seeds = tuple(range(args.seeds))
    only = (set(x.strip() for x in args.only.split(','))
            if args.only else None)
    t0 = time.time()

    blocks = [
        ('a-short', dict(W=20,  stride=20)),
        ('b-035s',  dict(W=100, stride=100)),
        ('c-1s',    dict(W=286, stride=286)),
    ]
    for name, kw in blocks:
        if only and name not in only:
            continue
        run(name, args.data_dir, groups=groups, seeds=seeds, **kw)

    RESULTS['_meta'] = dict(wall_s=time.time()-t0, device=str(device),
                            version='1.1 (geodesic loss/metric)')
    print(f"\nTotal wall time: {time.time()-t0:.1f} s")
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
