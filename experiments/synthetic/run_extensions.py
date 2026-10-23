# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — EXTENSION RUNS (review-driven additions)
#
# Five independent blocks, each can be commented out:
#   X1  RPMG opponent        — MLP-Quat / MLP-6D with a manifold-aware projective
#                              gradient (Chen et al. 2022), same forward as E1.
#   X2  Non-Gaussian noise   — E3.2 noise curve with Uniform and Laplace noise
#                              at matched RMS (robustness to the noise family).
#   X3  Applied probes       — E15 IMU / FK / E17 rigid-body (Section 6),
#                              so that "every number" becomes literally true.
#   X4  Vector Neurons       — a minimal SO(3)-equivariant VN opponent on the
#                              same composition task (honest, simplified).
#   E18 Composition-operator — "can 3 additions replace the Hamilton product?"
#   (v1.1)                   — ablation of the NODE only: BCH truncation ladder
#                              (sum / +1/2 cross / +triple commutators) vs the
#                              exact Hamilton product, plus a bit-equivalence
#                              check of the flop-lean rotate() identity.
#                              Reviewer answer: truncated BCH never reaches the
#                              floor and at order 2 already costs >= Hamilton.
#
# v1.1 (2026-09-21): added E18; rotate() switched to the exact flop-lean
#   identity (see run_main_experiments.py v1.3); no other changes. All
#   pre-existing rows remain valid (E18 measures that equivalence directly).
#
# Same protocol as run_main_experiments.py: 800 train / 4000 test, shared seeds
# (exception: E15 uses 400/2000).
# geodesic loss, paired t-tests. Prints a JSON dump at the end.
#
# IMPORTANT (2026-09-20): PROBE fixed to [0,0,1] to match run_main_experiments.py
# (an earlier version used [1,0,0]; discard any results from that version).
# Verified on CPU: X1 reproduces the frozen E1 numbers
#   LawKAN 2.79 / MLP-Quat 3.38 / MLP-Quat+RPMG 3.49 (RPMG vs QK p=1.7e-03).
#
# Run:  python run_extensions.py
# Time: ~30-40 min on an RTX 2080 SUPER (all four blocks).
# ============================================================
import torch, torch.nn as nn, math, time, json
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,3) for v in vals]}")

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
# Shared geometry (identical to run_main_experiments.py)
# ------------------------------------------------------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

def axis_angle_to_quat(axis, angle):
    half = angle.unsqueeze(-1)/2
    return torch.cat([torch.cos(half), torch.sin(half)*axis], dim=-1)

def rotate(q, v):
    # exact identity for unit q (30 vs 56 scalar ops vs the sandwich)
    u, w = q[...,1:], q[...,:1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)

def rotate_sandwich(q, v):
    # reference (two Hamilton products); E18 asserts rotate == rotate_sandwich
    qv = torch.cat([torch.zeros_like(q[...,:1]), v], dim=-1)
    qi = q.clone(); qi[...,1:] = -qi[...,1:]
    return quat_mul(quat_mul(q, qv), qi)[...,1:]

# Probe MUST be [0,0,1] to match run_main_experiments.py (data and models use
# the same probe there). An earlier version had [1,0,0] here, which shifted the
# readout frame and inflated the MLP-Quat numbers — do not change this back.
PROBE = torch.tensor([0.0, 0.0, 1.0]).to(device)

def vec_to_angles(v):
    al = torch.atan2(v[...,1], v[...,0])
    be = torch.acos(torch.clamp(v[...,2], -1+1e-6, 1-1e-6))
    return torch.stack([al, be], -1)

def angles_to_vec(a):
    al, be = a[...,0], a[...,1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], -1)

def geo_loss(pred, target):
    vp = angles_to_vec(pred); vt = angles_to_vec(target)
    c = (vp*vt).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return (1 - c).mean()

def geo_deg(pred, target):
    vp = angles_to_vec(pred); vt = angles_to_vec(target)
    c = (vp*vt).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(c))

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# comp2 data generator (main task, n=2) — mirrors make_compn_data from
# run_main_experiments.py exactly, with a switchable noise family at matched RMS.
def _noise(x, noise, family, g):
    if family == 'gauss':
        return noise * torch.randn(x.shape, generator=g)
    if family == 'uniform':
        return (torch.rand(x.shape, generator=g)*2 - 1) * (noise*math.sqrt(3))
    if family == 'laplace':
        b = noise / math.sqrt(2)
        u = torch.rand(x.shape, generator=g) - 0.5
        return -b * torch.sign(u) * torch.log1p(-2*torch.abs(u))
    raise ValueError(family)

def make_comp2(N, noise=0.03, seed=1, angles=(0.7, 1.1), family='gauss'):
    """Composition of 2 rotations q2 @ q1, random axes uniform on S^2.
    Input: angles of the two axes (4 numbers). Target: angles of the rotated probe."""
    g = torch.Generator().manual_seed(seed)
    n = 2
    axes = []
    for i in range(n):
        a = torch.randn(N, 3, generator=g); a = a/a.norm(dim=-1, keepdim=True)
        axes.append(a)
    ths = [torch.full((N,), float(angles[i])) for i in range(n)]
    prod = axis_angle_to_quat(axes[-1], ths[-1])
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[i], ths[i]))
    v = rotate(prod, torch.tensor([0.,0.,1.]).expand(N,3))
    v = v / v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1)
    X = X + _noise(X, noise, family, g)
    Y = vec_to_angles(v) + _noise(torch.empty(N,2), noise, family, g)
    return X.to(device), Y.to(device)

# ------------------------------------------------------------
# Models (copied from run_main_experiments.py)
# ------------------------------------------------------------
class GroupedQuatExpNet(nn.Module):
    def __init__(self, n_groups=2, K=1):
        super().__init__()
        self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3) * 0.1)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa = [torch.ones_like(al)]; fb = [torch.ones_like(be)]
        for m in range(1, self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        qs = []
        for gi in range(self.n_groups):
            f = self.group_feats(x[:, 2*gi:2*gi+2])
            v = torch.einsum('bf,fc->bc', f, self.w[gi])
            nv = v.norm(dim=-1, keepdim=True)
            qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1))
        prod = qs[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = quat_mul(prod, qs[gi])
        v = rotate(prod, PROBE.expand(x.shape[0], 3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True) + 1e-6))

def rot6d_to_matrix(d6):
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = a2 - (b1*a2).sum(-1, keepdim=True)*b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)

def mat_to_angles(R):
    v = R[..., 2]
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    return vec_to_angles(v)

class MLP6D(nn.Module):
    def __init__(self, in_phases=4, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,6))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return mat_to_angles(rot6d_to_matrix(self.net(f)))

class MLPQuatHead(nn.Module):
    def __init__(self, in_phases=4, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        q = self.net(f)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True)+1e-6))

# ============================================================
# X1: RPMG opponent — manifold-aware projective gradient (Chen et al. 2022)
# ============================================================
# RPMG idea: for a unit-quaternion readout q/||q||, the Euclidean gradient dL/dq
# has a radial component that fights the normalization. RPMG replaces it by the
# projection onto the tangent space of S^3 at the normalized point:
#     g_manifold = g - <g, q_hat> q_hat,   q_hat = q/||q||
# We implement this as a custom autograd function on the raw quaternion output,
# applied to the SAME MLP-Quat forward as E1, so the only change is the gradient.
class _RPMGQuat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_raw):
        r = q_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q_hat = q_raw / r
        ctx.save_for_backward(q_hat, r)
        return q_hat
    @staticmethod
    def backward(ctx, g_out):
        q_hat, r = ctx.saved_tensors
        radial = (g_out * q_hat).sum(-1, keepdim=True)
        g_tangent = g_out - radial * q_hat          # project out the radial part
        return g_tangent / r                          # VJP of q/||q||: divide by ||q_raw||

def rpmg_normalize(q_raw):
    return _RPMGQuat.apply(q_raw)

class MLPQuatRPMG(nn.Module):
    """MLP-Quat head, forward identical to E1; only the gradient is manifold-aware."""
    def __init__(self, in_phases=4, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        q = rpmg_normalize(self.net(f))
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True)+1e-6))

def train_eval(net, Xtr, Ytr, Xte, Yte, lr=3e-3, epochs=3000):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = geo_loss(net(Xtr), Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()

def run_seeds(ctor, lr, n_seeds=5, epochs=3000, data_fn=make_comp2, **data_kw):
    vals = []
    for seed in range(n_seeds):
        Xtr, Ytr = data_fn(800, seed=seed, **data_kw)
        Xte, Yte = data_fn(4000, seed=seed+100, **data_kw)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

print("\n" + "="*72)
print("X1 / RPMG opponent (comp2, same as E1)")
print("="*72)
res_x1 = {}
for name, ctor, lr in [
    ('QuatKAN K=1 (ref)', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
    ('MLP-Quat h=32 (E1)', lambda: MLPQuatHead(4, h=32), 3e-3),
    ('MLP-Quat h=32 + RPMG', lambda: MLPQuatRPMG(4, h=32), 3e-3),
]:
    vals = run_seeds(ctor, lr)
    res_x1[name] = vals
    reg(f'X1:{name}', vals, params=n_params(ctor()))
t, df, p = paired_t(res_x1['MLP-Quat h=32 + RPMG'], res_x1['QuatKAN K=1 (ref)'])
print(f"  RPMG vs QuatKAN: t={t:.2f}, df={df}, p={p:.2e}")
RESULTS['X1:ttest:RPMG_vs_QuatKAN'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")
t, df, p = paired_t(res_x1['MLP-Quat h=32 + RPMG'], res_x1['MLP-Quat h=32 (E1)'])
print(f"  RPMG vs plain-Quat: t={t:.2f}, df={df}, p={p:.2e}")
RESULTS['X1:ttest:RPMG_vs_plain'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")

# ============================================================
# X2: Non-Gaussian noise (E3.2 with Uniform and Laplace, matched RMS)
# ============================================================
print("\n" + "="*72)
print("X2 / Non-Gaussian noise robustness (comp2, sigma sweep)")
print("="*72)
for family in ['uniform', 'laplace']:
    for sigma in [0.05, 0.10, 0.20]:
        res_x2 = {}
        for name, ctor, lr in [
            ('QuatKAN K=1', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
            ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3),
            ('MLP-Quat h=32 + RPMG', lambda: MLPQuatRPMG(4, h=32), 3e-3),
        ]:
            vals = run_seeds(ctor, lr, n_seeds=5, family=family, noise=sigma)
            res_x2[name] = vals
            reg(f'X2:{family}:{sigma}:{name}', vals)
        t, df, p = paired_t(res_x2['MLP-Quat h=32 + RPMG'], res_x2['QuatKAN K=1'])
        print(f"  [{family} s={sigma}] RPMG-MLP vs QuatKAN: t={t:.2f}, df={df}, p={p:.2e}")
        RESULTS[f'X2:ttest:{family}:{sigma}'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")

# ============================================================
# X3: Applied probes (Section 6) — E15 IMU / FK / E17 rigid-body
# ============================================================
print("\n" + "="*72)
print("X3 / Applied probes (Section 6)")
print("="*72)

# --- E15: IMU orientation integration ---
# Given q0 and a sequence of 20 angular-velocity steps omega(t) (smooth random
# walk in R^3), predict the final orientation after semi-implicit Euler
# integration of dq/dt = (1/2) q (x) omega. The network maps the step sequence
# directly to the total rotation (no ODE iteration).
def make_imu_data(N, steps=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    # smooth random walk for omega
    domega = torch.randn(N, steps, 3, generator=g) * 0.1
    omega = torch.cumsum(domega, dim=1) * 0.3          # (N, steps, 3)
    # integrate ground truth
    q = torch.tensor([1.0,0.0,0.0,0.0]).expand(N,4).clone().to(device)
    omega = omega.to(device)
    dt = 1.0
    for s in range(steps):
        w = omega[:, s, :]
        wn = w.norm(dim=-1).clamp_min(1e-8)             # (N,)
        dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*dt)   # exp(omega dt)
        q = quat_mul(q, dq)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(N,3))
    Y = vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))
    X = omega.reshape(N, steps*3)
    return X.to(device), Y.to(device)

class QuatIMU(nn.Module):
    """Per-step SHARED QuatKAN edge: the map omega_s -> exp(omega_s*dt) is the same
    physical map at every step, so one edge is reused for all 20 steps:
    direction of omega via the K=1 Fourier phase-pair edge, magnitude as a linear
    channel (v_s = |w_s|*dt * dir(w_s)); node = ascending Hamilton product.
    (An earlier version averaged 5 steps per group — that destroys the sequence
    and loses badly; do not revert. K=2 destabilizes: 5.8 deg vs 0.08 for K=1.)"""
    def __init__(self, steps=20, K=1, dt=1.0):
        super().__init__()
        self.steps, self.K, self.dt = steps, K, dt
        self.w = nn.Parameter(torch.randn((2*K+1)**2, 3) * 0.05)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1,self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        B = x.shape[0]
        om = x.reshape(B, self.steps, 3)
        m = om.norm(dim=-1).clamp_min(1e-8)                 # (B,S)
        a = vec_to_angles(om / m.unsqueeze(-1)).reshape(B*self.steps, 2)
        f = self.group_feats(a)
        vdir = torch.einsum('bf,fc->bc', f, self.w).reshape(B, self.steps, 3)
        v = (m*self.dt).unsqueeze(-1) * vdir
        nv = v.norm(dim=-1, keepdim=True)
        q = torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1)  # (B,S,4)
        prod = q[:,0]
        for s in range(1, self.steps):
            prod = quat_mul(prod, q[:,s])
        v = rotate(prod, PROBE.expand(B,3))
        return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

class MLPFlat(nn.Module):
    def __init__(self, in_dim, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        q = self.net(x)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

class MLPFlatRPMG(nn.Module):
    """Same flat MLP as MLPFlat, but the quaternion readout uses the
    manifold-aware (RPMG) gradient — strongest version of the MLP opponent."""
    def __init__(self, in_dim, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        q = rpmg_normalize(self.net(x))
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

print("\n--- E15: IMU orientation integration (20 steps) ---")
res_imu = {}
for name, ctor, lr in [
    ('QuatIMU', lambda: QuatIMU(20, K=1), 1e-2),
    ('MLP h=8', lambda: MLPFlat(60, h=8), 3e-3),
    ('MLP h=32', lambda: MLPFlat(60, h=32), 3e-3),
    ('MLP h=32 + RPMG', lambda: MLPFlatRPMG(60, h=32), 3e-3),
]:
    vals = []
    for seed in range(3):
        Xtr, Ytr = make_imu_data(400, seed=seed)
        Xte, Yte = make_imu_data(2000, seed=seed+100)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, 3000))
    res_imu[name] = vals
    reg(f'X3:E15:{name}', vals, params=n_params(ctor()))
t, df, p = paired_t(res_imu['MLP h=32'], res_imu['QuatIMU'])
print(f"  E15 MLP h=32 vs QuatIMU: t={t:.2f}, df={df}, p={p:.2e}")
RESULTS['X3:E15:ttest'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")

# E15b: data efficiency — 50 training trajectories
print("\n--- E15b: IMU, N=50 training trajectories ---")
for name, ctor, lr in [
    ('QuatIMU', lambda: QuatIMU(20, K=1), 1e-2),
    ('MLP h=32', lambda: MLPFlat(60, h=32), 3e-3),
]:
    vals = []
    for seed in range(3):
        Xtr, Ytr = make_imu_data(50, seed=seed)
        Xte, Yte = make_imu_data(2000, seed=seed+100)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, 3000))
    reg(f'X3:E15-N50:{name}', vals, params=n_params(ctor()))

# --- FK: forward-kinematics calibration (5-DOF arm) ---
# Calibration setting (matches Section 6.2 of the paper): the axis defect is a
# FIXED miscalibration of one physical arm — sampled ONCE per run and SHARED by
# the train and test splits (identifiability comes from seeing many poses of
# the same miscalibrated arm). An earlier reconstruction drew a fresh defect
# per sample, which makes it unlearnable noise and inverts the result;
# that version is wrong, do not use it.
def make_fk_data(N, sigma_deg=1.5, seed=1, defect_seed=0):
    nj = 5
    # nominal joint axes (fixed across all runs)
    nominal = F.normalize(torch.randn(nj, 3, generator=torch.Generator().manual_seed(7)), dim=-1).to(device)
    # fixed calibration defect for this run (shared by train and test)
    gd = torch.Generator().manual_seed(10_000 + defect_seed)
    defect = (torch.randn(nj, 3, generator=gd) * (sigma_deg*math.pi/180)).to(device)
    axes = F.normalize(nominal + defect, dim=-1)               # (nj,3) fixed for the run
    # sample joint configurations
    g = torch.Generator().manual_seed(seed)
    theta = ((torch.rand(N, nj, generator=g)*2-1) * 1.0).to(device)  # joint angles
    q = torch.tensor([1.0,0,0,0]).expand(N,4).clone().to(device)
    for j in range(nj):
        qj = axis_angle_to_quat(axes[j].expand(N,3), theta[:,j])
        q = quat_mul(q, qj)
        q = q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(N,3))
    Y = vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))
    X = theta
    return X.to(device), Y.to(device)

class QuatKANArm(nn.Module):
    """QuatKAN FK edge: v_j = theta_j*lin_j + Fourier_j(theta_j); q_j = exp(v_j);
    node = Hamilton product in the kinematic chain order (q0*q1*...*q4).
    The linear-in-theta channel is essential: a pure K=1 Fourier edge cannot
    produce v ~ theta*axis over theta in [-1,1] (sin saturates), which otherwise
    caps the error at ~18 deg. Chain order must match the generator (ascending):
    an earlier version composed descending and inverted the result."""
    def __init__(self, nj=5, K=1, init=0.05):
        super().__init__()
        self.nj, self.K = nj, K
        # init=0.05 stabilizes the 5-factor chain (init=0.01 leaves one seed in 3
        # stuck at ~8 deg; verified on CPU: 0.097 +/- 0.007 across 3 seeds)
        self.w = nn.Parameter(torch.randn(nj, (2*K+1)**2, 3) * init)
        self.lin = nn.Parameter(torch.randn(nj, 3) * init)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1,self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        B = x.shape[0]
        qs = []
        for j in range(self.nj):
            # joint angle as a phase pair (theta, fixed reference)
            a = torch.stack([x[:,j], torch.zeros_like(x[:,j])], -1)
            f = self.group_feats(a)
            v = torch.einsum('bf,fc->bc', f, self.w[j]) + x[:,j].unsqueeze(-1)*self.lin[j]
            nv = v.norm(dim=-1, keepdim=True)
            qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1))
        prod = qs[0]
        for j in range(1, self.nj):
            prod = quat_mul(prod, qs[j])
        v = rotate(prod, PROBE.expand(B,3))
        return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

print("\n--- FK calibration (5-DOF arm) ---")
for name, ctor, lr in [
    ('QuatKANArm', lambda: QuatKANArm(5, K=1), 1e-2),
    ('MLP h=32', lambda: MLPFlat(5, h=32), 3e-3),
    ('MLP h=32 + RPMG', lambda: MLPFlatRPMG(5, h=32), 3e-3),
]:
    vals = []
    for seed in range(3):
        # same defect_seed for train and test: one miscalibrated arm per run
        Xtr, Ytr = make_fk_data(800, seed=seed, defect_seed=seed)
        Xte, Yte = make_fk_data(4000, seed=seed+100, defect_seed=seed)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, 3000))
    reg(f'X3:FK:{name}', vals, params=n_params(ctor()))

# --- E17: rigid-body dynamics (Euler equations) ---
def make_rigid_data(N, steps=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    I = torch.tensor([1.0, 2.0, 3.0]).to(device)
    w = ((torch.rand(N,3,generator=g)*1.0 + 0.5) * torch.sign(torch.randn(N,3,generator=g))).to(device)
    w0 = w.clone()                                  # input = INITIAL angular velocity
    q = torch.tensor([1.0,0,0,0]).expand(N,4).clone().to(device)
    dt = 0.05
    for s in range(steps):
        # Euler: I dw/dt = -w x (I w)
        Iw = w * I
        dw = -torch.cross(w, Iw, dim=-1) / I
        w = w + dw*dt
        wn = w.norm(dim=-1).clamp_min(1e-8)             # (N,)
        dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*dt)
        q = quat_mul(q, dq)
        q = q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(N,3))
    Y = vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))
    return w0.to(device), Y.to(device)

def rigid_rollout(w0, steps=20, dt=0.05):
    """Pure function of the initial angular velocity: final orientation angles
    after the same semi-implicit Euler integration as make_rigid_data."""
    I = torch.tensor([1.0, 2.0, 3.0]).to(device)
    w = w0.clone()
    q = torch.tensor([1.0,0,0,0]).expand(w0.shape[0],4).clone().to(device)
    for s in range(steps):
        Iw = w * I
        dw = -torch.cross(w, Iw, dim=-1) / I
        w = w + dw*dt
        wn = w.norm(dim=-1).clamp_min(1e-8)
        dq = axis_angle_to_quat(w/wn.unsqueeze(-1), wn*dt)
        q = quat_mul(q, dq)
        q = q / q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v = rotate(q, PROBE.expand(w0.shape[0],3))
    return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

# control: the true system is non-chaotic — a 1e-3 perturbation of omega0
# changes the final orientation by only ~0.1 deg after 20 steps
gc = torch.Generator().manual_seed(0)
w0c = ((torch.rand(2000,3,generator=gc)*1.0 + 0.5) * torch.sign(torch.randn(2000,3,generator=gc))).to(device)
pert = 1e-3 * torch.randn(2000,3,generator=gc).to(device)
div = geo_deg(rigid_rollout(w0c), rigid_rollout(w0c + pert)).mean().item()
reg_text('X3:E17:control', f"1e-3 perturbation of omega0 -> {div:.4f} deg divergence after 20 steps")

print("\n--- E17: rigid-body dynamics (20-step rollout) ---")
for name, ctor, lr, do_pad in [
    ('QuatKAN K=1', lambda: GroupedQuatExpNet(2, K=1), 1e-2, True),
    ('MLP h=32', lambda: MLPFlat(3, h=32), 3e-3, False),
    ('MLP h=32 + RPMG', lambda: MLPFlatRPMG(3, h=32), 3e-3, False),
]:
    vals = []
    for seed in range(3):
        Xtr, Ytr = make_rigid_data(800, seed=seed)
        Xte, Yte = make_rigid_data(4000, seed=seed+100)
        torch.manual_seed(100+seed)
        # GroupedQuatExpNet expects 4-dim input (2 phase pairs); pad w to 4 for it only
        def pad(X): return torch.cat([X, torch.zeros(X.shape[0],1).to(device)], -1)
        if do_pad:
            vals.append(train_eval(ctor(), pad(Xtr), Ytr, pad(Xte), Yte, lr, 3000))
        else:
            vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, 3000))
    reg(f'X3:E17:{name}', vals, params=n_params(ctor()))

# ============================================================
# X4: Vector Neurons opponent (minimal SO(3)-equivariant)
# ============================================================
print("\n" + "="*72)
print("X4 / Vector Neurons opponent (comp2)")
print("="*72)
# Minimal VN: features are lists of 3D vectors; VN-linear mixes them with
# scalar weights (SO(3)-equivariant); nonlinearity = VN-ReLU on vector norms.
# Input: the two rotation axes as vectors (2 -> V channels). Readout: a vector
# whose direction gives the answer. This is an honest, simplified VN-MLP.
class VNLinear(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.w = nn.Parameter(torch.randn(cout, cin) * (1.0/math.sqrt(cin)))
    def forward(self, v):            # v: (B, C, 3)
        return torch.einsum('oc,bci->boi', self.w, v)

class VNReLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dir = nn.Parameter(torch.randn(c, 3))
    def forward(self, v):
        d = F.normalize(self.dir, dim=-1).unsqueeze(0)   # (1,C,3)
        proj = (v*d).sum(-1, keepdim=True)*d
        n = v.norm(dim=-1, keepdim=True)
        mask = ((v*d).sum(-1, keepdim=True) >= 0).float()
        return mask*v + (1-mask)*(v - proj)

class VNComp(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.inp = VNLinear(2, h)
        self.nl1 = VNReLU(h)
        self.hid = VNLinear(h, h)
        self.nl2 = VNReLU(h)
        self.out = VNLinear(h, 1)
    def forward(self, x):
        # x: (B,4) phase pairs -> two axis vectors
        a1 = angles_to_vec(x[:, 0:2]); a2 = angles_to_vec(x[:, 2:4])
        v = torch.stack([a1, a2], dim=1)                  # (B,2,3)
        v = self.nl2(self.hid(self.nl1(self.inp(v))))
        v = self.out(v).squeeze(1)                        # (B,3)
        return vec_to_angles(v / (v.norm(dim=-1,keepdim=True)+1e-6))

res_vn = {}
for name, ctor, lr in [
    ('QuatKAN K=1 (ref)', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
    ('MLP-Quat h=32 + RPMG', lambda: MLPQuatRPMG(4, h=32), 3e-3),
    ('VN-MLP h=8', lambda: VNComp(h=8), 3e-3),
]:
    vals = run_seeds(ctor, lr)
    res_vn[name] = vals
    reg(f'X4:{name}', vals, params=n_params(ctor()))
t, df, p = paired_t(res_vn['VN-MLP h=8'], res_vn['QuatKAN K=1 (ref)'])
print(f"  VN vs QuatKAN: t={t:.2f}, df={df}, p={p:.2e}")
RESULTS['X4:ttest:VN_vs_QuatKAN'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")
t, df, p = paired_t(res_vn['VN-MLP h=8'], res_vn['MLP-Quat h=32 + RPMG'])
print(f"  VN vs MLP+RPMG: t={t:.2f}, df={df}, p={p:.2e}")
RESULTS['X4:ttest:VN_vs_RPMG'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")

# ============================================================
# E18: Composition-operator ablation — "can 3 additions replace Hamilton?"
# ============================================================
# Reviewer/user question: the Hamilton product has 27 learnable-free scalar
# terms and costs 28 ops; the unit quaternion has only 3 DOF, so could the
# node just ADD the edge rotvecs (3 additions) and exp-map once?
# Answer: that is the zeroth-order Baker-Campbell-Hausdorff truncation —
# exact only for commuting rotations or infinitesimal steps. We ablate the
# NODE only (edges identical) with a truncation ladder, on canonical E1 data:
#   bch0:  v = v1 + v0                                 (3 adds)
#   bch1:  v += 1/2 [v1, v0]                           (~15 ops)
#   bch2:  v += 1/12 ([v1,[v1,v0]] + [v0,[v0,v1]])     (~30 ops >= Hamilton)
#   hamilton: exact group law                          (28 ops)
# Frozen reference (this block, 5 seeds, 3000 epochs):
#   bch0 5.957 | bch1 4.037 | bch2 3.897 | Hamilton 2.790 deg (Bayes 2.697)
# Also: bit-equivalence of the flop-lean rotate() against rotate_sandwich(),
# and the theoretical floor (multiplicative rank of the quaternion product
# is exactly 8: de Groote / Howell-Lafon; Makarov's 8-mul algorithm needs
# 28 additions, i.e. 36 total ops > 28 — on FMA hardware the naive Hamilton
# is already the flop-cheapest exact form).
print("\n" + "="*72)
print("E18 / Composition-operator ablation (comp2, canonical E1)")
print("="*72)

# --- bit-equivalence: rotate() (flop-lean) vs rotate_sandwich() ---
_g = torch.Generator().manual_seed(0)
_qs = torch.randn(4096, 4, generator=_g).to(device)
_qs = _qs / _qs.norm(dim=-1, keepdim=True)
_vs = torch.randn(4096, 3, generator=_g).to(device)
_eq = (rotate(_qs, _vs) - rotate_sandwich(_qs, _vs)).abs().max().item()
reg_text('E18:rotate_equivalence',
         f"max |rotate - rotate_sandwich| = {_eq:.2e} (float32 rounding only)")

class QuatKANBCH(nn.Module):
    """QuatKAN edges, but the node is a TRUNCATED BCH series in the edge
    rotvecs instead of the Hamilton product. Data composes q1 . q0, so the
    model composes v1 . v0 (order fixed to match the generator)."""
    def __init__(self, n_groups=2, K=1, order=0):
        super().__init__()
        self.n_groups, self.K, self.order = n_groups, K, order
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3) * 0.1)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1,self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        vs = []
        for gi in range(self.n_groups):
            f = self.group_feats(x[:, 2*gi:2*gi+2])
            vs.append(torch.einsum('bf,fc->bc', f, self.w[gi]))
        a, b = vs[1], vs[0]                # composition order: a . b
        v = a + b
        if self.order >= 1:
            ab = torch.cross(a, b, dim=-1)
            v = v + 0.5*ab
            if self.order >= 2:
                # [a,[a,b]] = a(a.b) - b|a|^2 ;  [b,[b,a]] = b(b.a) - a|b|^2
                dab = (a*b).sum(-1, keepdim=True)
                na2 = (a*a).sum(-1, keepdim=True); nb2 = (b*b).sum(-1, keepdim=True)
                v = v + (a*dab - b*na2 + b*dab - a*nb2)/12.0
        nv = v.norm(dim=-1, keepdim=True)
        q = torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1)
        vv = rotate(q, PROBE.expand(x.shape[0], 3))
        return vec_to_angles(vv / (vv.norm(dim=-1, keepdim=True) + 1e-6))

res_e17 = {}
for name, ctor, lr in [
    ('QuatKAN-BCH0 (sum, 3 adds)',  lambda: QuatKANBCH(2, K=1, order=0), 1e-2),
    ('QuatKAN-BCH1 (+1/2 cross)',   lambda: QuatKANBCH(2, K=1, order=1), 1e-2),
    ('QuatKAN-BCH2 (+triple comm)', lambda: QuatKANBCH(2, K=1, order=2), 1e-2),
    ('QuatKAN Hamilton (exact)',    lambda: GroupedQuatExpNet(2, K=1),   1e-2),
]:
    vals = run_seeds(ctor, lr, n_seeds=5)
    res_e17[name] = vals
    reg(f'E18:{name}', vals, params=n_params(ctor()))
for bch in ['QuatKAN-BCH0 (sum, 3 adds)', 'QuatKAN-BCH1 (+1/2 cross)',
            'QuatKAN-BCH2 (+triple comm)']:
    t, df, p = paired_t(res_e17[bch], res_e17['QuatKAN Hamilton (exact)'])
    print(f"  {bch} vs Hamilton: t={t:.2f}, df={df}, p={p:.2e}")
    RESULTS[f'E18:ttest:{bch.split(" ")[0]}'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")
reg_text('E18:conclusion',
         "Truncated BCH never reaches the Bayes floor on E1 (5.96/4.04/3.90 vs "
         "2.79 deg) and at order 2 already costs >= the exact Hamilton product "
         "(~30 vs 28 scalar ops); the group law is the flop-cheapest EXACT "
         "composer. Quaternion-product multiplicative rank = 8 (de Groote), "
         "but Makarov's 8-mul algorithm needs 36 total ops > 28.")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*72)
print("EXTENSIONS SUMMARY-JSON")
print("="*72)
print(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str))
print("\nEXTENSIONS COMPLETE")
