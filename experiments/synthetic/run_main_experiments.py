# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — MAIN BENCHMARK RUN, v1.3 (2026-09-21)
# (formerly run_all_experiments.py, v1.0–v1.1; renamed in v1.2)
#
# This script is the source of every number in §4–§5 and Appendices C.1–C.16
# of the paper: the main synthetic benchmark, blocks E1–E14
# (E18 — the composition-operator ablation — lives in run_extensions.py).
# The applied probes of §6 (IMU, FK calibration, rigid-body), an independent
# RPMG rerun, non-Gaussian noise and Vector Neurons live in the companion
# script run_extensions.py; the GL+(2,R) transfer probe is the self-contained
# third script E16_gl2_composition.py. All three share the protocol:
# 800 train / 4000 test, shared seeds, geodesic loss, paired t-tests.
#
# v1.3: rotate() now uses the exact identity v + 2w(u x v) + 2u x (u x v)
#   instead of two Hamilton products (30 vs 56 scalar ops per application;
#   ~15% of the model's scalar FLOPs at K=1). Mathematically identical for
#   unit quaternions, so bit-level output shifts only at 1e-9 float noise.
#   The sandwich form is kept as rotate_sandwich() and E18 (in
#   run_extensions.py) asserts they agree. The frozen E1 numbers below were
#   produced with the sandwich form; v1.3 reproduces them within seed noise.
#   E2's LawKAN FLOPs now come from flops_convention.py (= 228, the paper's
#   Sec.3/Table 2 breakdown); the earlier literals 234 and 207 were convention
#   drift (see flops_convention.py).
#
# v1.2: renamed run_all_experiments.py -> run_main_experiments.py.
#   No code changes; the old name became inaccurate once the paper grew
#   beyond this file (§6 probes and the X-blocks live in the companions).
#
# v1.1: added the RPMG opponent (Chen et al. 2022) as ONE row in the headline
#   matrix: 'E1:MLP-Quat h=32 + RPMG' (+ its paired t-test vs LawKAN).
#   It is not added elsewhere on purpose: the CPU verification run shows RPMG
#   is statistically indistinguishable from plain MLP-Quat (p = 0.21), so the
#   plain head remains the reference opponent throughout the paper.
#   Pure addition: every pre-existing row is bit-reproducible from v1.0
#   (data generators use local torch.Generator; model init re-seeded per run).
#   Reference values (GPU run, B1-fixed RPMG VJP):
#     E1:MLP-Quat h=32 + RPMG ≈ 3.424 ± 0.147 deg  (plain: 3.357 ± 0.065;
#     LawKAN: 2.784 ± 0.015; RPMG vs LawKAN p ≈ 8.9e-04; RPMG vs plain p ≈ 0.21)
#
# Requirements:  torch, numpy, scipy (scipy only for p-values)
# Run:           python run_main_experiments.py
# Time:          ~35–45 min on an RTX 2080 SUPER; ~2–3 h on CPU.
# Output:        per-block lines + final SUMMARY-JSON dump —
#                paper tables are generated from it mechanically.
#
# Sanity check (frozen numbers, must reproduce within seed noise):
#   E1: LawKAN K=1   2.790 ± 0.018 deg
#   E1: MLP-Quat h=32 3.36 ± 0.06 deg    paired t=18.63, p=4.9e-05
#   Bayes floor (matched E1 draws, E40): 2.733 deg
#
# Documented protocol exceptions (Appendix A.4 of the paper):
#   1) E5 uses torch.manual_seed(50+s) instead of 100+s;
#   2) E6 uses torch.manual_seed(s), data generator seed fixed at 0;
#   3) E5 models train at lr=3e-2 with K=7 (Appendix A.3).
# ============================================================
import torch, torch.nn as nn, math, time, json
import torch.nn.functional as F
import numpy as np
from flops_convention import quatkan_flops

# --- pre-flight: scipy check (t-stats work without it; p-values need it) ---
try:
    import scipy  # noqa: F401
except ImportError:
    print("!" * 72)
    print("WARNING: scipy not found -> p-values will print as nan.")
    print("The p-values are paper numbers. Install with:  pip install scipy")
    print("!" * 72)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS = {}   # single registry: key -> dict(mean, std, per_seed, meta)
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.3f} ±{np.std(vals):.3f}  per-seed: {[round(v,2) for v in vals]}")

def reg_text(key, text):
    RESULTS[key] = dict(text=text)
    print(f"[{key}] {text}")

# ============================================================
# BLOCK 1: Geometry
# ============================================================
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

def axis_angle_to_quat(axis, angle):
    half = angle.unsqueeze(-1)/2
    return torch.cat([torch.cos(half), torch.sin(half)*axis], dim=-1)

def rotate(q, v):
    # v + 2*w*(u x v) + 2*u x (u x v): exact for unit q, cheaper than the
    # q*v*q* sandwich (30 vs 56 scalar ops) — E18's flop-lean variant
    u, w = q[...,1:], q[...,:1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)

def rotate_sandwich(q, v):
    # reference implementation (two Hamilton products); bit-equivalent check
    # against rotate() is asserted in E18
    qv = torch.cat([torch.zeros_like(q[...,:1]), v], dim=-1)
    qi = q.clone(); qi[...,1:] = -qi[...,1:]
    return quat_mul(quat_mul(q, qv), qi)[...,1:]

def vec_to_angles(v):
    return torch.stack([torch.atan2(v[...,1], v[...,0]),
                        torch.acos(torch.clamp(v[...,2], -1+1e-6, 1-1e-6))], dim=-1)

def angles_to_vec(a):
    al, be = a[...,0], a[...,1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)

def geo_loss(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return (1-cos).mean()

def geo_deg(pred, target):
    cos = (angles_to_vec(pred)*angles_to_vec(target)).sum(-1).clamp(-1+1e-6, 1-1e-6)
    return torch.rad2deg(torch.acos(cos))

PROBE = torch.tensor([0., 0., 1.], device=device)

def n_params(ctor):
    return sum(p.numel() for p in ctor().parameters())

# ============================================================
# BLOCK 2: Data — SINGLE generator
# ============================================================
def make_compn_data(N, n=2, noise=0.03, seed=1, angles=None):
    """Composition of n rotations q_n·...·q_1, random axes (uniform on S²).
    angles: tuple of length n (explicit) | None -> 0.7+0.2i | 'random' -> U[0.2,2.5]."""
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N,3,generator=g); a = a/a.norm(dim=-1,keepdim=True)
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
    v = rotate(prod, torch.tensor([0.,0.,1.]).expand(N,3))
    v = v / v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N,2*n,generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N,2,generator=g)
    return X.to(device), Y.to(device)

# canonical configurations (explicit angles — no defaults)
ANG2 = (0.7, 1.1)
ANG3 = (0.7, 1.1, 0.9)
comp2 = lambda N, **kw: make_compn_data(N, n=2, angles=ANG2, **kw)
comp3 = lambda N, **kw: make_compn_data(N, n=3, angles=ANG3, **kw)

def sample_axes_capped(N, beta_min, beta_max, g):
    """Area-uniform sampling within a polar band (corrected sampler, §5.4)."""
    cos_lo, cos_hi = math.cos(beta_max), math.cos(beta_min)
    cosb = torch.rand(N, generator=g) * (cos_hi - cos_lo) + cos_lo
    beta = torch.acos(cosb.clamp(-1, 1))
    alpha = torch.rand(N, generator=g) * 2 * math.pi - math.pi
    return torch.stack([torch.sin(beta)*torch.cos(alpha),
                        torch.sin(beta)*torch.sin(alpha), torch.cos(beta)], dim=-1)

def make_comp_banded(N, band1, band2, noise=0.03, seed=1, angles=ANG2):
    g = torch.Generator().manual_seed(seed)
    a1 = sample_axes_capped(N, *band1, g)
    a2 = sample_axes_capped(N, *band2, g)
    q1 = axis_angle_to_quat(a1, torch.full((N,), float(angles[0])))
    q2 = axis_angle_to_quat(a2, torch.full((N,), float(angles[1])))
    v = rotate(quat_mul(q2, q1), torch.tensor([0.,0.,1.]).expand(N,3))
    v = v / v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a1), vec_to_angles(a2)], -1) + noise*torch.randn(N,4,generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N,2,generator=g)
    return X.to(device), Y.to(device)

def make_null_data(N, noise=0.03, seed=1):
    """E7a null-task: same inputs, target WITHOUT the composition group law."""
    g = torch.Generator().manual_seed(seed)
    a1 = torch.randn(N,3,generator=g); a1 = a1/a1.norm(dim=-1,keepdim=True)
    a2 = torch.randn(N,3,generator=g); a2 = a2/a2.norm(dim=-1,keepdim=True)
    p1, p2 = vec_to_angles(a1), vec_to_angles(a2)
    al1, be1, al2, be2 = p1[:,0], p1[:,1], p2[:,0], p2[:,1]
    psi = al1 + al2
    phi = (be1 + be2).clamp(1e-4, math.pi-1e-4)
    Y = torch.stack([psi + 0.5*torch.sin(phi), phi], dim=-1)
    X = torch.stack([al1, be1, al2, be2], dim=-1) + noise*torch.randn(N,4,generator=g)
    Y = Y + noise*torch.randn(N,2,generator=g)
    return X.to(device), Y.to(device)

def make_zn(n, N=4096, noise=0.30, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n, (N,), generator=g); b = torch.randint(0, n, (N,), generator=g)
    ta = 2*math.pi*a/n + noise*torch.randn(N, generator=g)
    tb = 2*math.pi*b/n + noise*torch.randn(N, generator=g)
    tstar = 2*math.pi*((a+b) % n)/n
    return torch.stack([ta, tb], 1).to(device), tstar.to(device), ((a+b) % n).to(device)

def bayes_zn(n, noise=0.30, N=200000):
    g = torch.Generator().manual_seed(7)
    a = torch.randint(0, n, (N,), generator=g); b = torch.randint(0, n, (N,), generator=g)
    s = 2*math.pi*(a+b)/n + noise*math.sqrt(2)*torch.randn(N, generator=g)
    roots = 2*math.pi*torch.arange(n)/n
    pred = (1 - torch.cos(s.unsqueeze(-1) - roots)).argmin(-1)
    return (pred == (a+b) % n).float().mean().item()

# ============================================================
# BLOCK 3: Models
# ============================================================
class GroupedQuatExpNet(nn.Module):
    """QuatKAN: edge = Fourier function of a phase pair -> so(3), q = exp(v);
    node = Hamilton product in the data's composition order (last group first)."""
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

class GroupedQuatNormNet(nn.Module):
    """Variant with normalization instead of exp-map (E6, dead zones)."""
    def __init__(self, n_groups=2, K=1):
        super().__init__()
        self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 4) * 0.1)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa = [torch.ones_like(al)]; fb = [torch.ones_like(be)]
        for m in range(1, self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]
            fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        qs, norms = [], []
        for gi in range(self.n_groups):
            f = self.group_feats(x[:, 2*gi:2*gi+2])
            q = torch.einsum('bf,fc->bc', f, self.w[gi])
            norms.append(q.norm(dim=-1, keepdim=True))
            qs.append(q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        self.last_norm = norms[0].mean().item()
        prod = qs[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = quat_mul(prod, qs[gi])
        v = rotate(prod, PROBE.expand(x.shape[0], 3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True) + 1e-6))

class MLPComp(nn.Module):
    def __init__(self, in_phases=4, h=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        out = self.net(f)
        al = torch.atan2(out[:,1], out[:,0])
        be = torch.atan2(out[:,3].abs(), out[:,2])
        return torch.stack([al, be], -1)

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

class MLP9DSVD(nn.Module):
    def __init__(self, in_phases=4, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,9))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        M = self.net(f).view(-1,3,3)
        U, S, Vh = torch.linalg.svd(M)
        det = torch.det(U @ Vh)
        D = torch.diag_embed(torch.stack([torch.ones_like(det),
                                          torch.ones_like(det), det], -1))
        return mat_to_angles(U @ D @ Vh)

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

# --- RPMG opponent (Chen et al. 2022): manifold-aware projective gradient ---
# For a unit-quaternion readout q/||q|| the Euclidean gradient has a radial
# component that fights the normalization; RPMG replaces it by its projection
# onto the tangent space of S^3 at the normalized point:
#     g_manifold = g - <g, q_hat> q_hat,   q_hat = q/||q||
# Forward is identical to MLPQuatHead; only the backward pass changes.
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
        g_tangent = g_out - radial * q_hat
        return g_tangent / r

def rpmg_normalize(q_raw):
    return _RPMGQuat.apply(q_raw)

class MLPQuatRPMG(nn.Module):
    """MLP-Quat head, forward identical to MLPQuatHead; gradient is manifold-aware."""
    def __init__(self, in_phases=4, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,4))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        q = rpmg_normalize(self.net(f))
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True)+1e-6))

class MLPLieHead(nn.Module):
    """E11: MLP -> v in so(3) -> exp at the output (no group-law nodes)."""
    def __init__(self, in_phases=4, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases*2, h), nn.SiLU(),
                                 nn.Linear(h,h), nn.SiLU(), nn.Linear(h,3))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        v3 = self.net(f)
        nv = v3.norm(dim=-1, keepdim=True)
        q = torch.cat([torch.cos(nv), torch.sin(nv)*v3/(nv+1e-8)], dim=-1)
        v = rotate(q, PROBE.expand(x.shape[0],3))
        return vec_to_angles(v / (v.norm(dim=-1, keepdim=True)+1e-6))

class RealKANComp(nn.Module):
    def __init__(self, in_dim=4, out_dim=2, K=3):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(out_dim, in_dim, 2*K+1) * 0.1)
    def forward(self, x):
        k = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype)
        xp = x.unsqueeze(-1) * k
        feats = torch.cat([torch.cos(xp), torch.sin(xp), torch.ones_like(x.unsqueeze(-1))], dim=-1)
        return torch.einsum('bik,oik->bo', feats, self.w)

class RealKAN2Layer(nn.Module):
    def __init__(self, in_dim=4, hidden=4, out_dim=2, K=3):
        super().__init__()
        self.l1 = RealKANComp(in_dim, hidden, K)
        self.l2 = RealKANComp(hidden, out_dim, K)
    def forward(self, x):
        return self.l2(self.l1(x))

class RealKAN6D(nn.Module):
    def __init__(self, in_dim=4, K=3, h=8):
        super().__init__()
        self.K = K
        self.w1 = nn.Parameter(torch.randn(h, in_dim, 2*K+1) * 0.1)
        self.lin = nn.Linear(h, 6)
    def forward(self, x):
        k = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype)
        xp = x.unsqueeze(-1) * k
        feats = torch.cat([torch.cos(xp), torch.sin(xp), torch.ones_like(x.unsqueeze(-1))], dim=-1)
        hid = F.silu(torch.einsum('bik,oik->bo', feats, self.w1))
        return mat_to_angles(rot6d_to_matrix(self.lin(hid)))

class CircularKANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K=4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.randn(out_dim, in_dim, 2*K+1) * 0.3 / math.sqrt(in_dim*(2*K+1)))
    def forward(self, x):
        k = torch.arange(1, self.K+1, device=x.device, dtype=x.dtype)
        xp = x.unsqueeze(-1) * k
        feats = torch.cat([torch.cos(xp), torch.sin(xp), torch.ones_like(x.unsqueeze(-1))], dim=-1)
        return torch.einsum('bik,oik->bo', feats, self.w)

class CircularKAN(nn.Module):
    def __init__(self, widths, K=4):
        super().__init__()
        self.layers = nn.ModuleList([CircularKANLayer(a, b, K) for a, b in zip(widths[:-1], widths[1:])])
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

class GroupSumKAN(nn.Module):
    def __init__(self, K=3):
        super().__init__(); self.edge = CircularKANLayer(1, 1, K)
    def forward(self, x): return self.edge(x.sum(-1, keepdim=True))

def circ_loss(pred, target):
    return (1 - torch.cos(pred.squeeze(-1) - target)).mean()

def zn_readout(pred, n):
    roots = 2*math.pi*torch.arange(n, device=pred.device)/n
    return (1 - torch.cos(pred.squeeze(-1).unsqueeze(-1) - roots)).argmin(-1)

# ============================================================
# BLOCK 4: Protocol
# ============================================================
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

def run_seeds_raw(ctor, data_fn, lr, n_seeds, epochs=3000, **data_kw):
    vals = []
    for seed in range(n_seeds):
        Xtr, Ytr = data_fn(800, seed=seed, **data_kw)
        Xte, Yte = data_fn(4000, seed=seed+100, **data_kw)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

def paired_t(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float('nan')
    return float(t), n-1, float(p)

def main():
    # ============================================================
    # E1: Opponent matrix (§4, Table 1) — comp2 configuration
    # ============================================================
    print("\n" + "="*72)
    print("E1 / §4: Opponent matrix (n=2, angles 0.7/1.1, sigma=0.03, 800/4000)")
    print("="*72)
    matrix = [
        ('QuatKAN K=1',      lambda: GroupedQuatExpNet(2, K=1), 1e-2, 5),
        ('MLP-Quat h=32',    lambda: MLPQuatHead(4, h=32),      3e-3, 5),
        ('MLP-Quat h=32 + RPMG', lambda: MLPQuatRPMG(4, h=32),  3e-3, 5),
        ('MLP-Quat h=16',    lambda: MLPQuatHead(4, h=16),      3e-3, 3),
        ('MLP-6D h=32',      lambda: MLP6D(4, h=32),            3e-3, 3),
        ('MLP-6D h=16',      lambda: MLP6D(4, h=16),            3e-3, 5),
        ('MLP-9D+SVD h=16',  lambda: MLP9DSVD(4, h=16),         3e-3, 3),
        ('RealKAN-6D h=16',  lambda: RealKAN6D(4, K=3, h=16),   3e-3, 3),
        ('MLP-Quat h=8',     lambda: MLPQuatHead(4, h=8),       3e-3, 3),
        ('MLP-9D+SVD h=8',   lambda: MLP9DSVD(4, h=8),          3e-3, 3),
        ('MLP-6D h=8',       lambda: MLP6D(4, h=8),             3e-3, 3),
        ('RealKAN-6D h=8',   lambda: RealKAN6D(4, K=3, h=8),    3e-3, 3),
        ('MLP-Quat h=4',     lambda: MLPQuatHead(4, h=4),       3e-3, 3),
        ('MLP-9D+SVD h=4',   lambda: MLP9DSVD(4, h=4),          3e-3, 3),
        ('MLP-6D h=4',       lambda: MLP6D(4, h=4),             3e-3, 3),
        ('MLP h=4 (atan2)',  lambda: MLPComp(4, h=4),           3e-3, 5),
        ('RealKAN 2L K=3',   lambda: RealKAN2Layer(4, 4, 2, K=3), 3e-3, 3),
        ('RealKAN K=3',      lambda: RealKANComp(4, 2, K=3),    3e-3, 3),
    ]
    for name, ctor, lr, ns in matrix:
        vals = run_seeds_raw(ctor, comp2, lr, ns)
        reg(f'E1:{name}', vals, params=n_params(ctor), seeds=ns)

    # paired t-test for key pairs (shared seeds, M4)
    qk = RESULTS['E1:QuatKAN K=1']['per_seed']
    for opp in ['MLP-Quat h=32', 'MLP-Quat h=32 + RPMG', 'MLP-6D h=16']:
        t, df, p = paired_t(RESULTS[f'E1:{opp}']['per_seed'], qk)
        reg_text(f'E1:ttest:{opp}', f"t={t:.2f}, df={df}, p={p:.2e}")

    # ============================================================
    # E2: Wall-clock (§4.1)
    # ============================================================
    print("\n" + "="*72)
    print("E2 / §4.1: Wall-clock (batch 1024, 300 iters, 30 warmup)")
    print("="*72)
    def bench_net(net, x, y, lr=3e-3, iters=300, warmup=30):
        net = net.to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(warmup):
            opt.zero_grad(); geo_loss(net(x), y).backward(); opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            opt.zero_grad(); geo_loss(net(x), y).backward(); opt.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        tr = (time.time()-t0)/iters*1000
        with torch.no_grad():
            for _ in range(warmup): net(x)
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters): net(x)
            if device.type == 'cuda': torch.cuda.synchronize()
        inf = (time.time()-t0)/iters*1000
        return tr, inf

    torch.manual_seed(0)
    Xb, Yb = comp2(1024, seed=0)
    bench_rows = [
        ('QuatKAN K=1',   lambda: GroupedQuatExpNet(2, K=1), 1e-2, quatkan_flops(n_groups=2, K=1)),
        ('MLP h=4',       lambda: MLPComp(4, h=4),           3e-3, 128),
        ('MLP-6D h=16',   lambda: MLP6D(4, h=16),            3e-3, 960),
        ('MLP-6D h=32',   lambda: MLP6D(4, h=32),            3e-3, 2944),
        ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32),      3e-3, 2848),
    ]
    for name, ctor, lr, fl in bench_rows:
        net = ctor()
        tr, inf = bench_net(net, Xb, Yb, lr)
        reg_text(f'E2:{name}', f"params={n_params(ctor)}, FLOPs={fl}, train={tr:.2f} ms/it, infer={inf:.2f} ms/batch")

    # ============================================================
    # E3: Full battery against MLP-Quat h=32 (§4.2)
    # ============================================================
    print("\n" + "="*72)
    print("E3 / §4.2: Battery vs MLP-Quat h=32")
    print("="*72)

    # E3.1: n=3
    for name, ctor, lr in [('QuatKAN n=3', lambda: GroupedQuatExpNet(3, K=1), 1e-2),
                           ('MLP-Quat h=32', lambda: MLPQuatHead(6, h=32), 3e-3)]:
        vals = run_seeds_raw(ctor, comp3, lr, 3)
        reg(f'E3.1:{name}', vals)
    t, df, p = paired_t(RESULTS['E3.1:MLP-Quat h=32']['per_seed'], RESULTS['E3.1:QuatKAN n=3']['per_seed'])
    reg_text('E3.1:ttest', f"t={t:.2f}, df={df}, p={p:.2e}")

    # E3.2: noise curve
    for noise in [0.01, 0.05, 0.10, 0.20]:
        vq = run_seeds_raw(lambda: GroupedQuatExpNet(2, K=1), comp2, 1e-2, 5, epochs=2500, noise=noise)
        vm = run_seeds_raw(lambda: MLPQuatHead(4, h=32), comp2, 3e-3, 5, epochs=2500, noise=noise)
        reg(f'E3.2:{noise}:QuatKAN', vq)
        reg(f'E3.2:{noise}:MLP-Quat h=32', vm)

    # E3.3: sample efficiency vs top opponent
    Xte_se, Yte_se = comp2(4000, seed=100)
    for N in [100, 400, 1600, 3200]:
        for name, ctor, lr in [('QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                               ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3)]:
            vals = []
            for s in range(5):
                Xtr_s, Ytr_s = comp2(N, seed=s)
                torch.manual_seed(100+s)
                vals.append(train_eval(ctor(), Xtr_s, Ytr_s, Xte_se, Yte_se, lr, 2500))
            reg(f'E3.3:N{N}:{name}', vals)

    # E3.4: extrapolation (corrected sampler)
    TROP, POLE = (0.5, 2.6), (0.0, 0.3)
    Xtr_b, Ytr_b = make_comp_banded(1000, TROP, TROP, seed=0)
    tests_b = {
        'inside': make_comp_banded(2000, TROP, TROP, seed=100),
        'axis1_pole': make_comp_banded(2000, POLE, TROP, seed=200),
        'axis2_pole': make_comp_banded(2000, TROP, POLE, seed=300),
        'both_poles': make_comp_banded(2000, POLE, POLE, seed=400),
    }
    for name, ctor, lr in [('QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                           ('MLP h=4', lambda: MLPComp(4, h=4), 3e-3),
                           ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3)]:
        per = {k: [] for k in tests_b}
        for s in range(2):
            torch.manual_seed(100+s)
            net = ctor().to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            for ep in range(2500):
                opt.zero_grad()
                geo_loss(net(Xtr_b), Ytr_b).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                for k, (Xt, Yt) in tests_b.items():
                    per[k].append(geo_deg(net(Xt), Yt).mean().item())
        for k, v in per.items():
            reg(f'E3.4:{name}:{k}', v)

    # ============================================================
    # E4: Sample efficiency, three models (§5.3)
    # ============================================================
    print("\n" + "="*72)
    print("E4 / §5.3: Sample efficiency (N = 50..1600)")
    print("="*72)
    for N in [50, 100, 200, 400, 800, 1600]:
        for name, ctor, lr in [('QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                               ('MLP h=4', lambda: MLPComp(4, h=4), 3e-3),
                               ('MLP-6D h=16', lambda: MLP6D(4, h=16), 3e-3)]:
            vals = []
            for s in range(5):
                Xtr_s, Ytr_s = comp2(N, seed=s)
                torch.manual_seed(100+s)
                vals.append(train_eval(ctor(), Xtr_s, Ytr_s, Xte_se, Yte_se, lr, 2500))
            reg(f'E4:N{N}:{name}', vals)

    # ============================================================
    # E5: Z_n / BKT check (§5.5)
    # ============================================================
    print("\n" + "="*72)
    print("E5 / §5.5: Addition in Z_n, n = 3..8")
    print("="*72)
    for n in range(3, 9):
        Xtr, Ttr, Ctr = make_zn(n, N=400, noise=0.30, seed=2)
        Xte, Tte, Cte = make_zn(n, N=4000, noise=0.30, seed=99)
        for mname, mctor in [('GroupSum', lambda: GroupSumKAN(K=7)),
                             ('Additive', lambda: CircularKAN([2,1], K=7))]:
            accs, convs = [], []
            for s in range(2):
                torch.manual_seed(50+s)
                m = mctor().to(device)
                opt = torch.optim.Adam(m.parameters(), lr=3e-2)
                traj = []
                for ep in range(1500):
                    opt.zero_grad(); circ_loss(m(Xtr), Ttr).backward(); opt.step()
                    if ep % 50 == 0 or ep == 1499:
                        with torch.no_grad():
                            traj.append((zn_readout(m(Xte), n) == Cte).float().mean().item())
                final = traj[-1]
                t90 = 1/n + 0.9*(final - 1/n)
                convs.append(next((i*50 for i, a in enumerate(traj) if a >= t90), 1500))
                accs.append(final)
            reg(f'E5:n{n}:{mname}', accs, conv_epochs=convs, bayes=bayes_zn(n))

    # ============================================================
    # E6: Dead zones (§5.6) — 10 runs + trajectory min-norm trace (M5)
    # ============================================================
    print("\n" + "="*72)
    print("E6 / §5.6: Dead zones — q/||q|| vs exp-map (10 runs, with min-norm trace)")
    print("="*72)
    Xtr_dz, Ytr_dz = comp2(800, seed=0)
    Xte_dz, Yte_dz = comp2(4000, seed=100)
    for name, ctor, lr in [('normalization', lambda: GroupedQuatNormNet(2, K=1), 1e-2),
                           ('exp-map', lambda: GroupedQuatExpNet(2, K=1), 1e-2)]:
        fails, errs, norms_out, min_norms = 0, [], [], []
        for s in range(10):
            torch.manual_seed(s)
            net = ctor().to(device)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            min_norm = float('inf')
            for ep in range(3000):
                opt.zero_grad()
                geo_loss(net(Xtr_dz), Ytr_dz).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if hasattr(net, 'last_norm'):
                    min_norm = min(min_norm, net.last_norm)
            with torch.no_grad():
                d = geo_deg(net(Xte_dz), Yte_dz).mean().item()
            errs.append(d)
            if hasattr(net, 'last_norm'):
                norms_out.append(net.last_norm)
                min_norms.append(min_norm)
            if d > 10.0: fails += 1
        reg(f'E6:{name}', errs, fails=fails, final_norms=[round(x,3) for x in norms_out],
            min_norms=[round(x,4) for x in min_norms])

    # ============================================================
    # E7: Scope boundary (§5.7)
    # ============================================================
    print("\n" + "="*72)
    print("E7 / §5.7: null-task and random angles")
    print("="*72)
    for name, ctor, lr in [('QuatKAN K=1', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                           ('MLP h=4 (atan2)', lambda: MLPComp(4, h=4), 3e-3),
                           ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3)]:
        vals = run_seeds_raw(ctor, make_null_data, lr, 3)
        reg(f'E7a:{name}', vals, params=n_params(ctor))
    data_rand = lambda N, **kw: make_compn_data(N, n=2, angles='random', **kw)
    for name, ctor, lr in [('QuatKAN K=1', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                           ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3),
                           ('MLP h=4 (atan2)', lambda: MLPComp(4, h=4), 3e-3)]:
        vals = run_seeds_raw(ctor, data_rand, lr, 3)
        reg(f'E7b:{name}', vals, params=n_params(ctor))

    # ============================================================
    # E8: Scaling along the composition chain (§5.8)
    # ============================================================
    print("\n" + "="*72)
    print("E8 / §5.8: scaling along n (angles 0.7+0.2i)")
    print("="*72)
    for n in [2, 3, 4, 6, 8]:
        vq = run_seeds_raw(lambda: GroupedQuatExpNet(n, K=1), make_compn_data, 1e-2, 3, n=n)
        vm = run_seeds_raw(lambda: MLPQuatHead(2*n, h=32), make_compn_data, 3e-3, 3, n=n)
        v16 = run_seeds_raw(lambda: MLPComp(2*n, h=16), make_compn_data, 3e-3, 3, n=n)
        reg(f'E8:n{n}:QuatKAN', vq)
        reg(f'E8:n{n}:MLP-Quat h=32', vm)
        reg(f'E8:n{n}:MLP h=16', v16)

    # ============================================================
    # E9: Opponent LR sweep (§4, P18)
    # ============================================================
    print("\n" + "="*72)
    print("E9 / §4: Opponent LR sweep")
    print("="*72)
    for name, ctor in [('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32)),
                       ('MLP-6D h=16', lambda: MLP6D(4, h=16)),
                       ('MLP h=4', lambda: MLPComp(4, h=4))]:
        for lr in [1e-3, 3e-3, 1e-2]:
            vals = run_seeds_raw(ctor, comp2, lr, 3)
            reg(f'E9:{name}:lr{lr}', vals)

    # ============================================================
    # E10: Bayes floor (§4, M1)
    # ============================================================
    print("\n" + "="*72)
    print("E10 / §4: Bayes floor (Monte-Carlo posterior mean)")
    print("="*72)
    def bayes_floor(n, angles, N=2000, noise=0.03, M=256, seed=100):
        X, Y = make_compn_data(N, n=n, noise=noise, seed=seed, angles=angles)
        g = torch.Generator().manual_seed(seed + 777)
        draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
        al = draws[..., 0::2]; be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
        w = torch.sin(be).prod(-1)
        w = w / w.sum(1, keepdim=True)
        axes = torch.stack([torch.sin(be)*torch.cos(al),
                            torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)
        prod = axis_angle_to_quat(axes[:, :, -1], torch.full((N, M), float(angles[-1]), device=device))
        for i in range(n-2, -1, -1):
            prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i],
                            torch.full((N, M), float(angles[i]), device=device)))
        v = rotate(prod, PROBE.expand(N, M, 3))
        v = v / v.norm(dim=-1, keepdim=True)
        mean_dir = (v * w.unsqueeze(-1)).sum(1)
        mean_dir = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + 1e-8)
        pred = vec_to_angles(mean_dir)
        return geo_deg(pred, Y).mean().item()

    reg_text('E10:n2', f"bayes_floor={bayes_floor(2, ANG2):.3f} deg")
    reg_text('E10:n3', f"bayes_floor={bayes_floor(3, ANG3):.3f} deg")

    # E10b: Bayes floor for the random-angle (hidden, non-input theta) regime P16b
    def bayes_floor_random(n=2, N=2000, noise=0.03, M=2048, seed=100, lo=0.2, hi=2.5):
        X, Y = make_compn_data(N, n=n, noise=noise, seed=seed, angles='random')
        g = torch.Generator().manual_seed(seed + 777)
        draws = X.unsqueeze(1) + noise*torch.randn(N, M, 2*n, generator=g).to(device)
        al = draws[..., 0::2]; be = draws[..., 1::2].clamp(1e-4, math.pi-1e-4)
        w = torch.sin(be).prod(-1); w = w / w.sum(1, keepdim=True)
        axes = torch.stack([torch.sin(be)*torch.cos(al),
                            torch.sin(be)*torch.sin(al), torch.cos(be)], dim=-1)
        th = (lo + (hi-lo)*torch.rand(N, M, n, generator=g)).to(device)
        prod = axis_angle_to_quat(axes[:, :, -1], th[:, :, -1])
        for i in range(n-2, -1, -1):
            prod = quat_mul(prod, axis_angle_to_quat(axes[:, :, i], th[:, :, i]))
        v = rotate(prod, PROBE.expand(N, M, 3))
        v = v / v.norm(dim=-1, keepdim=True)
        mean_dir = (v * w.unsqueeze(-1)).sum(1)
        mean_dir = mean_dir / (mean_dir.norm(dim=-1, keepdim=True) + 1e-8)
        return geo_deg(vec_to_angles(mean_dir), Y).mean().item()

    reg_text('E10b:p16b-random',
             f"bayes_floor={bayes_floor_random(2):.3f} deg (hidden angles theta~U[0.2,2.5])")

    # ============================================================
    # E11: Lie baseline (§4, M7a, P19)
    # ============================================================
    print("\n" + "="*72)
    print("E11 / §4: Lie baseline")
    print("="*72)
    for name, ctor, lr in [('MLP-Lie h=32', lambda: MLPLieHead(4, h=32), 3e-3),
                           ('QuatKAN K=1 (ref)', lambda: GroupedQuatExpNet(2, K=1), 1e-2)]:
        vals = run_seeds_raw(ctor, comp2, lr, 10)
        reg(f'E11:{name}', vals, params=n_params(ctor))
    t, df, p = paired_t(RESULTS['E11:MLP-Lie h=32']['per_seed'], RESULTS['E11:QuatKAN K=1 (ref)']['per_seed'])
    reg_text('E11:ttest', f"t={t:.2f}, df={df}, p={p:.2e}")

    # ============================================================
    # E12: per-seed + paired t-test for key cells (M4)
    # ============================================================
    print("\n" + "="*72)
    print("E12 / M4: per-seed + paired t-test (5 seeds)")
    print("="*72)
    cells = [
        ('E1n2', comp2, [('QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                          ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3),
                          ('MLP-6D h=16', lambda: MLP6D(4, h=16), 3e-3)]),
        ('n3', comp3, [('QuatKAN', lambda: GroupedQuatExpNet(3, K=1), 1e-2),
                        ('MLP-Quat h=32', lambda: MLPQuatHead(6, h=32), 3e-3)]),
    ]
    for cell_name, data_fn, models in cells:
        res = {}
        for name, ctor, lr in models:
            vals = run_seeds_raw(ctor, data_fn, lr, 5)
            res[name] = vals
            reg(f'E12:{cell_name}:{name}', vals)
        for name in res:
            if name == 'QuatKAN': continue
            t, df, p = paired_t(res[name], res['QuatKAN'])
            reg_text(f'E12:{cell_name}:ttest:{name}', f"t={t:.2f}, df={df}, p={p:.2e}")
    res400 = {}
    for name, ctor, lr in [('QuatKAN', lambda: GroupedQuatExpNet(2, K=1), 1e-2),
                           ('MLP-Quat h=32', lambda: MLPQuatHead(4, h=32), 3e-3)]:
        vals = []
        for s in range(5):
            Xtr_s, Ytr_s = comp2(400, seed=s)
            torch.manual_seed(100+s)
            vals.append(train_eval(ctor(), Xtr_s, Ytr_s, Xte_se, Yte_se, lr, 2500))
        res400[name] = vals
        reg(f'E12:N400:{name}', vals)
    t, df, p = paired_t(res400['MLP-Quat h=32'], res400['QuatKAN'])
    reg_text('E12:N400:ttest', f"t={t:.2f}, df={df}, p={p:.2e}")

    # ============================================================
    # E13: Fourier-order ablation (§4, Q3, P20)
    # ============================================================
    print("\n" + "="*72)
    print("E13 / §4: K ablation")
    print("="*72)
    for K in [1, 2]:
        ctor = lambda: GroupedQuatExpNet(2, K=K)
        vals = run_seeds_raw(ctor, comp2, 1e-2, 5)
        reg(f'E13:K{K}', vals, params=n_params(ctor))
    t, df, p = paired_t(RESULTS['E13:K2']['per_seed'], RESULTS['E13:K1']['per_seed'])
    reg_text('E13:ttest', f"t={t:.2f}, df={df}, p={p:.2e}")

    # ============================================================
    # E14: n=6 per-seed, 10 seeds (§5.8, Q4, P22)
    # ============================================================
    print("\n" + "="*72)
    print("E14 / §5.8: n=6 per-seed (10 seeds)")
    print("="*72)
    vals = run_seeds_raw(lambda: GroupedQuatExpNet(6, K=1), make_compn_data, 1e-2, 10, n=6)
    reg('E14:n6:QuatKAN', vals, converged=sum(1 for v in vals if v < 20))

    # ============================================================
    # SUMMARY: machine-readable dump (source for the paper tables)
    # ============================================================
    print("\n" + "="*72)
    print("SUMMARY-JSON (machine-readable source for the paper tables)")
    print("="*72)
    print(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str))
    print("\nRUN COMPLETE")

if __name__ == '__main__':
    main()
