# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E22 — SE(3) COMPOSITION: the full LawKAN design on a non-compact group
#
# GL+(2,R) (E16) is non-compact, but exp: gl(2) -> GL+(2) is neither injective
# nor surjective, so only a node-only variant could be exact there. SE(3) is the
# complementary case: non-compact, yet exp: se(3) -> SE(3) IS surjective (screw
# theorem). So this is the first group on which the *full* LawKAN design --
# edge -> Lie algebra -> exp -> group product -- can be exact.
#
# Task (comp2 analogue). Each of n=2 rigid transforms is
#     g_i = (R(a_i, theta_i), p_i)        (fixed angles theta_i, random axis a_i,
#                                          random translation p_i ~ N(0, 0.5^2))
# and the target is the composition g = g_2 g_1 applied to four fixed probe
# points. Input per group: (alpha_i, beta_i, p_i) where (alpha,beta) are the axis
# phases. Readout: transformed probe points; metric: mean Euclidean error.
#
# Edge target. The exact edge must emit xi_i = log(g_i) with
#     omega_i = theta_i a_i
#     v_i     = V^{-1}(omega_i) p_i = p_i - (1/2) omega x p_i + d(theta) omega x (omega x p_i)
# where V is the SE(3) left Jacobian and d(theta) depends only on the fixed angle.
# Writing omega = theta a:
#     v_i = (1 - d theta^2) p_i  -  (theta/2)(a x p_i)  +  d theta^2 a (a . p_i)
# So the exact "analytic support" features are the axis components, the
# translation components, the cross products a x p, and a (a . p). A plain linear
# edge over [1, a, p] cannot express the products -> dense plateaus; a structured
# edge over the analytic support is exact.
#
# Run:  python E22_se3_composition.py            (~4-8 min on RTX 2080 SUPER)
#       python E22_se3_composition.py --quick    (CPU smoke test)
# ============================================================
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--seed_start', type=int, default=10)
ap.add_argument('--epochs', type=int, default=3000)
args, _unknown = ap.parse_known_args()
NSEEDS = 2 if args.quick else args.seeds
SEED_START = args.seed_start
EPOCHS = 300 if args.quick else args.epochs

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4e} ±{np.std(vals):.2e}  per-seed: {[f'{v:.2e}' for v in vals]}")

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
# SO(3) / SE(3) exp with series handling near theta = 0
# ------------------------------------------------------------
def hat(w):
    B = w.shape[:-1]
    W = torch.zeros(*B, 3, 3, device=w.device, dtype=w.dtype)
    W[...,0,1] = -w[...,2]; W[...,0,2] =  w[...,1]
    W[...,1,0] =  w[...,2]; W[...,1,2] = -w[...,0]
    W[...,2,0] = -w[...,1]; W[...,2,1] =  w[...,0]
    return W

def so3_coeffs(th):
    """a = sin th / th, b = (1 - cos th)/th^2, as (..,1,1) tensors, with series."""
    small = th < 1e-4
    th_s = torch.where(small, torch.ones_like(th), th)
    a = torch.where(small, 1 - th**2/6, torch.sin(th_s)/th_s)
    b = torch.where(small, 0.5 - th**2/24, (1 - torch.cos(th_s))/th_s**2)
    return a, b

def se3_exp(xi):
    """xi: (...,6) -> R (...,3,3), p (...,3)."""
    w, v = xi[..., :3], xi[..., 3:]
    th = w.norm(dim=-1, keepdim=True)                 # (...,1)
    W = hat(w)
    W2 = torch.matmul(W, W)
    a, b = so3_coeffs(th)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(W2)
    R = I + a[...,None]*W + b[...,None]*W2
    small = th < 1e-4
    c = torch.where(small, 1/6 - th**2/120, (th - torch.sin(torch.where(small, torch.ones_like(th), th))) /
                    (torch.where(small, torch.ones_like(th), th)**3))
    V = I + b[...,None]*W + c[...,None]*W2
    p = torch.einsum('bij,bj->bi', V, v)
    return R, p

def se3_compose(R1, p1, R2, p2):
    """g2 o g1."""
    return torch.matmul(R2, R1), torch.einsum('bij,bj->bi', R2, p1) + p2

PROBES = torch.tensor([[ 1., 1., 1.],
                       [ 1.,-1.,-1.],
                       [-1., 1.,-1.],
                       [-1.,-1., 1.]], device=device) / math.sqrt(3)   # (4,3)

def transform_points(R, p, pts=PROBES):
    pts = pts.to(dtype=R.dtype)
    return torch.einsum('bij,kj->bki', R, pts) + p.unsqueeze(1)

def point_loss(pred, target):
    return ((pred - target)**2).sum(-1).mean()          # MSE over points

def point_err(pred, target):
    return (pred - target).norm(dim=-1).mean().item()   # mean Euclidean error

# ------------------------------------------------------------
# Data: n=2 SE(3) transforms, fixed angles, random axes + translations
# ------------------------------------------------------------
ANG2 = (0.7, 1.1)
def make_se3_data(N, n=2, seed=1, angles=ANG2, sigma_p=0.5, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N, 3, generator=g); a = a/a.norm(dim=-1, keepdim=True)
        axes.append(a.to(device))
    ps = [(torch.randn(N, 3, generator=g) * sigma_p).to(device) for _ in range(n)]
    # build transforms (ascending composition, last applied leftmost)
    Rs, pv = [], []
    for i in range(n):
        th = torch.full((N,), float(angles[i]), device=device)
        w = th.unsqueeze(-1) * axes[i]
        # R from axis-angle (Rodrigues, robust)
        xi = torch.cat([w, torch.zeros_like(w)], -1)
        R, _ = se3_exp(xi)
        Rs.append(R); pv.append(ps[i])
    R, p = Rs[-1], pv[-1]
    for i in range(n-2, -1, -1):
        R, p = se3_compose(Rs[i], pv[i], R, p)     # g_{i} on the right
    Y = transform_points(R, p)                      # (N,4,3)
    phases = [torch.stack([torch.atan2(a[:,1], a[:,0]),
                           torch.acos(a[:,2].clamp(-1+1e-6, 1-1e-6))], -1) for a in axes]
    X = torch.cat([torch.cat([ph, ps[i]], -1) for i, ph in enumerate(phases)], dim=-1)
    if noise > 0:
        X = X + noise*torch.randn_like(X)
        Y = Y + noise*torch.randn_like(Y)
    return X.float().to(device), Y.float().to(device)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
def axis_feats(ph):
    al, be = ph[:,0], ph[:,1]
    return torch.stack([torch.sin(be)*torch.cos(al),
                        torch.sin(be)*torch.sin(al),
                        torch.cos(be)], dim=-1)          # (B,3) = unit axis

def cross(a, p):
    return torch.cross(a, p, dim=-1)

def dot(a, p):
    return (a*p).sum(-1, keepdim=True)

class SE3Dense(nn.Module):
    """Linear edge over [1, a(3), p(3)] -> se(3) (6) -> exp -> SE(3) product."""
    def __init__(self, n_groups=2):
        super().__init__(); self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, 7, 6) * 0.1)
    def group_feats(self, ph, p):
        a = axis_feats(ph)
        return torch.cat([torch.ones_like(p[:,:1]), a, p], -1)  # (B,7)
    def forward(self, x):
        R, t = None, None
        mats = []
        for gi in range(self.n_groups):
            ph = x[:, 5*gi:5*gi+2]; p = x[:, 5*gi+2:5*gi+5]
            f = self.group_feats(ph, p)
            xi = torch.einsum('bf,fc->bc', f, self.w[gi])
            Ri, pi = se3_exp(xi)
            mats.append((Ri, pi))
        R, t = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            R, t = se3_compose(mats[gi][0], mats[gi][1], R, t)
        return transform_points(R, t)

class SE3Structured(nn.Module):
    """Edge over the analytic support (axis, p, a x p, a(a.p)) -> se(3) -> exp."""
    def __init__(self, n_groups=2):
        super().__init__(); self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, 12, 6) * 0.1)
    def group_feats(self, ph, p):
        a = axis_feats(ph)
        axp = cross(a, p)
        adp = a * dot(a, p)
        return torch.cat([a, p, axp, adp], -1)      # (B,12)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            ph = x[:, 5*gi:5*gi+2]; p = x[:, 5*gi+2:5*gi+5]
            f = self.group_feats(ph, p)
            xi = torch.einsum('bf,fc->bc', f, self.w[gi])
            mats.append(se3_exp(xi))
        R, t = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            R, t = se3_compose(mats[gi][0], mats[gi][1], R, t)
        return transform_points(R, t)

class SE3NodeOnly(nn.Module):
    """Edge over [1,a,p] -> (quaternion 4, translation 3) directly; product. No exp."""
    def __init__(self, n_groups=2):
        super().__init__(); self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, 7, 7) * 0.1)
    def group_feats(self, ph, p):
        a = axis_feats(ph)
        return torch.cat([torch.ones_like(p[:,:1]), a, p], -1)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            ph = x[:, 5*gi:5*gi+2]; p = x[:, 5*gi+2:5*gi+5]
            out = torch.einsum('bf,fc->bc', self.group_feats(ph, p), self.w[gi])
            q = out[:, :4]; q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            mats.append((q, out[:, 4:]))
        q, t = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            t1 = mats[gi][1]
            t = torch.einsum('bij,bj->bi', R2_from_q(q), t1) + t
            q = quat_mul(q, mats[gi][0])
        R = quat_to_R(q)
        return transform_points(R, t)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], dim=-1)

def quat_to_R(q):
    w,x,y,z = q.unbind(-1)
    B = q.shape[0]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)], -1),
        torch.stack([2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
        torch.stack([2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)], -1)], -2)
    return R

def R2_from_q(q):
    return quat_to_R(q)

class MLPBaseline(nn.Module):
    """Directly regress the transformed probe points (4 x 3 = 12 outputs)."""
    def __init__(self, in_dim=10, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 12))
    def forward(self, x):
        return self.net(x).reshape(-1, 4, 3)

def n_params(m): return sum(p.numel() for p in m.parameters())

def train_eval(ctor, Xtr, Ytr, Xte, Yte, lr, epochs):
    net = ctor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad(); point_loss(net(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        return point_err(net(Xte), Yte)

def run(ctor, lr, n_seeds, epochs, **data_kw):
    vals = []
    for s in range(SEED_START, SEED_START + n_seeds):
        Xtr, Ytr = make_se3_data(800, seed=s, **data_kw)
        Xte, Yte = make_se3_data(4000, seed=s+100, **data_kw)
        torch.manual_seed(100+s)
        vals.append(train_eval(ctor, Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

# ------------------------------------------------------------
# exactness check for the structured edge
# ------------------------------------------------------------
def structured_exactness(n_groups=2, N=2000, seed=3):
    net = SE3Structured(n_groups).to(device).double()
    with torch.no_grad():
        net.w.zero_()
        for gi in range(n_groups):
            th = float(ANG2[gi])
            th_t = torch.tensor(th, dtype=torch.float64)
            small = th < 1e-4
            d = (1/th**2) - (1+math.cos(th))/(2*th*math.sin(th)) if not small else 1/12
            c = 1 - d*th**2
            # omega outputs 0..2 from axis features a (indices 0..2)
            net.w[gi, 0, 0] = th; net.w[gi, 1, 1] = th; net.w[gi, 2, 2] = th
            # v outputs 3..5: (1-d th^2) p (idx 3..5), -(th/2)(a x p) (idx 6..8), d th^2 a(a.p) (idx 9..11)
            for k in range(3):
                net.w[gi, 3+k, 3+k] = c
                net.w[gi, 6+k, 3+k] = -th/2
                net.w[gi, 9+k, 3+k] = d*th**2
        g = torch.Generator().manual_seed(seed)
        Xtr, Ytr = make_se3_data(N, seed=seed)
        # rebuild targets in float64 without the net's float32 path
        X = Xtr.double()
        pred = net(X)
        target = Ytr.double()
        err = (pred - target).norm(dim=-1).mean().item()
        denom = target.norm(dim=-1).mean().item()
    return err, err/denom

# ============================================================
print("\n" + "="*72)
print("E22 / SE(3) composition: full design (edge -> se(3) -> exp -> product)")
print("mean Euclidean error on 4 probe points")
print("="*72)

em, er = structured_exactness(2)
print(f"[E22:structured exactness] constructed analytic weights: mean err {em:.2e}, "
      f"relative {er:.2e} (expect ~1e-7)")

configs = [
    ('SE3-dense (linear edge + exp)',   lambda: SE3Dense(2),      1e-2),
    ('SE3-structured (analytic + exp)', lambda: SE3Structured(2), 1e-2),
    ('SE3-nodeonly (no exp)',           lambda: SE3NodeOnly(2),   1e-2),
    ('MLP h=16',                        lambda: MLPBaseline(10,16), 3e-3),
    ('MLP h=64',                        lambda: MLPBaseline(10,64), 3e-3),
]
res = {}
for name, ctor, lr in configs:
    vals = run(ctor, lr, NSEEDS, EPOCHS)
    res[name] = vals
    reg(f'E22:{name}', vals, params=n_params(ctor()))

print("\n--- MLP width sweep ---")
for h in [4, 8, 16, 32, 64]:
    vals = run(lambda h=h: MLPBaseline(10, h), 3e-3, NSEEDS, EPOCHS)
    reg(f'E22:MLP h={h}', vals, params=n_params(MLPBaseline(10,h)))

print("\n--- paired t-tests (structured vs others) ---")
for b in ['SE3-dense (linear edge + exp)', 'MLP h=16', 'MLP h=64']:
    t, df, p = paired_t(res['SE3-structured (analytic + exp)'], res[b])
    print(f"  structured vs {b}: t={t:.2f}, df={df}, p={p:.2e}")
    RESULTS[f'E22:ttest:structured vs {b}'] = dict(t=t, df=df, p=p)

print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=1, default=float))
print("\nE22 COMPLETE")
