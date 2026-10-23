# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E16 — GL+(2,R) composition probe (structure-transfer hypothesis)  [v2 FIXED]
#
# Hypothesis under test: the LawKAN design (Fourier phase edges -> Lie algebra
# -> exp-map -> multiplicative node) transfers from SU(2) to the general linear
# group GL+(2,R) at K=1. Each group element is a 2x2 matrix with det>0,
# parameterized by Iwasawa/polar coordinates:
#     g = R(theta) @ diag(exp(s1), exp(s2)) @ [[1, t],[0, 1]]
#   theta  in [0, 2pi)   rotation     (compact, periodic  -> Fourier-native)
#   s1,s2  in R          log-stretch  (non-compact        -> NOT Fourier-native)
#   t      in R          shear        (non-compact        -> NOT Fourier-native)
#
# The task: given two elements (4 params each), predict their product g2 @ g1.
# Target readout: the product matrix entries; metric: relative Frobenius.
#
# ---------------------------------------------------------------------------
# v2 (2026-09-24) FIXES A CONFIRMED BUG in v1. v1's GLKAN forward used only
# (theta, s1) per group -- the coordinates s2 and t were never read -- so the
# model was information-starved and could not represent the element, while the
# MLP baseline saw all 8 inputs and used ~70x more parameters. v1 therefore
# could not support its "structure does not transfer to non-compact groups"
# conclusion. v2:
#   * feeds ALL four coordinates per group;
#   * applies Fourier features only to the periodic theta, and LINEAR features
#     to the non-periodic (s1, s2, t);
#   * adds a node-only variant (edges emit the group element DIRECTLY, no
#     exp-map) so the effect of the multiplicative node is isolated from the
#     exp parameterization (exp is not surjective on GL+(2,R));
#   * reports the old buggy variant as a row, to measure the bug's effect;
#   * adds matched-parameter MLP baselines alongside the large MLP h=64.
# ---------------------------------------------------------------------------
#
# Run:  python E16_gl2_composition.py            (~8-12 min on an RTX 2080 SUPER)
#       python E16_gl2_composition.py --quick    (CPU smoke test: 2 seeds, 300 ep)
# ============================================================
import argparse, torch, torch.nn as nn, math, json, time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help='2 seeds, 300 epochs')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--epochs', type=int, default=3000)
args, _unknown = ap.parse_known_args()
NSEEDS = 2 if args.quick else args.seeds
EPOCHS = 300 if args.quick else args.epochs

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}  per-seed: {[round(v,3) for v in vals]}")

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
# Geometry: GL+(2,R) element <-> Iwasawa coordinates (theta, s1, s2, t)
# ------------------------------------------------------------
def params_to_mat(p):
    th, s1, s2, t = p.unbind(-1)
    c, sn = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)
    D = torch.zeros_like(R)
    D[..., 0, 0] = torch.exp(s1); D[..., 1, 1] = torch.exp(s2)
    H = torch.zeros_like(R)
    H[..., 0, 0] = 1.0; H[..., 0, 1] = t; H[..., 1, 1] = 1.0
    return R @ D @ H

PROBE = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # identity: images are the columns of g2@g1

def make_gl2_data(N, seed=1, stretch=0.6, shear=0.6):
    """Two GL+(2,R) elements -> images of the 2 probe vectors under g2 @ g1.
    Input: 8 numbers (4 coords x 2 elements). Target: 4 numbers (columns of product)."""
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(N, 2, generator=g) * 2 * math.pi
    s1 = torch.randn(N, 2, generator=g) * stretch
    s2 = torch.randn(N, 2, generator=g) * stretch
    t  = torch.randn(N, 2, generator=g) * shear
    X = torch.stack([th, s1, s2, t], dim=-1).reshape(N, 8)
    G1 = params_to_mat(X[:, 0:4]); G2 = params_to_mat(X[:, 4:8])
    M = G2 @ G1
    Y = M.reshape(N, 4)
    return X.float().to(device), Y.float().to(device)

def fro_rel(pred, Y):
    M = Y.reshape(-1, 2, 2); P = pred.reshape(-1, 2, 2)
    num = (P - M).norm(dim=(-2, -1)); den = M.norm(dim=(-2, -1)) + 1e-6
    return (num / den)

def fro_loss(pred, Y):
    return fro_rel(pred, Y).mean()

# ------------------------------------------------------------
# Edge features: Fourier on the periodic theta, LINEAR on non-periodic s1,s2,t.
# ------------------------------------------------------------
def edge_feats(theta, s1, s2, t, K=1):
    f = [torch.ones_like(theta)]
    for m in range(1, K+1):
        f += [torch.cos(m*theta), torch.sin(m*theta)]
    f += [s1, s2, t]
    return torch.stack(f, dim=-1)          # (B, 2K+3)

NFEAT = 2*1 + 1 + 3                         # K=1 -> 6 features per group

class GLKAN(nn.Module):
    """Corrected LawKAN transfer: edge = Fourier(theta)+linear(s,t) -> gl(2)
    (4 Lie-algebra coords); node = matrix exp -> GL+(2,R), then matrix product
    g2 @ g1.  `use_exp=False` emits the group element directly (node-only test)."""
    def __init__(self, n_groups=2, K=1, use_exp=True):
        super().__init__()
        self.n_groups, self.K, self.use_exp = n_groups, K, use_exp
        self.w = nn.Parameter(torch.randn(n_groups, 2*K+4, 4) * 0.1)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            p = x[:, 4*gi:4*gi+4]
            f = edge_feats(p[:, 0], p[:, 1], p[:, 2], p[:, 3], self.K)
            A = torch.einsum('bf,fc->bc', f, self.w[gi]).reshape(-1, 2, 2)
            mats.append(torch.matrix_exp(A) if self.use_exp else A)
        prod = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = prod @ mats[gi]
        return prod.reshape(-1, 4)

# ------------------------------------------------------------
# Analytic-support ("structured") edge for GL+(2,R), node-only.
# The exact element is g = R(theta) D(s1,s2) H(t), whose entries are
#     g11 = e^{s1} cos th
#     g21 = e^{s1} sin th
#     g12 = t e^{s1} cos th - e^{s2} sin th
#     g22 = t e^{s1} sin th + e^{s2} cos th
# so a LINEAR edge over the six analytic features
#     [e^{s1}c, e^{s1}s, e^{s2}c, e^{s2}s, t e^{s1}c, t e^{s1}s]
# represents g exactly. This is the GL+ analogue of the SU(2) "sparse" edge
# (keep only the analytic support), and it is what the dense linear feature set
# [1, cos th, sin th, s1, s2, t] cannot express.
# ------------------------------------------------------------
def structured_feats(p):
    th, s1, s2, t = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    c, s = torch.cos(th), torch.sin(th)
    e1, e2 = torch.exp(s1), torch.exp(s2)
    return torch.stack([e1*c, e1*s, e2*c, e2*s, t*e1*c, t*e1*s], dim=-1)  # (B,6)

class GLKANStructured(nn.Module):
    """Node-only GL+ composer with the analytic-support edge (6 features x 4)."""
    def __init__(self, n_groups=2):
        super().__init__()
        self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, 6, 4) * 0.1)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            f = structured_feats(x[:, 4*gi:4*gi+4])
            mats.append(torch.einsum('bf,fc->bc', f, self.w[gi]).reshape(-1, 2, 2))
        prod = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = prod @ mats[gi]
        return prod.reshape(-1, 4)

def structured_exactness(n_groups=2, N=2000, seed=3):
    """Construct the analytic weights (float64) and measure the exactness error."""
    net = GLKANStructured(n_groups).double()
    with torch.no_grad():
        net.w.zero_()
        for gi in range(n_groups):
            # output order is row-major (g11, g12, g21, g22)
            net.w[gi, 0, 0] = 1.0    # e^{s1}c      -> g11
            net.w[gi, 4, 1] = 1.0    # t e^{s1}c    -> g12
            net.w[gi, 3, 1] = -1.0   # -e^{s2}s     -> g12
            net.w[gi, 1, 2] = 1.0    # e^{s1}s      -> g21
            net.w[gi, 5, 3] = 1.0    # t e^{s1}s    -> g22
            net.w[gi, 2, 3] = 1.0    # e^{s2}c      -> g22
        g = torch.Generator().manual_seed(seed)
        th = torch.rand(N, n_groups, generator=g, dtype=torch.float64)*2*math.pi
        s1 = torch.randn(N, n_groups, generator=g, dtype=torch.float64)*0.6
        s2 = torch.randn(N, n_groups, generator=g, dtype=torch.float64)*0.6
        t  = torch.randn(N, n_groups, generator=g, dtype=torch.float64)*0.6
        X = torch.stack([th, s1, s2, t], -1).reshape(N, 4*n_groups)
        G = [params_to_mat(X[:, 4*i:4*i+4]) for i in range(n_groups)]
        Mtrue = G[-1]
        for i in range(n_groups-2, -1, -1):
            Mtrue = Mtrue @ G[i]
        pred = net(X.float().double()).reshape(N, 2, 2)
        rel = ((pred - Mtrue).norm(dim=(-2,-1)) /
               (Mtrue.norm(dim=(-2,-1)) + 1e-12))
    return rel.mean().item(), rel.max().item()

class GLKANOldBuggy(nn.Module):
    """v1 reference (BUG): uses only (theta, s1); Fourier over non-periodic s1."""
    def __init__(self, n_groups=2, K=1):
        super().__init__()
        self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 4) * 0.1)
    def pair_feats(self, u, v):
        fu = [torch.ones_like(u)]; fv = [torch.ones_like(v)]
        for m in range(1, self.K+1):
            fu += [torch.cos(m*u), torch.sin(m*u)]
            fv += [torch.cos(m*v), torch.sin(m*v)]
        return torch.stack([x*y for x in fu for y in fv], dim=-1)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            p = x[:, 4*gi:4*gi+4]
            f = self.pair_feats(p[:, 0], p[:, 1])   # only theta and s1
            A = torch.einsum('bf,fc->bc', f, self.w[gi]).reshape(-1, 2, 2)
            mats.append(torch.matrix_exp(A))
        prod = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = prod @ mats[gi]
        return prod.reshape(-1, 4)

class MLPBaseline(nn.Module):
    def __init__(self, in_dim=8, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 4))
    def forward(self, x):
        return self.net(x)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def train_eval(net, Xtr, Ytr, Xte, Yte, lr=3e-3, epochs=3000):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        fro_loss(net(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return fro_rel(net(Xte), Yte).mean().item()

def run_seeds(ctor, lr, n_seeds, epochs, **data_kw):
    vals = []
    for seed in range(n_seeds):
        Xtr, Ytr = make_gl2_data(800, seed=seed, **data_kw)
        Xte, Yte = make_gl2_data(4000, seed=seed+100, **data_kw)
        torch.manual_seed(100+seed)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

# ============================================================
# E16v2: GL+(2,R) composition
# ============================================================
print("\n" + "="*72)
print("E16v2 / GL+(2,R): corrected features + node-only + analytic support + MLPs")
print("relative Frobenius error of the predicted product (lower is better)")
print("="*72)

em, ex = structured_exactness(2)
print(f"[E16v2:structured exactness] constructed analytic weights: mean {em:.2e}, "
      f"max {ex:.2e} rel. Frobenius (expect ~1e-7)")

configs = [
    ('old GLKAN K=1 (BUG: s2,t ignored)', lambda: GLKANOldBuggy(2, K=1),             1e-2),
    ('GLKAN-exp K=1 (all inputs)',        lambda: GLKAN(2, K=1, use_exp=True),       1e-2),
    ('GLnode-noexp K=1 (all inputs)',     lambda: GLKAN(2, K=1, use_exp=False),      1e-2),
    ('GLnode-structured (analytic supp)', lambda: GLKANStructured(2),                1e-2),
    ('MLP h=3 (matched ~55p)',            lambda: MLPBaseline(8, h=3),               3e-3),
    ('MLP h=64 (unmatched, v1 baseline)', lambda: MLPBaseline(8, h=64),              3e-3),
]
res = {}
t0 = time.time()
for name, ctor, lr in configs:
    vals = run_seeds(ctor, lr, NSEEDS, EPOCHS)
    res[name] = vals
    reg(f'E16v2:{name}', vals, params=n_params(ctor()))
    print(f"      ({time.time()-t0:.0f}s elapsed)")

RESULTS['E16v2:structured exactness'] = dict(mean=em, max=ex)

print("\n--- MLP width sweep (Pareto: error vs parameters) ---")
mlp_sweep = {}
for h in [1, 2, 3, 4, 5, 6, 8]:
    vals = run_seeds(lambda h=h: MLPBaseline(8, h=h), 3e-3, NSEEDS, EPOCHS)
    p = n_params(MLPBaseline(8, h=h))
    mlp_sweep[h] = vals
    reg(f'E16v2:MLP h={h}', vals, params=p)

print("\n--- paired t-tests ---")
for a, b in [('GLKAN-exp K=1 (all inputs)', 'old GLKAN K=1 (BUG: s2,t ignored)'),
             ('GLnode-noexp K=1 (all inputs)', 'GLKAN-exp K=1 (all inputs)'),
             ('GLnode-structured (analytic supp)', 'GLnode-noexp K=1 (all inputs)'),
             ('GLnode-structured (analytic supp)', 'MLP h=3 (matched ~55p)'),
             ('GLnode-structured (analytic supp)', 'MLP h=64 (unmatched, v1 baseline)')]:
    t, df, p = paired_t(res[a], res[b])
    print(f"  {a}  vs  {b}:  t={t:.2f}, df={df}, p={p:.2e}")
    RESULTS[f'E16v2:ttest:{a} vs {b}'] = dict(t=t, df=df, p=p)

print("\nE16v2 COMPLETE")
print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=2, default=float))
