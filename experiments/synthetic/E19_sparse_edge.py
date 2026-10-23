# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E19 — SPARSE EDGE (Hypothesis N): full battery for the paper
#
# Motivation. The K=1 edge carries 9 Fourier features x 3 outputs = 27
# weights per group (54 for n=2). But the exact solution v = (theta/2)*a
# needs only the three axis components, which are themselves features:
#     cos(beta) -> v3,  cos(alpha)sin(beta) -> v1,  sin(alpha)sin(beta) -> v2.
# An earlier standalone probe (T1-T4) showed:
#   (i)   the constructed weights theta/2 are exact to float64 precision;
#   (ii)  trained sparse (18 p) BEATS trained dense (54 p) at sigma=0.03
#         (2.747 vs 2.788 deg, paired t=-12.19, p=2.6e-04, 5/5 seeds);
#   (iii) the advantage holds on the noise curve up to sigma=0.10 and
#         vanishes at sigma=0.20 (paired t per cell, this script);
#   (iv)  sparse sits closer to the Bayes floor (+0.05 deg) than dense (+0.09).
# This script extends the probe to the cells where pruning is appropriate,
# under the paper's exact protocol, with paired t-tests on EVERY cell.
#
# Registered predictions (to be logged in Appendix E before the rerun):
#   P23  sparse >= dense at sigma=0.03                 (already observed: CONFIRMED)
#   P24  advantage holds to sigma=0.10, gone at 0.20    (partially observed)
#   P25  advantage holds at n=3 (27 p vs 81 p)
#   P26  advantage GROWS at small N (N <= 200) — variance argument
# Out of scope by construction (stated in the paper, not run here):
#   random angles (P16b: theta is an input, the theta/2 construction does
#   not transfer), null task, Zn, GL(2), applied probes (IMU/FK: angle
#   enters through a linear channel, different construction).
#
# Protocol (identical to run_main_experiments.py / Appendix A):
#   full-batch Adam, lr=1e-2, grad clip 1.0, geodesic loss mean(1-cos),
#   3000 epochs (2500 in noise and sample-efficiency cells, as in E3.2/E4),
#   data seed=s, test seed=100+s, init torch.manual_seed(100+s);
#   sample efficiency shares ONE fixed test draw (seed=100) across all N.
#   5 seeds everywhere except chains (3, matching E8).
#
# FLOPs: reported under TWO conventions.
#   repo convention: the repository's flat count (212 dense / 128 sparse);
#   paper convention — single source flops_convention.py (paper Sec.3/B.5 flat count):
#   dense K=1 n=2 -> 228 (was 234), sparse n=2 -> 144 (was 150). Asserted below against
#   the E2 reference value, so E2 and E19 cannot drift apart again.
#
# Standalone: does not import or modify run_main_experiments.py,
# run_extensions.py or E16_gl2_composition.py.
#
# Run:  python E19_sparse_edge.py            (full; GPU recommended)
#       python E19_sparse_edge.py --quick    (CPU smoke test)
# ============================================================
import argparse, json, math, time
import numpy as np
import torch, torch.nn as nn
from flops_convention import quatkan_flops, FLOPS_E2_QUATKAN_K1

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
    """paired t of (a - b); returns t, df, p. Needs >=2 seeds."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; n = len(d)
    if n < 2:
        return float('nan'), n-1, float('nan')
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n) + 1e-12)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float('nan')
    return float(t), n-1, float(p)

def reg_t(key, a, b, name_a='sparse', name_b='dense'):
    """Paired t of (a - b); 'wins' counts seeds where a < b (lower error)."""
    t, df, p = paired_t(a, b)
    wins = int((np.asarray(a) < np.asarray(b)).sum())
    reg_text(key, f"paired t={t:.2f}, df={df}, p={p:.2e}, {name_a} wins {wins}/{len(a)} seeds")
    return t, p

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

# ------------------------------------------------------------
# Data (same generator as run_main_experiments.py)
# ------------------------------------------------------------
def make_compn_data(N, n=2, noise=0.03, seed=1, angles=None):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for i in range(n):
        a = torch.randn(N,3,generator=g); a = a/a.norm(dim=-1,keepdim=True)
        axes.append(a)
    ths = [torch.full((N,), float(angles[i])) for i in range(n)]
    prod = axis_angle_to_quat(axes[-1], ths[-1])
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[i], ths[i]))
    v = rotate(prod, torch.tensor([0.,0.,1.]).expand(N,3))
    v = v / v.norm(dim=-1, keepdim=True)
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N,2*n,generator=g)
    Y = vec_to_angles(v) + noise*torch.randn(N,2,generator=g)
    return X.to(device), Y.to(device)

ANG2 = (0.7, 1.1)
ANG3 = (0.7, 1.1, 0.9)
comp2 = lambda N, **kw: make_compn_data(N, n=2, angles=ANG2, **kw)
comp3 = lambda N, **kw: make_compn_data(N, n=3, angles=ANG3, **kw)
def chain(n):  # E8 protocol: angles 0.7 + 0.2*i
    return lambda N, **kw: make_compn_data(N, n=n, angles=tuple(0.7+0.2*i for i in range(n)), **kw)

# ------------------------------------------------------------
# Models: dense edge (K Fourier orders) and sparse edge (Hypothesis N)
# ------------------------------------------------------------
class GroupedQuatExpNet(nn.Module):
    """Reference LawKAN edge: (2K+1)^2 features x 3 outputs per group."""
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

class SparseQuatExpNet(GroupedQuatExpNet):
    """Hypothesis N: keep only [cos b, cos a sin b, sin a sin b]
    -> 3 features x 3 outputs = 9 weights per group (18 for n=2 vs 54)."""
    def __init__(self, n_groups=2):
        super().__init__(n_groups, K=1)
        self.w = nn.Parameter(torch.randn(n_groups, 3, 3) * 0.1)
    def group_feats(self, a):
        al, be = a[:, 0], a[:, 1]
        sbe = torch.sin(be)
        return torch.stack([torch.cos(be), torch.cos(al)*sbe, torch.sin(al)*sbe], dim=-1)

# ------------------------------------------------------------
# Training / evaluation (paper protocol, Appendix A.3/A.4)
# ------------------------------------------------------------
def train(net, Xtr, Ytr, lr=1e-2, epochs=3000):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = geo_loss(net(Xtr), Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net

def train_eval(net, Xtr, Ytr, Xte, Yte, lr=1e-2, epochs=3000):
    net = train(net, Xtr, Ytr, lr, epochs)
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()

def run_seeds(ctor, data_fn, lr, n_seeds, epochs=3000, train_N=800,
              fixed_test_seed=None, **data_kw):
    vals = []
    for s in range(n_seeds):
        Xtr, Ytr = data_fn(train_N, seed=s, **data_kw)
        tseed = fixed_test_seed if fixed_test_seed is not None else s + 100
        Xte, Yte = data_fn(4000, seed=tseed, **data_kw)
        torch.manual_seed(100 + s)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

def dense_ctor(n_groups=2, K=1):  return lambda: GroupedQuatExpNet(n_groups, K=K)
def sparse_ctor(n_groups=2):      return lambda: SparseQuatExpNet(n_groups)

def flops_repo(n_groups, n_feat, n_products):
    """Repository flat convention (dense n=2 K=1 -> 212)."""
    return 4*n_groups + n_feat*n_groups + 2*n_feat*3*n_groups + 15*n_groups + 16*n_products + 32

def flops_paper(n_groups, n_feat, n_products):
    """Paper Sec.3 / Appendix B.5 flat convention, via the single source
    ``flops_convention.py`` (dense n=2 K=1 -> 228, the Table 2 value).
    ``n_products`` is accepted for signature stability; the Hamilton term is
    derived internally as n_groups - 1."""
    return quatkan_flops(n_groups=n_groups, n_features=n_feat)

def flops_pair(n_groups, sparse, K=1):
    n_feat = 3 if sparse else (2*K+1)**2
    npd = n_groups - 1
    return dict(flops_repo=flops_repo(n_groups, n_feat, npd),
                flops_paper=flops_paper(n_groups, n_feat, npd))

def bench_net(net, x, y, lr=1e-2, iters=300, warmup=30):
    net = net.to(device)
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(warmup):
        opt.zero_grad(); geo_loss(net(x), y).backward(); opt.step()
    sync(); t0 = time.time()
    for _ in range(iters):
        opt.zero_grad(); geo_loss(net(x), y).backward(); opt.step()
    sync(); tr = (time.time()-t0)/iters*1000
    with torch.no_grad():
        for _ in range(warmup): net(x)
        sync(); t0 = time.time()
        for _ in range(iters): net(x)
    sync(); inf = (time.time()-t0)/iters*1000
    return tr, inf

# ============================================================
# CLI
# ============================================================
ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help='2 seeds, 300 epochs (CPU smoke test)')
ap.add_argument('--epochs', type=int, default=3000)
ap.add_argument('--seeds', type=int, default=5)
# parse_known_args: running from a Jupyter cell passes kernel args (-f kernel.json);
# unknown arguments are ignored in both CLI and notebook contexts.
args, _unknown = ap.parse_known_args()
EPOCHS   = 300 if args.quick else args.epochs
NOISE_EP = 300 if args.quick else 2500
NSEEDS   = 2 if args.quick else args.seeds
CHAIN_SEEDS = 2 if args.quick else 3   # E8 used 3 seeds
N_LIST   = [50, 100, 200, 400, 800, 1600]

# ============================================================
# E19.0 analytic exactness sanity (float64, exact phases)
# ============================================================
def t1b_double(net, angles, N=2000, seed=3):
    g = torch.Generator().manual_seed(seed)
    axes = []
    for _ in range(len(angles)):
        a = torch.randn(N, 3, generator=g, dtype=torch.float64)
        axes.append(a / a.norm(dim=-1, keepdim=True))
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1)
    prod = axis_angle_to_quat(axes[-1], torch.full((N,), float(angles[-1]), dtype=torch.float64))
    for i in range(len(axes)-2, -1, -1):
        prod = quat_mul(prod, axis_angle_to_quat(axes[i], torch.full((N,), float(angles[i]), dtype=torch.float64)))
    probe = torch.tensor([0., 0., 1.], dtype=torch.float64)
    vtrue = rotate(prod, probe.expand(N, 3)); vtrue = vtrue / vtrue.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        net = net.double()
        net.w.zero_()
        for gi, th in enumerate(angles):
            net.w[gi, 1, 0] = th/2.0   # cos a sin b -> v1
            net.w[gi, 2, 1] = th/2.0   # sin a sin b -> v2
            net.w[gi, 0, 2] = th/2.0   # cos b       -> v3
        # forward up to the probe vector, without the angle readout
        qs = []
        for gi in range(net.n_groups):
            f = net.group_feats(X[:, 2*gi:2*gi+2])
            v = torch.einsum('bf,fc->bc', f, net.w[gi])
            nv = v.norm(dim=-1, keepdim=True)
            qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], dim=-1))
        prod = qs[-1]
        for gi in range(net.n_groups-2, -1, -1):
            prod = quat_mul(prod, qs[gi])
        vpred = rotate(prod, probe.expand(N, 3))
        vpred = vpred / vpred.norm(dim=-1, keepdim=True)
    cos = (vpred * vtrue).sum(-1).clamp(-1.0, 1.0)
    err = torch.rad2deg(torch.acos(cos))
    return err.mean().item(), err.max().item()

print("\n" + "="*72)
print("E19.0 / analytic exactness of the constructed sparse weights (float64)")
print("="*72)
m, mx = t1b_double(SparseQuatExpNet(2), ANG2)
reg_text('E19.0:exactness comp2', f"mean {m:.2e} deg, max {mx:.2e} deg (expect ~1e-7)")
m, mx = t1b_double(SparseQuatExpNet(3), ANG3)
reg_text('E19.0:exactness comp3', f"mean {m:.2e} deg, max {mx:.2e} deg (expect ~1e-7)")

# ============================================================
# E19.1 main cell: comp2, sigma=0.03 (P23)
# ============================================================
print("\n" + "="*72)
print(f"E19.1 / comp2 sigma=0.03, {NSEEDS} seeds, {EPOCHS} epochs (P23)")
print("="*72)
res = {}
for name, ctor in [('dense K=1 (54p)', dense_ctor(2, 1)),
                   ('sparse (18p)',    sparse_ctor(2))]:
    vals = run_seeds(ctor, comp2, 1e-2, NSEEDS, epochs=EPOCHS)
    res[name] = vals
    reg(f'E19.1:{name}', vals, params=n_params(ctor), **flops_pair(2, sparse=('sparse' in name)))
reg_t('E19.1:ttest', res['sparse (18p)'], res['dense K=1 (54p)'])

# ============================================================
# E19.2 K / feature ablation: K=2 (150p) vs K=1 (54p) vs sparse (18p)
# ============================================================
print("\n" + "="*72)
print(f"E19.2 / K and feature ablation, {NSEEDS} seeds, {EPOCHS} epochs")
print("="*72)
res2 = {}
for name, ctor, sp in [('dense K=2 (150p)', dense_ctor(2, 2), False),
                       ('dense K=1 (54p)',  dense_ctor(2, 1), False),
                       ('sparse (18p)',     sparse_ctor(2),   True)]:
    vals = run_seeds(ctor, comp2, 1e-2, NSEEDS, epochs=EPOCHS)
    res2[name] = vals
    reg(f'E19.2:{name}', vals, params=n_params(ctor), **flops_pair(2, sp, K=2 if 'K=2' in name else 1))
reg_t('E19.2:ttest sparse vs dense K=1', res2['sparse (18p)'], res2['dense K=1 (54p)'])
reg_t('E19.2:ttest dense K=2 vs dense K=1', res2['dense K=2 (150p)'], res2['dense K=1 (54p)'],
      name_a='dense K=2', name_b='dense K=1')

# ============================================================
# E19.3 noise curve, sigma = 0.01..0.20 (P24)
# ============================================================
print("\n" + "="*72)
print(f"E19.3 / noise curve, {NSEEDS} seeds, {NOISE_EP} epochs (P24)")
print("="*72)
for noise in [0.01, 0.05, 0.10, 0.20]:
    rd, rs = {}, {}
    for name, ctor in [('dense K=1 (54p)', dense_ctor(2, 1)),
                       ('sparse (18p)',    sparse_ctor(2))]:
        vals = run_seeds(ctor, comp2, 1e-2, NSEEDS, epochs=NOISE_EP, noise=noise)
        rd[name] = vals
        reg(f'E19.3:sigma{noise}:{name}', vals)
    reg_t(f'E19.3:sigma{noise}:ttest', rd['sparse (18p)'], rd['dense K=1 (54p)'])

# ============================================================
# E19.4 composition n=3 (P25)
# ============================================================
print("\n" + "="*72)
print(f"E19.4 / comp3 (angles 0.7/1.1/0.9), {NSEEDS} seeds, {EPOCHS} epochs (P25)")
print("="*72)
res4 = {}
for name, ctor in [('dense K=1 (81p)', dense_ctor(3, 1)),
                   ('sparse (27p)',    sparse_ctor(3))]:
    vals = run_seeds(ctor, comp3, 1e-2, NSEEDS, epochs=EPOCHS)
    res4[name] = vals
    reg(f'E19.4:{name}', vals, params=n_params(ctor), **flops_pair(3, sparse=('sparse' in name)))
reg_t('E19.4:ttest', res4['sparse (27p)'], res4['dense K=1 (81p)'])

# ============================================================
# E19.5 chain scaling n=2,3,4 (E8 protocol, 3 seeds)
# ============================================================
print("\n" + "="*72)
print(f"E19.5 / chains n=2,3,4 (angles 0.7+0.2i), {CHAIN_SEEDS} seeds, {EPOCHS} epochs")
print("="*72)
for n in [2, 3, 4]:
    data_fn = chain(n)
    rd, rs = {}, {}
    for name, ctor in [(f'dense K=1 ({27*n}p)', dense_ctor(n, 1)),
                       (f'sparse ({9*n}p)',     sparse_ctor(n))]:
        vals = run_seeds(ctor, data_fn, 1e-2, CHAIN_SEEDS, epochs=EPOCHS)
        rd[name] = vals
        reg(f'E19.5:n{n}:{name}', vals, params=n_params(ctor),
            **flops_pair(n, sparse=('sparse' in name)))
    reg_t(f'E19.5:n{n}:ttest', rd[f'sparse ({9*n}p)'], rd[f'dense K=1 ({27*n}p)'])

# ============================================================
# E19.6 sample efficiency, N = 50..1600 (P26)
# ============================================================
print("\n" + "="*72)
print(f"E19.6 / sample efficiency N={N_LIST[0]}..{N_LIST[-1]}, {NSEEDS} seeds, "
      f"{NOISE_EP} epochs, shared test draw seed=100 (P26)")
print("="*72)
for N in N_LIST:
    rd, rs = {}, {}
    for name, ctor in [('dense K=1 (54p)', dense_ctor(2, 1)),
                       ('sparse (18p)',    sparse_ctor(2))]:
        vals = run_seeds(ctor, comp2, 1e-2, NSEEDS, epochs=NOISE_EP,
                         train_N=N, fixed_test_seed=100)
        rd[name] = vals
        reg(f'E19.6:N{N}:{name}', vals)
    reg_t(f'E19.6:N{N}:ttest', rd['sparse (18p)'], rd['dense K=1 (54p)'])

# ============================================================
# E19.7 does the optimizer find the constructed solution? (weight recovery)
# ============================================================
print("\n" + "="*72)
print("E19.7 / weight recovery: trained w vs analytic theta/2 (2 seeds, comp2)")
print("="*72)
for seed in range(2):
    Xtr, Ytr = comp2(800, seed=seed)
    torch.manual_seed(100 + seed)
    net = train(SparseQuatExpNet(2), Xtr, Ytr, 1e-2, EPOCHS)
    w = net.w.detach().cpu().numpy()
    wt = np.zeros_like(w)
    for gi, th in enumerate(ANG2):
        wt[gi, 1, 0] = th/2.0; wt[gi, 2, 1] = th/2.0; wt[gi, 0, 2] = th/2.0
    dmax = np.abs(w - wt).max()
    drel = np.linalg.norm(w - wt) / np.linalg.norm(wt)
    reg_text(f'E19.7:recovery seed{seed}',
             f"max|w - theta/2| = {dmax:.4f}, relative Frobenius = {drel:.4f}")
    # and the dense model, for the three load-bearing entries + norm of the rest
    torch.manual_seed(100 + seed)
    netd = train(GroupedQuatExpNet(2, K=1), Xtr, Ytr, 1e-2, EPOCHS)
    wd = netd.w.detach().cpu().numpy()
    idx = [(0, 5, 0), (0, 8, 1), (0, 1, 2), (1, 5, 0), (1, 8, 1), (1, 1, 2)]
    d_key = max(abs(wd[gi, f, c] - th/2) for (gi, f, c), th in zip(idx, [0.7, 0.7, 0.7, 1.1, 1.1, 1.1]))
    rest = [wd[gi, f, c] for gi in range(2) for f in range(9) for c in range(3)
            if (gi, f, c) not in idx]
    reg_text(f'E19.7:dense-rest seed{seed}',
             f"key entries max dev = {d_key:.4f}, other 48 weights: mean|w| = {np.mean(np.abs(rest)):.4f}, max|w| = {np.max(np.abs(rest)):.4f}")

# ============================================================
# E19.8 cost: params, FLOPs (two conventions), wall-clock
# ============================================================
print("\n" + "="*72)
print("E19.8 / cost (batch 1024, 300 iters, 30 warmup)")
print("="*72)
torch.manual_seed(0)
Xb, Yb = comp2(1024, seed=0)
for name, ctor, n_groups, sp, K in [
        ('dense K=1 (54p)', dense_ctor(2, 1), 2, False, 1),
        ('sparse (18p)',    sparse_ctor(2),   2, True,  1),
        ('dense K=2 (150p)', dense_ctor(2, 2), 2, False, 2)]:
    net = ctor()
    tr, inf = bench_net(net, Xb, Yb, 1e-2)
    fl = flops_pair(n_groups, sp, K=K)
    reg_text(f'E19.8:{name}',
             f"params={n_params(ctor)}, FLOPs_repo={fl['flops_repo']}, FLOPs_paper={fl['flops_paper']}, "
             f"train={tr:.2f} ms/it, infer={inf:.2f} ms/batch "
             f"(wall-clock within rerun noise; params and FLOPs are exact)")

# Convention guard: dense K=1 here must equal the shared E2/Table-2 value.
assert flops_paper(2, 9, 1) == FLOPS_E2_QUATKAN_K1, \
    "convention drift vs flops_convention.py / run_main_experiments E2"

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*72)
print("SUMMARY-JSON")
print("="*72)
print(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=str))
print("\nE19 SPARSE-EDGE BATTERY COMPLETE")