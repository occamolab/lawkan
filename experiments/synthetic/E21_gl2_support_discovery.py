# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E21 — SUPPORT DISCOVERY on GL+(2,R): can the analytic support be LEARNED?
#
# E16 v2 showed that a hand-derived analytic-support edge (6 features) is
# near-exact on GL+(2,R), while a dense linear edge over raw coordinates is not.
# This script tests whether that support is *discoverable* from data instead of
# hand-derived, via train -> prune -> retrain over a generic multiplicative
# dictionary.
#
# Dictionary. The full tensor product of simple univariate bases:
#     theta in {1, cos th, sin th}   (periodic)
#     s1    in {1, e^{s1}}           (non-periodic, multiplicative)
#     s2    in {1, e^{s2}}
#     t     in {1, t}
# -> 3*2*2*2 = 24 product features per group (a MultKAN-style edge). The exact
# GL+ element g = R D H is a LINEAR function of SIX of these 24 features:
#     e^{s1}cos th, e^{s1}sin th, e^{s2}cos th, e^{s2}sin th,
#     t e^{s1}cos th, t e^{s1}sin th.
# The node is the matrix product (exact). So discoverability reduces to: does
# magnitude pruning over the 24 generic features keep (a subset spanning) the
# analytic six, and does the pruned+retrained model reach the exact error?
#
# Procedure: train dense (24 features, group norm per feature) once per seed,
# then for each budget m keep the top-m features and retrain (mask fixed).
# A pruning curve shows the minimum support that is sufficient.
#
# Run:  python E21_gl2_support_discovery.py            (~4-8 min on RTX 2080S)
#       python E21_gl2_support_discovery.py --quick    (CPU smoke test)
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
    print(f"[{key}] mean {np.mean(vals):.5f} ±{np.std(vals):.5f}  per-seed: {[round(v,4) for v in vals]}")

def reg_text(key, text):
    RESULTS[key] = dict(text=text)
    print(f"[{key}] {text}")

# ------------------------------------------------------------
# Geometry (same as E16 v2)
# ------------------------------------------------------------
def params_to_mat(p):
    th, s1, s2, t = p.unbind(-1)
    c, sn = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2)
    D = torch.zeros_like(R); D[...,0,0] = torch.exp(s1); D[...,1,1] = torch.exp(s2)
    H = torch.zeros_like(R); H[...,0,0] = 1.0; H[...,0,1] = t; H[...,1,1] = 1.0
    return R @ D @ H

def make_gl2_data(N, seed=1, stretch=0.6, shear=0.6):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(N,2,generator=g)*2*math.pi
    s1 = torch.randn(N,2,generator=g)*stretch
    s2 = torch.randn(N,2,generator=g)*stretch
    t  = torch.randn(N,2,generator=g)*shear
    X = torch.stack([th,s1,s2,t],-1).reshape(N,8)
    G1 = params_to_mat(X[:,0:4]); G2 = params_to_mat(X[:,4:8])
    return X.float().to(device), (G2@G1).reshape(N,4).float().to(device)

def fro_rel(pred, Y):
    M = Y.reshape(-1,2,2); P = pred.reshape(-1,2,2)
    return ((P-M).norm(dim=(-2,-1)) / (M.norm(dim=(-2,-1))+1e-6))

def fro_loss(pred, Y):
    return fro_rel(pred, Y).mean()

# ------------------------------------------------------------
# Generic multiplicative dictionary (24 features) + labels
# ------------------------------------------------------------
TH_L, S1_L, S2_L, T_L = ['1','cos','sin'], ['1','e1'], ['1','e2'], ['1','t']
def dict_feats(p):
    th, s1, s2, t = p[:,0], p[:,1], p[:,2], p[:,3]
    Th = [torch.ones_like(th), torch.cos(th), torch.sin(th)]
    S1 = [torch.ones_like(s1), torch.exp(s1)]
    S2 = [torch.ones_like(s2), torch.exp(s2)]
    T  = [torch.ones_like(t), t]
    feats = [a*b*c*d for a in Th for b in S1 for c in S2 for d in T]
    return torch.stack(feats, dim=-1)          # (B, 24)
FEATURE_LABELS = [f"{a}*{b}*{c}*{d}" for a in TH_L for b in S1_L for c in S2_L for d in T_L]
ANALYTIC = {'cos*e1*1*1', 'sin*e1*1*1', 'cos*1*e2*1', 'sin*1*e2*1',
            'cos*e1*1*t', 'sin*e1*1*t'}      # the six analytic support features

NF = len(FEATURE_LABELS)                       # 24

class GLDict(nn.Module):
    """Node-only composer over the generic dictionary, with a hard feature mask."""
    def __init__(self, n_groups=2, n_feat=NF):
        super().__init__()
        self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, n_feat, 4)*0.1)
        self.register_buffer('mask', torch.ones(n_groups, n_feat, 1))
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            f = dict_feats(x[:, 4*gi:4*gi+4])
            w = self.w[gi] * self.mask[gi]
            mats.append(torch.einsum('bf,fc->bc', f, w).reshape(-1,2,2))
        prod = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = prod @ mats[gi]
        return prod.reshape(-1,4)

# analytic-support reference (6 features)
def structured_feats(p):
    th, s1, s2, t = p[:,0], p[:,1], p[:,2], p[:,3]
    c, s = torch.cos(th), torch.sin(th); e1, e2 = torch.exp(s1), torch.exp(s2)
    return torch.stack([e1*c, e1*s, e2*c, e2*s, t*e1*c, t*e1*s], -1)
class GLStructured(nn.Module):
    def __init__(self, n_groups=2):
        super().__init__(); self.n_groups = n_groups
        self.w = nn.Parameter(torch.randn(n_groups, 6, 4)*0.1)
    def forward(self, x):
        mats = []
        for gi in range(self.n_groups):
            f = structured_feats(x[:, 4*gi:4*gi+4])
            mats.append(torch.einsum('bf,fc->bc', f, self.w[gi]).reshape(-1,2,2))
        prod = mats[-1]
        for gi in range(self.n_groups-2, -1, -1):
            prod = prod @ mats[gi]
        return prod.reshape(-1,4)

class MLPBaseline(nn.Module):
    def __init__(self, in_dim=8, h=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,h), nn.ReLU(),
                                 nn.Linear(h,h), nn.ReLU(), nn.Linear(h,4))
    def forward(self, x): return self.net(x)

def n_params(m): return sum(p.numel() for p in m.parameters())

def train_eval(net, Xtr, Ytr, Xte, Yte, lr=1e-2, epochs=3000):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad(); fro_loss(net(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    with torch.no_grad():
        return fro_rel(net(Xte), Yte).mean().item()

def run(ctor, lr, n_seeds, epochs):
    vals = []
    for s in range(SEED_START, SEED_START + n_seeds):
        Xtr, Ytr = make_gl2_data(800, seed=s)
        Xte, Yte = make_gl2_data(4000, seed=s+100)
        torch.manual_seed(100+s)
        vals.append(train_eval(ctor(), Xtr, Ytr, Xte, Yte, lr, epochs))
    return vals

# ============================================================
print("\n" + "="*72)
print("E21 / GL+(2,R) support discovery: prune the 24-feature dictionary")
print("="*72)

# references
reg('E21:structured (6 analytic feats)', run(lambda: GLStructured(), 1e-2, NSEEDS, EPOCHS),
    params=n_params(GLStructured()))
reg('E21:dense (24 feats)', run(lambda: GLDict(), 1e-2, NSEEDS, EPOCHS),
    params=n_params(GLDict()))
reg('E21:MLP h=8', run(lambda: MLPBaseline(8,8), 3e-3, NSEEDS, EPOCHS), params=n_params(MLPBaseline(8,8)))

# pruning curve
BUDGETS = [2, 3, 4, 5, 6, 8, 10, 12, 24]
print("\n--- pruning curve (keep top-m features by group norm, retrain) ---")
curve = {m: [] for m in BUDGETS}
selected_labels = {}
for s in range(SEED_START, SEED_START + NSEEDS):
    Xtr, Ytr = make_gl2_data(800, seed=s)
    Xte, Yte = make_gl2_data(4000, seed=s+100)
    # 1) train the full dictionary
    torch.manual_seed(100+s)
    dense = GLDict().to(device)
    opt = torch.optim.Adam(dense.parameters(), lr=1e-2)
    for ep in range(EPOCHS):
        opt.zero_grad(); fro_loss(dense(Xtr), Ytr).backward()
        torch.nn.utils.clip_grad_norm_(dense.parameters(), 1.0); opt.step()
    with torch.no_grad():
        w = dense.w.detach()
        score = w.norm(dim=-1).sum(0)              # (24,) over outputs and groups
        order = torch.argsort(score, descending=True).cpu().tolist()
    if s == SEED_START:
        selected_labels = {m: [FEATURE_LABELS[i] for i in order[:m]] for m in BUDGETS}
        top6 = selected_labels[6]
        print(f"  seed0 top-6 selected: {top6}")
        print(f"  analytic six        : {sorted(ANALYTIC)}")
        print(f"  top-6 == analytic?  : {set(top6) == ANALYTIC}")
    # 2) retrain for each budget
    for m in BUDGETS:
        keep = set(order[:m])
        net = GLDict().to(device)
        with torch.no_grad():
            for gi in range(net.n_groups):
                for f in range(NF):
                    net.mask[gi, f, 0] = 1.0 if f in keep else 0.0
        torch.manual_seed(100+s)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for ep in range(EPOCHS):
            opt.zero_grad(); fro_loss(net(Xtr), Ytr).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad():
            curve[m].append(fro_rel(net(Xte), Yte).mean().item())

for m in BUDGETS:
    reg(f'E21:pruned m={m}', curve[m], kept=selected_labels.get(m, [])[:m])

print("\n--- selected support at m=6 (seed0) ---")
reg_text('E21:selected@6', ", ".join(selected_labels.get(6, [])))

print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=1, default=float))
print("\nE21 COMPLETE")
