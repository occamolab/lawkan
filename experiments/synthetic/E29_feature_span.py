# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E29: feature-span / tensor-extension theorem, confirming experiment.
#
# Claim (feature span): with an exact group-law node, the architecture represents
# the composition EXACTLY on the product set iff log(chi) in rowspan(F). Corollaries:
#   (K=1)   fixed angles:  F = {1,cos a,sin a}@{1,cos b,sin b} contains (theta/2)a;
#   (tensor) VARIABLE angles: adding the theta channel, F = {1,theta}@{1,cos a,sin a}@{1,cos b,sin b}
#            contains (theta/2)a for arbitrary theta  -> exact for arbitrary elements;
#   (floor) any missing feature group => positive floor independent of training.
#
# Task: n group elements, each supplied as (alpha, beta, theta). Target = composed
# rotation R = prod_j exp(theta_j a_j), a_j = (sin b cos a, sin b sin a, cos b).
# Readout is FULL SO(3) (two orthonormal probes), not a single probe direction.
#
# Edges compared (per group, Linear(feat_dim -> 3)):
#   tensor      {1,theta}@{1,cos a,sin a}@{1,cos b,sin b}   18   (exact for varying theta)
#   fixed       {1,cos a,sin a}@{1,cos b,sin b}              9   (exact only for fixed theta)
#   no_beta     {1,theta}@{1,cos a,sin a}                    6
#   no_alpha    {1,theta}@{1,cos b,sin b}                    6
#
# Run: python E29_feature_span.py
# Output: repro/json/e29.20260926-featurespan.json
# ============================================================
import torch, torch.nn as nn, time, json, os, math
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
EPOCHS = int(os.environ.get('E29_EPOCHS', 2000))
E1 = torch.tensor([1., 0., 0.], device=device)
E2 = torch.tensor([0., 1., 0.], device=device)

# ---------------- geometry ----------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)
def expmap(v):                       # v (N,3) -> unit quaternion, half-angle
    nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = nv/2
    return torch.cat([torch.cos(half), torch.sin(half)*v/nv], -1)
def rotate(q, v):
    u, w = q[..., 1:], q[..., :1]
    t = 2.0*torch.cross(u, v, dim=-1)
    return v + w*t + torch.cross(u, t, dim=-1)
def axis_of(a, b):                   # alpha,beta -> unit axis
    return torch.stack([torch.sin(b)*torch.cos(a), torch.sin(b)*torch.sin(a), torch.cos(b)], -1)
def rot_cols(R):                     # R (N,3,3) -> (p1,p2) = (R e1, R e2)
    return R[:, :, 0], R[:, :, 1]
def R_from_cols(p1, p2):             # rebuild orthonormal R from two columns, Gram-Schmidt
    e1 = p1/p1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    u2 = p2 - (e1*p2).sum(-1, keepdim=True)*e1
    e2 = u2/u2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)
def geo_deg(Rp, Rt):
    M = torch.bmm(Rp.transpose(1, 2), Rt)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cosang = ((tr - 1.0)/2.0).clamp(-1+1e-6, 1-1e-6)
    return torch.acos(cosang)*(180.0/math.pi)

# ---------------- feature maps (per group element: alpha,beta,theta) ----------------
def feats_tensor(x):                 # (N,3) -> (N,18):  t in {1,th}, A in {1,ca,sa}, B in {1,cb,sb}
    a, b, th = x[:,0], x[:,1], x[:,2]
    ca, sa, cb, sb = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b)
    T = [torch.ones_like(th), th]; A = [torch.ones_like(a), ca, sa]; B = [torch.ones_like(b), cb, sb]
    return torch.stack([t*Aij*Bkl for t in T for Aij in A for Bkl in B], -1)   # order: t,A,B
def feats_fixed(x):                  # (N,9): drop theta
    a, b = x[:,0], x[:,1]
    ca, sa, cb, sb = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b)
    A = [torch.ones_like(a), ca, sa]; B = [torch.ones_like(b), cb, sb]
    return torch.stack([Aij*Bkl for Aij in A for Bkl in B], -1)
def feats_no_beta(x):                # (N,6): {1,th}@{1,ca,sa}
    a, th = x[:,0], x[:,2]; ca, sa = torch.cos(a), torch.sin(a)
    return torch.stack([t*Aj for t in [torch.ones_like(th), th] for Aj in [torch.ones_like(a), ca, sa]], -1)
def feats_no_alpha(x):               # (N,6): {1,th}@{1,cb,sb}
    b, th = x[:,1], x[:,2]; cb, sb = torch.cos(b), torch.sin(b)
    return torch.stack([t*Bj for t in [torch.ones_like(th), th] for Bj in [torch.ones_like(b), cb, sb]], -1)

EDGES = {'tensor': (feats_tensor,18), 'fixed': (feats_fixed,9),
         'no_beta': (feats_no_beta,6), 'no_alpha': (feats_no_alpha,6)}

# ---------------- frozen data ----------------
def make_data(N, n, seed):
    g = torch.Generator().manual_seed(seed)
    al = (torch.rand(N, n, generator=g)*2*math.pi - math.pi).to(device)   # alpha ~ U[-pi,pi]
    be = torch.acos(1 - 2*torch.rand(N, n, generator=g)).to(device)       # beta area-uniform [0,pi]
    th = (torch.rand(N, n, generator=g)*2.3 + 0.2).to(device)             # theta ~ U[0.2,2.5]
    x = torch.stack([al, be, th], -1).to(device)                      # (N,n,3)
    q = torch.zeros(N, 4, device=device); q[:, 0] = 1.0
    for j in range(n):
        v = (th[:, j:j+1]/2) * axis_of(al[:, j], be[:, j])            # (theta/2) a
        q = quat_mul(q, expmap(v))
    R = torch.stack([rotate(q, E1.expand(N,3)), rotate(q, E2.expand(N,3)),
                     torch.cross(rotate(q, E1.expand(N,3)), rotate(q, E2.expand(N,3)), dim=-1)], -1)
    # target readout = full rotation columns (p1,p2,p3)
    return x, R

# ---------------- model ----------------
class QuatKAN(nn.Module):
    """per-group edge (features -> so(3)) + EXACT node (exp + Hamilton) + full-SO(3) readout."""
    def __init__(self, edge='tensor', n=3, shared=False):
        super().__init__()
        fn, d = EDGES[edge]
        self.fn, self.d, self.n, self.edge = fn, d, n, edge
        if shared:
            self.W = nn.Parameter(torch.zeros(d, 3).normal_(std=0.05))
        else:
            self.W = nn.ParameterList([nn.Parameter(torch.zeros(d,3).normal_(std=0.05)) for _ in range(n)])
    def forward(self, x):
        N = x.shape[0]
        q = torch.zeros(N, 4, device=x.device); q[:, 0] = 1.0
        for j in range(self.n):
            W = self.W if isinstance(self.W, nn.Parameter) else self.W[j]
            v = self.fn(x[:, j, :]) @ W
            q = quat_mul(q, expmap(v))
        p1 = rotate(q, E1.to(q.dtype).expand(N,3)); p2 = rotate(q, E2.to(q.dtype).expand(N,3))
        p3 = torch.cross(p1, p2, dim=-1)
        return torch.stack([p1, p2, p3], -1)
    def construct(self):
        """analytic tensor-extension weights: v=(theta/2)a via the three theta*axis features."""
        assert not isinstance(self.W, nn.Parameter), "construct() only for per-group edges"
        # feature index (t,A,B): idx = t*9 + A*3 + B ; need t=1 and A=ca/B=sb(14), A=sa/B=sb(17), A=1/B=cb(10)
        for j in range(self.n):
            self.W[j].data.zero_()
            self.W[j].data[14, 0] = 0.5   # theta * cos a * sin b  -> v_x
            self.W[j].data[17, 1] = 0.5   # theta * sin a * sin b  -> v_y
            self.W[j].data[10, 2] = 0.5   # theta * 1    * cos b   -> v_z

# ---------------- train / eval ----------------
def train_eval(ctor, n, seed, epochs=EPOCHS, lr=1e-2, ntr=2000, nte=5000):
    torch.manual_seed(100+seed)
    model = ctor().to(device)
    xtr, Rtr = make_data(ntr, n, seed)
    xte, Rte = make_data(nte, n, seed+1000)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        Rp = model(xtr)
        loss = (1 - ((Rp.transpose(1,2).bmm(Rtr))[:,0,0]+(Rp.transpose(1,2).bmm(Rtr))[:,1,1]+(Rp.transpose(1,2).bmm(Rtr))[:,2,2]-1)/2).mean()
        if not torch.isfinite(loss): return float('nan')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        return float(geo_deg(model(xte), Rte).mean().item())

def main():
    t0 = time.time(); RESULTS = {}
    SEEDS = list(range(50, 55))

    # (1) constructed-weights exactness (float64), tensor edge
    m = QuatKAN('tensor', n=3).double().to(device); m.construct()
    xte, Rte = make_data(2000, 3, 1050)
    with torch.no_grad():
        Rp = m(xte.double())
        maxerr = float((Rp - Rte.double()).abs().max().item())
    RESULTS['E29:constructed_tensor_maxerr'] = maxerr
    print(f"[constructed tensor] max|R-R_true| = {maxerr:.2e} (exact if ~0)")

    # (2) trained edges, n=3
    for edge in ['tensor','fixed','no_beta','no_alpha']:
        vals = [train_eval(lambda: QuatKAN(edge, n=3), 3, s) for s in SEEDS]
        RESULTS[f'E29:trained_{edge}_n3'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                                 per_seed=vals, params=sum(p.numel() for p in QuatKAN(edge,3).parameters()),
                                                 held_out_seeds=SEEDS)
        print(f"[trained {edge:8s} n=3] geodesic {np.mean(vals):.3f} ±{np.std(vals):.3f} deg")

    # (3) blind floors grow with n: fixed edge, n=2,3,4 (tensor edge as reference)
    for n in [2,3,4]:
        for edge in ['tensor','fixed']:
            vals = [train_eval(lambda: QuatKAN(edge, n=n), n, s) for s in SEEDS]
            RESULTS[f'E29:{edge}_n{n}'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                               per_seed=vals, params=sum(p.numel() for p in QuatKAN(edge,n).parameters()))
            print(f"[{edge:6s} n={n}] geodesic {np.mean(vals):.3f} ±{np.std(vals):.3f} deg")

    RESULTS['E29:wall_s'] = time.time()-t0
    save_dump(RESULTS, __file__, seeds=SEEDS, epochs=EPOCHS)
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
