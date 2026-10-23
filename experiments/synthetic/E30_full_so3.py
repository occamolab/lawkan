# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E30: the E1 opponent matrix on a FULL SO(3) readout
# (insight #2: replace the 2-DOF probe readout by the full rotation).
#
# Same task as E1 (composition n=2, fixed angles 0.7/1.1 rad, random axes, input
# noise 0.03 rad), but the TARGET is the composed rotation q_true (a full element
# of SO(3)), and the target carries a small rotation noise in the Lie algebra.
# Loss = 1 - |<q_pred, q_true>| (sign-invariant); metric = SO(3) geodesic (deg).
# Models now output the full rotation, so the comparison is apples-to-apples.
#
# Run: python E30_full_so3.py
# Output: repro/json/e30.20260926-fullso3.json
# ============================================================
import torch, torch.nn as nn, math, time, json, os
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
dtype = torch.float32
EPOCHS = int(os.environ.get("E30_EPOCHS", 3000))

# ---------------- geometry ----------------
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1); w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)
def expmap(v):
    nv = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat([torch.cos(nv), torch.sin(nv)*v/nv], -1)
def angles_to_vec(a):
    al, be = a[..., 0], a[..., 1]
    return torch.stack([torch.sin(be)*torch.cos(al), torch.sin(be)*torch.sin(al), torch.cos(be)], -1)
def vec_to_angles(v):
    al = torch.atan2(v[..., 1], v[..., 0])
    be = torch.acos(torch.clamp(v[..., 2], -1+1e-6, 1-1e-6))
    return torch.stack([al, be], -1)
def rot6d_to_matrix(d6):
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1/a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = a2 - (b1*a2).sum(-1, keepdim=True)*b1
    b2 = b2/b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)
def mat_to_quat(R):
    """Stable matrix -> quaternion (Shepperd's largest-component branch)."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    w = torch.sqrt((1.0 + tr).clamp_min(1e-12)) / 2
    q_w = torch.stack([
        w,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * w.clamp_min(1e-8)),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * w.clamp_min(1e-8)),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * w.clamp_min(1e-8)),
    ], -1)

    x = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-12)) / 2
    q_x = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * x.clamp_min(1e-8)),
        x,
        (R[:, 0, 1] + R[:, 1, 0]) / (4 * x.clamp_min(1e-8)),
        (R[:, 0, 2] + R[:, 2, 0]) / (4 * x.clamp_min(1e-8)),
    ], -1)

    y = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(1e-12)) / 2
    q_y = torch.stack([
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * y.clamp_min(1e-8)),
        (R[:, 0, 1] + R[:, 1, 0]) / (4 * y.clamp_min(1e-8)),
        y,
        (R[:, 1, 2] + R[:, 2, 1]) / (4 * y.clamp_min(1e-8)),
    ], -1)

    z = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(1e-12)) / 2
    q_z = torch.stack([
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * z.clamp_min(1e-8)),
        (R[:, 0, 2] + R[:, 2, 0]) / (4 * z.clamp_min(1e-8)),
        (R[:, 1, 2] + R[:, 2, 1]) / (4 * z.clamp_min(1e-8)),
        z,
    ], -1)

    use_x = (m00 >= m11) & (m00 >= m22) & (tr <= 0)
    use_y = (m11 > m00) & (m11 >= m22) & (tr <= 0)
    use_z = (m22 > m00) & (m22 > m11) & (tr <= 0)
    q = torch.where((tr > 0)[:, None], q_w, q)
    q = torch.where(use_x[:, None], q_x, q)
    q = torch.where(use_y[:, None], q_y, q)
    q = torch.where(use_z[:, None], q_z, q)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(q[:, :1] < 0, -q, q)
def quat_geo_loss(qp, qt):
    return (1 - (qp*qt).sum(-1).abs().clamp(0, 1)).mean()
def quat_geo_deg(qp, qt):
    c = (qp*qt).sum(-1).abs().clamp(0, 1-1e-6)
    return 2*torch.acos(c)*(180.0/math.pi)

# ---------------- data (E1 task, full-rotation target) ----------------
ANG2 = (0.7, 1.1); NOISE = 0.03
def make_compn_data(N, n=2, noise=NOISE, seed=1, angles=None, rnoise=NOISE):
    g = torch.Generator().manual_seed(seed)
    angles = angles if angles is not None else (ANG2 if n == 2 else (0.7, 1.1, 0.9))
    axes = [torch.randn(N, 3, generator=g).to(device) for _ in range(n)]
    axes = [a/a.norm(dim=-1, keepdim=True) for a in axes]
    X = torch.cat([vec_to_angles(a) for a in axes], dim=-1) + noise*torch.randn(N, 2*n, generator=g).to(device)
    prod = expmap((angles[-1]/2)*axes[-1])                       # q_n
    for i in range(n-2, -1, -1):
        prod = quat_mul(prod, expmap((angles[i]/2)*axes[i]))     # prod (*) q_i
    dr = rnoise*torch.randn(N, 3, generator=g).to(device)        # target rotation noise
    qt = quat_mul(prod, expmap(dr))
    return X, qt

# ---------------- models (full-rotation readout) ----------------
def mlp(sizes, in_phases):
    net = nn.Sequential()
    return net

class GroupedQuatExpNet(nn.Module):        # LawKAN-SU(2)
    def __init__(self, n_groups=2, K=1):
        super().__init__()
        self.n_groups, self.K = n_groups, K
        self.w = nn.Parameter(torch.randn(n_groups, (2*K+1)**2, 3)*0.1)
    def group_feats(self, a):
        al, be = a[:,0], a[:,1]
        fa=[torch.ones_like(al)]; fb=[torch.ones_like(be)]
        for m in range(1, self.K+1):
            fa += [torch.cos(m*al), torch.sin(m*al)]; fb += [torch.cos(m*be), torch.sin(m*be)]
        return torch.stack([x*y for x in fa for y in fb], dim=-1)
    def forward(self, x):
        qs=[]
        for gi in range(self.n_groups):
            f = self.group_feats(x[:, 2*gi:2*gi+2]); v = torch.einsum('bf,fc->bc', f, self.w[gi])
            nv = v.norm(dim=-1, keepdim=True); qs.append(torch.cat([torch.cos(nv), torch.sin(nv)*v/(nv+1e-8)], -1))
        prod = qs[-1]
        for gi in range(self.n_groups-2, -1, -1): prod = quat_mul(prod, qs[gi])
        return prod/prod.norm(dim=-1, keepdim=True).clamp_min(1e-8)

class _RPMGQuat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_raw):
        r = q_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8); q_hat = q_raw/r
        ctx.save_for_backward(q_hat, r); return q_hat
    @staticmethod
    def backward(ctx, g_out):
        q_hat, r = ctx.saved_tensors
        return (g_out - (g_out*q_hat).sum(-1, keepdim=True)*q_hat)/r

def head_net(in_dim, h, out_dim):
    return nn.Sequential(nn.Linear(in_dim, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out_dim))

class MLPQuat(nn.Module):
    def __init__(self, h=32, rpmg=False):
        super().__init__(); self.net = head_net(8, h, 4); self.rpmg = rpmg
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1); q = self.net(f)
        q = _RPMGQuat.apply(q) if self.rpmg else q/q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return q

class MLP6Dhead(nn.Module):
    def __init__(self, h=32):
        super().__init__(); self.net = head_net(8, h, 6)
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1); return mat_to_quat(rot6d_to_matrix(self.net(f)))

class MLP9DSVD(nn.Module):
    def __init__(self, h=16):
        super().__init__(); self.net = head_net(8, h, 9)
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        U,S,Vh = torch.linalg.svd(self.net(f).view(-1,3,3)); det = torch.det(U@Vh)
        D = torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], -1))
        return mat_to_quat(U@D@Vh)

class MLPLieHead(nn.Module):
    def __init__(self, h=32):
        super().__init__(); self.net = head_net(8, h, 3)
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1); return expmap(self.net(f))

# ---------------- train / eval ----------------
def train_eval(ctor, seed, lr, epochs=EPOCHS, ntr=800, nte=4000):
    torch.manual_seed(100+seed)
    Xtr, Ytr = make_compn_data(ntr, seed=seed)
    Xte, Yte = make_compn_data(nte, seed=seed+100)
    m = ctor().to(device); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        m.train(); opt.zero_grad(); loss = quat_geo_loss(m(Xtr), Ytr); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss): return float('nan'), sum(p.numel() for p in m.parameters())
    m.eval()
    with torch.no_grad(): err = float(quat_geo_deg(m(Xte), Yte).mean().item())
    return err, sum(p.numel() for p in m.parameters())

def main():
    t0=time.time(); RESULTS={}
    # Bayes floor: geodesic of the target noise itself (predicting q = center)
    ref = []
    for seed in range(100,105):
        Xte, Yte = make_compn_data(4000, seed=seed)
        q0 = torch.tensor([1.,0,0,0], device=device).expand(Xte.shape[0],4)
        # floor = E[geodesic(center, noisy target)] ; center ~ the noise-free true rotation
        xc, qc = make_compn_data(4000, seed=seed, rnoise=0.0)
        ref.append(float(quat_geo_deg(qc, Yte).mean().item()))
    RESULTS['E30:bayes_floor_deg'] = float(np.mean(ref))
    print(f"[Bayes floor, full SO(3)] {np.mean(ref):.3f} deg")

    configs = [
        ('E30:QuatKAN-SU(2)', lambda: GroupedQuatExpNet(2,1), 1e-2),
        ('E30:MLP-Quat h=32', lambda: MLPQuat(32, False), 3e-3),
        ('E30:MLP-Quat h=32 + RPMG', lambda: MLPQuat(32, True), 3e-3),
        ('E30:MLP-6D h=32', lambda: MLP6Dhead(32), 3e-3),
        ('E30:MLP-9D-SVD h=16', lambda: MLP9DSVD(16), 3e-3),
        ('E30:Lie head h=32', lambda: MLPLieHead(32), 3e-3),
    ]
    for name, ctor, lr in configs:
        vals, ps = [], 0
        for seed in range(5):
            e, ps = train_eval(ctor, seed, lr); vals.append(e)
        RESULTS[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=vals,
                             params=int(ps), lr=lr, held_out_seeds=[0,1,2,3,4])
        print(f"[{name:26s}] geodesic {np.mean(vals):.3f} ±{np.std(vals):.3f} deg | params {ps}")
    RESULTS['E30:wall_s']=time.time()-t0
    save_dump(RESULTS, __file__, seeds=5, epochs=EPOCHS)
    print("\n===== SUMMARY-JSON ====="); print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
