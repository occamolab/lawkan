# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# E23 — SE(3) FEATURE ABLATION (registered blind test RT-1)
#
# Registered prediction (the experiment journal, commit d2af621, BEFORE this run).
# The exact SE(3) edge target is xi = log(g) with
#     v = (1 - d th^2) p  -  (th/2)(a x p)  +  d th^2 a (a . p)
# so the analytic support is {a, p, a x p, a(a.p)}. Ablating feature groups must
# produce positive representational floors; both non-linear groups are necessary.
#
# Predictions:
#   (1) err(S4=full) < 1e-2
#   (2) err(S1={a,p}) > 0.3
#   (3) err(S3={a,p,a(a.p)}) > err(S2={a,p,a x p}) > 0, with ratio in [4,12]
#   (4) err(S2) scales linearly with sigma_p: err(S2;1.0)/err(S2;0.5) = 2.0 +- 0.5
#
# Run:  python E23_se3_feature_ablation.py --seed_start 20 --seeds 5
# ============================================================
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--seed_start', type=int, default=20)
ap.add_argument('--epochs', type=int, default=3000)
args, _unknown = ap.parse_known_args()
NSEEDS = 2 if args.quick else args.seeds
SEED_START = args.seed_start
EPOCHS = 300 if args.quick else args.epochs

RESULTS = {}
def reg(key, vals, **meta):
    vals=[float(v) for v in vals]
    RESULTS[key]=dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4e} ±{np.std(vals):.2e}  per-seed: {[f'{v:.2e}' for v in vals]}")
def reg_text(key, text):
    RESULTS[key]=dict(text=text); print(f"[{key}] {text}")
def paired_t(a,b):
    d=np.array(a)-np.array(b); n=len(d)
    t=d.mean()/(d.std(ddof=1)/math.sqrt(n)+1e-12)
    try:
        from scipy import stats; p=2*stats.t.sf(abs(t),df=n-1)
    except Exception: p=float('nan')
    return float(t), n-1, float(p)

# --- SE(3) geometry (same as E22) ---
def hat(w):
    B=w.shape[:-1]; W=torch.zeros(*B,3,3,device=w.device,dtype=w.dtype)
    W[...,0,1]=-w[...,2]; W[...,0,2]=w[...,1]; W[...,1,0]=w[...,2]; W[...,1,2]=-w[...,0]; W[...,2,0]=-w[...,1]; W[...,2,1]=w[...,0]
    return W
def so3_coeffs(th):
    small=th<1e-4; ths=torch.where(small,torch.ones_like(th),th)
    a=torch.where(small,1-th**2/6,torch.sin(ths)/ths)
    b=torch.where(small,0.5-th**2/24,(1-torch.cos(ths))/ths**2)
    return a,b
def se3_exp(xi):
    w,v=xi[...,:3],xi[...,3:]; th=w.norm(dim=-1,keepdim=True)
    W=hat(w); W2=torch.matmul(W,W); a,b=so3_coeffs(th)
    I=torch.eye(3,device=xi.device,dtype=xi.dtype).expand_as(W2)
    R=I+a[...,None]*W+b[...,None]*W2
    small=th<1e-4; ths=torch.where(small,torch.ones_like(th),th)
    c=torch.where(small,1/6-th**2/120,(ths-torch.sin(ths))/ths**3)
    V=I+b[...,None]*W+c[...,None]*W2
    return R, torch.einsum('bij,bj->bi',V,v)
def se3_compose(R1,p1,R2,p2):
    return torch.matmul(R2,R1), torch.einsum('bij,bj->bi',R2,p1)+p2
PROBES=torch.tensor([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],device=device)/math.sqrt(3)
def transform_points(R,p):
    return torch.einsum('bij,kj->bki',R,PROBES.to(R.dtype))+p.unsqueeze(1)
def point_loss(pred,target): return ((pred-target)**2).sum(-1).mean()
def point_err(pred,target): return (pred-target).norm(dim=-1).mean().item()

ANG2=(0.7,1.1)
def make_se3_data(N, n=2, seed=1, sigma_p=0.5):
    g=torch.Generator().manual_seed(seed); axes=[]; ps=[]
    for _ in range(n):
        a=torch.randn(N,3,generator=g); a=a/a.norm(dim=-1,keepdim=True); axes.append(a.to(device))
        ps.append((torch.randn(N,3,generator=g)*sigma_p).to(device))
    Rs,pv=[],[]
    for i in range(n):
        th=torch.full((N,),float(ANG2[i]),device=device)
        xi=torch.cat([th.unsqueeze(-1)*axes[i], torch.zeros_like(axes[i])],-1)
        R,_=se3_exp(xi); Rs.append(R); pv.append(ps[i])
    R,p=Rs[-1],pv[-1]
    for i in range(n-2,-1,-1): R,p=se3_compose(Rs[i],pv[i],R,p)
    Y=transform_points(R,p)
    ph=[torch.stack([torch.atan2(a[:,1],a[:,0]), torch.acos(a[:,2].clamp(-1+1e-6,1-1e-6))],-1) for a in axes]
    X=torch.cat([torch.cat([ph[i],ps[i]],-1) for i in range(n)],-1)
    return X.float().to(device), Y.float().to(device)

def axis_feats(ph):
    al,be=ph[:,0],ph[:,1]
    return torch.stack([torch.sin(be)*torch.cos(al), torch.sin(be)*torch.sin(al), torch.cos(be)],-1)
def all_feats(ph,p):
    a=axis_feats(ph); axp=torch.cross(a,p,dim=-1); adp=a*(a*p).sum(-1,keepdim=True)
    return torch.cat([a,p,axp,adp],-1)     # 12

SUBSETS = {
    'S1={a,p}':        [0,1,2,3,4,5],
    'S2={a,p,axp}':    [0,1,2,3,4,5,6,7,8],
    'S3={a,p,adp}':    [0,1,2,3,4,5,9,10,11],
    'S4=full':         [0,1,2,3,4,5,6,7,8,9,10,11],
}
class SE3Subset(nn.Module):
    def __init__(self, idxs, n_groups=2):
        super().__init__(); self.idxs=idxs; self.n_groups=n_groups
        self.w=nn.Parameter(torch.randn(n_groups, len(idxs), 6)*0.1)
    def forward(self, x):
        mats=[]
        for gi in range(self.n_groups):
            ph=x[:,5*gi:5*gi+2]; p=x[:,5*gi+2:5*gi+5]
            f=all_feats(ph,p)[:, self.idxs]
            xi=torch.einsum('bf,fc->bc',f,self.w[gi])
            mats.append(se3_exp(xi))
        R,t=mats[-1]
        for gi in range(self.n_groups-2,-1,-1): R,t=se3_compose(mats[gi][0],mats[gi][1],R,t)
        return transform_points(R,t)

def run(idxs, sigma_p):
    vals=[]
    for s in range(SEED_START, SEED_START+NSEEDS):
        Xtr,Ytr=make_se3_data(800,seed=s,sigma_p=sigma_p)
        Xte,Yte=make_se3_data(4000,seed=s+100,sigma_p=sigma_p)
        torch.manual_seed(100+s)
        net=SE3Subset(idxs).to(device); opt=torch.optim.Adam(net.parameters(),lr=1e-2)
        for _ in range(EPOCHS):
            opt.zero_grad(); point_loss(net(Xtr),Ytr).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
        with torch.no_grad(): vals.append(point_err(net(Xte),Yte))
    return vals

print("\nE23 / SE(3) feature-group ablation (noiseless)")
res={}
for name,idxs in SUBSETS.items():
    res[name]=run(idxs, 0.5); reg(f'E23:{name}:sp0.5', res[name])
# sigma_p scaling of S2
res_s2={0.25:run(SUBSETS['S2={a,p,axp}'],0.25), 1.00:run(SUBSETS['S2={a,p,axp}'],1.00)}
reg('E23:S2:sp0.25', res_s2[0.25]); reg('E23:S2:sp1.0', res_s2[1.00])

print("\n--- checks vs registered predictions ---")
s4 = RESULTS['E23:S4=full:sp0.5']['mean']
s1 = RESULTS['E23:S1={a,p}:sp0.5']['mean']
s2 = RESULTS['E23:S2={a,p,axp}:sp0.5']['mean']
s3 = RESULTS['E23:S3={a,p,adp}:sp0.5']['mean']
reg_text('E23:check:1', f"S4 mean {s4:.2e} (<1e-2? {s4 < 1e-2})")
reg_text('E23:check:2', f"S1 mean {s1:.3f} (>0.3? {s1 > 0.3})")
r32 = s3 / s2
reg_text('E23:check:3', f"S3/S2 = {r32:.2f} (predicted [4,12])")
# Registered P38 ratio is 1.0/0.5 (2.0 +/- 0.5); the sp0.25 cell is diagnostic only.
ratio = RESULTS['E23:S2:sp1.0']['mean'] / RESULTS['E23:S2={a,p,axp}:sp0.5']['mean']
reg_text('E23:check:4', f"S2 err(1.0)/err(0.5) = {ratio:.2f} (predicted 2.0 +/- 0.5)")
t, df, p = paired_t(res['S3={a,p,adp}'], res['S2={a,p,axp}'])
reg_text('E23:S3_vs_S2', f"t={t:.2f}, p={p:.2e}")

print("\n===== SUMMARY-JSON =====")
print(json.dumps(RESULTS, indent=1, default=float))
print("\nE23 COMPLETE")
