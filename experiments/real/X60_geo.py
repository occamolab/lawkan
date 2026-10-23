#!/usr/bin/env python3
"""X60 -- X53-geo: RBF on the group (pair metric), not on feature coordinates.

Cells: metric='geo' (geodesic) and metric='chord' (quaternion chordal).
Global slot keeps the chart-dependent 30-dim Fourier basis (its nature, L2).
Centers = seeded subset of training input pairs (no k-means).
Two registered axes: floor (final <=1.15) and spikes (max stays ~13).
"""
import argparse, copy, os, sys
os.environ['FEAT'] = 'fourier'
import numpy as np, torch, torch.nn as nn
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA = os.path.join(_HERE, 'data', 'calib_data.pickle')
if not os.path.exists(_DATA):
    _DATA = os.path.join(_REPO, 'data', 'calib_data.pickle')
import X7_robot_calibration as X7, X24_fr3_gate as X24, X34_fr3_kernel as X34
import X7_gn_ls as GN
from repro.dump import save_dump
DT = torch.float64
L_MM = 50.0


def _mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def _rot_d2(qq, qc, metric):
    if metric == 'geo':
        rel = _mul(qc, qq * torch.tensor([1., -1., -1., -1.], dtype=qq.dtype))
        w = rel[..., 0].abs().clamp(max=1.0 - 1e-7)
        return (2.0 * torch.acos(w)) ** 2
    d1 = ((qq - qc) ** 2).sum(-1)
    d2 = ((qq + qc) ** 2).sum(-1)
    return torch.minimum(d1, d2)


def pair_d2(qa, ta, qb, tb, cq0, ct0, cq1, ct1, metric):
    da = (_rot_d2(qa[:, None, :], cq0[None, :, :], metric)
          + ((ta[:, None, :] - ct0[None, :, :]).norm(dim=-1) / L_MM) ** 2)
    db = (_rot_d2(qb[:, None, :], cq1[None, :, :], metric)
          + ((tb[:, None, :] - ct1[None, :, :]).norm(dim=-1) / L_MM) ** 2)
    return da + db


class GeoPsi(nn.Module):
    def __init__(self, pairs, bw, metric):
        super().__init__()
        self.global_lin = nn.Linear(30, 6).double()
        with torch.no_grad():
            self.global_lin.weight.zero_()
            self.global_lin.bias.zero_()
        self.metric = metric
        self.bw = float(bw)
        for nm, v in zip(('cq0', 'ct0', 'cq1', 'ct1'), pairs):
            self.register_buffer(nm, v.clone())
        self.Wk = nn.Parameter(torch.zeros(pairs[0].shape[0], 6, dtype=DT))

    def forward(self, A, B):
        qa, ta = A
        qb, tb = B
        f = X34.feats30(A, B).reshape(-1, 30)
        out = self.global_lin(f)
        d2 = pair_d2(qa.reshape(-1, 4), ta.reshape(-1, 3),
                     qb.reshape(-1, 4), tb.reshape(-1, 3),
                     self.cq0, self.ct0, self.cq1, self.ct1, self.metric)
        out = out + torch.exp(-d2 / (2 * self.bw ** 2)) @ self.Wk
        return 0.1 * out


def collect_pairs(m, q):
    got = []

    def hook(mod, inp, out):
        A, B = inp
        got.append(tuple(x.detach().cpu().reshape(-1, x.shape[-1])
                         for p in (A, B) for x in p))

    h = m.psi.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.tensor(q, dtype=DT))
    h.remove()
    return [torch.cat(x, 0) for x in zip(*got)]


def train(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr, seed, ck=25):
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi.Wk.requires_grad_(True)
    m.psi.global_lin.weight.requires_grad_(True)
    m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    trace = {}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        qp, tp = m(q_tr)
        dp = (tp - ot_tr).norm(dim=-1)
        dot = (qp * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % ck == 0:
            trace[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--k', type=int, default=20)
    a = ap.parse_args()
    q_np, obs_np = X7.load_philnad_dataset(_DATA)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1000 + a.seed).permutation(len(q_np))
    tr, te = perm[:a.N], perm[a.N:]
    q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
    q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
    exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, 4000, 5e-3, a.seed)
    pairs = collect_pairs(copy.deepcopy(exact), q_tr)
    n_pairs = pairs[0].shape[0]
    sel = np.random.default_rng(1000 + a.seed).permutation(n_pairs)[:a.k]
    sel = torch.tensor(sel)
    cpairs = [p[sel] for p in pairs]
    RES = {'cells': {}}
    for metric in ('geo', 'chord'):
        d2 = pair_d2(pairs[0], pairs[1], pairs[2], pairs[3],
                     pairs[0], pairs[1], pairs[2], pairs[3], metric)
        off = d2[~torch.eye(n_pairs, dtype=torch.bool)]
        bw = float(torch.sqrt(off.median()))
        m = copy.deepcopy(exact)
        m.psi = GeoPsi(cpairs, bw, metric)
        tr_ = train(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                    a.epochs, 5e-3, a.seed)
        eps = sorted(tr_)
        vals = np.array([tr_[e] for e in eps])
        print(f'{metric:5s} bw {bw:.4f}  max {vals.max():5.2f} at {eps[int(vals.argmax())]}'
              f'  final {vals[-1]:.3f}', flush=True)
        print('   ' + ' '.join(f'{e}:{tr_[e]:.2f}' for e in eps), flush=True)
        RES['cells'][metric] = {'bw': float(bw), 'max': float(vals.max()),
                                'at': int(eps[int(vals.argmax())]), 'final': float(vals[-1]),
                                'trace': {int(e): float(tr_[e]) for e in eps}}
    save_dump(RES, __file__, args=a, seeds=1, epochs=a.epochs)


if __name__ == '__main__':
    main()
