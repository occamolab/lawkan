#!/usr/bin/env python3
"""X59 -- EMA over all training: averaged psi trace (beta=0.99, eval every 25).

Prediction (registered): the EMA trace has no spike at 500-1000 (max <= 5)
where the raw trajectory spikes -> transient sharpness is a trajectory property.
"""
import argparse, copy, os, sys
os.environ['FEAT'] = 'quat'
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_DATA = os.path.join(_HERE, 'data', 'calib_data.pickle')
if not os.path.exists(_DATA):
    _DATA = os.path.join(_REPO, 'data', 'calib_data.pickle')
import X7_robot_calibration as X7, X24_fr3_gate as X24, X33_fr3_twoslot as X33
import X34_fr3_kernel as X34, X7_gn_ls as GN
from scipy.cluster.vq import kmeans2
from repro.dump import save_dump
DT = torch.float64


def train_ema(m, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te, epochs, lr, seed,
              beta=0.99, ck=25):
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi.Wk.requires_grad_(True)
    m.psi.global_lin.weight.requires_grad_(True)
    m.psi.global_lin.bias.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0)
    ema = [p.detach().clone() for p in params]
    out = {}
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        qp, tp = m(q_tr)
        dp = (tp - ot_tr).norm(dim=-1)
        dot = (qp * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema, params):
                e.mul_(beta).add_(p, alpha=1 - beta)
        if ep % ck == 0:
            saved = [p.detach().clone() for p in params]
            with torch.no_grad():
                for p, e in zip(params, ema):
                    p.copy_(e)
            out[ep] = GN.eval_pos(m, q_te, oq_te, ot_te)[0]
            with torch.no_grad():
                for p, s in zip(params, saved):
                    p.copy_(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=2000)
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
    feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
    RES = {'cells': {}}
    for name, k in (('k=20', 20), ('k=n', len(q_tr))):
        cen, _ = kmeans2(feats.numpy(), k, minit='++', seed=a.seed)
        centers = torch.tensor(cen, dtype=DT)
        d = torch.cdist(centers, centers)
        md = float(torch.median(d[d > 0]))
        ts = X33.make_two_slot(exact, centers, md)
        r = train_ema(ts, q_tr, oq_tr, ot_tr, q_te, oq_te, ot_te,
                      a.epochs, 5e-3, a.seed)
        eps = sorted(r)
        vals = np.array([r[e] for e in eps])
        print(f'{name}: EMA trace max {vals.max():5.2f}  at {eps[int(vals.argmax())]}'
              f'  final {vals[-1]:.3f}', flush=True)
        print('   ' + ' '.join(f'{e}:{r[e]:.2f}' for e in eps), flush=True)
        RES['cells'][name] = {'max': float(vals.max()), 'at': int(eps[int(vals.argmax())]),
                              'final': float(vals[-1]),
                              'trace': {int(e): float(r[e]) for e in eps}}
    save_dump(RES, __file__, args=a, seeds=1, epochs=a.epochs)


if __name__ == '__main__':
    main()
