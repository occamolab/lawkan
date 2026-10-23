#!/usr/bin/env python3
"""X53 -- socket convergence: does psi overshoot on the FR3 unit / Kinova?

Same socket task as X36 (MUKCa), checkpoint grid, exact edge + isolated psi
(global head and two-slot) for the FR3 unit and Kinova -- the two platforms
where the correction does not win. The question: is the failure a
convergence/optimization effect (overshoot like the FR3 arm) or a form limit?
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import X36_recount_socket as X36          # noqa: E402
import X32_kernel as X32                  # noqa: E402
import X10_fk_calib_real as X10           # noqa: E402
import MK1_mukca as MK                    # noqa: E402
from repro.dump import save_dump          # noqa: E402

DT = torch.float64
CK = list(range(100, 4001, 100))
RESULTS = {}


def fit_ckpt(m, q0, q1, q0_te, q1_te, steps, lr, seed, ck, l2=0.0):
    torch.manual_seed(seed)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0)
    q0t = torch.tensor(q0, dtype=DT)
    q1t = torch.tensor(q1, dtype=DT)
    out = {}
    for ep in range(1, steps + 1):
        opt.zero_grad()
        loss = MK.socket_loss(m, q0t, q1t)
        if l2:
            loss = loss + l2 * sum((q ** 2).sum() for q in m.psi.global_parameters())
        loss.backward()
        opt.step()
        sched.step()
        if ep in ck:
            out[ep] = MK.evaluate(m, q0, q1, q0_te, q1_te)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default='fr3,kinova')
    ap.add_argument('--seeds', type=int, default=0,
                    help='0 = per-platform (FR3 unit 5, else 10), as in Table 4')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.platforms, args.seeds, args.steps = 'fr3', 1, 300
    t0 = time.time()
    for plat in args.platforms.split(','):
        n_seeds = args.seeds or (5 if plat == 'fr3' else 10)
        xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
        q0, q1 = MK.load_sockets(plat)
        res = {k: [] for k in ('global', 'two_slot')}
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

            ex = X10.ScrewCalib(xi, M0, mode='dense')
            ex.dxi.requires_grad_(False)
            MK.fit(ex, q0_tr, q1_tr, args.steps, 1e-2, s)
            edge_state = copy.deepcopy(ex.state_dict())

            # isolated global psi on the frozen exact edge
            gl = X32.build_iso(xi, M0, edge_state, s)
            res['global'].append(fit_ckpt(gl, q0_tr, q1_tr, q0_te, q1_te,
                                          args.steps, args.lr, s, CK))
            # two-slot on the same frozen edge
            probe = X32.build_iso(xi, M0, edge_state, s)
            feats = X36.collect_feats(probe.psi, lambda: probe(torch.tensor(
                np.concatenate([q0_tr, q1_tr], 0), dtype=DT)))
            centers, bw = X36.centers_bw(feats, 20, s)
            ts = X36.build_twoslot_boxed(xi, M0, edge_state, s, centers, bw)
            res['two_slot'].append(fit_ckpt(ts, q0_tr, q1_tr, q0_te, q1_te,
                                            args.steps, args.lr, s, CK))
            print(f'[X53 {plat} seed {s}] done', flush=True)
        for k, runs in res.items():
            for ep in CK:
                vals = [r[ep] for r in runs if r is not None and ep in r]
                if vals:
                    RESULTS[f'X53:{plat}:{k}:ep{ep}'] = dict(
                        mean=float(np.mean(vals)), per_seed=[float(v) for v in vals])
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args, seeds=args.seeds,
                  extra_meta={'note': 'X53 socket convergence (FR3 unit, Kinova)'})
    print('dump ->', p)


if __name__ == '__main__':
    main()
