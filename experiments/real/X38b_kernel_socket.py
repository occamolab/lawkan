#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X38b_kernel_socket.py — Kinova: screw edge + kernel-only psi vs pure screw edge.
(registered before run)

Motivation. X38 gave the 2x2 on Kinova for the two-slot head:
  {reduced, per-joint screw} base x {psi-off, two-slot}.
On Kinova the classical screw edge with the exact law (psi=0) reproduces the
analytical estimator (1.978 vs reference 1.98 = parity by construction), while the
two-slot psi HURTS on top of it (2.397; +0.209 on the healthy seeds, p=0.036). The
one cell not yet measured is the OTHER head: the local KERNEL (X32). On the reduced
base the kernel was the only head below the dense screw (Table 6: 1.914 < 1.98).
So the open question: does the kernel add anything on top of the correctly
constructed (per-joint screw) edge?

Minimal probe (per seed, same splits as X38/X36; Kinova, 10 seeds):
  dense            : pure per-joint screw edge, no psi (= the analytical estimator);
  iso_pj_kernel    : the SAME frozen screw edge + kernel-only psi (X32 head).

Registered prediction (plan 2026-10-16):
  P-K1: iso_pj_kernel <= dense (the screw edge already carries the classical
        mismatch; a kernel residual must not lose -- if it is < dense there is a
        non-classical remainder the kernel captures).

Output: repro/json/X38b_kernel_socket.<ts>.json via repro/dump.
Run:  python X38b_kernel_socket.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import X10_fk_calib_real as X10            # noqa: E402
import MK1_mukca as MK                      # noqa: E402
import X32_kernel as X32                    # noqa: E402
import X36_recount_socket as X36            # noqa: E402
from repro.dump import save_dump            # noqa: E402

torch.set_num_threads(int(os.environ.get('X38_THREADS', '2')))

DT = torch.float64
RESULTS = {}
SOCKET = ['kinova']
MODELS = ('dense', 'iso_pj_kernel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default=','.join(SOCKET))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.steps = 1, 300
    t0 = time.time()

    for plat in args.platforms.split(','):
        xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
        q0, q1 = MK.load_sockets(plat)
        print(f'[X38b {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}, '
              f'{args.seeds} seeds', flush=True)
        rms = {k: [] for k in MODELS}
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

            # pure per-joint screw edge, no psi (= the analytical estimator)
            dn = X10.ScrewCalib(xi, M0, mode='dense')
            MK.fit(dn, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['dense'].append(MK.evaluate(dn, q0_tr, q1_tr, q0_te, q1_te)[0])
            dense_state = copy.deepcopy(dn.state_dict())

            # kernel centers/bw from the pair features (base-independent)
            probe = X32.build_iso(xi, M0, dense_state, s)
            feats = X36.collect_feats(probe.psi, lambda: probe(torch.tensor(
                np.concatenate([q0_tr, q1_tr], 0), dtype=DT)))
            centers, bw = X36.centers_bw(feats, args.k, s)

            # kernel-only psi on the frozen screw edge
            pk = X32.build_iso(xi, M0, dense_state, s, kernel=(centers, bw))
            X32.fit(pk, q0_tr, q1_tr, args.steps, args.lr, s, l2=0.0)
            rms['iso_pj_kernel'].append(MK.evaluate(pk, q0_tr, q1_tr, q0_te, q1_te)[0])

            print(f'[X38b {plat} seed {s}] dense {rms["dense"][-1]:.3f} | '
                  f'+kernel {rms["iso_pj_kernel"][-1]:.3f} mm (bw {bw:.3f}, n0 {n0})',
                  flush=True)

        for k in MODELS:
            a = np.array(rms[k])
            RESULTS[f'X38b:{plat}:{k}:rms_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                max=float(a.max()), min=float(a.min()),
                per_seed=[float(x) for x in a])
        a, b = np.array(rms['iso_pj_kernel']), np.array(rms['dense'])
        t, p = stats.ttest_rel(a, b)
        RESULTS[f'X38b:{plat}:paired:iso_pj_kernel_vs_dense'] = dict(
            diff=float(a.mean() - b.mean()), t=float(t), p=float(p))

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=args.steps,
              extra_meta={'head': 'kernel-only (X32, no global slot) on the screw edge',
                          'note': 'P-K1: screw edge + kernel <= pure screw edge (parity); '
                                  'the two-slot harm (X38) was its global slot'})


if __name__ == '__main__':
    main()
