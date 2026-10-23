#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X38c_screw_heads.py — the remaining heads (global, kernel) on the frozen per-joint
screw edge, for the second column-group of Table 8. (registered before run)

Context. X38 gave the screw-base cells psi=0 (`dense`) and two-slot (`iso_pj`) on all
four MUKCa units; X38b gives kernel-only on Kinova. To mirror the reduced-edge group
of Table 8 (psi0 / global / kernel / two-slot) on the screw base, this probe fills
`global` (all platforms) and `kernel` (KUKA/Panda/FR3-unit; Kinova comes from X38b).

Setup (per seed, same splits as X38/X36; frozen fitted per-joint screw as the base):
  dense          : pure per-joint screw edge, no psi (the base / reference);
  iso_pj_global  : global head only (platform MLP, kernel slot frozen at zero);
  iso_pj_kernel  : kernel-only head (X32 kernel).

Registered predictions (plan 2026-10-16):
  P-C1: iso_pj_global <= dense on every platform (nesting -> no representational loss;
        as with the two-slot (X38), parity within noise expected);
  P-C2: iso_pj_kernel <= dense on every platform (kernel adds no loss on the nested base).

Output: repro/json/X38c_screw_heads.<ts>.json via repro/dump.
Run:  python X38c_screw_heads.py [--quick]
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
SOCKET = ['kuka14', 'panda_1', 'fr3', 'kinova']
MODELS = ('dense', 'iso_pj_global', 'iso_pj_kernel')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--platforms', default=','.join(SOCKET))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--fr3_seeds', type=int, default=5)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.fr3_seeds, args.steps = 1, 1, 300
    t0 = time.time()

    for plat in args.platforms.split(','):
        xi, M0 = MK.platform_tensors(MK.PLATFORMS[plat])
        q0, q1 = MK.load_sockets(plat)
        nseed = args.fr3_seeds if plat == 'fr3' else args.seeds
        print(f'[X38c {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}, '
              f'{nseed} seeds', flush=True)
        rms = {k: [] for k in MODELS}
        for s in range(nseed):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

            # per-joint screw edge (the node class of the analytical estimator)
            dn = X10.ScrewCalib(xi, M0, mode='dense')
            MK.fit(dn, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['dense'].append(MK.evaluate(dn, q0_tr, q1_tr, q0_te, q1_te)[0])
            dense_state = copy.deepcopy(dn.state_dict())

            # kernel centers/bw from the pair features (base-independent)
            probe = X32.build_iso(xi, M0, dense_state, s)
            feats = X36.collect_feats(probe.psi, lambda: probe(torch.tensor(
                np.concatenate([q0_tr, q1_tr], 0), dtype=DT)))
            centers, bw = X36.centers_bw(feats, args.k, s)

            # global-only head on the frozen screw base (kernel slot frozen at zero)
            pg = X36.build_twoslot_boxed(xi, M0, dense_state, s, centers, bw)
            pg.psi.Wk.requires_grad_(False)
            X36.fit_twoslot_socket(pg, q0_tr, q1_tr, args.steps, args.lr, s, l2=0.0)
            rms['iso_pj_global'].append(MK.evaluate(pg, q0_tr, q1_tr, q0_te, q1_te)[0])

            # kernel-only head on the frozen screw base
            pk = X32.build_iso(xi, M0, dense_state, s, kernel=(centers, bw))
            X32.fit(pk, q0_tr, q1_tr, args.steps, args.lr, s, l2=0.0)
            rms['iso_pj_kernel'].append(MK.evaluate(pk, q0_tr, q1_tr, q0_te, q1_te)[0])

            print(f'[X38c {plat} seed {s}] dense {rms["dense"][-1]:.3f} | '
                  f'+global {rms["iso_pj_global"][-1]:.3f} +kernel {rms["iso_pj_kernel"][-1]:.3f} mm '
                  f'(bw {bw:.3f}, n0 {n0})', flush=True)

        for k in MODELS:
            a = np.array(rms[k])
            RESULTS[f'X38c:{plat}:{k}:rms_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                max=float(a.max()), min=float(a.min()),
                per_seed=[float(x) for x in a])
        for k in ('iso_pj_global', 'iso_pj_kernel'):
            a, b = np.array(rms[k]), np.array(rms['dense'])
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'X38c:{plat}:paired:{k}_vs_dense'] = dict(
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p))

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=args.steps,
              extra_meta={'head': 'global / kernel on the frozen per-joint screw edge',
                          'note': 'P-C1/P-C2: fill the screw column-group of Table 8; '
                                  'two-slot (iso_pj) and psi0 (dense) come from X38, '
                                  'Kinova kernel from X38b'})


if __name__ == '__main__':
    main()
