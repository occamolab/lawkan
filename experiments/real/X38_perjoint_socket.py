#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X38_perjoint_socket.py — paradigm-consistent socket test with the FINAL two-slot
psi head (X36): embed the KNOWN per-joint constant-screw structure as the base
(the same class as the classical reference) and learn only the psi residual on
top; compare with the reduced-edge base used in Table 6. (registered before run)

Motivation. In the MUKCa socket protocol the classical estimator is the full
per-joint screw ("dense screw", 55p) — the known *form* of the kinematic
mismatch (a constant screw per joint). Table 6, however, uses the *reduced* base
(tool + joint biases, 13p) and lets psi learn the rest; where the remainder is
classical (FR3-unit) psi cannot represent it and "loses" to the dense screw.
By our own paradigm ("impose the known structure, learn only the remainder") the
consistent base is the per-joint screw itself: then the model nests the classical
estimator (psi = 0 is available) and cannot lose on representational grounds.

Head: the FINAL two-slot psi (X36 TwoSlotPsiSocket = platform global head + local
kernel), matching the `two-slot` row of Table 6; dose from X36.dose_for(n_tr).

Setup (per platform, per seed, same splits as MK1/X36):
  dense     : full per-joint screw (ScrewCalib mode='dense'), fit to socket loss;
  iso_pj    : two-slot psi on the FROZEN fitted dense base (per-joint edge + psi);
  iso_edgeb : two-slot psi on the FROZEN reduced base (Table 6's LawKAN base).

Registered predictions (plan 2026-10-15):
  P-NEST-1: iso_pj <= dense on every platform (nesting: parity where the
            mismatch is classical, never a representational loss);
  P-NEST-2: on FR3-unit iso_pj < iso_edgeb (embedding the known per-joint
            structure removes the Table-6 boundary);
  P-NEST-3: on KUKA/Panda iso_edgeb remains competitive (the reduced base is a
            data-efficient alternative where psi can cover the gap).

Output: repro/json/X38_perjoint_socket.<ts>.json via repro/dump.
Run:  python X38_perjoint_socket.py [--quick]
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
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X10_fk_calib_real as X10            # noqa: E402
import MK1_mukca as MK                      # noqa: E402
import X32_kernel as X32                    # noqa: E402
import X36_recount_socket as X36            # noqa: E402
from repro.dump import save_dump            # noqa: E402

torch.set_num_threads(int(os.environ.get('X38_THREADS', '2')))

DT = torch.float64
RESULTS = {}
SOCKET = ['kuka14', 'panda_1', 'fr3', 'kinova']

def fit_twoslot_seq(m, q0_tr, q1_tr, steps, lr, seed):
    """Table 4 / X36seq two-slot config: sequential-on-residual, dose 0."""
    for p in m.parameters():
        p.requires_grad_(False)
    m.psi.Wk.requires_grad_(True)
    X36.fit_twoslot_socket(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0)
    m.psi.Wk.requires_grad_(False)
    for p in m.psi.global_parameters():
        p.requires_grad_(True)
    X36.fit_twoslot_socket(m, q0_tr, q1_tr, steps, lr, seed, l2=0.0)
    return m


MODELS = ('edge_b', 'dense', 'iso_pj', 'iso_edgeb')


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
        print(f'[X38 {plat}] sockets {q0.shape[0]}/{q1.shape[0]}, dof {xi.shape[0]}, '
              f'{nseed} seeds', flush=True)
        rms = {k: [] for k in MODELS}
        for s in range(nseed):
            rng = np.random.default_rng(1000 + s)
            p0, p1 = rng.permutation(len(q0)), rng.permutation(len(q1))
            n0, n1 = int(0.7 * len(q0)), int(0.7 * len(q1))
            q0_tr, q0_te = q0[p0[:n0]], q0[p0[n0:]]
            q1_tr, q1_te = q1[p1[:n1]], q1[p1[n1:]]

            # full per-joint screw (classical same-class estimator)
            dn = X10.ScrewCalib(xi, M0, mode='dense')
            MK.fit(dn, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['dense'].append(MK.evaluate(dn, q0_tr, q1_tr, q0_te, q1_te)[0])
            dense_state = copy.deepcopy(dn.state_dict())

            # reduced-edge base (Table 6's LawKAN base)
            eb = X10.ScrewCalib(xi, M0, mode='dense')
            eb.dxi.requires_grad_(False)
            MK.fit(eb, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['edge_b'].append(MK.evaluate(eb, q0_tr, q1_tr, q0_te, q1_te)[0])
            edge_state = copy.deepcopy(eb.state_dict())

            # kernel centers/bw from the pair features (base-independent)
            probe = X32.build_iso(xi, M0, edge_state, s)
            feats = X36.collect_feats(probe.psi, lambda: probe(torch.tensor(
                np.concatenate([q0_tr, q1_tr], 0), dtype=DT)))
            centers, bw = X36.centers_bw(feats, args.k, s)

            # psi (two-slot) on the frozen PER-JOINT (dense) base -- paradigm-consistent
            pj = X36.build_twoslot_boxed(xi, M0, dense_state, s, centers, bw)
            fit_twoslot_seq(pj, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['iso_pj'].append(MK.evaluate(pj, q0_tr, q1_tr, q0_te, q1_te)[0])

            # psi (two-slot) on the frozen REDUCED base -- Table 6 control
            ie = X36.build_twoslot_boxed(xi, M0, edge_state, s, centers, bw)
            fit_twoslot_seq(ie, q0_tr, q1_tr, args.steps, args.lr, s)
            rms['iso_edgeb'].append(MK.evaluate(ie, q0_tr, q1_tr, q0_te, q1_te)[0])

            print(f'[X38 {plat} seed {s}] edge_b {rms["edge_b"][-1]:.3f} '
                  f'dense {rms["dense"][-1]:.3f} | +psi: pj {rms["iso_pj"][-1]:.3f} '
                  f'edgeb {rms["iso_edgeb"][-1]:.3f} mm '
                  f'(bw {bw:.3f}, n0 {n0})', flush=True)

        for k in MODELS:
            a = np.array(rms[k])
            RESULTS[f'X38:{plat}:{k}:rms_mm'] = dict(
                mean=float(a.mean()), std=float(a.std()),
                max=float(a.max()), min=float(a.min()),
                per_seed=[float(x) for x in a])
        for k, ref in (('dense', 'edge_b'), ('iso_pj', 'dense'),
                       ('iso_edgeb', 'edge_b'), ('iso_pj', 'iso_edgeb'),
                       ('iso_pj', 'edge_b'), ('iso_edgeb', 'dense')):
            a, b = np.array(rms[k]), np.array(rms[ref])
            t, p = stats.ttest_rel(a, b)
            RESULTS[f'X38:{plat}:paired:{k}_vs_{ref}'] = dict(
                diff=float(a.mean() - b.mean()), t=float(t), p=float(p))

    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    save_dump(RESULTS, __file__, quick=args.quick, args=args,
              seeds=args.seeds, epochs=args.steps,
              extra_meta={'head': 'two-slot (X36, seq + dose 0 = Table-6 config)',
                          'note': 'P-NEST: per-joint-edge base nests the classical '
                                  'screw (no loss); embedding the known structure '
                                  'removes the FR3-unit boundary'})


if __name__ == '__main__':
    main()
