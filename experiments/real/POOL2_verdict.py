#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
POOL2_verdict.py — pooled cross-platform claim under the FINAL heads (registered
2026-10-15, before the recompute). Recomputes the behavioral P-POOL / P-POOL-EXT
claim (not new data).

Cells (final recommended head per setting):
  ABB N=80   : global (MLP) isolated vs exact      (RC1)
  KUKA front : two-slot isolated vs edge_b         (X36)
  Kinova     : local kernel isolated vs edge_b     (X36; kernel-only = goff)

Kinova extension gate: paired vs edge_b, p<0.05 and diff<0.
primary  = Fisher {ABB, KUKA}         (df=4)
extended = Fisher {ABB, KUKA, Kinova} (df=6)

Predictions:
  POOL-2a: Kinova gate PASSES under the kernel head (old MLP gate failed, p=0.26);
  POOL-2b: extended pooled met -> "across three platforms";
  POOL-2c: primary (ABB+KUKA) stays met.

Output: repro/json/POOL2_verdict.<ts>.json via repro/dump.
Run:  python POOL2_verdict.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.dump import save_dump  # noqa: E402

RESULTS = {}
JSON = os.path.join(_REPO, 'repro', 'json')


def latest(prefix):
    fs = sorted(Path(JSON).glob(prefix + '*.json'))
    if not fs:
        raise FileNotFoundError(prefix)
    return json.loads(fs[-1].read_text(encoding='utf-8'))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    return float(a.mean() - b.mean()), float(t), float(p)


def fisher(ps):
    chi2 = -2.0 * float(np.sum(np.log(np.asarray(ps, float))))
    return chi2, float(stats.chi2.sf(chi2, df=2 * len(ps)))


def main():
    x36 = latest('X36_recount_socket.')
    rc1 = latest('RC1_robotcali.')

    # ABB N=80: global (isolated) vs exact
    d_abb, t_abb, p_abb = paired(rc1['RC1:N80:isolated']['per_seed'],
                                 rc1['RC1:N80:exact']['per_seed'])
    print(f'[POOL2] ABB N=80 (global vs exact): {d_abb:+.3f} mm, p={p_abb:.4g}')

    # KUKA front: two-slot vs edge_b
    d_k, t_k, p_k = paired(x36['X36:kuka14:twoslot:rms_mm']['per_seed'],
                           x36['X36:kuka14:edge_b:rms_mm']['per_seed'])
    print(f'[POOL2] KUKA front (two-slot vs edge_b): {d_k:+.3f} mm, p={p_k:.4g}')

    # Kinova: kernel (goff) vs edge_b
    d_v, t_v, p_v = paired(x36['X36:kinova:goff:rms_mm']['per_seed'],
                           x36['X36:kinova:edge_b:rms_mm']['per_seed'])
    gate = bool(p_v < 0.05 and d_v < 0)
    print(f'[POOL2] Kinova (kernel vs edge_b): {d_v:+.3f} mm, p={p_v:.4g} '
          f'-> gate {"PASS" if gate else "FAIL"}')

    # primary + extended Fisher
    chi2, pp = fisher([p_abb, p_k])
    primary = bool(p_abb < 0.05 and p_k < 0.05 and pp < 0.01)
    print(f'[POOL2] primary Fisher (ABB,KUKA): chi2={chi2:.2f}, p={pp:.4g} '
          f'-> {"MET" if primary else "NOT met"}')

    chi2e = pe = None
    extended = bool(primary and gate)
    if extended:
        chi2e, pe = fisher([p_abb, p_k, p_v])
        print(f'[POOL2] extended Fisher (+Kinova): chi2={chi2e:.2f}, p={pe:.4g} '
              f'-> MET (across three platforms)')
    else:
        print('[POOL2] extended not claimed')

    RESULTS['POOL2:abb'] = dict(diff=d_abb, p=p_abb)
    RESULTS['POOL2:kuka'] = dict(diff=d_k, p=p_k)
    RESULTS['POOL2:kinova'] = dict(diff=d_v, p=p_v, gate=gate)
    RESULTS['POOL2:primary'] = dict(chi2=chi2, p=pp, met=primary)
    RESULTS['POOL2:extended'] = dict(claimed=extended, chi2=chi2e, p=pe,
                                     composition=['abb_N80', 'kuka_front', 'kinova_front'],
                                     heads=['global', 'two-slot', 'kernel'])
    print('dump:', save_dump(RESULTS, __file__))


if __name__ == '__main__':
    main()
