#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
POOL_verdict.py — P-POOL / P-POOL-EXT verdicts (registered 2026-10-04, before the runs).

Primary pooled criterion (unchanged):
  ABB N=80 paired isolated vs exact (RC1)  +  KUKA front paired isolated vs edge_b (MK1)
  per-platform p < 0.05 with the registered direction, Fisher over the two p-values,
  pooled p < 0.01  ->  "across two robot platforms".

Extension rule P-POOL-EXT (registered):
  Kinova front is an *extension cell*.  It enters the *extended* pooled claim only if it
  passes its own per-platform gate (paired isolated vs edge_b, p < 0.05, direction psi+).
  Extended pooled = Fisher over {ABB N=80, KUKA front, Kinova front} (chi2, df=6) and is
  always reported with its explicit composition.  KUKA cross-view cells are within-robot
  replication: reported for information, never pooled.

Dumps are resolved per key across all non-smoke dumps (latest dump containing the key),
so runs that cover disjoint platform sets are combined correctly.

Run:  python POOL_verdict.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.dump import save_dump  # noqa: E402

RESULTS = {}
JSON = os.path.join(_REPO, 'repro', 'json')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


_MANIFEST = None


def canonical(block):
    """Resolve a block through repro/canonical.json (never by 'latest filename')."""
    global _MANIFEST
    if _MANIFEST is None:
        with open(os.path.join(_REPO, 'repro', 'canonical.json'), encoding='utf-8') as fh:
            _MANIFEST = json.load(fh)
    entry = _MANIFEST.get(block)
    if not entry or not entry.get('file'):
        raise FileNotFoundError(f'canonical.json: no entry for {block!r}')
    path = os.path.join(JSON, entry['file'])
    if not os.path.exists(path):
        raise FileNotFoundError(f'canonical dump missing: {entry["file"]}')
    return _load(path)


def canonical_with_key(block, key):
    d = canonical(block)
    if key not in d:
        raise FileNotFoundError(f'{block}: key {key!r} missing in canonical dump')
    return d


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_rel(a, b)
    return float(a.mean() - b.mean()), float(t), float(p)


def fisher(ps):
    chi2 = -2.0 * float(np.sum(np.log(np.asarray(ps, float))))
    return chi2, float(stats.chi2.sf(chi2, df=2 * len(ps)))


def main():
    # ---------------- ABB ----------------
    rc = canonical('rc1')
    abb_sizes = sorted({int(k.split(':')[1][1:]) for k in rc
                        if k.startswith('RC1:N') and k.endswith(':exact')})
    abb = {}
    for n in abb_sizes:
        iso = rc[f'RC1:N{n}:isolated']['per_seed']
        ex = rc[f'RC1:N{n}:exact']['per_seed']
        d, t, p = paired(iso, ex)
        abb[n] = (d, p)
        print(f'[POOL] ABB N={n}: isolated-exact {d:+.3f} mm, p={p:.4f}')
    abb_p = abb[80][1] if 80 in abb else abb[abb_sizes[-1]][1]

    # ---------------- KUKA front ----------------
    mk = canonical_with_key('mk1', 'MK1:kuka14:iso:rms_mm')
    iso_k = mk['MK1:kuka14:iso:rms_mm']['per_seed']
    eb_k = mk['MK1:kuka14:edge_b:rms_mm']['per_seed']
    d_k, t_k, p_k = paired(iso_k, eb_k)
    print(f'[POOL] KUKA front: isolated-edge_b {d_k:+.3f} mm, p={p_k:.4g}')

    # ---------------- Kinova (extension cell) ----------------
    ext = {}
    try:
        mkk = canonical_with_key('kinova', 'MK1:kinova:iso:rms_mm')
        iso_v = mkk['MK1:kinova:iso:rms_mm']['per_seed']
        eb_v = mkk['MK1:kinova:edge_b:rms_mm']['per_seed']
        dn_v = mkk['MK1:kinova:dense:rms_mm']['per_seed']
        d_v, t_v, p_v = paired(iso_v, eb_v)
        dvd, tvd, pvd = paired(iso_v, dn_v)
        gate = (p_v < 0.05) and (d_v < 0)
        ext = dict(diff=d_v, p=p_v, dense_diff=dvd, dense_p=pvd,
                   gate=bool(gate))
        print(f'[POOL] Kinova: isolated-edge_b {d_v:+.3f} mm, p={p_v:.4g} '
              f'(vs dense {dvd:+.3f}, p={pvd:.4g}) -> gate {"PASS" if gate else "FAIL"}')
    except FileNotFoundError as e:
        print(f'[POOL] Kinova: not available yet ({e})')

    # ---------------- within-robot replication (informational) ----------------
    views = {}
    for tag in ('kuka14_left_back', 'kuka14_raised_left'):
        try:
            mkv = canonical_with_key('mk1_views', f'MK1:{tag}:iso:rms_mm')
            iv = mkv[f'MK1:{tag}:iso:rms_mm']['per_seed']
            ev = mkv[f'MK1:{tag}:edge_b:rms_mm']['per_seed']
            dv, tv, pv = paired(iv, ev)
            views[tag] = dict(diff=dv, p=pv, wins=bool(dv < 0 and pv < 0.05))
            print(f'[POOL] {tag}: isolated-edge_b {dv:+.3f} mm, p={pv:.4g} '
                  f'({"win" if dv < 0 else "harm"})')
        except FileNotFoundError:
            print(f'[POOL] {tag}: not available yet')

    # ---------------- primary pooled (registered) ----------------
    chi2, pooled_p = fisher([abb_p, p_k])
    met = (abb_p < 0.05) and (p_k < 0.05) and (pooled_p < 0.01)
    print(f'[POOL] primary Fisher (ABB N=80, KUKA front): chi2={chi2:.2f}, '
          f'p={pooled_p:.4g} -> criterion {"MET" if met else "NOT met"}')

    # ---------------- extended pooled (P-POOL-EXT) ----------------
    ext_claimed = bool(met and ext.get('gate'))
    if ext_claimed:
        chi2e, pooled_pe = fisher([abb_p, p_k, ext['p']])
        print(f'[POOL] extended Fisher (+Kinova): chi2={chi2e:.2f}, p={pooled_pe:.4g} '
              f'-> extended criterion MET')
    else:
        chi2e, pooled_pe = None, None
        print('[POOL] extended pooled not claimed '
              + ('(primary not met)' if not met else '(Kinova gate failed)'))

    RESULTS['POOL:abb'] = {f'N{n}': dict(diff=abb[n][0], p=abb[n][1]) for n in abb_sizes}
    RESULTS['POOL:kuka14'] = dict(diff=d_k, p=p_k)
    RESULTS['POOL:fisher'] = dict(chi2=float(chi2), p=pooled_p, met=bool(met))
    RESULTS['POOL:kinova_ext'] = ext
    RESULTS['POOL:views_replication'] = views
    RESULTS['POOL:extended'] = dict(claimed=ext_claimed, chi2=chi2e, p=pooled_pe,
                                    composition=['abb_N80', 'kuka14_front', 'kinova_front'])
    out = save_dump(RESULTS, __file__)
    print('dump:', out)


if __name__ == '__main__':
    main()
