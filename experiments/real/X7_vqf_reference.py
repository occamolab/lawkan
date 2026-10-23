#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X7_vqf_reference.py — P-X7: classical AHRS fusion reference (VQF 6D) on BROAD.

Runs the VQF filter (Laidig & Seel, Information Fusion 2023; the same author as
the BROAD dataset) on gyro + accelerometer of each BROAD trial and evaluates the
identical window-relative geodesic metric used in X5, so the applied IMU table
gains one classical fusion reference row.  The input class differs from our
gyro-only probe (VQF uses gravity) — this is a reference/boundary row, not a
like-for-like competitor.  The per-trial-bias strapdown integrator is recomputed
on the same windows as a sanity check against the canonical X5 numbers.

Registered expectation P-X7: VQF is better at >= 1 s (gravity bounds the drift),
at the noise-limited floor at 0.07 s; on the disturbed group the gap shrinks
(structured disturbances reach the accelerometer).

Run:  python X7_vqf_reference.py --data_dir /path/to/broad/data_hdf5 [--groups undisturbed]
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
import vqf
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X5_broad_real_imu as X5      # noqa: E402
from repro.dump import save_dump
from repro.data_guard import skip_if_missing    # noqa: E402

RESULTS = {}
BLOCKS = {
    'a-short': dict(W=20, stride=20),
    'b2-035s': dict(W=100, stride=25),
    'b-1s': dict(W=286, stride=71),
    'c-5s': dict(W=1430, stride=143),
    'd-short': dict(W=20, stride=20),
    'd-035s': dict(W=100, stride=100),
}


def load_acc(path):
    with h5py.File(path, 'r') as f:
        return np.asarray(f['imu_acc'][()], dtype=np.float64)


def window_pairs(trial, W, stride, min_move_frac=0.5):
    """Same window-selection rules as X5.build_windows (indices only)."""
    quat = torch.from_numpy(trial['quat']).to(torch.float64)
    move = trial['move']
    N = len(quat)
    qn = quat.norm(dim=-1)
    valid = torch.isfinite(quat).all(dim=-1) & ((qn - 1.0).abs() < 1e-2)
    pairs = []
    for s in range(0, N - W + 1, stride):
        e = s + W
        if move[s:e].mean() < min_move_frac:
            continue
        if not (valid[s] and valid[e]):
            continue
        pairs.append((s, e))
    return pairs, quat


def run(name, data_dir, groups, W, stride):
    """Evaluate VQF 6D and the per-trial-bias strapdown on the *same* window set
    as X5: windows with bias subtracted, trial-level split 800/2000 (seed 12345)."""
    tag = '+'.join(groups)
    files = X5.list_broad_files(data_dir, groups)
    assert files, f'no trials for {groups}'
    per_trial = []                    # (name, pairs, T-id), bias-subtracted gyro, VQF quats
    for ti, (path, gr) in enumerate(files):
        tr = X5.load_broad_trial(path)
        fs = tr['fs']
        acc = load_acc(path)
        out = vqf.offlineVQF(tr['gyr'].astype(np.float64), acc,
                             np.full_like(acc, np.nan), 1.0 / fs)
        pairs, quat = window_pairs(tr, W, stride)
        if not pairs:
            continue
        bias, _ = X5.estimate_bias(tr['gyr'], tr['move'], margin_s=0.5, fs=fs)
        gb = torch.from_numpy(tr['gyr'] - bias.astype(np.float32)).to(torch.float64)
        per_trial.append(dict(ti=ti, name=tr['name'], pairs=pairs, quat=quat,
                              gb=gb, qv=out['quat6D'], fs=fs))
    assert per_trial, f'no windows for {groups}'

    T = torch.tensor([t['ti'] for t in per_trial for _ in t['pairs']])
    tr_idx, te_idx = X5.trial_split(T, 800, 2000)      # same split as the X5 table rows
    offsets, pos = [], 0
    for t in per_trial:
        offsets.append(pos)
        pos += len(t['pairs'])

    vqf_errs, stb_errs = [], []
    trial_v, trial_s = [], []
    for i in te_idx.tolist():
        k = 0
        while offsets[k + 1] <= i if k + 1 < len(offsets) else False:
            k += 1
        t = per_trial[k]
        s, e = t['pairs'][i - offsets[k]]
        gt = X5.quat_mul(X5.quat_conj(t['quat'][s]), t['quat'][e])
        vqf_errs.append(float(X5.geo_deg_q(
            X5.quat_mul(X5.quat_conj(torch.tensor(t['qv'][s], dtype=torch.float64)),
                        torch.tensor(t['qv'][e], dtype=torch.float64)).unsqueeze(0),
            gt.unsqueeze(0))[0]))
        q_stb = X5.strapdown_integrate(t['gb'][s:e].unsqueeze(0), 1.0 / t['fs'])
        stb_errs.append(float(X5.geo_deg_q(q_stb.cpu(), gt.unsqueeze(0))[0]))

    v = np.array(vqf_errs); s = np.array(stb_errs)
    tv, pv = stats.ttest_rel(v, s)
    RESULTS[f"X7:{tag}:{name}:VQF6D"] = dict(mean=float(v.mean()), std=float(v.std()),
                                       n=int(len(v)))
    RESULTS[f"X7:{tag}:{name}:Strapdown(bias)"] = dict(mean=float(s.mean()), std=float(s.std()),
                                                 n=int(len(s)))
    RESULTS[f"X7:{tag}:{name}:paired:VQF_vs_Strapdown"] = dict(
        diff=float(v.mean() - s.mean()), t=float(tv), p=float(pv))
    print(f"[X7 {tag} {name}] VQF6D {v.mean():.4f} deg  vs  strapdown(bias) {s.mean():.4f} "
          f"(test windows {len(v)}, paired p={pv:.2e})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--groups', default='disturbed')
    ap.add_argument('--blocks', default=None,
                    help='comma list; default: undisturbed -> a-short,b2-035s,b-1s,c-5s; '
                         'disturbed -> d-short,d-035s')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    groups = tuple(g.strip() for g in args.groups.split(',') if g.strip())
    if args.blocks:
        names = [b.strip() for b in args.blocks.split(',') if b.strip()]
    else:
        names = (['a-short'] if args.quick else
                 (['d-short', 'd-035s'] if 'disturbed' in groups
                  else ['a-short', 'b2-035s', 'b-1s', 'c-5s']))
    if args.quick:
        names = names[:1]
    t0 = time.time()
    for name in names:
        run(name, args.data_dir, groups, **BLOCKS[name])
    RESULTS['_meta'] = dict(wall_s=time.time() - t0, groups=list(groups),
                            filter='vqf.offlineVQF quat6D, mag=NaN')
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args)
    print(f'\nTotal wall time: {time.time()-t0:.1f} s; dump: {out}')


if __name__ == '__main__':
    main()
