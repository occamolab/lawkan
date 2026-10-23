#!/usr/bin/env python3
"""X47 — socket: joint vs sequential two-slot, matched seeds.

Runs X36_recount_socket twice (seq off/on) with the same seeds and dose 0,
then compares the two-slot heads per platform (paired t-test) and the
global-slot collapse counts. Registered before the run: X33 (armcal) showed
joint ~ seq; on Kinova joint may collapse.

Run:  python X47_socket_joint_vs_seq.py   (writes repro/json/X47_*.json)
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(HERE))
SEEDS = '0,1,2,3,4,5,6,7,8,9'
FR3_SEEDS = '0,1,2,3,4'
PLATS = 'kuka14,panda_1,kinova,fr3'


def newest_x36(after):
    f = [x for x in glob.glob(os.path.join(_REPO, 'repro', 'json', 'X36_*.json'))
         if 'smoke' not in x and os.path.getmtime(x) >= after]
    if not f:
        raise RuntimeError('no new X36 dump found')
    return sorted(f, key=os.path.getmtime)[-1]


def run(seq):
    cmd = [sys.executable, os.path.join(HERE, 'X36_recount_socket.py'),
           '--platforms', PLATS, '--socket-seeds', SEEDS,
           '--fr3-seeds', FR3_SEEDS, '--dose-override', '0',
           '--skip-abb', '--heads', 'twoslot']
    if seq:
        cmd.append('--seq')
    t0 = time.time()
    print('[X47] running', 'sequential' if seq else 'joint', flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)
    return newest_x36(t0)


def seeds(d, key):
    return np.asarray(d[key]['per_seed'], dtype=float)


def main():
    joint = json.load(open(run(False)))
    seq = json.load(open(run(True)))
    out = {'X47:protocol': dict(
        platforms=PLATS, seeds=SEEDS, fr3_seeds=FR3_SEEDS,
        dose='0',
        note='X47: socket joint vs sequential two-slot, matched seeds')}
    for p in PLATS.split(','):
        a = seeds(joint, f'X36:{p}:twoslot:rms_mm')
        b = seeds(seq, f'X36:{p}:twoslot:rms_mm')
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        t, pv = stats.ttest_rel(b, a)          # sequential vs joint
        ec = np.asarray(joint[f'X36:{p}:edge_b:rms_mm']['per_seed'])[:n]
        out[f'X47:{p}'] = dict(
            joint_mean=float(a.mean()), seq_mean=float(b.mean()),
            diff=float((b - a).mean()), t=float(t), p=float(pv),
            n=int(n),
            joint_collapse=int((a > ec).sum()),
            seq_collapse=int((b > ec).sum()))
        print(f'[X47 {p}] joint {a.mean():.3f} seq {b.mean():.3f} '
              f'diff {b.mean()-a.mean():+.3f} p={pv:.3f} '
              f'collapse {int((a>ec).sum())}/{int((b>ec).sum())}', flush=True)
    path = os.path.join(_REPO, 'repro', 'json',
                        'X47_socket_joint_vs_seq.' + __import__('datetime').datetime.now()
                        .strftime('%Y%m%d-%H%M%S') + '.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('[X47] dump ->', path, flush=True)


if __name__ == '__main__':
    main()
