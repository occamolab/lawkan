#!/usr/bin/env python3
"""X61-all -- run X61_confirm for every config/seed and merge into one dump."""
import argparse, json, os, subprocess, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE) if os.path.basename(_HERE) == 'quatkan' else os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, _REPO):
    if p not in sys.path: sys.path.insert(0, p)
from repro.dump import save_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=200)
    a = ap.parse_args()
    runs = []
    for cfg in ('base', 'gonly', 'kernel'):
        for seed in [int(s) for s in a.seeds.split(',')]:
            out = subprocess.check_output(
                [sys.executable, os.path.join(_HERE, 'X61_confirm.py'),
                 '--cfg', cfg, '--seed', str(seed), '--N', str(a.N), '--epochs', str(a.epochs)],
                text=True)
            runs.append(json.loads(out.strip().splitlines()[-1]))
            print(f'{cfg} seed {seed} done', flush=True)
    save_dump({'runs': runs}, __file__, args=a, seeds=len(runs), epochs=a.epochs)


if __name__ == '__main__':
    main()
