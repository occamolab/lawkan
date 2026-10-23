#!/usr/bin/env python3
"""E40 -- matched Bayes floor for the E1 test draws.

The headline "above the floor" gap should use the floor evaluated on the same
test sets as E1 (N=4000, test seeds 100-104), not a separate N=2000 draw.
Averages the intrinsic (Karcher) floor over those five draws.
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
def _find_repo(start):
    d = start
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, 'repro')):
        d = os.path.dirname(d)
    return d
_REPO = _find_repo(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path: sys.path.insert(0, _p)
from E33_floor_reported_metric import posterior_draws, intrinsic_floor, ANG2
from repro.dump import save_dump


def main():
    floors = []
    for s in range(100, 105):
        v, w, Y = posterior_draws(4000, seed=s, noise=0.03, M=512, n=2,
                                  angles=ANG2, random_angles=False)
        floors.append(float(intrinsic_floor(v, w, Y)))
        print(f'[floor seed {s}] {floors[-1]:.4f}', flush=True)
    man = json.load(open(os.path.join(_REPO, 'repro', 'canonical.json')))
    run = json.load(open(os.path.join(_REPO, 'repro', 'json', man['run_main']['file'])))
    law = np.array(run['E1:QuatKAN K=1']['per_seed'])
    f = np.array(floors)
    res = {'E40:floor': dict(mean=float(f.mean()), std=float(f.std(ddof=1)),
                             per_seed=floors, N=4000, M=512),
           'E40:gap': dict(mean=float((law - f).mean()),
                           per_seed=[float(x) for x in (law - f)])}
    print('floor', res['E40:floor']['mean'], '+-', res['E40:floor']['std'],
          '| gap', res['E40:gap']['mean'], flush=True)
    save_dump(res, __file__, seeds=5, epochs=0)


if __name__ == '__main__':
    main()
