#!/usr/bin/env python3
"""E39 -- sample efficiency of tuned opponents (fairness audit for the 32x claim).

Curves at N = 50..1600 (5 seeds, 3000 epochs, 4000 test): LawKAN node+KAN,
tuned MLP-6D h=16 (lr 1e-2), tuned MLP-Quat h=64 (lr 3e-3).
"""
import os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
def _find_repo(start):
    d = start
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, 'repro')):
        d = os.path.dirname(d)
    return d
_REPO = _find_repo(_HERE)
for _p in (_HERE, _REPO):
    if _p not in sys.path: sys.path.insert(0, _p)
import E31_node_isolation as E
from E38_tuned_opponents import MLP6D
from repro.dump import save_dump

SIZES = (50, 100, 200, 400, 800, 1600)
CELLS = [
    ('LawKAN', lambda: E.NodeKANEdge(2, K=1), 1e-2),
    ('MLP6D16_tuned', lambda: MLP6D(4, h=16), 1e-2),
    ('MLPQuat64_tuned', lambda: E.MLPQuatHead(4, h=64), 3e-3),
]


def main():
    res = {}
    for name, ctor, lr in CELLS:
        for n in SIZES:
            vals = [E.train_eval(ctor, s, lr, epochs=3000, ntr=n, nte=4000) for s in range(5)]
            res[f'E39:N{n}:{name}'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                           per_seed=vals, n=n, lr=lr)
            print(f'{name:16s} N={n:4d} {np.mean(vals):.3f} ± {np.std(vals):.3f}', flush=True)
    save_dump(res, __file__, seeds=5, epochs=3000)


if __name__ == '__main__':
    main()
