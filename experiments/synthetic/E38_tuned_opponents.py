#!/usr/bin/env python3
"""E38 -- fairness audit: tuned opponents and per-edge learning-rate checks.

Cells (same protocol as E1/E31: 800/4000, 3000 epochs, 5 seeds):
  LawKAN node+KAN at lr 1e-2 (headline), 3e-3, 1e-3
  MLP-Quat h=64 at its best lr (3e-3)      -> tuned strongest opponent
  MLP-6D h=16 at its best lr (1e-2)        -> tuned matrix opponent
  MLP h=4 (atan2) at its best lr (1e-2)
  node + MLP edge h=32 at lr 1e-3          -> per-edge lr for Table 2
"""
import os, sys
import numpy as np
import torch, torch.nn as nn

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
from repro.dump import save_dump


def rot6d_to_matrix(d6):
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


def mat_to_angles(R):
    v = R[..., 2]
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    return E.vec_to_angles(v)


class MLP6D(nn.Module):
    def __init__(self, in_phases=4, h=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases * 2, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 6))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return mat_to_angles(rot6d_to_matrix(self.net(f)))


class MLPComp(nn.Module):
    def __init__(self, in_phases=4, h=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_phases * 2, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 4))
    def forward(self, x):
        f = torch.cat([torch.cos(x), torch.sin(x)], -1)
        out = self.net(f)
        al = torch.atan2(out[:, 1], out[:, 0])
        be = torch.atan2(out[:, 3].abs(), out[:, 2])
        return torch.stack([al, be], -1)


CELLS = [
    ('LawKAN lr1e-2', lambda: E.NodeKANEdge(2, K=1), 1e-2),
    ('LawKAN lr3e-3', lambda: E.NodeKANEdge(2, K=1), 3e-3),
    ('LawKAN lr1e-3', lambda: E.NodeKANEdge(2, K=1), 1e-3),
    ('MLPQuat64 lr3e-3', lambda: E.MLPQuatHead(4, h=64), 3e-3),
    ('MLP6D16 lr1e-2', lambda: MLP6D(4, h=16), 1e-2),
    ('MLP4 lr1e-2', lambda: MLPComp(4, h=4), 1e-2),
    ('nodeMLP32 lr1e-3', lambda: E.NodeMLPEdge(2, h=32), 1e-3),
]


def main():
    res = {}
    for name, ctor, lr in CELLS:
        vals = [E.train_eval(ctor, s, lr, epochs=3000) for s in range(5)]
        res[f'E38:{name}'] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                  per_seed=vals, params=int(E.n_params(ctor)), lr=lr)
        print(f'{name:18s} lr={lr:g} {np.mean(vals):.4f} ± {np.std(vals):.4f} '
              f'params {E.n_params(ctor)}', flush=True)
    save_dump(res, __file__, seeds=5, epochs=3000)


if __name__ == '__main__':
    main()
