# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E33c: numerically stable metric loss (P55, after P53 refutation)
#
# E33b showed that mean arccos(dot) did not improve LawKAN (2.7916 vs 2.7837)
# and hurt the MLP (3.56 vs 3.36) -- likely because d(acos)/dtheta ~ 1/theta is
# unbounded at zero. Here the SAME metric is written in stable form:
#   theta = 2 * asin(||v_pred - v_true|| / 2)  (bounded gradient),
# same minimizer. Models: LawKAN K=1 (54p), LawKAN K=3 (294p), MLP-Quat h=32.
# Protocol: comp2, 800/4000, 3000 epochs, seeds 0-4 (as E33b).
# Pre-registration: P55 in the experiment journal (2026-09-27).
#
# Run: python E33c_metric_loss_stable.py [--quick]
# Output: repro/json/E33c_metric_loss_stable.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from E31_node_isolation import (MLPQuatHead, NodeKANEdge, angles_to_vec, device,
                                geo_deg, make_comp2, paired_t)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

E31_DUMP = load_canon('e31')
E33_DUMP = load_canon('e33')

MODELS = [
    ('E33c:QuatKAN K=1, stable metric', lambda: NodeKANEdge(2, K=1), 1e-2),
    ('E33c:QuatKAN K=3, stable metric', lambda: NodeKANEdge(2, K=3), 1e-2),
    ('E33c:MLP-Quat h=32, stable metric', lambda: MLPQuatHead(4, h=32), 3e-3),
]


def stable_metric_loss(pred, target):
    vp = angles_to_vec(pred)
    vt = angles_to_vec(target)
    d = (vp - vt).norm(dim=-1)/2.0
    return 2.0*torch.asin(d.clamp(0, 1-1e-7)).mean()


def train_eval(ctor, seed, lr, epochs, ntr, nte):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    net = ctor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = stable_metric_loss(net(Xtr), Ytr)
        if not torch.isfinite(loss):
            return float('nan')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=0)
    args = ap.parse_args()
    epochs, ns = args.epochs, args.seeds
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, ntr, nte = 50, 1, 64, 128

    floor = E33_DUMP['E33:n2:intrinsic']['mean']
    onecos = E31_DUMP['E31:node+KAN (K=1)']['mean']
    print(f"E33c | stable metric loss | floor={floor:.3f} | 1-cos KAN={onecos:.3f}", flush=True)
    t0 = time.time()
    results = {}
    for name, ctor, lr in MODELS:
        vals = [train_eval(ctor, args.seed_start+s, lr, epochs, ntr, nte)
                for s in range(ns)]
        results[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                             per_seed=vals, lr=lr,
                             held_out_seeds=list(range(args.seed_start,
                                                      args.seed_start+ns)))
        print(f"[{name:36s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f}", flush=True)

    if not args.quick:
        k1 = results['E33c:QuatKAN K=1, stable metric']
        k3 = results['E33c:QuatKAN K=3, stable metric']
        mlp = results['E33c:MLP-Quat h=32, stable metric']
        t1, df1, p1 = paired_t(k1['per_seed'], k3['per_seed'])
        t2, df2, p2 = paired_t(k1['per_seed'], mlp['per_seed'])
        results['E33c:ttest:K1 vs K3'] = dict(text=f"t={t1:.2f}, df={df1}, p={p1:.2e}")
        results['E33c:ttest:K1 vs MLP'] = dict(text=f"t={t2:.2f}, df={df2}, p={p2:.2e}")
        p551 = k1['mean'] <= 2.40
        p552 = (k1['mean'] <= k3['mean']) and (k1['mean'] <= mlp['mean']) and (p2 < 0.05)
        p553 = k1['mean'] >= 2.70
        results['E33c:P55.1'] = dict(
            text=f"{'PASS' if p551 else 'FAIL'}: K=1 stable metric {k1['mean']:.4f} <= 2.40 "
                 f"(1-cos {onecos:.3f})")
        results['E33c:P55.2'] = dict(
            text=f"{'PASS' if p552 else 'FAIL'}: K=1 {k1['mean']:.4f} <= K=3 {k3['mean']:.4f} "
                 f"and <= MLP {mlp['mean']:.4f} (K1-vs-MLP p={p2:.2e})")
        results['E33c:P55.3'] = dict(
            text=f"{'PASS' if p553 else 'FAIL'}: stable-metric K=1 {k1['mean']:.4f} >= 2.70 "
                 f"-> the gap to the metric floor is a model/data property, not the loss")
        for k in ['E33c:P55.1', 'E33c:P55.2', 'E33c:P55.3']:
            print(f"  {k}: {results[k]['text']}", flush=True)

    results['E33c:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(floor=floor, one_cos_ref=onecos,
                              loss='2*asin(||v_pred-v_true||/2)'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
