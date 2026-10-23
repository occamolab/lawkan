# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E32: minimal opponent tuning (Tier 4, reviewer #1)
#
# Grid: MLPQuatHead h in {16,32,64} x lr in {1e-3,3e-3,1e-2}, 3 seeds,
# E1 protocol (comp2 0.7/1.1, sigma=0.03, 800/4000, 3000 epochs).
# Report the BEST cell; compare against the LawKAN reference (E1, 5 seeds)
# and the single-exp control MLP-Lie h=32 (E11, 10 seeds).
# Pre-registration: P50 in the experiment journal (2026-09-27).
#
# Run: python E32_opponent_tuning.py [--quick]
# Output: repro/json/E32_opponent_tuning.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from E31_node_isolation import (MLPQuatHead, geo_loss, geo_deg, make_comp2,
                                paired_t)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# references from canonical dumps repro/json/run_main.20260927-165803.json
QK_REF = 2.7836864948272706      # E1:QuatKAN K=1 (5 seeds)
LIE_REF = 3.2792752385139465     # E11:MLP-Lie h=32 (10 seeds)

GRID_H = [16, 32, 64]
GRID_LR = [1e-3, 3e-3, 1e-2]


def train_eval(ctor, seed, lr, epochs, ntr, nte):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    net = ctor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = geo_loss(net(Xtr), Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        return geo_deg(net(Xte), Yte).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed_start', type=int, default=0)
    args = ap.parse_args()
    epochs, ns = args.epochs, args.seeds
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, ntr, nte = 50, 1, 64, 128

    print(f"Device: {device} | E32 opponent tuning | epochs={epochs} seeds={ns}", flush=True)
    t0 = time.time()
    results = {}
    grid = {}
    for h in GRID_H:
        for lr in GRID_LR:
            vals = [train_eval(lambda: MLPQuatHead(4, h=h), args.seed_start+s, lr,
                               epochs, ntr, nte) for s in range(ns)]
            key = f'E32:h={h},lr={lr:g}'
            grid[(h, lr)] = vals
            results[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                                per_seed=vals, h=h, lr=lr,
                                held_out_seeds=list(range(args.seed_start,
                                                         args.seed_start+ns)))
            print(f"[{key:18s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f} deg", flush=True)

    best = min(results.items(), key=lambda kv: kv[1]['mean'])
    best_key, best_v = best
    results['E32:best_cell'] = dict(text=f"{best_key} = {best_v['mean']:.4f} deg "
                                         f"(QuatKAN ref {QK_REF:.4f})")
    p501 = (best_v['mean'] >= 2.80) and (best_v['mean'] - QK_REF >= 0.15)
    p502 = best_v['mean'] >= 3.00
    p503 = best_v['mean'] >= LIE_REF - 0.10
    results['E32:P50.1'] = dict(text=f"{'PASS' if p501 else 'FAIL'}: best {best_v['mean']:.4f} "
                                     f">= QuatKAN+0.15 ({QK_REF+0.15:.4f})")
    results['E32:P50.2'] = dict(text=f"{'PASS' if p502 else 'FAIL'}: best {best_v['mean']:.4f} >= 3.00")
    results['E32:P50.3'] = dict(text=f"{'PASS' if p503 else 'FAIL'}: best {best_v['mean']:.4f} "
                                     f">= MLP-Lie-0.10 ({LIE_REF-0.10:.4f})")
    print(f"[best] {best_key}: {best_v['mean']:.4f} deg | P50.1 {p501} P50.2 {p502} P50.3 {p503}",
          flush=True)

    results['E32:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(qk_ref=QK_REF, lie_ref=LIE_REF))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
