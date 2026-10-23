# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E31b: capacity bracket for the MLP edge (P51)
#
# E31 left a gap: node+MLP h=2 (1 layer, 38 p) fails (22.9 deg) while
# h=8 (2 layers, 278 p) succeeds (2.97 deg). This run adds h=4 in both
# depth variants to locate the capacity/depth threshold:
#   node+MLP h=4 (2 layers, 110 p)  -- same family as h=8/32
#   node+MLP h=4 (1 layer,  70 p)   -- same family as h=2 (1 layer)
# Protocol identical to E31 (comp2, 800/4000, 3000 epochs, seeds 0-4,
# lr 3e-3); controls and node variants are NOT rerun -- per-seed values are
# read from the canonical E31 dump (same seeds/protocol; determinism of the
# protocol is established: E31 node+KAN equals E1 bit-for-bit).
# Pre-registration: P51 in the experiment journal (2026-09-27).
#
# Run: python E31b_node_mlp_h4.py [--quick]
# Output: repro/json/E31b_node_mlp_h4.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import os
import time

import numpy as np

from E31_node_isolation import (NodeMLPEdge, device, geo_loss, geo_deg,
                                make_comp2, n_params, paired_t)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

E31_DUMP = load_canon('e31')

VARIANTS = [
    ('E31b:node+MLP h=4 (2L, 110p)', lambda: NodeMLPEdge(2, h=4, hidden_layers=2), 3e-3),
    ('E31b:node+MLP h=4 (1L, 70p)', lambda: NodeMLPEdge(2, h=4, hidden_layers=1), 3e-3),
]


def train_eval(ctor, seed, lr, epochs, ntr, nte):
    import torch
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
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=0)
    args = ap.parse_args()
    epochs, ns = args.epochs, args.seeds
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, ntr, nte = 50, 1, 64, 128

    ref = E31_DUMP
    controls = ['E31:plain MLP-Quat h=32', 'E31:single-exp MLP-Lie h=32']
    best_node = min(ref[k]['mean'] for k in
                    ['E31:node+KAN (K=1)', 'E31:node+MLP h=8', 'E31:node+MLP h=32'])
    best_control = min(ref[k]['mean'] for k in controls)
    print(f"E31b | epochs={epochs} seeds={ns} | bracket ({best_node:.3f}, {best_control:.3f})",
          flush=True)

    t0 = time.time()
    results = {}
    for name, ctor, lr in VARIANTS:
        vals = [train_eval(ctor, args.seed_start+s, lr, epochs, ntr, nte)
                for s in range(ns)]
        results[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                             per_seed=vals, params=int(n_params(ctor)), lr=lr,
                             held_out_seeds=list(range(args.seed_start,
                                                      args.seed_start+ns)))
        print(f"[{name:30s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f} | params "
              f"{n_params(ctor)}", flush=True)

    if not args.quick:
        for name, _, _ in VARIANTS:
            for cname in controls:
                t, df, p = paired_t(results[name]['per_seed'], ref[cname]['per_seed'])
                results[f'E31b:ttest:{name} vs {cname}'] = dict(
                    text=f"t={t:.2f}, df={df}, p={p:.2e}")
                print(f"[ttest] {name} vs {cname}: p={p:.2e}", flush=True)
        v2 = results[VARIANTS[0][0]]['mean']
        v1 = results[VARIANTS[1][0]]['mean']
        p511 = (v2 < ref[controls[0]]['mean']) and (v2 < ref[controls[1]]['mean']) \
            and (best_node < v2 < best_control)
        t, df, p = paired_t(results[VARIANTS[1][0]]['per_seed'],
                            results[VARIANTS[0][0]]['per_seed'])
        p512 = v1 > v2
        results['E31b:P51.1'] = dict(text=f"{'PASS' if p511 else 'FAIL'}: 2L h=4 "
                                         f"{v2:.4f} in ({best_node:.3f}, {best_control:.3f})")
        results['E31b:P51.2'] = dict(text=f"{'PASS' if p512 else 'FAIL'}: 1L h=4 "
                                         f"{v1:.4f} > 2L h=4 {v2:.4f} "
                                         f"(paired t={t:.2f}, p={p:.2e})")
        print(f"[P51.1] {results['E31b:P51.1']['text']}", flush=True)
        print(f"[P51.2] {results['E31b:P51.2']['text']}", flush=True)

    results['E31b:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(ref_dump='e31'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
