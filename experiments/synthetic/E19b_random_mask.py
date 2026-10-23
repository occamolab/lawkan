# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E19b: random equal-size mask control (P63; renumbered from P56 after the E34/E35 registrations)
#
# The claim "sparse 18 p beats dense 54 p" (E19/E31c) admits a simpler
# explanation: fewer parameters. This control keeps a RANDOM 3-feature subset
# per group (same index set in both groups, exactly the sparse edge's 18 active
# parameters) of the dense K=1 dictionary and trains from scratch.
# R=5 random masks per seed x 5 seeds = 25 runs; references (sparse, dense K=1)
# are read from the canonical dumps. Protocol identical to E31/E19
# (comp2, 800/4000, 3000 epochs, lr 1e-2, seeds 0-4).
# Pre-registration: P63 in the experiment journal (2026-09-27).
#
# Run: python E19b_random_mask.py [--quick]
# Output: repro/json/E19b_random_mask.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import os
import time

import numpy as np
import torch

from E31_node_isolation import (device, geo_deg, geo_loss, make_comp2,
                                paired_t)
from E31d_pruning_curve import MaskedKANEdge
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

N_FEAT = 9
KEEP = 3
DRAWS = 5


def canon(block):
    return load_canon(block)


def train_eval(mask, seed, epochs, ntr, nte):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    net = MaskedKANEdge(2, K=1).to(device)
    with torch.no_grad():
        net.mask.zero_()
        for g in range(net.n_groups):
            net.mask[g, mask] = 1.0
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
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
    ap.add_argument('--draws', type=int, default=DRAWS)
    args = ap.parse_args()
    epochs, ns, nd = args.epochs, args.seeds, args.draws
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, nd, ntr, nte = 50, 1, 2, 64, 128

    e31 = canon('e31')
    e31c = canon('e31c')
    sparse = e31c['E31c:node+sparse (18p)']
    dense = e31['E31:node+KAN (K=1)']
    print(f"E19b | random 18-p masks | epochs={epochs} seeds={ns} draws={nd} | "
          f"sparse={sparse['mean']:.4f} dense={dense['mean']:.4f}", flush=True)

    t0 = time.time()
    results = {}
    per_seed_means, all_runs, used = [], [], []
    for s in range(args.seed_start, args.seed_start+ns):
        draw_vals = []
        for r in range(nd):
            g = torch.Generator().manual_seed(5000 + 10*s + r)
            mask = torch.sort(torch.randperm(N_FEAT, generator=g)[:KEEP]).values
            used.append(sorted(mask.tolist()))
            v = train_eval(mask, s, epochs, ntr, nte)
            draw_vals.append(v)
        per_seed_means.append(float(np.mean(draw_vals)))
        all_runs.extend(draw_vals)
        print(f"[seed {s}] draws {[round(v,3) for v in draw_vals]} | mean "
              f"{np.mean(draw_vals):.4f}", flush=True)

    results['E19b:random-mask (18p)'] = dict(
        mean=float(np.mean(all_runs)), std=float(np.std(all_runs)),
        per_seed=per_seed_means, all_runs=[float(v) for v in all_runs],
        masks=used, params=18)
    print(f"[random-mask 18p] {np.mean(all_runs):8.4f} ±{np.std(all_runs):.4f} "
          f"(per-seed means std {np.std(per_seed_means):.4f})", flush=True)

    if not args.quick:
        t1, df1, p1 = paired_t(per_seed_means, sparse['per_seed'])
        t2, df2, p2 = paired_t(per_seed_means, dense['per_seed'])
        results['E19b:ttest:random vs sparse'] = dict(
            text=f"t={t1:.2f}, df={df1}, p={p1:.2e}")
        results['E19b:ttest:random vs dense'] = dict(
            text=f"t={t2:.2f}, df={df2}, p={p2:.2e}")
        mean = results['E19b:random-mask (18p)']['mean']
        p561 = (mean >= 3.5) and (p1 < 0.05)
        p562 = mean >= (dense['mean'] - 0.05)
        results['E19b:P63.1'] = dict(
            text=f"{'PASS' if p561 else 'FAIL'}: random {mean:.4f} >= 3.5 and worse than "
                 f"sparse {sparse['mean']:.4f} (p={p1:.2e})")
        results['E19b:P63.2'] = dict(
            text=f"{'PASS' if p562 else 'FAIL'}: random {mean:.4f} >= dense-0.05 "
                 f"({dense['mean']-0.05:.4f})")
        for k in ['E19b:P63.1', 'E19b:P63.2']:
            print(f"  {k}: {results[k]['text']}", flush=True)

    results['E19b:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(reference_sparse=sparse['mean'],
                              reference_dense=dense['mean'], draws=nd))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
