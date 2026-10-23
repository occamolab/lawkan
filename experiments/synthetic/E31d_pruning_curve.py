# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E31d: automatic pruning of a generic dictionary on comp2 (P54)
#
# Train-prune-retrain in the E21 recipe, now on the rotation task (comp2) with
# the E31 node fixed: train the dense K=2 dictionary (25 features/group,
# 150 p), rank features WITHIN each group by the norm of their trained weights,
# keep the top-m per group and retrain the masked model from scratch for
# m in {1,2,3,4,6,9,12,25}. Automatic (ranking comes from training), no manual
# feature choice -- not tuning. Pre-registration: P54 (the experiment journal).
#
# Run: python E31d_pruning_curve.py [--quick]
# Output: repro/json/E31d_pruning_curve.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import os
import time

import numpy as np
import torch

from E31_node_isolation import (NodeKANEdge, device, geo_deg, geo_loss,
                                make_comp2, paired_t)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

BUDGETS = [1, 2, 3, 4, 6, 9, 12, 25]
# K=2 feature vector is 5x5 per group; the K=1 analytic support
# {(0,1),(1,2),(2,2)} maps to indices 1, 1*5+2=7, 2*5+2=12.
ANALYTIC = {1, 7, 12}


def _canon(block):
    return load_canon(block)


class MaskedKANEdge(NodeKANEdge):
    """Dense K=2 edge with a hard per-(group, feature) mask (feature side)."""
    def __init__(self, n_groups=2, K=2):
        super().__init__(n_groups, K=K)
        self.register_buffer('mask', torch.ones(n_groups, (2*K+1)**2))

    def edge_so3(self, a):
        f = self.group_feats(a) * self.mask[self._gi]
        return torch.einsum('bf,fc->bc', f, self.w[self._gi])


def train(net, Xtr, Ytr, epochs, lr=1e-2):
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = geo_loss(net(Xtr), Ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


def run_seed(seed, epochs, ntr, nte):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    dense = train(NodeKANEdge(2, K=2), Xtr, Ytr, epochs)
    with torch.no_grad():
        out = {'dense': geo_deg(dense(Xte), Yte).mean().item()}
        score = dense.w.detach().norm(dim=-1)                      # (g, nf)
        order = torch.argsort(score, dim=-1, descending=True)      # per group
    for m in BUDGETS:
        keep = order[:, :m]
        torch.manual_seed(100+seed)
        net = MaskedKANEdge(2, K=2)
        with torch.no_grad():
            net.mask.zero_()
            for g in range(net.n_groups):
                net.mask[g, keep[g]] = 1.0
        net = train(net, Xtr, Ytr, epochs)
        with torch.no_grad():
            out[m] = geo_deg(net(Xte), Yte).mean().item()
        if m == 3:
            out['keep3'] = [sorted(keep[g].cpu().tolist()) for g in range(keep.shape[0])]
    return out


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
        budgets = [1, 3, 9]
    else:
        budgets = BUDGETS

    print(f"E31d | automatic pruning curve | epochs={epochs} seeds={ns}", flush=True)
    t0 = time.time()
    runs = [run_seed(args.seed_start+s, epochs, ntr, nte) for s in range(ns)]
    results = {}
    results['E31d:dense K=2 (150p)'] = dict(
        mean=float(np.mean([r['dense'] for r in runs])),
        std=float(np.std([r['dense'] for r in runs])),
        per_seed=[r['dense'] for r in runs], params=150)
    for m in budgets:
        vals = [r[m] for r in runs]
        results[f'E31d:pruned m={m} ({6*m}p)'] = dict(
            mean=float(np.mean(vals)), std=float(np.std(vals)),
            per_seed=vals, params=6*m, kept_m=m)
        print(f"[m={m:2d} ({6*m:3d} p)] {np.mean(vals):8.4f} ±{np.std(vals):.4f}", flush=True)
    print(f"[dense K=2]     {results['E31d:dense K=2 (150p)']['mean']:8.4f}", flush=True)

    if not args.quick:
        hit = 0
        for r in runs:
            if r['keep3'] and all(set(k) == ANALYTIC for k in r['keep3']):
                hit += 1
        results['E31d:P54.3'] = dict(
            text=f"{'PASS' if hit >= 4 else 'FAIL'}: top-3 = analytic support for "
                 f"{hit}/{ns} seeds ({sorted(ANALYTIC)})")

        e31 = _canon('e31')['E31:node+KAN (K=1)']
        rm = _canon('run_main')['E13:K2']
        m3 = results['E31d:pruned m=3 (18p)']
        t1, df1, p1 = paired_t(m3['per_seed'], e31['per_seed'])
        t2, df2, p2 = paired_t(m3['per_seed'], rm['per_seed'])
        results['E31d:ttest:m3 vs K=1'] = dict(text=f"t={t1:.2f}, df={df1}, p={p1:.2e}")
        results['E31d:ttest:m3 vs K=2'] = dict(text=f"t={t2:.2f}, df={df2}, p={p2:.2e}")
        best_m = min(budgets, key=lambda m: results[f'E31d:pruned m={m} ({6*m}p)']['mean'])
        best = results[f'E31d:pruned m={best_m} ({6*best_m}p)']
        p541 = (best['mean'] <= 2.80) and (6*best_m <= 30)
        p542 = (m3['mean'] <= e31['mean']) and (p1 < 0.05) and (m3['mean'] < rm['mean'])
        results['E31d:P54.1'] = dict(
            text=f"{'PASS' if p541 else 'FAIL'}: best m={best_m} ({6*best_m} p) "
                 f"{best['mean']:.4f} <= 2.80 at <= 30 p")
        results['E31d:P54.2'] = dict(
            text=f"{'PASS' if p542 else 'FAIL'}: m=3 {m3['mean']:.4f} <= K=1 "
                 f"{e31['mean']:.4f} (p={p1:.2e}) and < K=2 {rm['mean']:.4f}")
        for k in ['E31d:P54.1', 'E31d:P54.2', 'E31d:P54.3']:
            print(f"  {k}: {results[k]['text']}", flush=True)

    results['E31d:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(dictionary='dense K=2 (25 features/group, 150p)',
                              recipe='E21 train-prune-retrain, per-group top-m by |w|'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
