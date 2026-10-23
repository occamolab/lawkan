# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E31c: analytic-support (sparse) edge under the E31 protocol (P52)
#
# Left anchor of the Fourier family for the capacity figure: the 18-parameter
# edge keeps only [cos b, cos a sin b, sin a sin b] (the analytic support,
# SparseQuatExpNet from E19) under the SAME node and protocol as E31/E31b.
# Controls and dense KAN are NOT rerun: per-seed values come from the canonical
# E31 dump (same seeds/protocol; node+KAN equals E1 bit-for-bit).
# Pre-registration: P52 in the experiment journal (2026-09-27).
#
# Run: python E31c_sparse_edge.py [--quick]
# Output: repro/json/E31c_sparse_edge.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from E31_node_isolation import (NodeModel, device, geo_loss, geo_deg,
                                make_comp2, n_params, paired_t)
import sys as _sys
from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parents[2])
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
from repro.dump import save_dump
from repro.canon import load as load_canon

E31_DUMP = load_canon('e31')


class NodeSparseEdge(NodeModel):
    """Analytic support: [cos b, cos a sin b, sin a sin b] -> so(3), 9 p/group."""
    def __init__(self, n_groups=2):
        super().__init__(n_groups)
        self.w = nn.Parameter(torch.randn(n_groups, 3, 3) * 0.1)

    def group_feats(self, a):
        al, be = a[:, 0], a[:, 1]
        sbe = torch.sin(be)
        return torch.stack([torch.cos(be), torch.cos(al)*sbe, torch.sin(al)*sbe], dim=-1)

    def edge_so3(self, a):
        return torch.einsum('bf,fc->bc', self.group_feats(a), self.w[self._gi])


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
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=0)
    args = ap.parse_args()
    epochs, ns = args.epochs, args.seeds
    ntr, nte = 800, 4000
    if args.quick:
        epochs, ns, ntr, nte = 50, 1, 64, 128

    ref = E31_DUMP
    controls = ['E31:plain MLP-Quat h=32', 'E31:single-exp MLP-Lie h=32']
    print(f"E31c | analytic-support edge (18 p) | epochs={epochs} seeds={ns}", flush=True)
    t0 = time.time()
    results = {}
    vals = [train_eval(NodeSparseEdge, args.seed_start+s, 1e-2, epochs, ntr, nte)
            for s in range(ns)]
    name = 'E31c:node+sparse (18p)'
    results[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                         per_seed=vals, params=int(n_params(NodeSparseEdge)), lr=1e-2,
                         held_out_seeds=list(range(args.seed_start, args.seed_start+ns)))
    print(f"[{name:30s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f} | params "
          f"{n_params(NodeSparseEdge)}", flush=True)

    if not args.quick:
        p_ok = True
        for cname in controls:
            t, df, p = paired_t(results[name]['per_seed'], ref[cname]['per_seed'])
            results[f'E31c:ttest:sparse vs {cname}'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")
            p_ok = p_ok and (p < 0.01) and (results[name]['mean'] < ref[cname]['mean'])
            print(f"[ttest] sparse vs {cname}: p={p:.2e}", flush=True)
        dense = ref['E31:node+KAN (K=1)']
        t, df, p = paired_t(results[name]['per_seed'], dense['per_seed'])
        p522 = (results[name]['mean'] <= dense['mean']) and (p < 0.05)
        results['E31c:P52.1'] = dict(
            text=f"{'PASS' if p_ok else 'FAIL'}: sparse {results[name]['mean']:.4f} below both "
                 f"controls (p<0.01)")
        results['E31c:P52.2'] = dict(
            text=f"{'PASS' if p522 else 'FAIL'}: sparse {results[name]['mean']:.4f} <= dense K=1 "
                 f"{dense['mean']:.4f} (paired t={t:.2f}, p={p:.2e})")
        print(f"[P52.1] {results['E31c:P52.1']['text']}", flush=True)
        print(f"[P52.2] {results['E31c:P52.2']['text']}", flush=True)

    results['E31c:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(ref_dump='e31'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
