# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN - E33b: training on the reported metric itself (P53)
#
# P53 was registered against the then-reported metric floor 1.902 deg (later
# corrected to 2.697; 1.902 is the posterior spread). Keep the registered
# threshold for the verdict and report the corrected floor separately. This run
# trains the SAME models with the
# reported metric (mean arccos) to test whether the +0.88 deg gap is the loss
# mismatch rather than the architecture.
# Protocol: comp2, 800/4000, 3000 epochs, seeds 0-4; LawKAN lr 1e-2,
# MLP-Quat h=32 lr 3e-3; loss = mean(arccos(cos)) in radians.
# Pre-registration: P53 in the experiment journal (2026-09-27).
#
# Run: python E33b_metric_loss.py [--quick]
# Output: repro/json/E33b_metric_loss.<ts>.json (smoke: -smoke)
# ============================================================
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from E31_node_isolation import (MLPQuatHead, NodeKANEdge, device, geo_deg,
                                make_comp2, paired_t)
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
    ('E33b:QuatKAN K=1, metric loss', lambda: NodeKANEdge(2, K=1), 1e-2),
    ('E33b:MLP-Quat h=32, metric loss', lambda: MLPQuatHead(4, h=32), 3e-3),
]


def metric_loss(pred, target):
    cos = (torch.stack([torch.sin(pred[..., 1])*torch.cos(pred[..., 0]),
                        torch.sin(pred[..., 1])*torch.sin(pred[..., 0]),
                        torch.cos(pred[..., 1])], -1)
           * torch.stack([torch.sin(target[..., 1])*torch.cos(target[..., 0]),
                          torch.sin(target[..., 1])*torch.sin(target[..., 0]),
                          torch.cos(target[..., 1])], -1)).sum(-1)
    return torch.acos(cos.clamp(-1+1e-7, 1-1e-7)).mean()


def train_eval(ctor, seed, lr, epochs, ntr, nte):
    Xtr, Ytr = make_comp2(ntr, seed=seed)
    Xte, Yte = make_comp2(nte, seed=seed+100)
    torch.manual_seed(100+seed)
    net = ctor().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = metric_loss(net(Xtr), Ytr)
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

    floor = E33_DUMP['E33:n2:posterior_spread']['mean']       # registered metric floor
    floor_corrected = E33_DUMP['E33:n2:intrinsic']['mean']    # after the E33 correction
    ref = E31_DUMP
    print(f"E33b | metric loss | registered floor={floor:.3f} | corrected={floor_corrected:.3f} "
          f"| epochs={epochs} seeds={ns}", flush=True)
    t0 = time.time()
    results = {}
    for name, ctor, lr in MODELS:
        vals = [train_eval(ctor, args.seed_start+s, lr, epochs, ntr, nte)
                for s in range(ns)]
        results[name] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                             per_seed=vals, lr=lr,
                             held_out_seeds=list(range(args.seed_start,
                                                      args.seed_start+ns)))
        print(f"[{name:34s}] {np.mean(vals):8.4f} ±{np.std(vals):.4f}", flush=True)

    if not args.quick:
        qk = results['E33b:QuatKAN K=1, metric loss']
        mlp = results['E33b:MLP-Quat h=32, metric loss']
        t, df, p = paired_t(qk['per_seed'], mlp['per_seed'])
        results['E33b:ttest:QuatKAN vs MLP'] = dict(text=f"t={t:.2f}, df={df}, p={p:.2e}")
        p531 = qk['mean'] <= 2.10
        p532 = qk['mean'] <= floor + 0.25
        p533 = (mlp['mean'] <= 3.05) and (qk['mean'] < mlp['mean']) and (p < 0.05)
        results['E33b:P53.1'] = dict(text=f"{'PASS' if p531 else 'FAIL'}: metric-loss QuatKAN "
                                         f"{qk['mean']:.4f} <= 2.10 (1-cos {ref['E31:node+KAN (K=1)']['mean']:.3f})")
        results['E33b:P53.2'] = dict(text=f"{'PASS' if p532 else 'FAIL'}: {qk['mean']:.4f} <= "
                                         f"registered floor+0.25 ({floor+0.25:.3f})")
        results['E33b:P53.2:corrected_floor'] = dict(
            text=f"post-hoc with corrected floor 2.697: "
                 f"{'PASS' if qk['mean'] <= floor_corrected + 0.25 else 'FAIL'} "
                 f"({qk['mean']:.4f} <= {floor_corrected + 0.25:.3f})")
        results['E33b:P53.3'] = dict(text=f"{'PASS' if p533 else 'FAIL'}: MLP {mlp['mean']:.4f} <= 3.05 "
                                         f"and QuatKAN<MLP p={p:.2e}")
        for k in ['E33b:P53.1', 'E33b:P53.2', 'E33b:P53.3']:
            print(f"  {k}: {results[k]['text']}", flush=True)

    results['E33b:wall_s'] = time.time() - t0
    save_dump(results, __file__, quick=args.quick, args=args, seeds=ns,
              seed_start=args.seed_start, epochs=epochs,
              extra_meta=dict(floor_registered=floor, floor_corrected=floor_corrected, ref_dump='e31'))
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(results, indent=2, default=float))


if __name__ == '__main__':
    main()
