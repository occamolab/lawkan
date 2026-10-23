# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# X9 — IMU tuning-sensitivity & determinism (real BROAD data)
#
# Purpose. The paper claims the structured composer is a tiny, deterministic,
# TUNING-FREE component. X5/X8 so far fixed lr=3e-3 for every model, so that
# claim was not demonstrated. X9 sweeps lr over {1e-3, 3e-3, 1e-2} and reports,
# per model, the spread across lr and the seed std, on the real-data windows
# with the trial-level split. Designed to show:
#   * structured (QuatIMU sparse/dense) is lr-robust, sparse is seed-deterministic;
#   * flat MLP / residual MLP are lr-sensitive and their best lr still does not
#     beat the structured model / the training-free integrator.
#
# GRU is NOT swept (kept as a single completeness row in the paper).
# c-5s sweep is reduced to sparse/dense/resid (flat MLP is ~100 deg regardless).
#
# Run:  python X9_imu_tuning_sensitivity.py --data_dir /path/to/broad/data_hdf5
# ============================================================
import argparse, json, math, os, sys, time
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.data_guard import skip_if_missing  # noqa: E402
import X5_broad_real_imu as X5
import X8_broad_seq as X8
from X5_broad_real_imu import (make_dataset, trial_split, QuatIMU, MLPFlat,
                               train_eval, strapdown_integrate, geo_deg_q)

device = X5.device
print(f"Device: {device}")

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}")

LR_GRID = [1e-3, 3e-3, 1e-2]
SETTINGS = [
    ('b-1s', 286, 286 // 4, ['sparse', 'dense', 'flat', 'resid']),
    ('c-5s', 1430, 1430 // 10, ['sparse', 'dense', 'resid']),
]
SEEDS = (0, 1, 2)
EPOCHS = 3000

def ctor_for(kind, W, dt, din):
    if kind == 'sparse': return lambda: QuatIMU(W, K=1, dt=dt, bias_dim=3, sparse=True)
    if kind == 'dense':  return lambda: QuatIMU(W, K=1, dt=dt, bias_dim=3)
    if kind == 'flat':   return lambda: MLPFlat(din, h=32)
    if kind == 'resid':  return lambda: X8.ResidMLP(din, h=32)
    raise ValueError(kind)

def run():
    global EPOCHS, SEEDS, LR_GRID
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    if args.quick:
        EPOCHS, SEEDS, LR_GRID = 300, (0,), [1e-3, 1e-2]
    t0 = time.time()
    for name, W, stride, kinds in SETTINGS:
        print(f"\n===== X9:{name}  (W={W}) =====")
        X, Y, B, fs, T = make_dataset(args.data_dir, W=W, stride=stride, groups=('disturbed',))
        tr, te = trial_split(T, 800, 2000)
        print(f"  trial split: {len(T[tr].unique())} train / {len(T[te].unique())} test trials; "
              f"{len(tr)}/{len(te)} windows")
        Xtr, Ytr, Btr = X[tr], Y[tr], B[tr]
        Xte, Yte, Bte = X[te], Y[te], B[te]
        dt = 1.0 / fs
        din = W * 3 + 3
        # training-free floor
        q_te = strapdown_integrate(Xte.reshape(-1, W, 3), dt)
        reg(f'X9:{name}:integrator', geo_deg_q(q_te, Yte).cpu().numpy(), params=0)
        Qtr = strapdown_integrate(Xtr.reshape(-1, W, 3), dt)
        Qte = strapdown_integrate(Xte.reshape(-1, W, 3), dt)
        for kind in kinds:
            ctor = ctor_for(kind, W, dt, din)
            for lr in LR_GRID:
                errs = []
                for s in SEEDS:
                    torch.manual_seed(100 + s)          # seed init
                    m = ctor()
                    if kind == 'resid':
                        e = X8.train_eval_resid(m, Xtr, Ytr, Btr, Qtr, Xte, Yte, Bte, Qte,
                                                epochs=EPOCHS, lr=lr, seed=100 + s)
                    else:
                        e = train_eval(m, Xtr, Ytr, Btr, Xte, Yte, Bte,
                                       epochs=EPOCHS, lr=lr, seed=100 + s)
                    errs.append(e)
                reg(f'X9:{name}:{kind}:lr{lr:g}', errs,
                    params=sum(p.numel() for p in ctor().parameters()))
            # spread across lr and seed
            lrs = [RESULTS[f'X9:{name}:{kind}:lr{lr:g}']['mean'] for lr in LR_GRID]
            seedstds = [RESULTS[f'X9:{name}:{kind}:lr{lr:g}']['std'] for lr in LR_GRID]
            RESULTS[f'X9:{name}:{kind}:summary'] = dict(
                best=min(lrs), worst=max(lrs), lr_spread=max(lrs) - min(lrs),
                max_seed_std=max(seedstds))
            print(f"  -> {name}/{kind}: best {min(lrs):.3f} worst {max(lrs):.3f} "
                  f"lr_spread {max(lrs)-min(lrs):.3f}  max_seed_std {max(seedstds):.3f}")
    RESULTS['_meta'] = dict(wall_s=time.time() - t0, device=str(device))
    print(f"\nTotal wall time: {time.time()-t0:.1f} s")
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    run()
