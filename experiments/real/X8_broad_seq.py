# SPDX-License-Identifier: AGPL-3.0-or-later
# ============================================================
# LawKAN — X8: sequential & residual baselines on BROAD (companion to X5)
#
# Why this experiment (registered predictions P30/P31, see
# P28_P29_registration.md):
#   The X5 flat-MLP failure at long horizons has exactly two known fixes,
#   and they are the two ways to BUY the composition structure:
#     1. bake it into the architecture (QuatIMU: per-step edge + Hamilton
#        node — "structure is free": 18 params, one parallel pass);
#     2. learn it via recurrence (GRU/LSTM — "sequence is expensive": the
#        hidden state must LEARN the group law; ~4k params, 1430 sequential
#        steps);
#   flat one-shot regression is the control point that has neither ("flat:
#   no way", already measured in X5: 88.9/100.0 deg at c-5s).
#   X8 measures fix 2 (GRUIntegrator) and the standard residual trick
#   (ResidMLP predicts a LOCAL correction to the integrator output, the
#   TLIO/SINS-error philosophy).
#
# Protocol: IDENTICAL to X5 v1.1 — same windows (same strides), same split
# (seed 12345, 800/2000), same seeds (100+s), same geodesic loss/ metric,
# same bias feature. Only the trainable models differ.
#
# Requirements: torch, numpy, h5py (same env as X5). Run from the directory
# containing X5_broad_real_imu.py:
#   python X8_broad_seq.py --data_dir /path/to/broad/data_hdf5 --only b-1s,c-5s
# ============================================================
import argparse, time, json, os, sys
import numpy as np
import torch, torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from repro.data_guard import skip_if_missing  # noqa: E402
import X5_broad_real_imu as X5          # dataset builder, integrator, metric
from X5_broad_real_imu import (strapdown_integrate, geo_deg_q, geo_rad_q,
                               rotvec_to_quat, quat_mul)

device = X5.device
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

RESULTS = {}
def reg(key, vals, **meta):
    vals = [float(v) for v in vals]
    RESULTS[key] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                        per_seed=vals, **meta)
    print(f"[{key}] mean {np.mean(vals):.4f} ±{np.std(vals):.4f}")

# ------------------------------------------------------------
# Trainable models
# ------------------------------------------------------------
class GRUIntegrator(nn.Module):
    """Sequential baseline (P30): GRU over the window, hidden state -> rotvec.
    The bias estimate is appended to every step (same information as X5)."""
    def __init__(self, W, bias_dim=0, h=32):
        super().__init__()
        self.W = W
        self.gru = nn.GRU(input_size=3 + bias_dim, hidden_size=h,
                          batch_first=True)
        self.head = nn.Linear(h, 3)
    def forward(self, x, bias=None):
        B = x.shape[0]
        s = x.reshape(B, self.W, 3)
        if bias is not None:
            s = torch.cat([s, bias.unsqueeze(1).expand(-1, self.W, -1)], dim=-1)
        _, hn = self.gru(s)
        return rotvec_to_quat(self.head(hn[-1])).to(torch.float64)

class ResidMLP(nn.Module):
    """Residual baseline (P31): flat MLP predicting a LOCAL correction delta
    to the (training-free) integrator output:  q_pred = q_int * exp(delta).
    q_int is precomputed per window and passed in — the model only learns
    the correction, exactly the TLIO / SINS-error philosophy."""
    def __init__(self, din, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 3))
    def forward(self, x, bias=None, q_int=None):
        if bias is not None:
            x = torch.cat([x, bias], dim=-1)
        dq = rotvec_to_quat(self.net(x))
        return quat_mul(q_int, dq).to(torch.float64)

# ------------------------------------------------------------
# Training / eval (same protocol as X5.train_eval)
# ------------------------------------------------------------
def train_eval_gru(model, Xtr, Ytr, Btr, Xte, Yte, Bte,
                   epochs=3000, lr=3e-3, seed=0, tuned=False, mb=128):
    # tuned=True: the P30b registered "fair industrial recipe" — gradient
    # clipping 1.0, lr 1e-3 with cosine decay, up to 3x epochs, early stop
    # on train-loss plateau. Same DATA protocol; only optimization changes.
    # Mini-batch in FIXED order (no shuffling, no extra RNG): a full-batch
    # GRU at W=1430 OOMs on 8 GB (cuDNN stores every hidden state for
    # backward). The loss is the exact dataset mean, identical to X5.
    torch.manual_seed(seed)
    model = model.to(device)
    n = Xtr.shape[0]
    if tuned:
        epochs, lr = 3 * epochs, 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if tuned else None)
    best, patience = float('inf'), 0
    for ep in range(epochs):
        opt.zero_grad()
        tot = 0.0
        for i in range(0, n, mb):
            j = min(i + mb, n)
            loss = (geo_rad_q(model(Xtr[i:j], Btr[i:j]), Ytr[i:j]) ** 2).mean()
            (loss * (j - i) / n).backward()
            tot += loss.item() * (j - i) / n
        if tuned:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if tuned:
            if tot < best - 1e-7:
                best, patience = tot, 0
            else:
                patience += 1
            if patience > 1500:      # early stop on plateau
                break
    with torch.no_grad():
        errs = []
        for i in range(0, Xte.shape[0], 500):
            errs.append(geo_deg_q(model(Xte[i:i+500], Bte[i:i+500]),
                                  Yte[i:i+500]))
        return torch.cat(errs).mean().item()

def train_eval_resid(model, Xtr, Ytr, Btr, Qtr, Xte, Yte, Bte, Qte,
                     epochs=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = (geo_rad_q(model(Xtr, Btr, Qtr), Ytr) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return geo_deg_q(model(Xte, Bte, Qte), Yte).mean().item()

# ------------------------------------------------------------
# Driver: one setting = one window length (identical to X5)
# ------------------------------------------------------------
def run_setting(name, W, stride, data_dir, seeds=(0, 1, 2),
                n_train=800, n_test=2000, epochs=3000, gru_tuned=False):
    print(f"\n===== X8:{name}  (W={W}) =====")
    X, Y, B, fs, T = X5.make_dataset(data_dir, W=W, stride=stride,
                                     groups=('disturbed',))
    n = len(X)
    tr, te = X5.trial_split(T, n_train, n_test)      # trial-level, same as X5
    print(f"  trial split: {len(T[tr].unique())} train trials / {len(T[te].unique())} test trials; "
          f"{len(tr)} train / {len(te)} test windows")
    Xtr, Ytr, Btr = X[tr], Y[tr], B[tr]
    Xte, Yte, Bte = X[te], Y[te], B[te]
    dt = 1.0 / fs

    # --- training-free floor (same as X5) ---
    q_te = strapdown_integrate(Xte.reshape(-1, W, 3), dt)
    reg(f'X8:{name}:BiasIntegrator', geo_deg_q(q_te, Yte).cpu().numpy(), params=0)

    # --- integrator outputs for the residual model (train + test) ---
    Qtr = strapdown_integrate(Xtr.reshape(-1, W, 3), dt)
    Qte = strapdown_integrate(Xte.reshape(-1, W, 3), dt)

    din = W * 3 + 3                                # + bias feature
    for label, kind in [('GRU h=32', 'gru'), ('ResidMLP h=32', 'resid')]:
        errs = []
        for s in seeds:
            torch.manual_seed(100 + s)          # seed model init (was drawn pre-seed)
            if kind == 'gru':
                m = GRUIntegrator(W, bias_dim=3, h=32)
                errs.append(train_eval_gru(m, Xtr, Ytr, Btr, Xte, Yte, Bte,
                                           epochs=epochs, seed=100 + s,
                                           tuned=gru_tuned))
            else:
                m = ResidMLP(din, h=32)
                errs.append(train_eval_resid(m, Xtr, Ytr, Btr, Qtr,
                                             Xte, Yte, Bte, Qte,
                                             epochs=epochs, seed=100 + s))
        npar = sum(p.numel() for p in m.parameters())
        reg(f'X8:{name}:{label}', errs, params=npar)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.environ.get('BROAD', os.path.join(os.path.dirname(_REPO), 'broad', 'data_hdf5')))
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--only', default=None,
                    help="comma-separated blocks (choices: b-1s, c-5s)")
    ap.add_argument('--gru_tuned', action='store_true',
                    help="P30b registered fair-recipe GRU (clip 1.0, lr 1e-3 "
                         "cosine, 3x epochs, early stop)")
    args = ap.parse_args()
    if skip_if_missing(args.data_dir, 'BROAD data_dir'):
        return
    seeds = tuple(range(args.seeds))
    only = set(args.only.split(',')) if args.only else None
    t0 = time.time()

    if only is None or 'b-1s' in only:
        run_setting('b-1s', W=286, stride=286 // 4, data_dir=args.data_dir,
                    seeds=seeds, gru_tuned=args.gru_tuned)
    if only is None or 'c-5s' in only:
        run_setting('c-5s', W=1430, stride=1430 // 10, data_dir=args.data_dir,
                    seeds=seeds, gru_tuned=args.gru_tuned)

    RESULTS['_meta'] = dict(wall_s=time.time() - t0, device=str(device))
    print(f"\nTotal wall time: {time.time() - t0:.1f} s")
    print("\n===== SUMMARY-JSON =====")
    print(json.dumps(RESULTS, indent=2, default=float))

if __name__ == '__main__':
    main()
