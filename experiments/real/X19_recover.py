#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""X19_recover.py — recover the X19 (P-TAU) verdicts after an aggregation bug.

The X19 training loop completed (seeds 0-4 x N=20/40/60/80) but the aggregation
crashed because the generic reference was read from the latest G2bc dump, which
is now the cluster-split dump (G2fC keys).  This script:
  1) parses the per-seed physics/exact numbers from the X19 log;
  2) reads the generic (G2f, random-split) per-seed values from the canonical
     5-seed G2bc dump;
  3) recomputes the paired stats and TOST(10%) exactly as X19 would have;
  4) retrains ONE model (seed 4, N=80) to capture the P-TAU3/P-TAU4 form checks.
Run: LAWKAN_THREADS=1 python X19_recover.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from X19_tau_psi import two_phase_tau, tau_tensor   # noqa: E402
import X7_robot_calibration as X7                   # noqa: E402
import E34_x7_recipe as E34                         # noqa: E402
import torch                                         # noqa: E402
from repro.dump import save_dump                     # noqa: E402

DT = torch.float64
RESULTS = {}


def parse_log(path):
    rows = {}
    pat = re.compile(r'\[X19 seed (\d+) n=(\d+)\] exact ([\d.]+) physics ([\d.]+)')
    for line in open(path):
        m = pat.search(line)
        if m:
            s, n, ex, ph = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
            rows.setdefault(n, {})[s] = (ex, ph)
    return rows


def main():
    log = os.environ.get('X19_LOG', os.path.join(
        _REPO, 'repro', 'provenance', 'x19_full.log'))
    rows = parse_log(log)
    with open(os.path.join(_REPO, 'repro', 'canonical.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    gen = json.load(open(os.path.join(_REPO, 'repro', 'json',
                                      man['g2bc_fig']['file']), encoding='utf-8'))
    assert 'G2f:fourier:n20:isolated' in gen
    for n in sorted(rows):
        seeds = sorted(rows[n])
        ex = np.array([rows[n][s][0] for s in seeds])
        ph = np.array([rows[n][s][1] for s in seeds])
        g = np.array(gen[f'G2f:fourier:n{n}:isolated']['per_seed'])[:len(seeds)]
        RESULTS[f'X19:n{n}:exact_own'] = dict(mean=float(ex.mean()), std=float(ex.std()),
                                              per_seed=[float(x) for x in ex])
        RESULTS[f'X19:n{n}:physics'] = dict(mean=float(ph.mean()), std=float(ph.std()),
                                            per_seed=[float(x) for x in ph])
        RESULTS[f'X19:n{n}:generic'] = dict(mean=float(g.mean()), std=float(g.std()),
                                            per_seed=[float(x) for x in g])
        t, p = stats.ttest_rel(ph, ex)
        RESULTS[f'X19:paired:n{n}:physics_vs_exact'] = dict(
            diff=float(ph.mean() - ex.mean()), t=float(t), p=float(p))
        t, p = stats.ttest_rel(ph, g)
        RESULTS[f'X19:paired:n{n}:physics_vs_generic'] = dict(
            diff=float(ph.mean() - g.mean()), t=float(t), p=float(p))
        tol = 0.10 * g.mean()
        se = (ph - g).std(ddof=1) / np.sqrt(len(ph))
        half = stats.t.ppf(0.95, len(ph) - 1) * se
        RESULTS[f'X19:TOST10:n{n}'] = dict(
            margin=float(tol), ci_lo=float((ph - g).mean() - half),
            ci_hi=float((ph - g).mean() + half),
            pass_=bool(abs((ph - g).mean()) + half < tol))
        print(f'[recover n={n}] exact {ex.mean():.3f} physics {ph.mean():.3f} '
              f'generic {g.mean():.3f} | phys-generic {ph.mean()-g.mean():+.3f} '
              f'TOST {"pass" if RESULTS[f"X19:TOST10:n{n}"]["pass_"] else "fail"}', flush=True)

    # form/gains: retrain one model (seed 4, N=80)
    import roboticstoolbox as rtb
    import X19_tau_psi as X19m
    X19m._PANDA_DH[0] = rtb.models.DH.Panda()
    q_np, obs_np = X7.load_philnad_dataset(os.path.join(_REPO, 'data', 'calib_data.pickle'))
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    perm = np.random.default_rng(1004).permutation(len(q_np))
    tr = perm[:80]
    _, ph = two_phase_tau(C, E, q_all[tr], oq_all[tr], ot_all[tr], 4000, 5e-3, 4)
    with torch.no_grad():
        tg = torch.linspace(-15, 15, 121, dtype=DT)
        y = ph.psi.h(tg.unsqueeze(-1)).squeeze(-1) + ph.psi.alpha[0] * tg
        odd = float((y + y.flip(0)).abs().mean() / y.abs().mean().clamp_min(1e-12))
        dy = y[1:] - y[:-1]
        mono = float((dy >= -1e-6).float().mean())
        RESULTS['X19:form:h_oddness'] = odd
        RESULTS['X19:form:h_monotone_frac'] = mono
        RESULTS['X19:gains:alpha'] = ph.psi.alpha.detach().cpu().numpy().tolist()
    print(f'[recover] form: oddness {odd:.3f}, monotone {mono:.2f}, '
          f'alpha {np.round(RESULTS["X19:gains:alpha"],3).tolist()}', flush=True)
    RESULTS['X19:recovered'] = dict(note='per-seed values parsed from x19_full.log; '
                                    'form/gains from a seed-4 N=80 retrain')
    save_dump(RESULTS, __file__)
    print('saved', RESULTS['X19:recovered'])


if __name__ == '__main__':
    main()
