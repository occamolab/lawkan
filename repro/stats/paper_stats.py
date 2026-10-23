#!/usr/bin/env python3
"""Derived statistics quoted in the paper, recomputed from canonical dumps.

Outputs (stdout + repro/stats/paper_stats.json):
  table3_p     paired t-tests, row 7 (two-slot, X35) vs row 6 (exact, X24), per N
  table3_boot  paired seed bootstrap, row 7 minus row 1 (GN, X7bench), per N
  t1_t2        X61 T1 (q14-global vs base, 90% CI) and T2 (|EMA-raw| final)
  bh_socket    Benjamini-Hochberg over the 11 numeric tests of Table 4
"""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = json.load(open(os.path.join(ROOT, 'repro', 'canonical.json')))
J = os.path.join(ROOT, 'repro', 'json')


def dump(key):
    return json.load(open(os.path.join(J, C[key]['file'])))


OUT = {}

# ---- Table 3: paired p-values and bootstrap CI -------------------------------
x35, x24, x7b = dump('x35'), dump('x24'), dump('x7bench')
t3 = {}
rng = np.random.default_rng(0)
for n in (20, 40, 60, 80):
    two = np.array(x35[f'X35:n{n}:isolated']['per_seed'])
    ex = np.array(x24[f'X24:n{n}:exact']['per_seed'])
    gn = np.array([p[0] for p in x7b[f'X7bench:n{n}:gn']['per_seed']])
    t, p = stats.ttest_rel(two, ex)
    d = two - gn
    idx = rng.integers(0, len(d), (20000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    t3[str(n)] = {'row7_minus_row6': float((two - ex).mean()), 'p': float(p),
                  'row7_minus_row1': float(d.mean()),
                  'boot95': [float(lo), float(hi)], 'diff_per_seed': d.tolist()}
OUT['table3'] = t3

# two-slot vs the classical GN optimum (x7bench gn), paired
gn_tests = {}
for n in (20, 40, 60, 80):
    two = np.array(x35[f'X35:n{n}:isolated']['per_seed'])
    gn = np.array([p[0] for p in x7b[f'X7bench:n{n}:gn']['per_seed']])
    t, pv = stats.ttest_rel(two, gn)
    gn_tests[str(n)] = {'diff': float((two - gn).mean()), 'p': float(pv)}
OUT['table3_gn'] = gn_tests

# ---- X61: T1 / T2 ------------------------------------------------------------
runs = []
for key in ('x61', 'x61b'):
    runs += dump(key).get('runs', [])
res = {}
for r in runs:
    raw = {int(k): v for k, v in r['raw'].items()}
    ema = {int(k): v for k, v in r['ema'].items()}
    res[(r['cfg'], r['seed'])] = (raw[max(raw)], ema[max(ema)])
seeds = sorted({s for (c, s) in res})
cfgs = ('base', 'gonly', 'kernel')
means, t2 = {}, {}
for c in cfgs:
    raw = np.array([res[(c, s)][0] for s in seeds])
    ema = np.array([res[(c, s)][1] for s in seeds])
    means[c] = {'raw_final': raw.tolist(), 'mean': float(raw.mean()),
                'sd': float(raw.std(ddof=1))}
    t2[c] = float(np.abs(ema - raw).max())
d = np.array([res[('gonly', s)][0] - res[('base', s)][0] for s in seeds])
n = len(d); m = d.mean(); h = stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n)
OUT['x61'] = {'table3_like': means, 'T1': {'diff_mean': float(m), 'ci90': [float(m - h), float(m + h)],
            'equivalent_011': bool(m - h > -0.11 and m + h < 0.11), 'n': n},
              'T2_max_abs': t2}

# ---- BH over the socket table ------------------------------------------------
x38, x38b = dump('x38'), dump('x38b')
tests = {
 'reduced edge KUKA': x38['X38:kuka14:paired:iso_edgeb_vs_edge_b']['p'],
 'reduced dense KUKA': x38['X38:kuka14:paired:iso_edgeb_vs_dense']['p'],
 'reduced edge Panda': x38['X38:panda_1:paired:iso_edgeb_vs_edge_b']['p'],
 'reduced dense Panda': x38['X38:panda_1:paired:iso_edgeb_vs_dense']['p'],
 'reduced edge Kinova': x38['X38:kinova:paired:iso_edgeb_vs_edge_b']['p'],
 'reduced dense Kinova': x38['X38:kinova:paired:iso_edgeb_vs_dense']['p'],
 'reduced edge FR3': x38['X38:fr3:paired:iso_edgeb_vs_edge_b']['p'],
 'reduced dense FR3': x38['X38:fr3:paired:iso_edgeb_vs_dense']['p'],
 'per-joint KUKA': x38['X38:kuka14:paired:iso_pj_vs_dense']['p'],
 'per-joint Panda': x38['X38:panda_1:paired:iso_pj_vs_dense']['p'],
 'per-joint Kinova': x38b['X38b:kinova:paired:iso_pj_kernel_vs_dense']['p'],
}
names = list(tests); p = np.array([tests[k] for k in names]); m = len(p)
order = np.argsort(p); q = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]
    prev = min(prev, p[i] * m / (rank + 1)); q[i] = prev
OUT['bh_socket'] = {k: {'p': float(tests[k]), 'q': float(q[i])} for i, k in enumerate(names)}

out_path = os.environ.get('PAPER_STATS_OUT', os.path.join(ROOT, 'repro', 'stats', 'paper_stats.json'))
json.dump(OUT, open(out_path, 'w'), indent=1)
print(json.dumps(OUT, indent=1))
