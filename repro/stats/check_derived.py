#!/usr/bin/env python3
"""Gate for the derived statistics displayed in the paper."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fd, tmp = tempfile.mkstemp(suffix='.json')
os.close(fd)
env = dict(os.environ, PAPER_STATS_OUT=tmp)
subprocess.check_call([sys.executable, os.path.join(ROOT, 'repro', 'stats', 'paper_stats.py')],
                      stdout=subprocess.DEVNULL, env=env)
d = json.load(open(tmp))
os.unlink(tmp)
checks = []


def chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    checks.append((name, got, want, ok))


for n, p in (('40', 0.0278), ('60', 0.0071), ('80', 0.0027)):
    chk(f'T3 p N={n}', d['table3'][n]['p'], p, 0.002)
for n, lo, hi in (('40', -0.249, -0.082), ('60', -0.244, -0.060), ('80', -0.263, -0.162)):
    chk(f'T3 boot lo N={n}', d['table3'][n]['boot95'][0], lo, 0.03)
    chk(f'T3 boot hi N={n}', d['table3'][n]['boot95'][1], hi, 0.03)
for n, p in (('40', 0.0397), ('60', 0.0615), ('80', 0.0017)):
    chk(f'T3 vs GN p N={n}', d['table3_gn'][n]['p'], p, 0.005)
chk('T1 mean', d['x61']['T1']['diff_mean'], -0.062, 0.01)
chk('T1 ci lo', d['x61']['T1']['ci90'][0], -0.114, 0.01)
chk('T1 ci hi', d['x61']['T1']['ci90'][1], -0.011, 0.01)
chk('T2 max', max(d['x61']['T2_max_abs'].values()), 0.0141, 0.002)
chk('BH Kinova edge', d['bh_socket']['reduced edge Kinova']['q'], 0.483, 0.01)
chk('BH pj Kinova', d['bh_socket']['per-joint Kinova']['q'], 0.111, 0.01)

bad = 0
for name, got, want, ok in checks:
    print(f"{'OK  ' if ok else 'FAIL'} {name:20s} got {got:+.4f} want {want:+.4f}")
    bad += not ok
print(f"{len(checks)-bad}/{len(checks)} OK")
sys.exit(1 if bad else 0)
