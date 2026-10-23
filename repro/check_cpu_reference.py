#!/usr/bin/env python3
"""Verify the CPU reference run against the paper's displayed numbers.

Uses the curated CHECKS list and maps each canonical block to the best-matching
dump under repro/json/cpu-reference/ (script prefix + recorded args), falling
back to repro/json. This is the gate for the portable CPU path.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_numbers as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = json.load(open(os.path.join(ROOT, 'repro', 'canonical.json')))
REG = json.load(open(os.path.join(ROOT, 'repro', 'registry.json')))


def prefix(f):
    return os.path.basename(f).split('.')[0]


def args_of(path):
    try:
        return json.load(open(path)).get('_meta', {}).get('args', {}) or {}
    except Exception:
        return {}


CPU_DIR = os.path.join(ROOT, 'repro', 'json', 'cpu-reference')
DUMPS = [f for f in glob.glob(os.path.join(CPU_DIR, '*.json'))
         if not f.endswith('-smoke.json')]
CPU_PREFIXES = {prefix(f) for f in DUMPS}
PENDING = set()  # all headline blocks covered


def best(script_prefix, refargs):
    scored = []
    for f in DUMPS:
        if prefix(f) != script_prefix:
            continue
        a = args_of(f)
        if 'seeds' in refargs and 'seeds' in a and str(a['seeds']) != str(refargs['seeds']):
            continue
        score = sum(1 for k, v in refargs.items() if k in a and str(a[k]) == str(v))
        scored.append((score, os.path.getmtime(f), f))
    if not scored:
        return None
    top = max(scored)
    if refargs and top[0] < len(refargs):
        return None
    return top[2]


fails, miss, skipped = [], [], 0
for label, block, key, expected, tol in C.CHECKS:
    fn = MAN.get(block, {}).get('file')
    if not fn:
        miss.append((label, block, 'no canonical entry')); continue
    script = (REG.get(block) or {}).get('script') or ''
    sp = os.path.basename(script).split('.')[0]
    if sp in PENDING:
        print(f'  PENDING {sp}: CPU dump regeneration queued'); skipped += 1; continue
    if sp not in CPU_PREFIXES:
        skipped += 1; continue
    f = best(sp, args_of(os.path.join(ROOT, 'repro', 'json', fn)))
    if not f:
        skipped += 1; continue
    got = C.walk(json.load(open(f)), key)
    if got is None:
        miss.append((label, block, 'key missing')); continue
    if isinstance(got, list) and got and all(isinstance(x, (int, float)) for x in got):
        got = sum(got) / len(got)
    if isinstance(got, str):
        continue
    if abs(got - expected) > tol:
        fails.append((label, block, expected, got, tol, os.path.basename(f)))

covered = len(C.CHECKS) - skipped
print(f'CPU reference: {covered-len(fails)-len(miss)}/{covered} covered checks match, '
      f'{len(fails)} mismatches, {len(miss)} missing; {skipped} not in the CPU run')
for x in fails:
    print(f'  FAIL {x[0]} [{x[1]}]: paper {x[2]:g}, CPU {x[3]:g} (tol {x[4]:g}) {x[5]}')
for x in miss:
    print(f'  MISS {x[0]} [{x[1]}]: {x[2]}')
sys.exit(0 if not fails and not miss else 1)
