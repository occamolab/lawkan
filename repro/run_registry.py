#!/usr/bin/env python3
"""Registry-driven reproduction runner.

Examples:
  python repro/run_registry.py --tier headline --dry-run
  python repro/run_registry.py --tier headline --mode reference
  python repro/run_registry.py --only x46,x35 --mode reference
  python repro/run_registry.py --tier full --mode smoke
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'repro'))
import runtime  # noqa: E402

REG = json.load(open(os.path.join(ROOT, 'repro', 'registry.json')))
SMOKE = {'mk1', 'mk1_fr3', 'rc1', 'x6b', 'x7vqf_disturbed', 'x16iso', 'x10'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='headline', choices=['smoke', 'headline', 'full', 'all'])
    ap.add_argument('--only', default='')
    ap.add_argument('--mode', default='reference', choices=['reference', 'gpu', 'smoke'])
    ap.add_argument('--device', default=None, choices=[None, 'cpu', 'cuda', 'auto'])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--allow-missing', action='store_true')
    ap.add_argument('--log-dir', default=os.path.join(ROOT, 'repro', 'logs'))
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True)

    only = [s.strip() for s in a.only.split(',') if s.strip()]
    if a.tier == 'smoke':
        sel = [k for k in REG if k in SMOKE]
    elif a.tier == 'headline':
        sel = [k for k in REG if REG[k]['tier'] == 'headline']
    elif a.tier == 'full':
        sel = list(REG)
    else:
        sel = list(REG)
    if only:
        sel = [k for k in only if k in REG]

    ts = time.strftime('%Y%m%d-%H%M%S')
    ok = fail = skipped = 0
    for key in sorted(sel):
        e = REG[key]
        if e['status'] != 'ok' and not a.allow_missing:
            print(f'SKIP {key}: {e["status"]}'); skipped += 1; continue
        if not e.get('script'):
            print(f'SKIP {key}: no script'); skipped += 1; continue
        try:
            mode = 'smoke' if a.tier == 'smoke' else a.mode
            dev, dt = runtime.resolve(mode, e, a.device, None)
            extra = ['--quick'] if mode == 'smoke' and key in SMOKE else []
            args = runtime.expand_args(e.get('args', [])) + extra
        except SystemExit as ex:
            print(f'SKIP {key}: {ex}'); skipped += 1; continue
        cmd = [sys.executable, os.path.join(ROOT, e['script'])] + args
        env = dict(os.environ)
        if dev == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''   # scripts pick device at import
        print(f'[{key}] {dev}/{dt}: ' + ' '.join(cmd[1:]))
        if a.dry_run: continue
        log = os.path.join(a.log_dir, f'{key}.{ts}.log')
        t0 = time.time()
        with open(log, 'w') as fh:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        wall = time.time() - t0
        print(f'  rc={rc} wall={wall/60:.1f} min log={os.path.relpath(log, ROOT)}')
        ok += rc == 0; fail += rc != 0
    print(f'\nselected {len(sel)}: ok={ok} fail={fail} skipped={skipped}')
    return 1 if fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
