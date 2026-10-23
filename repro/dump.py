#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single dump helper: every experiment writes its SUMMARY-JSON through this.

Convention (2026-09-27):
  * default path  repro/json/<script>.<YYYYMMDD-HHMMSS>.json
  * --quick runs  repro/json/<script>.<YYYYMMDD-HHMMSS>-smoke.json
    (a smoke run can never overwrite a canonical dump)
  * `_meta` is merged in: git HEAD/branch, date, seeds, seed_start, epochs,
    argparse dict, quick flag, torch version.

Fixed output filenames are forbidden; use `--out` only for explicit overrides.
"""
import json
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _git(root):
    def run(*a):
        try:
            return subprocess.check_output(
                ['git', '-C', root, *a], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return 'unknown'
    return run('rev-parse', 'HEAD'), run('rev-parse', '--abbrev-ref', 'HEAD')


def save_dump(results, script_path, *, quick=False, args=None, seeds=None,
              seed_start=None, epochs=None, out=None, extra_meta=None):
    """Write `results` to a dated JSON dump and return the path.

    script_path : pass __file__ from the caller.
    out         : explicit override (use sparingly, e.g. repro harness).
    """
    root = _REPO  # dumps always land in <repo>/repro/json, wherever the script lives
    name = os.path.splitext(os.path.basename(script_path))[0]
    ts = time.strftime('%Y%m%d-%H%M%S')
    if out is None:
        suffix = '-smoke' if quick else ''
        out = os.path.join(root, 'repro', 'json', f'{name}.{ts}{suffix}.json')
    head, branch = _git(root)
    try:
        import torch
        torch_v = torch.__version__
    except Exception:
        torch_v = 'unknown'
    meta = dict(script=os.path.basename(script_path), git_head=head,
                git_branch=branch, date=time.strftime('%Y-%m-%d %H:%M:%S'),
                quick=bool(quick), seeds=seeds, seed_start=seed_start,
                epochs=epochs, torch=torch_v)
    if args is not None:
        meta['args'] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                            else str(v)) for k, v in vars(args).items()}
    if extra_meta:
        meta.update(extra_meta)
    data = dict(results)
    prev = data.get('_meta', {})
    data['_meta'] = {**prev, **meta}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=float)
    print('Saved:', out, flush=True)
    return out
