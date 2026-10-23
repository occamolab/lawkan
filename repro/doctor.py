#!/usr/bin/env python3
"""Environment preflight for the reproduction harness."""
import argparse, importlib, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--strict", action="store_true", help="fail on missing optional data")
args = ap.parse_args()
ok = True


def check(label, cond, detail=''):
    global ok
    print(f"{'OK  ' if cond else 'FAIL'}  {label:28s} {detail}")
    ok = ok and bool(cond)


print(f"python {sys.version.split()[0]}  ({sys.executable})")
for mod in ('numpy', 'scipy', 'torch'):
    try:
        m = importlib.import_module(mod)
        check(mod, True, getattr(m, '__version__', ''))
    except Exception as e:
        check(mod, False, str(e))

try:
    import torch
    check('cuda (optional)', True, 'available' if torch.cuda.is_available() else 'not available')
except Exception:
    pass

calib = os.environ.get('CALIB', os.path.join(ROOT, 'data', 'calib_data.pickle'))
check('FR3 data', os.path.exists(calib), calib)
for var, hint in (('BROAD', 'BROAD data_hdf5'), ('ABB_XLSX', 'ABB RobotCali xlsx'),
                  ('LAWKAN_DATA', 'multi-dataset dir (ipa/pauljac)')):
    val = os.environ.get(var)
    present = bool(val and os.path.exists(val))
    if args.strict:
        check(var, present, val or f"unset ({hint})")
    else:
        print(f"{'OK  ' if present else 'MISS'}  {var:28s} "
              f"{val or f'unset ({hint})'}")
check('git', shutil.which('git') is not None)
sys.exit(0 if ok else 1)
