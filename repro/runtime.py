#!/usr/bin/env python3
"""Device/dtype and data-path resolution for the reproduction harness."""
import os
import shutil

DATA_ENV = ('BROAD', 'CALIB', 'ABB_XLSX', 'LAWKAN_DATA')
HINTS = {
    'BROAD': 'BROAD data_hdf5 directory (set BROAD)',
    'CALIB': 'FR3 calib_data.pickle (set CALIB)',
    'ABB_XLSX': 'ABB RobotCali xlsx (set ABB_XLSX; not redistributed)',
    'LAWKAN_DATA': 'data directory with philnad/ipa/pauljac (set LAWKAN_DATA)',
}


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    'CALIB': os.path.join(ROOT, 'data', 'calib_data.pickle'),
    'LAWKAN_DATA': os.path.join(ROOT, 'data'),
    'ABB_XLSX': os.path.join(ROOT, 'data', 'robotcali', 'RobotCali-new.xlsx'),
}


def expand_args(args):
    out = []
    for a in args:
        a = a.strip('"').strip("'")
        for name, default in DEFAULTS.items():
            if '${%s}' % name in a and name not in os.environ and os.path.exists(default):
                os.environ[name] = default
        b = os.path.expandvars(a)
        if '${' in b:
            name = b.split('${', 1)[1].split('}', 1)[0]
            raise SystemExit(f'unset data variable {name}: {HINTS.get(name, "")}')
        out.append(b)
    return out


def resolve(mode, entry, device=None, dtype=None):
    dev = device or entry.get('device', 'cpu')
    if device is None and mode in ('smoke',):
        dev = 'cpu'
    if device is None and mode == 'gpu':
        dev = 'cuda'
    if device is None and mode == 'reference':
        dev = entry.get('device', 'cpu')
    dt = dtype or entry.get('dtype', 'float64')
    if dt == 'auto':
        dt = 'float32' if dev == 'cuda' else 'float64'
    if dev == 'cuda' and not cuda_available():
        raise SystemExit(f'block requires CUDA but none is available (mode={mode})')
    return dev, dt
