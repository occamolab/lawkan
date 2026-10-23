#!/usr/bin/env python3
"""Check the external ABB RobotCali xlsx (name, size and SHA256)."""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, 'data', 'robotcali', 'RobotCali-new.xlsx')
EXPECTED_SIZE = 73894
EXPECTED_SHA256 = '724dbbeb90f6469dc1ce9273618df3326a0791aa8a118c79c3c8e59dcf83d7aa'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default=os.environ.get('ABB_XLSX', DEFAULT))
    args = ap.parse_args()
    if not os.path.exists(args.path):
        print(f'MISSING {args.path}')
        print('ABB RobotCali is not redistributed (no license); see REPRODUCE.md, section Data.')
        return 1
    h = hashlib.sha256()
    with open(args.path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    size, digest = os.path.getsize(args.path), h.hexdigest()
    print(f'path   {args.path}')
    print(f'size   {size} (expected {EXPECTED_SIZE})')
    print(f'sha256 {digest}')
    print(f'       (expected {EXPECTED_SHA256})')
    ok = size == EXPECTED_SIZE and digest == EXPECTED_SHA256
    print('OK' if ok else 'MISMATCH')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
