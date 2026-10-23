#!/usr/bin/env python3
"""Resolve experiment dumps through repro/canonical.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, 'repro', 'canonical.json')


def file(block):
    with open(MANIFEST, encoding='utf-8') as fh:
        entry = json.load(fh)[block]
    return os.path.join(ROOT, 'repro', 'json', entry['file'])


def load(block):
    with open(file(block), encoding='utf-8') as fh:
        return json.load(fh)
