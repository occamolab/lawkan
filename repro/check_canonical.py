#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""check_canonical.py -- verify the canonical dump manifest.

Every entry of repro/canonical.json must resolve to a dump that (a) exists on
disk and (b) is tracked by git, so a fresh clone can regenerate every paper
table. Tools must resolve through the manifest, never by "latest filename".

Run:  python repro/check_canonical.py
Exit: 0 -- every entry OK; 1 -- missing/untracked/pending entries.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "repro", "json")
MAN = os.path.join(ROOT, "repro", "canonical.json")


def main():
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
        check=True).stdout.split())
    with open(MAN, encoding="utf-8") as fh:
        man = json.load(fh)

    problems = 0
    total = 0
    for block, entry in sorted(man.items()):
        if block.startswith("_"):
            continue
        total += 1
        if block.lower().startswith(("e27", "e28")) or str(entry.get("file", "")).upper().startswith(("E27", "E28")):
            print(f"FORBIDDEN   {block:16s} excluded from the public artifact")
            problems += 1
            continue
        name = entry.get("file")
        if not name:
            print(f"PENDING     {block:16s} {entry.get('note', 'no dump yet')}")
            problems += 1
            continue
        rel = os.path.join("repro", "json", name)
        on_disk = os.path.exists(os.path.join(J, name))
        in_git = rel in tracked
        if on_disk and in_git:
            status = "OK"
        elif not on_disk:
            status = "NOT-ON-DISK"
            problems += 1
        else:
            status = "UNTRACKED"
            problems += 1
        print(f"{status:11s} {block:16s} {name}")

    print(f"\n{total} blocks, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
