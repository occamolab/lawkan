# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attach provenance `_meta` to harness-extracted canonical dumps.

Dumps written through repro/dump.py already carry `_meta`; dumps extracted by
repro/extract_json.py (run_all.sh) do not. This script stamps the latter with
the git HEAD recorded in repro/logs/env.<ts>.txt plus the run timestamp parsed
from the dump filename. Idempotent: files that already have `_meta` are skipped.

Usage:
    python repro/add_meta.py           # stamp, report
    python repro/add_meta.py --check   # report only
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
JSONDIR = os.path.join(ROOT, "json")
LOGDIR = os.path.join(ROOT, "logs")
NAME_RE = re.compile(r"^(?P<stem>.+?)\.(?P<ts>\d{8}-\d{6})(?P<smoke>-smoke)?\.json$")


def env_head(ts):
    path = os.path.join(LOGDIR, f"env.{ts}.txt")
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"^git HEAD:\s*(\S+)\s+(\S+)", line)
            if m:
                return m.group(1), m.group(2)
    return None, None


def main():
    check = "--check" in sys.argv
    canonical_path = os.path.join(ROOT, "canonical.json")
    with open(canonical_path) as fh:
        canonical = json.load(fh)

    stamped, skipped, missing = 0, 0, 0
    for block, entry in sorted(canonical.items()):
        if not isinstance(entry, dict) or not entry.get("file"):
            continue
        path = os.path.join(JSONDIR, entry["file"])
        if not os.path.exists(path):
            print(f"[missing] {block}: {entry['file']}")
            missing += 1
            continue
        with open(path) as fh:
            dump = json.load(fh)
        if "_meta" in dump:
            skipped += 1
            continue
        m = NAME_RE.match(entry["file"])
        ts = m.group("ts") if m else None
        head, short = env_head(ts) if ts else (None, None)
        date = entry.get("run_date") or (
            f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}+03:00"
            if ts else None
        )
        meta = {
            "source": "repro/run_all.sh (extract_json.py)",
            "date": date,
            "git_head": head,
            "git_head_short": short,
            "env_log": f"repro/logs/env.{ts}.txt" if ts else None,
            "note": entry.get("note"),
        }
        if check:
            print(f"[would stamp] {block}: {entry['file']} head={short}")
            stamped += 1
            continue
        dump["_meta"] = meta
        with open(path, "w") as fh:
            json.dump(dump, fh, ensure_ascii=False, indent=2)
        print(f"[stamped] {block}: {entry['file']} head={short}")
        stamped += 1
    print(f"\nstamped {stamped}, already had _meta {skipped}, missing {missing}")


if __name__ == "__main__":
    main()
