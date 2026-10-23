#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate paper tables from the canonical SUMMARY-JSON dumps.

Resolution order (2026-09-27):
  1. repro/canonical.json   -- the single manifest: block -> dump file;
  2. fallback: latest <prefix>.<ts>.json (with a warning).

`repro/json/_stale/` is never read. New experiment prefixes are included.

Usage:
    python repro/make_tables.py                 # canonical dumps
    python repro/make_tables.py --out tables.md
    python repro/make_tables.py --prefix run_main
"""
import argparse
import glob
import json
import os
import re

PREFIXES = ["run_main", "run_ext", "e16v2", "e19", "e20", "e21", "e22",
            "e23", "e24", "e25", "e25b", "e27c", "e27d", "e27e", "e28",
            "e29", "e30", "x5", "x6", "x7", "x8", "x8t", "x9", "x10", "bayes"]


def load_manifest(d):
    path = os.path.join(os.path.dirname(d), "canonical.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def latest_dump(d, prefix):
    ts = re.compile(rf"^{re.escape(prefix)}\.(\d{{8}}-\d{{6}})(-smoke)?\.json$")
    cands = []
    for f in glob.glob(os.path.join(d, "*.json")):
        m = ts.match(os.path.basename(f))
        if m and not m.group(2):
            cands.append((m.group(1), f))
    if not cands:
        return None
    return sorted(cands)[-1][1]


def resolve(d, manifest, prefix):
    entry = manifest.get(prefix)
    if entry and entry.get("file"):
        f = os.path.join(d, entry["file"])
        if os.path.exists(f):
            return f, None
        return None, f"canonical file missing: {entry['file']}"
    if entry and entry.get("file") is None:
        return None, f"no canonical dump yet ({entry.get('note', 'pending')})"
    f = latest_dump(d, prefix)
    if f is None:
        return None, "no dump found"
    return f, f"no manifest entry; fell back to latest {os.path.basename(f)}"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(here, "json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()

    manifest = load_manifest(args.dir)
    prefixes = [args.prefix] if args.prefix else sorted(
        k for k in manifest if not k.startswith("_"))
    if not prefixes:
        prefixes = PREFIXES
    out = []
    for p in prefixes:
        f, warn = resolve(args.dir, manifest, p)
        if f is None:
            out.append(f"## {p}\n\n_{warn}_\n")
            continue
        if warn:
            out.append(f"> WARNING [{p}]: {warn}")
        d = json.load(open(f, encoding="utf-8"))
        out.append(f"## {p}  (source: `{os.path.relpath(f, os.path.dirname(here))}`)\n")
        out.append("| key | mean | std | params |")
        out.append("|---|---|---|---|")
        texts = []
        for k in sorted(d):
            v = d[k]
            if not isinstance(v, dict):
                continue
            if "mean" in v:
                params = v.get("params", "")
                mean = v["mean"]
                if isinstance(mean, (int, float)):
                    out.append(f"| {k} | {mean:.5g} | {v.get('std', '')} | {params} |")
                else:
                    out.append(f"| {k} | {mean} | {v.get('std', '')} | {params} |")
            elif "text" in v:
                texts.append((k, v["text"]))
        out.append("")
        if texts:
            out.append("_text entries:_")
            for k, t in texts:
                out.append(f"- **{k}**: {t}")
            out.append("")
    md = "\n".join(out)
    print(md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
        print(f"\n[make_tables] wrote {args.out}")


if __name__ == "__main__":
    main()
