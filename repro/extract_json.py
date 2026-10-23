#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Extract the machine-readable SUMMARY-JSON object from a script log.

Usage: extract_json.py <log_file> <out_json>
Finds the LAST occurrence of a line containing 'SUMMARY-JSON', then reads the
first JSON object that follows (brace-matched, string-aware) and validates it.
Exit codes: 0 ok, 2 no marker, 3 no '{', 4 unbalanced, 5 parse error.
"""
import sys, json

log, out = sys.argv[1], sys.argv[2]
txt = open(log, encoding="utf-8", errors="replace").read()

idx = txt.rfind("SUMMARY-JSON")
if idx < 0:
    print("NO_SUMMARY"); sys.exit(2)

j = txt.find("{", idx)
if j < 0:
    print("NO_JSON"); sys.exit(3)

depth = 0; instr = False; esc = False; end = -1
for k in range(j, len(txt)):
    c = txt[k]
    if instr:
        if esc: esc = False
        elif c == "\\": esc = True
        elif c == '"': instr = False
    else:
        if c == '"': instr = True
        elif c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = k + 1; break
if end < 0:
    print("UNBALANCED"); sys.exit(4)

try:
    d = json.loads(txt[j:end])
except Exception as e:
    print("PARSE_ERR", e); sys.exit(5)

json.dump(d, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("OK", len(d))
