# Reproducibility harness

Run the experiment battery and regenerate the machine-readable dumps that the
paper tables are built from.

## Usage

```bash
bash repro/run_all.sh stage1            # paper core: run_main, run_extensions, E16, E19, bayes
bash repro/run_all.sh stage2            # real data: X5, X6, X8, X8t, X9, X7
bash repro/run_all.sh stage3            # expansion (registered seeds): E20..E24
bash repro/run_all.sh stage4            # state-conditioned edges: E25, E25b
bash repro/run_all.sh stage5            # variable-angle theorem + full-SO(3): E29, E30
bash repro/run_all.sh stage2 x7         # optional <name> filter: run one experiment

bash repro/run_all.sh stage6            # three-platform FK calibration (X10, P48)
bash repro/run_all.sh stage7            # phase-2: E31 (P49), E32 (P50), E33
bash repro/run_all.sh stage8            # phase-2 cont.: E31b/c/d, E33b/c, E34

bash repro/run_all.sh stage9            # MUKCa socket family: KUKA/Panda/Kinova (10 seeds), FR3 unit, cross-view
bash repro/run_all.sh stage10           # ABB RobotCali range-only (10 seeds)
bash repro/run_all.sh stage11           # X6b (TinyGC-style), X7 vqf reference, X16 isolated
bash repro/run_all.sh stage12           # appendix diagnostics: G2a/G2bc/G3/G4, X17, S8
bash repro/run_all.sh stage13           # validity package: X18/X18b, X19, X20/X21, POOL
bash repro/run_all.sh stage14           # revision round: X22-X34
bash repro/run_all.sh stage15           # pole-fix branch: X35-X61 (registry-driven)

bash repro/run_all.sh headline          # main-text tables/figures end to end (CPU reference, overnight)
bash repro/run_all.sh headline-smoke    # minutes-scale sanity of the quick-capable headline blocks
```

Outputs (under `repro/`):
- `logs/<name>.<ts>.log` — full stdout/stderr (git-ignored)
- `json/<name>.<ts>.json` — extracted SUMMARY-JSON (committed)
- `logs/env.<ts>.txt` — environment snapshot (git HEAD, torch/CUDA, GPU)
- `MANIFEST.<ts>.tsv` — script, command, exit code, wall time, json path

Wall times on the reference GPU (RTX 2080 SUPER): `run_main` ≈ **103 min**,
`run_extensions` ≈ 44 min, `E19` ≈ 51 min, `E16v2` ≈ 6 min, `E20–E24` ≈ 10–47 min,
`E25` ≈ 32 min, `E25b` ≈ 13 min, `X5` ≈ 31 min, `X6` ≈ 10 min, `X8`/`X8t` ≈ 12/35 min,
`X9` ≈ 47 min, `X7` ≈ 52 min,
`E29` ≈ 15 min, `E30` ≈ 10 min.
New blocks (stage9–11, headline): `MK1` front trio at 10 seeds ≈ 70 min, `MK1` FR3 unit ≈ 15 min,
`MK1` cross-view (2 sessions) ≈ 25 min, `RC1` at 10 seeds ≈ 210 min, `X6b` ≈ 12 min,
`X7` vqf ≈ 4 min (both groups), `X16` isolated ≈ 15 min; the full `headline` preset is an
overnight CPU run (the original GPU timings were 8–10 h). The `x16iso` block regenerates
Fig. 3(b) and the G.4 numbers; `mk1`/`mk1views`/`kinova` feed Table 4 and the attribution map.
`x7cal` denotes the older `X7_robot_calibration.py` (stage2); `x7vqf` is the VQF reference
(stage11).

## Canonical dumps

`canonical.json` is the single source of truth: it maps every experiment block
cited in the paper (120 blocks) to the one dump that the text/tables must match.
Tools (`make_tables.py`, `check_numbers.py`, `POOL_verdict.py` and the figure scripts)
resolve through the manifest, never by "latest filename" (a few figure scripts pin an
explicit dump; see `tools/figures/README.md`).
The X19 verdicts additionally use the archived training log
`repro/provenance/x19_full.log` (the original aggregation crashed post-training). Superseded dumps are moved to
`json/_stale/` and are not cited.

Self-dumping scripts (all newer probes: `E20`--`E40`, `X5`--`X61`, `S*`, `G*`, `MK1`,
`RC1`, `POOL`) go through `repro/dump.py`: dumps are dated (`<script>.<ts>.json`),
carry a `_meta` block (git HEAD, date, seeds, args), and any run in smoke/quick
mode is suffixed `-smoke` so it can never overwrite a canonical dump.
Harness-extracted dumps get their `_meta` stamped by `repro/add_meta.py` from
the matching `logs/env.<ts>.txt`.

## Paper tables

```bash
python repro/make_tables.py                 # markdown tables for every experiment (canonical dumps)
python repro/make_tables.py --prefix run_main --out tables.md
```

`make_tables.py` iterates the blocks of `canonical.json` and emits
`key | mean | std | params` rows plus the text entries; if a block has no entry
it warns and falls back to the latest non-`_stale` dump. This is the
"mechanical" JSON → table step referenced in the top-level README. The manifest
itself is verified by `repro/check_canonical.py` (every entry must exist on disk
and be tracked by git).

## Provenance caveat

Dumps produced **before 2026-09-25** (e.g. the early `e19.20260924-082107.json`) record
`git HEAD: 6b9f8dc` in their `env.<ts>.txt`; that commit was **removed from
history** during a cleanup (removal of accidentally committed non-English files),
so it is no longer reachable. Those env logs are historical only. Dumps produced
from 2026-09-25 on (e.g. the canonical `run_main.20260927-165803.json`, head `48a2402`) carry
the current, reachable HEAD and are fully traceable.
