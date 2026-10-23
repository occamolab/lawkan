# Contributing

Thanks for helping. This repository is the reproduction artifact for the LawKAN paper: the
priority is that every number stays checkable.

- Before opening a pull request, run:

  ```bash
  python repro/check_canonical.py   # every canonical dump exists and is tracked
  python repro/check_numbers.py    # paper headline numbers vs the canonical dumps
  ```

- Do not edit files in `repro/json/` by hand: dumps are written by the scripts through
  `repro/dump.py` and pinned in `repro/canonical.json`.
- Keep the repository in English (code, comments, docs).
- For a number that disagrees with your rerun, open the reproduction-mismatch issue
  template first, with your `run.log`.
