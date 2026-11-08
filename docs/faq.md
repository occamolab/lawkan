# FAQ

## What is LawKAN?

A self-contained group-law layer for **composition under a known group law**. The **node** composes
elements by the exact group law and has no parameters itself; the **edge** learns only
the local step (a map from the input chart into the group's increment); and a small
**zero-initialized correction to the law** (ψ) modifies it only where the hardware deviates from
the law. It is trained in isolation. Written as a layer: `edge → (ψ) → exp → group product`.

## Is LawKAN only for rotations?

No. Composition on SU(2) (rotation composition) is the leading instance, but the paper develops and tests
the same recipe on three groups — compact SU(2), non-compact GL⁺(2,ℝ), mixed SE(3) — and a
different group law follows the same recipe with a different chart, edge target and node
realization. The README's compact example shows the SU(2) case only for brevity.

## How is this different from a KAN?

A KAN splits a layer into learnable univariate edges and additive nodes; that split is exact
for addition and wrong for multiplication. LawKAN keeps the learned edges but makes the
**node** the known group law (a non-commutative product), so composition is never learned.
There is a pre-training criterion for exactness (the edge must carry the right coordinates
for the step), a law-match corollary, a pruning rule (keep the chart's analytic support),
and the zero-initialized law correction.

## What exactly is learned, and what is fixed?

The chart and the group law are fixed; the node applies the exact law and has no parameters
by default. The edge learns the local increment (its coefficients are fitted from data), and
the optional zero-initialized ψ is the only learned deformation of the law. The architecture
is written `edge → (ψ) → exp → group product`; the edge does not relearn composition.

## How small is the model?

The compact SU(2) instance uses a 54-parameter edge at K=1 (18 parameters in the pruned
analytic-support edge), and the node adds no parameters. The FR3 law correction is 306
parameters. On the synthetic matrix this is 27× fewer parameters than the strongest matrix
opponent and 92× fewer than the strongest tuned opponent — the point of the paper is a small,
structured model, not a large one.

## Is this just an equivariant or quaternion network?

No. Equivariance constrains how features transform; it does not by itself supply the exact
non-commutative composition map. LawKAN fixes the composition operation in the node and
learns only the local increment; a quaternion or 6D output head is a parameterization choice,
not the same architectural claim. The paper tests compact SU(2), non-compact GL⁺(2,ℝ) and
mixed SE(3) with the same recipe.

## Doesn't the correction just compensate for a weak edge?

That is the compensation hypothesis, and it is tested directly. ψ is trained with the exact
edge frozen (isolation); joint training is only a control and stays within 0.07 mm of
isolation on the FR3 arm. The correction-vs-edge decomposition finds 88% of the correction's
effect field cannot be absorbed by the edge (G3/G3b), and a node-only learned law without the
per-joint edge overfits at small N. The correction is law-level in this setting, not a richer
edge in disguise.

## What compute do I need to reproduce it?

The headline path is CPU-first and runs overnight on four cores; the `smoke` tier takes
minutes and checks that every headline block starts and dumps. GPU (`--mode gpu`) is optional
and faster, but the portable reference numbers were recorded on CPU. Full-tier runs, including
the pole-fix branch, take longer.

## What is the prediction protocol?

Every experiment pre-specified its prediction in the authors’ internal experiment log before the run, and refuted predictions are kept and reported. The paper reports **130 verdicts** (59 main battery + 30 validity + 30 revision
+ 11 pole-fix branch); the public log has 119 per-entry rows and the paper's Appendix K
summarizes the 11 branch probes.

## Is LawKAN available as a pip package?

Not yet. This repository is the research artifact: scripts, pinned dumps, registry-driven
runner, checks and figure generators. Install with `pip install -r requirements.txt` and run
from the repository root; the span check (`tools/span_check.py`) is the quickest way to test
whether the recipe applies to your own task.

## What are the headline results?

See [results.md](results.md). In short: on the synthetic composition battery LawKAN uses
27× fewer parameters than the strongest matrix opponent (92× fewer than the strongest tuned
opponent) and sits 0.05° above the matched Bayes floor. On real arm calibration the isolated
306-parameter two-slot correction matches or beats both the exact model and the same-class
Gauss–Newton fit at every training size (1.12 mm vs 1.33 mm at N=80; significant at N=40/80,
within noise at N=20/60). It replicates across three robot vendors and two measurement
protocols (pooled p < 10⁻¹⁰). Where the classical method already carries the same law, the
exact node reaches parity on real sensors and robots.

## Is the correction a proof that the law is right?

No. It starts exact (zero-initialized) and moves only where the data require it, so its
deviation from zero is a *test*, not a proof: on data that obey the law it gives no held-out gain. Its shape
localizes which inputs carry the deviation. Joint training of edge and ψ is a compensation
control, not the recommended protocol.

## Where does it fail?

- Tasks with no group law (a plain MLP wins).
- Hidden rotation angle unrelated to any input (information-limited).
- The Kinova Gen3 lite correction is seed-unstable (2/10 collapse; TOST fails) — localized and
  fixed in the revision round and pole-fix branch, where a two-slot global+local head
  removes the collapse (X22–X61); the dense classical screw class still wins on the FR3 unit
  — measured attribution boundaries.
- The dense IMU edge is seed-bimodal at long horizons; the pruned 18-parameter edge is the
  deterministic one.

## How do I reproduce the paper numbers?

```bash
python repro/run_registry.py --tier headline --mode reference   # CPU reference, overnight
bash repro/run_all.sh headline-smoke  # minutes-scale sanity
python repro/check_canonical.py       # 120 canonical blocks, all tracked
python repro/check_numbers.py         # 171 paper numbers vs the dumps
python repro/stats/check_derived.py   # derived statistics (18 checks)
python repro/check_cpu_reference.py   # portable CPU-run verification
```

Data prerequisites and the paper-item → command map are in `REPRODUCE.md`.

## Why do some dump keys say `QuatKAN`?

`QuatKAN` is the internal historical spelling kept in class names and JSON keys as
provenance; the family is called LawKAN.

## What is the license?

AGPL-3.0-or-later applies to the original code and documentation: research use is free;
derivative code must remain open. Third-party datasets retain their original licenses (MIT or
CC BY 4.0); ABB RobotCali has no explicit license and is not redistributed. See
`THIRD_PARTY.md` and `data/FR3_LICENSE`. Commercial licensing of the original code is available
on request.

## How should I cite it?

Use `CITATION.cff`; the paper reference and BibTeX will be added when the preprint is posted.
