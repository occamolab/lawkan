# LawKAN: group-law layers for composition — when imposing a known law pays off

*Code and reproduction artifact: <https://github.com/occamolab/lawkan>. The preprint
link is added upon posting; see `CITATION.cff`. This mirrors the manuscript abstract; the
typeset version is in the paper.*

## Abstract

Many tasks compose transformations by a known rule: chained rotations in robotics and
navigation. Learned models relearn it, compounding error; classical estimators hardcode it and
miss hardware deviations. We impose the known structure and learn only the remainder: an exact
non-commutative group law sits in the node of a self-contained layer, its edges learn local
increments, and a zero-initialized correction ψ learns unknown deviations of the law. A
pre-training criterion predicts exact representability. On fixed-angle SU(2) composition, a
54-parameter dense edge is the most accurate dense model, at 27× fewer parameters than the
strongest fixed-matrix baseline, and lies 0.05° above the Bayes floor. On real robot arms, an
isolated 306-parameter ψ reaches 1.12–1.14 mm versus 1.29–1.33 mm for Gauss–Newton at N ≥ 60, and
the recipe transfers across three vendors and two protocols. The recipe matches classical fits
where the data obey the law and loses where the imposed law is not the task's.

**Data and code.** The code is available under AGPL-3.0-or-later; third-party datasets retain
their own licenses. See [`../THIRD_PARTY.md`](../THIRD_PARTY.md) for the full list. The ABB RobotCali
dataset has no explicit license and is not redistributed.

## Keywords

Kolmogorov–Arnold networks · KAN · group law · transformation composition · Lie groups ·
geometric deep learning · robot-arm calibration · inertial navigation · reproducibility ·
parameter efficiency · tiny models.

## Where to find what in this repository

- Headline numbers per probe: [`results.md`](results.md)
- Questions and scope: [`faq.md`](faq.md); terms: [`glossary.md`](glossary.md)
- Script-by-script map to the paper: [`blocks.md`](blocks.md)
- Registered-predictions log (prediction → outcome → verdict): [`predictions.md`](predictions.md)
- Reproduction guide and data: [`../REPRODUCE.md`](../REPRODUCE.md)
