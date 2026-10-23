# LawKAN: group-law layers for composition — when imposing a known law pays off

*Code and reproduction artifact: <https://anonymous.4open.science/r/lawkan-6A4A/>. The preprint
link is added upon posting; see `CITATION.cff`. This is a condensed repository abstract; the
manuscript abstract is in the paper.*

## Abstract

Many tasks compose transformations with a known rule: chained rotations in robotics and
navigation. Learned models relearn it, compounding error; classical estimators hard-code it but
miss hardware deviations. We impose the known structure and learn only the remainder. An exact
non-commutative group law sits in the node of a self-contained layer; its edges learn local
increments, and a zero-initialized correction ψ learns unknown deviations. Where data obey the
law, the model is the classical estimator it embeds; where hardware deviates, ψ removes the floor
those estimators leave. A pre-training criterion predicts exact representability. On fixed-angle
SU(2) composition, a 54-parameter dense edge is the most accurate dense model with 27× fewer
parameters than the strongest fixed-matrix baseline (92× fewer than the tuned one) and 0.05°
above the Bayes floor; pruning to 18 gives 2.75°. On real arms, an isolated 306-parameter ψ
reaches 1.12–1.14 mm versus 1.29–1.33 mm for Gauss–Newton (exact 1.37–1.40 mm), transferring
across three vendors and two protocols. On sensors and arms it matches classical fits under the
same law and fails without one.

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
