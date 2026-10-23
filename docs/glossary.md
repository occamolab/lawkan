# Glossary

**LawKAN** — the architecture family in this paper: a self-contained group-law layer whose
node applies a known group law exactly and whose edge learns only the local increment,
optionally with a zero-initialized law correction ψ. It keeps the KAN edge/node split, but
the node composes by a non-commutative group law rather than adding. The layer is written
`edge → (ψ) → exp → group product`.

**chart (χ)** — a coordinate system on the group; a map from the input to a group element,
used by the edge as its feature space.

**edge** — the layer's learned map: a function from the input chart to the group's increment
whose coefficients are fitted from data (a Lie-algebra vector where the exponential is global,
otherwise a group element).

**node** — the fixed part of the layer: it applies the group law exactly (exp, then the
group product). It has no parameters itself; a zero-initialized correction $\psi$ modifies it only
where the hardware deviates from the law, and at $\psi{=}0$ the node is exact.

**group law** — the rule for composing two group elements into a third (e.g. the Hamilton
product on unit quaternions, matrix multiplication on GL⁺(2,ℝ), the rigid-motion product on
SE(3)). Non-commutative in general.

**increment (ξ)** — the local step emitted by the edge; the node composes it.

**analytic support** — the set of chart features that the exact construction needs
(e.g. three axis components on SU(2) at fixed rotation amounts). The pruning rule keeps
only this support.

**span criterion** — a pre-training check: with the exact node, the composition is
representable exactly iff the edge features span the chart's logarithm (Proposition 1 in
the paper). If a needed component is missing, there is a positive floor independent of
training.

**law-match corollary** — the complement: if the node's law is wrong, no edge can represent
the composition exactly (a training-independent floor remains).

**ψ (law correction)** — a small zero-initialized learned correction applied to the law in
the node. It starts exact, engages only where the hardware deviates from the law, and is trained
in isolation on the frozen exact edge; its deviation from zero is a test, not a proof, of
the law.

**isolation (protocol)** — training only ψ with the exact edge frozen; joint training of
edge and ψ is the compensation control (non-identifiable, can drift into compensation).

**kernel (local) ψ head** — a ψ parameterized as a sum of RBF bases centred on the training
inputs. Being local, it removes the global head's seed collapse without weight decay
(X32) and improves the small-N FR3 result (X34).

**two-slot ψ head** — the recommended correction parameterization: a global (linear/Fourier)
trend plus a local RBF-kernel term, both zero-initialized. The kernel wins the data-starved
corner and the global trend the data-rich one, so the two-slot head dominates both pure heads
at every training size (X33).

**global (linear/Fourier) ψ head** — the pure global correction: a linear or Fourier trend
over the chart, zero-initialized. It is accurate when data are plentiful and smooth but is
seed-unstable in the data-starved corner (the Kinova collapse, X25/X32).

**node-only ψ** — a learned law correction with no per-joint edge (the 198-parameter Fourier
row in Table 3); it overfits at small N, showing that the learned per-joint edge matters.

**relaxed node** — a node whose law is itself learned (the X13/X14 relaxation probes), used
to test whether the imposed law is the right one; it is a diagnostic, not the recommended
architecture.

**sequential / joint two-slot** — the two ways to combine the global and kernel slots in the
recommended ψ head: kernel first, then the global slot on the residual (sequential, declared
in Table 4), or both fitted together (joint). They differ within noise on KUKA/Panda (X47).

**capacity vs locality (of the correction)** — the two levers found in the revision round: the
global head's seed instability is controlled by capacity (an L2 weight-decay window, X26) and
removed by locality (the kernel head, X32–X34).

**exact composer / exact node** — the architecture with the exact law in the node and no
ψ; the baseline whose floor ψ removes on deformed hardware.

**LawKAN-SU(2) / LawKAN-GL⁺(2,ℝ) / LawKAN-SE(3)** — the three named group instances:
compact rotation composition, non-compact 2×2 linear composition, and mixed rigid-motion
composition. **LawKAN-SE(3)** with the per-joint edge is the arm-calibration model.

**exact edge** — the edge that carries the analytic support of the step, so that with the
exact node the composition is representable exactly (the K=1 SU(2) edge is the leading case).

**per-joint edge** — one SE(3) edge per arm joint (the 54-parameter FR3 base), the model
whose deformation ψ corrects.

**sparse / analytic-support edge** — the pruned edge keeping only the features the exact
construction needs (18 parameters on SU(2)); it matches the dense edge where the support is
sufficient and is the deterministic IMU variant.

**pole / operand chart singularity** — the axis–angle chart of the accumulated pose
branches when its rotation crosses π (the quaternion scalar part changes sign); the
correction itself stays small. Quaternion features remove the branch; the residual spike is
a training-trajectory effect.

**EMA / Polyak averaging (deployment)** — averaging the ψ weights over the training
trajectory. It removes the transient spikes in every configuration tested and leaves the
final error within 0.02 mm, so it is the deployment-side fix (X59, X61).

**branch-safe variant** — quaternion features with the global slot only. It avoids the
operand chart branch and shows no measured cost at the tested budget, though equivalence at
the ±10% margin was not established (X61).

**T1 / T2 (branch closing tests)** — T1 tests equivalence of the branch-safe variant at a
±10% margin; T2 tests that EMA leaves the final error unchanged. T2 was confirmed; T1
equivalence was not established (X61).

**QuatIMU / LawKANArm (code names)** — historical class names for the exact/sparse SU(2)
edge and the per-joint SE(3) arm model. The paper uses the descriptive names; the class
names appear in scripts and dump keys.

**QuatKAN** — the internal historical spelling kept in some class names and JSON dump keys
as provenance; the family is called LawKAN.
