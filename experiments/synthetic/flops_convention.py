# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single source of truth for QuatKAN FLOP accounting.

Flat convention (paper Sec.3 / Appendix B.5 / Table 2): one full forward pass
per example; every transcendental (sin, cos, sqrt, division) charged as 1 FLOP;
multiply and add each charged as 1 FLOP. Identical convention for all models.

Reference row -- QuatKAN K=1, n_groups=2:
    Fourier features   2 * (4 + 9)      =  26
    feature->so(3)     2 * 3*(2*9 - 1)  = 102
    exp-maps           2 * 15           =  30
    Hamilton product   28 * (2 - 1)     =  28
    probe rotation     30 (v1.3 identity; 56 for the q*v*q* sandwich)
    output head        12
    ------------------------------------------------------------
    total                               = 228 FLOPs/example   (Table 2)

History (convention drift resolved 2026-09-23): the literal 234 in
run_main_experiments.py v1.2 / ``flops_paper`` in E19 used a 108-op einsum
(2*nf*3 instead of 3*(2*nf-1)); the v1.3 literal 207 then subtracted the
flop-lean probe-rotation saving a second time, although the paper's 228 count
already used it. Both scripts now derive the number from this module, so they
cannot diverge again.
"""

# E2 reference value for LawKAN K=1, n_groups=2 -- asserted against E19.8.
FLOPS_E2_QUATKAN_K1 = None  # set after the function definition


def quatkan_flops(n_groups=2, K=1, n_features=None, rotate="lean"):
    """Analytic per-example forward FLOPs of QuatKAN under the flat convention.

    n_groups : number of Fourier edge groups (composition length n)
    K        : Fourier order (features per group = (2K+1)**2)
    n_features : override the per-group feature count (e.g. 3 for the sparse edge)
    rotate   : "lean" (v1.3 unit-quaternion identity, 30 ops, default)
               or "sandwich" (q*v*q*, 56 ops)
    """
    nf = (2 * K + 1) ** 2 if n_features is None else n_features
    n_products = n_groups - 1
    features = n_groups * (4 * K + nf)      # trig (2K per phase x 2) + feature products
    einsum = n_groups * 3 * (2 * nf - 1)    # feature -> so(3), mult + add
    exp = 15 * n_groups                     # exp-map (norm, cos/sin, scale)
    hamilton = 28 * n_products              # 28 scalar ops per Hamilton product
    rotation = 30 if rotate == "lean" else 56
    head = 12
    return features + einsum + exp + hamilton + rotation + head


FLOPS_E2_QUATKAN_K1 = quatkan_flops(n_groups=2, K=1)

def node_mlp_edge_flops(n_groups=2, h=8, hidden_layers=2):
    """Per-example forward FLOPs of the exp+Hamilton node with per-group MLP edges.

    Same flat convention as quatkan_flops: a dense layer costs 2*in*out
    (mult+add per weight; as the MLP rows in E2), trig features cost 1 each,
    activations are free; the shared node overhead (exp 15/group, Hamilton 28
    per product, probe rotation 30, output head 12) is identical to the
    QuatKAN count.  Reference: h=8, 2 layers, n=2 -> 588 FLOPs/example.
    """
    per_group = 4 + 2*4*h + (hidden_layers-1)*2*h*h + 2*h*3
    node = 15*n_groups + 28*(n_groups-1) + 30 + 12
    return per_group*n_groups + node
