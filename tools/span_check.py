#!/usr/bin/env python3
"""Apply the span criterion (Sec. 3.3) to a dataset.

The criterion is a five-step check (Sec. 5.6):
 1. fix the group and the chart;
 2. write the target's chart logarithm (the algebra element);
 3. test the edge's span: fit the edge features to that logarithm; the
    held-out residual is the *span gap*;
 4. for a deviating law, the same test on the deviation eta and the psi
    features (the law-match criterion);
 5. verdict: gap at the noise floor -> the span is full (representable);
    a gap above it -> a training-independent floor.

This script implements steps 2-5. It is deliberately a *linear* probe: a
least-squares readout from the features to the logarithm tells us whether the
features can represent it at all, independent of the architecture's training.

Usage:
    python tools/span_check.py --demo
    python tools/span_check.py --data X.npy --target Q.npy --feats fourier
"""
import argparse

import numpy as np

# ---------------------------------------------------------------- SU(2) tools


def qmul(a, b):
    """Hamilton product, scalar-first (..., 4)."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qexp(v):
    """Axis-angle vector -> unit quaternion (the chart logarithm's inverse)."""
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([np.cos(th / 2), k * v], -1)


def qlog(q):
    """Unit quaternion -> rotation vector (the chart logarithm)."""
    q = q * np.sign(q[..., :1] + 1e-30)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12) * ang, 2 * v)


# ---------------------------------------------------------------- features


def fourier(axes, order=1, drop=None):
    """Group features of an axis encoded by (azimuth, polar) phases."""
    a = np.arctan2(axes[..., 1], axes[..., 0])
    b = np.arccos(np.clip(axes[..., 2], -1, 1))
    fa, fb = [np.ones_like(a)], [np.ones_like(b)]
    for m in range(1, order + 1):
        fa += [np.cos(m * a), np.sin(m * a)]
        fb += [np.cos(m * b), np.sin(m * b)]
    F = np.stack([x * y for x in fa for y in fb], -1)
    if drop is not None:
        F = np.delete(F, drop, axis=-1)   # a span-gap control
    return F


# ---------------------------------------------------------------- the test


def span_gap(F, Y, train=0.7, ridge=1e-8):
    """Held-out residual of the best linear readout F -> Y (the span gap)."""
    n = len(F)
    cut = int(n * train)
    Xtr = np.c_[F[:cut], np.ones(cut)]
    Xte = np.c_[F[cut:], np.ones(n - cut)]
    W = np.linalg.solve(Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1]),
                        Xtr.T @ Y[:cut])
    res = Xte @ W - Y[cut:]
    return float(np.sqrt((res ** 2).sum(-1).mean())), W


def report(name, gap, floor):
    verdict = 'full span (representable)' if gap <= 1.5 * floor else 'floor'
    print(f'  {name:<34s} gap {gap:8.4f}  floor {floor:8.4f}  -> {verdict}')


def demo():
    """The edge's increment on SU(2): features -> chart logarithm."""
    rng = np.random.default_rng(0)
    n = 2000
    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    ang = 0.7
    Y = ang * axes                                # step 2: chart log of the step
    sigma = 0.03
    noisy = axes + sigma * rng.normal(size=axes.shape)
    noisy /= np.linalg.norm(noisy, axis=-1, keepdims=True)
    print('span criterion demo (SU(2) increment, Fourier K=1, 9 features)')
    gap_true, _ = span_gap(fourier(axes), Y)       # noiseless input
    gap_noisy, _ = span_gap(fourier(noisy), Y)     # the noise floor
    report('Fourier K=1, true axes', gap_true, gap_noisy)
    print(f'  {"Fourier K=1, noisy axes":<34s} gap {gap_noisy:8.4f}  '
          f'floor {gap_noisy:8.4f}  -> at the noise floor')
    # Features are ordered [1, cos a, sin a] x [1, cos b, sin b]; indices 3-8
    # are the azimuth-dependent (alpha) terms.
    gap_drop, _ = span_gap(fourier(noisy, drop=[3, 4, 5, 6, 7, 8]), Y)
    report('Fourier minus the alpha terms', gap_drop, gap_noisy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--data', help='features X.npy (n, d)')
    ap.add_argument('--target', help='chart logarithm Y.npy (n, m)')
    args = ap.parse_args()
    if args.demo or not args.data:
        demo()
        return
    F = np.load(args.data)
    Y = np.load(args.target)
    gap, _ = span_gap(F, Y)
    print(f'span gap {gap:.6f}  (compare with the task noise floor)')


if __name__ == '__main__':
    main()
