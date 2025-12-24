"""Numerical helpers (smooth max, stability utilities)."""

from __future__ import annotations

import numpy as np

def softmax(a: np.ndarray, b: np.ndarray, beta: float = 20.0) -> np.ndarray:
    """Numerically stable differentiable approximation of max(a, b) using log-sum-exp."""
    ba = beta * a
    bb = beta * b
    m = np.maximum(ba, bb)
    s = m + np.log(np.exp(ba - m) + np.exp(bb - m))
    return (a + b) / 2.0 + 0.5 * s / beta


def softmax_sqrt_ab(a: np.ndarray, b: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Alternative softmax approximation using a smooth sqrt approximation.

    This implements a smooth approximation of max(a, b) by operating on the
    difference x = a - b and returning b + 0.5*(x + sqrt(beta^2 + x^2)).
    The `beta` parameter controls smoothing (smaller -> closer to hard max).
    """
    # Use eps = 1/beta so that as beta -> +inf the smoothing -> 0 and
    # the expression converges to the hard max: b + 0.5*(x + |x|) = max(a,b).
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = a - b
    # Guard beta to be positive and non-zero
    beta_f = float(beta)
    if beta_f <= 0.0:
        # fallback: return simple average when beta non-positive
        return 0.5 * (a + b)
    eps = 1.0 / beta_f
    return b + 0.5 * (x + np.sqrt(x * x + eps * eps))


def softmax_pair(a: np.ndarray, b: np.ndarray, beta: float = 20.0) -> np.ndarray:
    """Pairwise softmax matching the inline log-sum-exp used in rollback.

    Implements: V = b + (1/beta) * log(1 + exp(beta*(a-b))) in a numerically stable way.
    This converges to max(a,b) as beta->+inf.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = beta * (a - b)
    m_stable = np.maximum(0.0, x)
    return b + (m_stable + np.log(np.exp(-m_stable) + np.exp(x - m_stable))) / float(beta)


# Module-level softmax function (can be reassigned for experiments)
# Default: numerically stable log-sum-exp version
SOFTMAX_FN = softmax_pair



