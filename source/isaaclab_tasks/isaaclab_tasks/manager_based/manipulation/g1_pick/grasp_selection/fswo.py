# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FSWO — Frictionless Self-balancing Wrench Optimizer (force-closure score).

Implements Section 4 of ``OPTIMIZER_IMPLEMENTATION_SPEC.md`` (the force-closure
stage of "Lightning Grasp", Yin & Abbeel, arXiv:2511.07418, Eq. 1).

Scores ONE grasp from its ``k`` frictionless contacts ``{(p_i, n_i)}`` (contact
position, INWARD unit normal, both in the cube/object frame):

    force of contact i  :  alpha_i * n_i           (alpha_i >= 0)
    torque of contact i :  alpha_i * (p_i x n_i)

    minimize   alpha^T Q alpha      Q = N N^T + lam * (T T^T)
    s.t.       alpha_i >= 0,  max_i alpha_i = 1

    score S = -(minimum)  in (-inf, 0];  S = 0 <=> perfect force closure.

CPU-only, no Isaac Lab / Isaac Sim imports.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

__all__ = ["psd_sqrt", "fswo_score", "fswo_alpha"]


def psd_sqrt(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric PSD square root ``L`` of ``Q`` such that ``Q = L @ L = L.T @ L``.

    Tiny/negative eigenvalues (numerical noise on a Gram matrix) are clipped to 0.
    """
    w, V = np.linalg.eigh(Q)  # Q is symmetric PSD -> real eigendecomposition
    w = np.clip(w, eps, None) - eps
    return V @ np.diag(np.sqrt(w)) @ V.T


def _gram(contacts: np.ndarray, lam: float) -> np.ndarray:
    """Gram matrix ``Q_ij = n_i . n_j + lam * (tau_i . tau_j)`` of the contact wrenches."""
    p, n = contacts[:, :3], contacts[:, 3:]
    tau = np.cross(p, n)  # (k,3) torque about the object origin of a unit normal force
    return n @ n.T + lam * (tau @ tau.T)


def fswo_score(contacts: np.ndarray, lam: float = 1.0) -> float:
    """Force-closure score of a contact set.

    Args:
        contacts: ``(k, 6)`` rows ``[px, py, pz, nx, ny, nz]`` with INWARD unit
            normals, expressed in the object frame (torques are taken about the
            object origin).
        lam: torque-vs-force weight ``lambda``. 1.0 = Lightning-Grasp default;
            raise it to punish torque (rocking) imbalance more.

    Returns:
        ``S in (-inf, 0]``; ``0`` = the contacts admit a perfectly self-balancing
        non-zero frictionless load (force closure), more negative = worse.
    """
    contacts = np.asarray(contacts, dtype=float)
    if contacts.ndim != 2 or contacts.shape[1] != 6:
        raise ValueError(f"contacts must be (k, 6), got {contacts.shape}.")
    if contacts.shape[0] < 2:
        raise ValueError(f"FSWO needs >= 2 contacts, got {contacts.shape[0]}.")

    Q = _gram(contacts, lam)
    # The design matrix MUST be the square root L (Q = L @ L), never Q itself:
    # only then is ||L alpha||^2 == alpha^T Q alpha an exact identity, so NNLS's
    # complementary slackness matches the QP's. See spec Section 4.4.
    L = psd_sqrt(Q)

    k = contacts.shape[0]
    best_val = np.inf
    for j in range(k):  # enumerate which contact attains max_i alpha_i = 1
        mask = np.arange(k) != j
        # min_{x>=0} || L[:,~j] x + L[:,j] ||^2 ; rnorm = ||L alpha||
        alpha_free, rnorm = nnls(L[:, mask], -L[:, j])
        # Score from the NNLS RESIDUAL, not from forming alpha^T Q alpha: the two are
        # identical in exact arithmetic (||L alpha||^2 == alpha^T Q alpha), but when the
        # contact set is near-degenerate NNLS can return |alpha| ~ 1e11 along a null
        # direction of Q, and the quadratic form then cancels catastrophically (observed:
        # -3.5e7, i.e. a positive "score", where the true residual is ~1e-33). ||L alpha||
        # is a sum of squares, so it stays non-negative and accurate.
        best_val = min(best_val, float(rnorm) ** 2)
    return -best_val


def fswo_alpha(contacts: np.ndarray, lam: float = 1.0) -> tuple[float, np.ndarray]:
    """Same as :func:`fswo_score` but also returns the optimal contact magnitudes.

    Useful for debugging/inspection: ``alpha`` shows which contacts actually carry
    the self-balancing internal load.
    """
    contacts = np.asarray(contacts, dtype=float)
    if contacts.shape[0] < 2:
        raise ValueError(f"FSWO needs >= 2 contacts, got {contacts.shape[0]}.")
    Q = _gram(contacts, lam)
    L = psd_sqrt(Q)
    k = contacts.shape[0]
    best_val, best_alpha = np.inf, np.zeros(k)
    for j in range(k):
        mask = np.arange(k) != j
        alpha_free, rnorm = nnls(L[:, mask], -L[:, j])
        alpha = np.zeros(k)
        alpha[mask] = alpha_free
        alpha[j] = 1.0
        val = float(rnorm) ** 2  # see fswo_score: residual, not alpha^T Q alpha
        if val < best_val:
            best_val, best_alpha = val, alpha
    return -best_val, best_alpha
