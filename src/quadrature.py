"""
Gauss-Legendre quadrature for PRZZ kappa computation.

This module provides 1D and 2D numerical integration on [0,1] and [0,1]^2.
All integrands in the PRZZ framework integrate over these domains.

Key design choices:
- 1D nodes/weights are cached via lru_cache and returned as READ-ONLY arrays
- 2D grids use indexing="ij" to avoid silent broadcasting bugs
- 2D grids are NOT cached (O(n^2) memory)
"""

from __future__ import annotations
import functools
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple


@functools.lru_cache(maxsize=16)
def gauss_legendre_01(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (nodes, weights) for n-point Gauss-Legendre quadrature on [0,1].

    Maps from standard [-1,1] interval:
    - x_01 = 0.5 * (x + 1)
    - w_01 = 0.5 * w

    Uses numpy.polynomial.legendre.leggauss() for [-1,1] nodes/weights.

    Args:
        n: Number of quadrature points (positive integer)

    Returns:
        Tuple of (nodes, weights) arrays, each of length n.
        Arrays are READ-ONLY (writeable=False) to prevent mutation of cached data.

    Raises:
        ValueError: if n < 1
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")

    # Get nodes and weights on [-1, 1]
    nodes_11, weights_11 = leggauss(n)

    # Map to [0, 1]: x_01 = 0.5*(x+1), w_01 = 0.5*w
    nodes = 0.5 * (nodes_11 + 1.0)
    weights = 0.5 * weights_11

    # Make arrays read-only to prevent accidental mutation of cached data
    nodes.flags.writeable = False
    weights.flags.writeable = False

    return nodes, weights


def tensor_grid_2d(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (U, T, W) for 2D tensor product quadrature on [0,1]^2.

    Creates meshgrid from 1D Gauss-Legendre nodes with product weights.

    Args:
        n: Number of quadrature points per dimension (total n^2 points)

    Returns:
        Tuple of (U, T, W) where:
        - U[i,j] = u_i (first coordinate grid)
        - T[i,j] = t_j (second coordinate grid)
        - W[i,j] = w_i * w_j (product weight grid)

        All arrays have shape (n, n).

    Note:
        Uses indexing="ij" so that U varies along axis 0 and T varies along axis 1.
        This avoids silent broadcasting bugs that occur with default "xy" indexing.

        NOT cached because 2D grids are O(n^2) memory.
    """
    nodes, weights = gauss_legendre_01(n)

    # Create 2D grids with indexing="ij"
    # U[i,j] = nodes[i], T[i,j] = nodes[j]
    U, T = np.meshgrid(nodes, nodes, indexing="ij")

    # Product weights: W[i,j] = weights[i] * weights[j]
    W = np.outer(weights, weights)

    return U, T, W


def tensor_grid_3d(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, Y, Z, W) for 3D tensor product quadrature on [0,1]^3.

    Args:
        n: Number of quadrature points per dimension (total n^3 points)

    Returns:
        Tuple of (X, Y, Z, W) where:
        - X[i,j,k] = x_i (first coordinate grid)
        - Y[i,j,k] = y_j (second coordinate grid)
        - Z[i,j,k] = z_k (third coordinate grid)
        - W[i,j,k] = w_i * w_j * w_k (product weight grid)

        All arrays have shape (n, n, n).

    Note:
        Uses indexing="ij" for consistency with 2D grids.
        NOT cached because 3D grids are O(n^3) memory.
    """
    nodes, weights = gauss_legendre_01(n)

    # Create 3D grids with indexing="ij"
    X, Y, Z = np.meshgrid(nodes, nodes, nodes, indexing="ij")

    # Product weights: W[i,j,k] = weights[i] * weights[j] * weights[k]
    W = np.einsum("i,j,k->ijk", weights, weights, weights)

    return X, Y, Z, W
