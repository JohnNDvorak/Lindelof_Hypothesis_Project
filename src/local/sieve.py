"""
Fast arithmetic sieve arrays for mollifier coefficient generation.

Provides:
- Prime sieve (Eratosthenes)
- Mobius function mu(n)
- von Mangoldt function Lambda(n)
- Dirichlet convolutions (mu * Lambda), (mu * Lambda * Lambda)

Key identity: (mu * Lambda)(n) = -mu(n) * log(n)
This follows from the Mobius inversion formula applied to sum_{d|n} Lambda(d) = log(n).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class SieveArrays:
    """Precomputed arithmetic arrays up to N.

    Attributes:
        N: Maximum index
        primes: List of primes up to N
        is_prime: Boolean array, is_prime[n] = True iff n is prime
        mobius: int8 array, mobius[n] = mu(n)
        von_mangoldt: float64 array, von_mangoldt[n] = Lambda(n)
        mu_star_Lambda: float64 array, mu_star_Lambda[n] = (mu * Lambda)(n) = -mu(n)*log(n)
        mu_star_Lambda_Lambda: Optional float64 array for psi_3
    """
    N: int
    primes: np.ndarray  # dtype int64
    is_prime: np.ndarray  # dtype bool
    mobius: np.ndarray  # dtype int8
    von_mangoldt: np.ndarray  # dtype float64
    mu_star_Lambda: np.ndarray  # dtype float64
    mu_star_Lambda_Lambda: Optional[np.ndarray] = None  # dtype float64


def prime_sieve(nmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Eratosthenes sieve up to nmax.

    Args:
        nmax: Upper bound (inclusive)

    Returns:
        (primes, is_prime) where:
        - primes: 1D array of primes <= nmax
        - is_prime: Boolean array of length nmax+1
    """
    if nmax < 2:
        return np.array([], dtype=np.int64), np.zeros(nmax + 1, dtype=bool)

    is_prime = np.ones(nmax + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(nmax**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    primes = np.nonzero(is_prime)[0].astype(np.int64)
    return primes, is_prime


def mobius_array(nmax: int, primes: np.ndarray) -> np.ndarray:
    """Compute Mobius function mu(n) for n = 0, 1, ..., nmax.

    mu(n) = (-1)^k if n is a product of k distinct primes
    mu(n) = 0 if n has a squared prime factor

    Args:
        nmax: Upper bound
        primes: Array of primes (from prime_sieve)

    Returns:
        int8 array of length nmax+1 with mu[n] = mobius(n)
    """
    if nmax < 1:
        return np.zeros(nmax + 1, dtype=np.int8)

    mu = np.zeros(nmax + 1, dtype=np.int8)
    mu[1] = 1

    # Use smallest prime factor approach with factorization
    # For each n >= 2, compute mu(n) by tracking prime factors
    for n in range(2, nmax + 1):
        m = n
        mu_val = 1
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                m //= p
                mu_val = -mu_val
                if m % p == 0:
                    # p^2 divides n
                    mu_val = 0
                    break
        if mu_val != 0 and m > 1:
            # m is a prime factor
            mu_val = -mu_val
        mu[n] = mu_val

    return mu


def von_mangoldt_array(nmax: int, primes: np.ndarray) -> np.ndarray:
    """Compute von Mangoldt function Lambda(n) for n = 0, 1, ..., nmax.

    Lambda(n) = log(p) if n = p^k for some prime p and k >= 1
    Lambda(n) = 0 otherwise

    Args:
        nmax: Upper bound
        primes: Array of primes

    Returns:
        float64 array of length nmax+1 with Lambda[n] = von_mangoldt(n)
    """
    Lambda = np.zeros(nmax + 1, dtype=np.float64)

    for p in primes:
        if p > nmax:
            break
        log_p = np.log(float(p))
        pk = p
        while pk <= nmax:
            Lambda[pk] = log_p
            pk *= p

    return Lambda


def mu_star_lambda(nmax: int, mu: np.ndarray) -> np.ndarray:
    """Compute (mu * Lambda)(n) = -mu(n) * log(n) for n = 1, ..., nmax.

    This uses the exact identity from Mobius inversion.

    Args:
        nmax: Upper bound
        mu: Mobius array

    Returns:
        float64 array of length nmax+1
    """
    result = np.zeros(nmax + 1, dtype=np.float64)
    # log(1) = 0, so result[1] = 0
    for n in range(2, nmax + 1):
        result[n] = -float(mu[n]) * np.log(float(n))
    return result


def mu_star_lambda_lambda(
    nmax: int,
    mu_L: np.ndarray,
    primes: np.ndarray
) -> np.ndarray:
    """Compute (mu * Lambda * Lambda)(n) for n = 1, ..., nmax.

    This is ((mu * Lambda) * Lambda)(n).

    Uses convolution: h(n) = sum_{d|n} g(d) * Lambda(n/d)
    where g = mu * Lambda.

    Args:
        nmax: Upper bound
        mu_L: (mu * Lambda) array
        primes: Array of primes

    Returns:
        float64 array of length nmax+1
    """
    result = np.zeros(nmax + 1, dtype=np.float64)

    # Convolve g = mu_L with Lambda
    # For each prime power pk, Lambda(pk) = log(p)
    # h(n) = sum_{pk | n} g(n/pk) * log(p)
    for p in primes:
        if p > nmax:
            break
        log_p = np.log(float(p))
        pk = p
        while pk <= nmax:
            # Add g(m) * log(p) for all m where m * pk <= nmax
            for m in range(1, nmax // pk + 1):
                result[m * pk] += mu_L[m] * log_p
            pk *= p

    return result


def compute_sieve_arrays(nmax: int, include_psi3: bool = False) -> SieveArrays:
    """Compute all arithmetic sieve arrays up to nmax.

    Args:
        nmax: Maximum index
        include_psi3: If True, also compute mu * Lambda * Lambda

    Returns:
        SieveArrays dataclass with all precomputed arrays
    """
    primes, is_prime = prime_sieve(nmax)
    mu = mobius_array(nmax, primes)
    Lambda = von_mangoldt_array(nmax, primes)
    muL = mu_star_lambda(nmax, mu)

    muLL = None
    if include_psi3:
        muLL = mu_star_lambda_lambda(nmax, muL, primes)

    return SieveArrays(
        N=nmax,
        primes=primes,
        is_prime=is_prime,
        mobius=mu,
        von_mangoldt=Lambda,
        mu_star_Lambda=muL,
        mu_star_Lambda_Lambda=muLL,
    )
