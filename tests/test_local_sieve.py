"""
Tests for the arithmetic sieve arrays.
"""

import numpy as np
import pytest
from src.local.sieve import (
    prime_sieve,
    mobius_array,
    von_mangoldt_array,
    mu_star_lambda,
    mu_star_lambda_lambda,
    compute_sieve_arrays,
    SieveArrays,
)


class TestPrimeSieve:
    """Tests for prime_sieve function."""

    def test_prime_count_100(self):
        """pi(100) = 25."""
        primes, is_prime = prime_sieve(100)
        assert len(primes) == 25

    def test_prime_count_1000(self):
        """pi(1000) = 168."""
        primes, is_prime = prime_sieve(1000)
        assert len(primes) == 168

    def test_first_primes(self):
        """Check first few primes."""
        primes, _ = prime_sieve(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert list(primes) == expected

    def test_is_prime_array(self):
        """is_prime[n] should be True iff n is prime."""
        primes, is_prime = prime_sieve(20)

        assert is_prime[2] == True
        assert is_prime[3] == True
        assert is_prime[4] == False
        assert is_prime[5] == True
        assert is_prime[6] == False
        assert is_prime[7] == True
        assert is_prime[1] == False
        assert is_prime[0] == False

    def test_small_nmax(self):
        """Handle small nmax values."""
        primes, is_prime = prime_sieve(1)
        assert len(primes) == 0

        primes, is_prime = prime_sieve(2)
        assert list(primes) == [2]


class TestMobius:
    """Tests for Mobius function."""

    def test_specific_values(self):
        """Test known Mobius values."""
        primes, _ = prime_sieve(30)
        mu = mobius_array(30, primes)

        assert mu[1] == 1     # mu(1) = 1
        assert mu[2] == -1    # mu(2) = -1 (one prime factor)
        assert mu[3] == -1    # mu(3) = -1
        assert mu[4] == 0     # mu(4) = 0 (4 = 2^2)
        assert mu[5] == -1
        assert mu[6] == 1     # mu(6) = (-1)^2 = 1 (two prime factors)
        assert mu[8] == 0     # mu(8) = 0 (8 = 2^3)
        assert mu[9] == 0     # mu(9) = 0 (9 = 3^2)
        assert mu[10] == 1    # mu(10) = (-1)^2 = 1
        assert mu[30] == -1   # mu(30) = (-1)^3 = -1 (30 = 2*3*5)

    def test_sum_over_divisors(self):
        """sum_{d|n} mu(d) = 0 for n > 1."""
        primes, _ = prime_sieve(100)
        mu = mobius_array(100, primes)

        for n in range(2, 101):
            divisor_sum = sum(mu[d] for d in range(1, n+1) if n % d == 0)
            assert divisor_sum == 0, f"Failed for n={n}"


class TestVonMangoldt:
    """Tests for von Mangoldt function."""

    def test_prime_powers(self):
        """Lambda(p^k) = log(p)."""
        primes, _ = prime_sieve(100)
        Lambda = von_mangoldt_array(100, primes)

        # Lambda(2) = log(2)
        assert np.isclose(Lambda[2], np.log(2))
        # Lambda(4) = log(2) (4 = 2^2)
        assert np.isclose(Lambda[4], np.log(2))
        # Lambda(8) = log(2)
        assert np.isclose(Lambda[8], np.log(2))
        # Lambda(27) = log(3)
        assert np.isclose(Lambda[27], np.log(3))

    def test_non_prime_powers(self):
        """Lambda(n) = 0 if n is not a prime power."""
        primes, _ = prime_sieve(100)
        Lambda = von_mangoldt_array(100, primes)

        # Lambda(1) = 0
        assert Lambda[1] == 0
        # Lambda(6) = 0 (6 = 2*3, not a prime power)
        assert Lambda[6] == 0
        # Lambda(10) = 0
        assert Lambda[10] == 0
        # Lambda(12) = 0
        assert Lambda[12] == 0

    def test_sum_over_divisors(self):
        """sum_{d|n} Lambda(d) = log(n)."""
        primes, _ = prime_sieve(50)
        Lambda = von_mangoldt_array(50, primes)

        for n in range(1, 51):
            if n == 1:
                continue  # log(1) = 0
            divisor_sum = sum(Lambda[d] for d in range(1, n+1) if n % d == 0)
            assert np.isclose(divisor_sum, np.log(n)), f"Failed for n={n}"


class TestMuStarLambda:
    """Tests for (mu * Lambda)(n) = -mu(n) * log(n)."""

    def test_identity(self):
        """(mu * Lambda)(n) = -mu(n) * log(n)."""
        primes, _ = prime_sieve(100)
        mu = mobius_array(100, primes)
        muL = mu_star_lambda(100, mu)

        for n in range(2, 101):
            expected = -float(mu[n]) * np.log(float(n))
            assert np.isclose(muL[n], expected), f"Failed for n={n}"

    def test_at_primes(self):
        """(mu * Lambda)(p) = log(p) for prime p."""
        primes, is_prime = prime_sieve(100)
        mu = mobius_array(100, primes)
        muL = mu_star_lambda(100, mu)

        for p in primes:
            # mu(p) = -1, so -mu(p)*log(p) = log(p)
            assert np.isclose(muL[p], np.log(float(p))), f"Failed for p={p}"


class TestMuStarLambdaLambda:
    """Tests for (mu * Lambda * Lambda)."""

    def test_basic_computation(self):
        """(mu * Lambda * Lambda) should be computable."""
        primes, _ = prime_sieve(100)
        mu = mobius_array(100, primes)
        muL = mu_star_lambda(100, mu)
        muLL = mu_star_lambda_lambda(100, muL, primes)

        # Should have same length
        assert len(muLL) == 101

        # muLL[1] = 0 (no prime power divides 1 except trivially)
        assert muLL[1] == 0


class TestComputeSieveArrays:
    """Tests for compute_sieve_arrays function."""

    def test_basic(self):
        """compute_sieve_arrays returns correct structure."""
        sieve = compute_sieve_arrays(100)

        assert isinstance(sieve, SieveArrays)
        assert sieve.N == 100
        assert len(sieve.primes) == 25
        assert len(sieve.mobius) == 101
        assert len(sieve.von_mangoldt) == 101
        assert len(sieve.mu_star_Lambda) == 101
        assert sieve.mu_star_Lambda_Lambda is None

    def test_with_psi3(self):
        """compute_sieve_arrays with include_psi3=True."""
        sieve = compute_sieve_arrays(100, include_psi3=True)

        assert sieve.mu_star_Lambda_Lambda is not None
        assert len(sieve.mu_star_Lambda_Lambda) == 101

    def test_consistency(self):
        """Arrays should be consistent with each other."""
        sieve = compute_sieve_arrays(50)

        # Check is_prime matches primes list
        for p in sieve.primes:
            assert sieve.is_prime[p]

        # Count primes
        assert np.sum(sieve.is_prime) == len(sieve.primes)
