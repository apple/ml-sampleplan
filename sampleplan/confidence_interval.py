#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Tuple

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import binom, hypergeom, norm


def hypergeometric_proportion_interval_exact(
    confidence: float, population_size: int, sample_size: int, num_successes: int
) -> Tuple[float, float]:
    total_count = round(num_successes / sample_size * population_size)
    q1 = (1.0 - confidence) / 2
    q2 = (1.0 + confidence) / 2
    cl = hypergeom.ppf(q1, M=population_size, n=total_count, N=sample_size)
    cu = hypergeom.ppf(q2, M=population_size, n=total_count, N=sample_size)

    return cl / sample_size, cu / sample_size


def binomial_proportion_interval_exact_midp(alpha: float, sample_size: int, num_successes: int) -> Tuple[float, float]:
    def _fn_low(pl: float):
        lower_limit = 0.5 * binom.pmf(num_successes, sample_size, pl) + 1 - binom.cdf(num_successes, sample_size, pl)
        return lower_limit - alpha / 2

    def _fn_up(pu: float):
        upper_limit = 0.5 * binom.pmf(num_successes, sample_size, pu) + binom.cdf(num_successes - 1, sample_size, pu)
        return upper_limit - alpha / 2

    low = 0.0
    up = 1.0

    if num_successes != 0:
        low = root_scalar(_fn_low, bracket=(0.0, num_successes / sample_size)).root

    if num_successes != sample_size:
        up = root_scalar(_fn_up, bracket=(num_successes / sample_size, 1.0)).root

    return low, up


# Sample sizes


# Exact


def sample_size_exact_binomial(p0: float, alpha: float, ci_half_width: float):
    pl = max(0.0, p0 - ci_half_width)
    pu = min(1.0, p0 + ci_half_width)

    def _fn(n):
        x = round(p0 * n)
        lower_limit = 1 - binom.cdf(x - 1, round(n), pl)
        upper_limit = binom.cdf(x, round(n), pu)
        return lower_limit + upper_limit

    sol = root_scalar(lambda n: _fn(n) - alpha, bracket=(1, 1_000_000))

    return math.ceil(sol.root)


def sample_size_exact_hypergeometric(lot_size: int, p0: float, alpha: float, ci_half_width: float):
    pl = max(0.0, p0 - ci_half_width)
    pu = min(1.0, p0 + ci_half_width)

    if any([np.isclose(p0, 0.0), np.isclose(p0, 1.0)]):
        raise ValueError(f"p0 is too close to 0 or 1")

    def _fn(n):
        x = round(p0 * n)
        lower_limit = 1 - hypergeom.cdf(x - 1, lot_size, round(lot_size * pl), round(n))
        upper_limit = hypergeom.cdf(x, lot_size, round(lot_size * pu), round(n))
        result = lower_limit + upper_limit

        return result

    sol = root_scalar(lambda n: _fn(n) - alpha, bracket=(1, lot_size))

    if not sol.converged:
        raise Exception(sol)

    return math.ceil(sol.root)


# Exact mid-P


def sample_size_exact_mid_point_binomial(p0: float, alpha: float, ci_half_width: float):
    pl = max(0.0, p0 - ci_half_width)
    pu = min(1.0, p0 + ci_half_width)

    def _fn(n):
        x = round(p0 * n)
        lower_limit = 0.5 * binom.pmf(x, n, pl) + 1 - binom.cdf(x, n, pl)
        upper_limit = 0.5 * binom.pmf(x, n, pu) + binom.cdf(x - 1, n, pu)
        return lower_limit + upper_limit

    sol = root_scalar(lambda n: _fn(n) - alpha, bracket=(1, 1_000_000))

    return math.ceil(sol.root)


def sample_size_exact_mid_point_hypergeometric(lot_size: int, p0: float, alpha: float, ci_half_width: float):
    pl = max(0.0, p0 - ci_half_width)
    pu = min(1.0, p0 + ci_half_width)

    if any([np.isclose(p0, 0.0), np.isclose(p0, 1.0)]):
        raise ValueError(f"p0 is too close to 0 or 1")

    def _fn(n):
        x = round(p0 * n)
        lower_limit = (
            0.5 * hypergeom.pmf(x, lot_size, lot_size * pl, n) + 1 - hypergeom.cdf(x, lot_size, lot_size * pl, n)
        )
        upper_limit = 0.5 * hypergeom.pmf(x, lot_size, lot_size * pu, n) + hypergeom.cdf(
            x - 1, lot_size, lot_size * pu, n
        )
        return lower_limit + upper_limit

    sol = root_scalar(lambda n: _fn(n) - alpha, bracket=(1, lot_size))

    return math.ceil(sol.root)


# Approx


def sample_size_agresti_coull_binomial(p: float, alpha: float, half_width: float) -> int:
    # As this has no closed-form solution, we need to solve it numerically

    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Agresti%E2%80
    # %93Coull_interval
    def _agresti_coull(n: int):
        n_s = p * n

        z = norm.ppf(1 - alpha / 2)
        z_squared = z * z

        n_tilda = n + z_squared
        p_tilda = 1 / n_tilda * (n_s + 0.5 * z_squared)

        bounds = z * math.sqrt(p_tilda * (1 - p_tilda) / n_tilda)

        return bounds

    sol = root_scalar(lambda n: _agresti_coull(n) - half_width, bracket=(1, 1_000_000))

    return math.ceil(sol.root)
