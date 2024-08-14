#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Given k (number of successes) and n (sample size), return cumulative probability mass
from typing import Callable

import scipy.stats

Cdf = Callable[[int, int], float]
# Given k (number of successes) and n (sample size), return  probability mass
Pmf = Callable[[int, int], float]

BinomialCdfGenerator = lambda p: lambda k, n: scipy.stats.binom.cdf(k, n, p)
BinomialPmfGenerator = lambda p: lambda k, n: scipy.stats.binom.pmf(k, n, p)
BinomialLogPmfGenerator = lambda p: lambda k, n: scipy.stats.binom.logpmf(k, n, p)

HypergeometricCdfGenerator = lambda p, lot_size: lambda k, n: scipy.stats.hypergeom.cdf(
    k, lot_size, round(lot_size * p), n
)
HypergeometricPmfGenerator = lambda p, lot_size: lambda k, n: scipy.stats.hypergeom.pmf(
    k, lot_size, round(lot_size * p), n
)
