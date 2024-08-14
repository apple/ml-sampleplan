#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import scipy


def binomial_test_exact(
    p0: float, p1: float, alpha: float, beta: float, alternative: str = "lower", n_min: int = 5, limit: int = 10_000
):
    # https://reliabilityanalyticstoolkit.appspot.com/sample_size
    # http://ndl.ethernet.edu.et/bitstream/123456789/30101/1/24..pdf
    # https://www.r-bloggers.com/2020/01/calculating-the-required-sample-size-for-a-binomial-test-in-r/
    # https://nshi.jp/en/js/onebinom/
    # https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
    # https://pages.stat.wisc.edu/~st571-1/10-power-4.pdf

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Invalid value given for alternative: [{alternative}]")

    actual_alpha = 1.0
    actual_beta = 1.0

    for n in range(n_min, limit + 1):
        if actual_alpha <= alpha and actual_beta <= beta:
            return n

        for r in range(n - 1, -1, -1):
            actual_alpha = scipy.stats.binom.cdf(r, n, p0)
            if actual_alpha <= alpha:
                break

        actual_beta = 1 - scipy.stats.binom.cdf(r, n, p1)
    else:
        raise ValueError(f"Did not find sample size after looking up to n = {limit}")
