#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest

from sampleplan.acceptance_sampling import DoubleSamplingPlan

# Single
from sampleplan.acceptance_sampling.util import (
    BinomialCdfGenerator,
    BinomialPmfGenerator,
)


@pytest.mark.parametrize(
    "p1, p2, alpha, beta, expected_c1, expected_c2, expected_n, expected_asn_p1, expected_asn_p2, "
    "expected_asn_curtailed_p1, expected_asn_curtailed_p2",
    [
        (0.01, 0.05, 0.05, 0.1, 0, 3, 69, 103.155, 104.599, 101.982, 85.538),
        (0.01, 0.09, 0.05, 0.1, 0, 2, 32, 40.601, 44.545, 40.342, 37.512),
        (0.05, 0.1, 0.15, 0.075, 4, 12, 95, 144.137, 172.368, 140.205, 128.086),
        (0.007, 0.02, 0.02, 0.1, 4, 11, 440, 526.993, 776.049, 524.742, 582.769),
    ],
)
def test_acceptance_sampling_double_binomial(
    p1: float,
    p2: float,
    alpha: float,
    beta: float,
    expected_c1: int,
    expected_c2: int,
    expected_n: int,
    expected_asn_p1: float,
    expected_asn_p2: float,
    expected_asn_curtailed_p1: float,
    expected_asn_curtailed_p2: float,
):
    # Test cases are obtained from
    # https://acceptancesampling.com/
    plan = DoubleSamplingPlan.binomial(p1, p2, alpha, beta)

    asn_p1 = plan.average_sample_size(p1)
    asn_p2 = plan.average_sample_size(p2)

    asn_curtailed_p1 = plan.average_sample_size_curtailed(p1)
    asn_curtailed_p2 = plan.average_sample_size_curtailed(p2)

    assert plan.c1 == expected_c1
    assert plan.c2 == expected_c2
    assert plan.n == expected_n

    assert asn_p1 == pytest.approx(expected_asn_p1, abs=0.1)
    assert asn_p2 == pytest.approx(expected_asn_p2, abs=0.1)

    assert asn_curtailed_p1 == pytest.approx(expected_asn_curtailed_p1, abs=0.1)
    assert asn_curtailed_p2 == pytest.approx(expected_asn_curtailed_p2, abs=0.1)


@pytest.mark.parametrize(
    "p1, p2, alpha, beta, lot_size, expected_c1, expected_c2, expected_n",
    [
        (0.01, 0.05, 0.05, 0.1, 100_000, 0, 3, 69),
        (0.01, 0.09, 0.05, 0.1, 100_000, 0, 2, 32),
        (0.05, 0.1, 0.15, 0.075, 100_000, 4, 12, 95),
    ],
)
def test_acceptance_sampling_double_hypergeometric(
    p1: float, p2: float, alpha: float, beta: float, lot_size: int, expected_c1: int, expected_c2: int, expected_n: int
):
    # We do not have plan numbers from other people for double sampling hypergeometric,
    # so we cheat and use a huge lot size so that we approximate the hypergeometric
    # distribution with the binomial one and get the same plans.
    plan = DoubleSamplingPlan.hypergeometric(p1, p2, alpha, beta, lot_size)

    assert plan.c1 == expected_c1
    assert plan.c2 == expected_c2
    assert plan.n == expected_n


@pytest.mark.parametrize(
    "p, c1, c2, n1, n2, expected",
    [
        (0.03, 1, 3, 50, 100, 88.20),
        (0.06, 1, 3, 50, 100, 95.73),
        (0.02, 0, 3, 50, 50, 80.90),
        (0.08, 0, 3, 50, 50, 70.49),
        (0.06, 2, 4, 75, 150, 129.6),
        (0.09, 2, 4, 75, 150, 98.1),
    ],
)
def test_acceptance_sampling_double_binomial_average_acceptance_full(
    p: float, c1: int, c2: int, n1: int, n2: int, expected: float
):
    # Numbers are taken from Table 2 of
    # The Average Sample Number for Truncated Single and Double Attributes Acceptance Sampling Plans
    # Author(s): C. C. Craig
    # Source: Technometrics, Vol. 10, No. 4 (Nov., 1968), pp. 685-692

    cdf = BinomialCdfGenerator(p)

    asn = DoubleSamplingPlan._average_sample_size(cdf, c1=c1, c2=c2, n1=n1, n2=n2)

    assert asn == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize(
    "p, c1, c2, n1, n2, expected",
    [
        (0.03, 1, 3, 50, 100, 68.99),
        (0.06, 1, 3, 50, 100, 61.32),
        (0.03, 0, 3, 50, 50, 79.66),
        (0.08, 0, 3, 50, 50, 58.09),
        (0.06, 2, 4, 75, 150, 83.89),
        (0.09, 2, 4, 75, 150, 77.33),
    ],
)
def test_acceptance_sampling_double_binomial_average_acceptance_curtailed_partial(
    p: float, c1: int, c2: int, n1: int, n2: int, expected: float
):
    # Numbers are taken from Table 2 of
    # The Average Sample Number for Truncated Single and Double Attributes Acceptance Sampling Plans
    # Author(s): C. C. Craig
    # Source: Technometrics, Vol. 10, No. 4 (Nov., 1968), pp. 685-692

    cdf = BinomialCdfGenerator(p)
    pmf = BinomialPmfGenerator(p)

    asn = DoubleSamplingPlan._average_sample_size_curtailed(p, cdf, pmf, c1=c1, c2=c2, n1=n1, n2=n2)

    assert asn == pytest.approx(expected, abs=0.1)
