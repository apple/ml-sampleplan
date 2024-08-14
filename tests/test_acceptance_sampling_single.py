#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest

from sampleplan.acceptance_sampling import SingleSamplingPlan


@pytest.mark.parametrize(
    "p1, p2, alpha, beta, expected_c, expected_n",
    [
        (0.01, 0.08, 0.05, 0.1, 2, 65),
        (0.0065, 0.02, 0.05, 0.1, 7, 587),
        (0.05, 0.15, 0.05, 0.075, 7, 80),
    ],
)
def test_acceptance_sampling_single_binomial(
    p1: float, p2: float, alpha: float, beta: float, expected_c: int, expected_n: int
):
    # Test cases are obtained from
    # https://acceptancesampling.com/
    plan = SingleSamplingPlan.binomial(p1, p2, alpha, beta)

    assert plan.c == expected_c
    assert plan.n == expected_n


@pytest.mark.parametrize(
    "p1, p2, alpha, beta, lot_size, expected_c, expected_n",
    [
        (0.05, 0.1, 0.15, 0.075, 1000, 10, 154),
        (0.08, 0.1, 0.05, 0.1, 500, 35, 391),
        (0.0065, 0.02, 0.02, 0.1, 6000, 8, 634),
    ],
)
def test_acceptance_sampling_single_hypergeometric(
    p1: float, p2: float, alpha: float, beta: float, lot_size: int, expected_c: int, expected_n: int
):
    # Test cases are obtained by comparing ourselves to the 'AcceptanceSampling' R package
    plan = SingleSamplingPlan.hypergeometric(p1, p2, alpha, beta, lot_size)

    assert plan.c == expected_c
    assert plan.n == expected_n
