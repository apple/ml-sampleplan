#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional, Tuple

import pytest

from sampleplan.acceptance_sampling import DoubleSamplingPlan, SingleSamplingPlan

# Single
from sampleplan.acceptance_sampling.sequential import (
    HypergeometricSequentialSamplingPlan,
    compute_boundaries,
)


def test_compute_boundaries():
    # Equation 2.15 in "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"
    alpha = 0.05
    beta = 0.1

    boundaries = compute_boundaries(alpha, beta)

    assert boundaries.alpha == alpha
    assert boundaries.beta == beta
    assert boundaries.log_lower_bound == pytest.approx(-2.25129)
    assert boundaries.log_upper_bound == pytest.approx(2.89037)


def test_computing_critical_limits_hypergeometric():
    # Table 2.1 in "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

    alpha = 0.05
    beta = 0.1
    N = 100
    p1 = 25 / N
    p2 = 40 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, alpha, beta, N)

    cl = plan.compute_lower_critical_limit(32)
    cu = plan.compute_upper_critical_limit(32)

    assert cl == 8
    assert cu == 14


@pytest.mark.parametrize(
    "n, cl, cu, initial_cl, initial_cu",
    [
        (1, None, None, None, None),
        (2, None, None, None, None),
        (3, None, None, None, None),
        (4, None, None, None, None),
        (5, None, None, None, None),
        (6, None, 6, None, None),
        (6, None, 6, None, 6),
        (7, None, 7, None, 6),
        (7, None, 7, None, 7),
        (8, None, 7, None, 7),
        (9, None, 7, None, 7),
        (10, 0, 7, None, 7),
        (10, 0, 7, 0, 7),
        (11, 0, 8, 0, 7),
        (11, 0, 8, 0, 8),
        (12, 0, 8, 0, 8),
        (13, 1, 8, 0, 8),
        (13, 1, 8, 1, 8),
        (14, 1, 9, 1, 8),
        (14, 1, 9, 1, 9),
        (15, 1, 9, 1, 9),
        (17, 2, 9, 1, 9),
        (17, 2, 9, 2, 9),
        (18, 3, 10, 2, 9),
        (18, 3, 10, 3, 10),
        (20, 3, 10, 3, 10),
        (21, 4, 10, 3, 10),
        (21, 4, 10, 4, 10),
        (22, 4, 11, 4, 10),
        (22, 4, 11, 4, 11),
        (23, 4, 11, 4, 11),
        (24, 5, 11, 4, 11),
        (24, 5, 11, 5, 11),
        (25, 5, 12, 5, 11),
        (25, 5, 12, 5, 12),
        (26, 5, 12, 5, 12),
        (27, 6, 12, 5, 12),
        (27, 6, 12, 6, 12),
        (28, 6, 12, 6, 12),
        (29, 6, 13, 6, 12),
        (29, 6, 13, 6, 13),
        (30, 7, 13, 6, 13),
        (30, 7, 13, 7, 13),
        (31, 7, 13, 7, 13),
        (32, 8, 14, 7, 13),
        (32, 8, 14, 8, 14),
        (33, 8, 14, 8, 14),
        (34, 8, 14, 8, 14),
        (35, 9, 14, 8, 14),
        (35, 9, 14, 9, 14),
        (36, 9, 15, 9, 14),
        (36, 9, 15, 9, 15),
        (37, 9, 15, 9, 15),
        (38, 10, 15, 9, 15),
        (38, 10, 15, 10, 15),
        (39, 10, 16, 10, 15),
        (39, 10, 16, 10, 16),
        (40, 10, 16, 10, 16),
        (41, 11, 16, 10, 16),
        (41, 11, 16, 11, 16),
        (41, 11, 16, 11, 16),
        (42, 11, 16, None, None),
        (43, 11, 17, None, None),
        (44, 12, 17, None, None),
        (45, 12, 17, None, None),
        (46, 13, 17, None, None),
        (47, 13, 18, None, None),
        (48, 13, 18, None, None),
        (49, 14, 18, None, None),
        (50, 14, 19, None, None),
        (51, 14, 19, None, None),
        (52, 15, 19, None, None),
        (53, 15, 19, None, None),
        (54, 15, 20, None, None),
        (55, 16, 20, None, None),
        (56, 16, 20, None, None),
        (57, 16, 21, None, None),
        (58, 17, 21, None, None),
        (59, 17, 21, None, None),
        (60, 17, 21, None, None),
        (61, 18, 22, None, None),
        (62, 18, 22, None, None),
        (63, 19, 22, None, None),
        (64, 19, 22, None, None),
        (65, 19, 23, None, None),
        (66, 20, 23, None, None),
        (67, 20, 23, None, None),
        (68, 20, 23, None, None),
        (69, 21, 24, None, None),
        (70, 21, 24, None, None),
        (71, 21, 24, None, None),
        (72, 22, 24, None, None),
        (73, 22, 25, None, None),
        (74, 22, 25, None, None),
        (75, 23, 25, None, None),
        (76, 23, 25, None, None),
        (77, 23, 26, None, None),
        (78, 24, 26, None, None),
        (79, 24, 26, None, None),
        (80, 24, 26, None, None),
        (81, 25, 26, None, None),
    ],
)
def test_computing_critical_limits_multiple(n: int, cl: int, cu: int, initial_cl: int, initial_cu: int):
    # Table 2.2 in "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

    alpha = 0.05
    beta = 0.1
    N = 100
    p1 = 25 / N
    p2 = 40 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, alpha, beta, N)

    actual_cl = plan.compute_lower_critical_limit(n, initial_guess=initial_cl)
    actual_cu = plan.compute_upper_critical_limit(n, initial_guess=initial_cu)

    assert actual_cl == cl
    assert actual_cu == cu


@pytest.mark.parametrize(
    "N, d1, d2, cutoff",
    [
        (30, 10, 20, 13),
        (50, 2, 12, 19),
        (50, 20, 30, 28),
        (100, 5, 20, 29),
    ],
)
def test_fixed_size_test_cutoff(N: int, d1: int, d2: int, cutoff: int):
    # Section 4 figures in "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"
    # Some of their values seem to be wrong, we are sure that our algorithm is correct for
    # computing SingleSamplingPlan.hypergeometric, so we only have a subet of plans here that
    # are in Meeker
    p1 = d1 / N
    p2 = d2 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, 0.05, 0.1, N)

    actual_cutoff = plan.compute_fixed_test_truncation()

    assert actual_cutoff == cutoff


@pytest.mark.parametrize(
    "N, d1, d2, cutoff, limits",
    [
        # Test Plan 1
        (
            30,
            5,
            15,
            13,
            [
                (0, None, None),
                (1, None, None),
                (2, None, None),
                (3, None, 3),
                (4, None, 3),
                (5, 0, 4),
                (6, 0, 4),
                (7, 1, 4),
                (8, 1, 4),
                (9, 1, 5),
                (10, 2, 5),
                (11, 2, 5),
                (12, 3, 5),  # Region change
                (13, 4, 5),
            ],
        )
    ],
)
def test_fixed_size_test_truncation(
    N: int, d1: int, d2: int, cutoff: int, limits: List[Tuple[int, Optional[int], Optional[int]]]
):
    # This tests the truncation rules for Wald regions given in Section 3.2 of
    # "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

    p1 = d1 / N
    p2 = d2 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, 0.05, 0.1, N)

    region = plan.compute_truncated_wald_region(cutoff)

    for n, cl, cu in limits:
        actual_cl = region.lower_limits[n]
        actual_cu = region.upper_limits[n]

        assert actual_cl == cl, f"n = {n}"
        assert actual_cu == cu, f"n = {n}"


@pytest.mark.parametrize(
    "N, d1, d2, D, cutoff, P",
    [
        # Test Plan from Table 3.3, "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975", D = 30
        (
            100,
            25,
            40,
            30,
            81,
            [
                (6, 0.00000, 0.00050),
                (7, 0.00000, 0.00000),
                (8, 0.00000, 0.00057),
                (9, 0.00000, 0.00194),
                (10, 0.02292, 0.00442),
                (11, 0.00000, 0.00000),
                (12, 0.00000, 0.00216),
                (13, 0.03453, 0.00519),
                (14, 0.00000, 0.00000),
                (15, 0.00000, 0.00234),
                (16, 0.03904, 0.00546),
                (17, 0.00000, 0.00937),
                (18, 0.05970, 0.000000),
                (19, 0.00000, 0.00360),
                (20, 0.00000, 0.00772),
                (21, 0.04973, 0.01242),
                (22, 0.000000, 0.00000),
                (23, 0.000000, 0.004553),
                (24, 0.04511, 0.00943),
                (25, 0.00000, 0.00000),
                (26, 0.00000, 0.00372),
                (27, 0.04157, 0.00799),
                (28, 0.00000, 0.01272),
                (29, 0.00000, 0.00000),
                (30, 0.03858, 0.00453),
                (31, 0.00000, 0.00932),
                (32, 0.05321, 0.00000),
                (33, 0.00000, 0.00359),
                (34, 0.00000, 0.0076),
                (35, 0.04175, 0.0121),
                (36, 0.00000, 0.00000),
                (37, 0.00000, 0.00419),
                (38, 0.03673, 0.00862),
                (39, 0.00000, 0.00000),
                (40, 0.00000, 0.00324),
                (41, 0.03321, 0.00693),
                (42, 0.00000, 0.01086),
                (43, 0.00000, 0.00000),
            ],
        ),
        # Test Plan 1 p. 75 "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975", D = 0
        (
            30,
            5,
            15,
            0,
            13,
            [
                (5, 1.00000, 0.00000),
                (6, 0.00000, 0.00000),
                (7, 0.00000, 0.00000),
                (8, 0.00000, 0.00000),
                (9, 0.00000, 0.00000),
                (10, 0.00000, 0.00000),
                (11, 0.00000, 0.00000),
                (12, 0.00000, 0.00000),
                (13, 0.00000, 0.00000),
            ],
        ),
        # Test Plan 1 p. 75 "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975", D = 10
        (
            30,
            5,
            15,
            10,
            13,
            [
                (3, 0.00000, 0.02956),
                (4, 0.00000, 0.06568),
                (5, 0.10880, 0.00000),
                (6, 0.00000, 0.02688),
                (7, 0.13599, 0.05376),
                (8, 0.00000, 0.07698),
                (9, 0.00000, 0.00000),
                (10, 0.08385, 0.02634),
                (11, 0.00000, 0.04724),
                (12, 0.09639, 0.06074),
                (13, 0.12522, 0.06261),
            ],
        ),
        # Test Plan 1 p. 75 "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975", D = 20
        (
            30,
            5,
            15,
            20,
            13,
            [
                (3, 0.000000, 0.28079),
                (4, 0.00006, 0.31199),
                (5, 0.00177, 0.0000),
                (6, 0.00000, 0.14682),
                (7, 0.00147, 0.1305),
                (8, 0.00000, 0.07696),
                (9, 0.00000, 0.00000),
                (10, 0.00013, 0.02632),
                (11, 0.00000, 0.01574),
                (12, 0.00005, 0.00578),
                (13, 0.00017, 0.00135),
            ],
        ),
        # Test Plan 1 p. 75 "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975", D = 25
        (
            30,
            5,
            15,
            25,
            13,
            [
                (3, 0.00000, 0.56650),
                (4, 0.00000, 0.31472),
                (5, 0.00001, 0.00009),
                (6, 0.0000, 0.08522),
                (7, 0.00000, 0.02841),
                (8, 0.00000, 0.00479),
                (9, 0.00000, 0.00000),
                (10, 0.000000, 0.00036),
                (11, 0.00000, 0.00000),
                (12, 0.00000, 0.00000),
                (13, 0.00000, 0.00000),
            ],
        ),
    ],
)
def test_decisive_sample_numbers(N: int, d1: int, d2: int, D: int, cutoff: int, P: List[Tuple[int, float, float]]):
    p1 = d1 / N
    p2 = d2 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, 0.05, 0.1, N)
    props = plan.average_sample_number(D, cutoff)

    for n, P0, P1 in P:
        actual_P0 = props.A0[n]
        actual_P1 = props.A1[n]

        assert actual_P0 == pytest.approx(P0, abs=0.0001), f"n = {n}"
        assert actual_P1 == pytest.approx(P1, abs=0.0001), f"n = {n}"


@pytest.mark.parametrize(
    "N, d1, d2, cutoff, D, expected_asn",
    [
        # Table 3.4 in Meeker
        (100, 25, 40, 81, 20, 21.4806),
        (100, 25, 40, 81, 25, 28.4823),
        (100, 25, 40, 81, 30, 35.7704),
        (100, 25, 40, 81, 35, 35.8717),
        (100, 25, 40, 81, 40, 29.7936),
        (100, 25, 40, 81, 45, 23.6276),
        # Test Plan 1 in Meeker
        (30, 5, 15, 13, 0, 5.0000),
        (30, 5, 15, 13, 5, 7.4386),
        (30, 5, 15, 13, 10, 8.9493),
        (30, 5, 15, 13, 20, 5.0482),
        (30, 5, 15, 13, 25, 3.7104),
    ],
)
def test_average_sample_number_truncated(N: int, d1: int, d2: int, cutoff: int, D: int, expected_asn: float):
    p1 = d1 / N
    p2 = d2 / N

    plan = HypergeometricSequentialSamplingPlan(p1, p2, 0.05, 0.1, N)
    props = plan.average_sample_number(D, cutoff)

    assert props.average_sample_number == pytest.approx(expected_asn, abs=0.01)
