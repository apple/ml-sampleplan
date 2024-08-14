#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest

from sampleplan.acceptance_sampling.sequential import BinomialSequentialSamplingPlan


@pytest.mark.parametrize(
    "p1, p2, expected_h2, expected_h1, expected_s, expected_asn0, expected_asn1, expected_asn_p1, expected_asn_s, expected_asn_p2",
    [
        (0.005, 0.01, 4.1398, 3.2245, 0.007216, 447, 5, 1289, 1863, 1222),
        (0.005, 0.02, 2.0624, 1.6064, 0.01084, 149, 3, 244, 309, 185),
        (0.005, 0.03, 1.5906, 1.2389, 0.01400, 89, 2, 122, 143, 82),
        (0.005, 0.04, 1.3664, 1.0643, 0.01693, 63, 2, 79, 87, 49),
        (0.005, 0.05, 1.2305, 0.9585, 0.01970, 49, 2, 58, 61, 33),
        (0.005, 0.06, 1.1371, 0.8857, 0.02237, 40, 2, 45, 46, 25),
        (0.005, 0.07, 1.0679, 0.8318, 0.02496, 34, 2, 37, 36, 19),
        (0.010, 0.03, 2.5829, 2.0118, 0.01824, 111, 3, 216, 290, 181),
        (0.010, 0.04, 2.0397, 1.5887, 0.02172, 74, 3, 120, 153, 92),
        (0.010, 0.05, 1.7510, 1.3639, 0.02499, 55, 2, 81, 98, 58),
        (0.010, 0.06, 1.5678, 1.2211, 0.02811, 44, 2, 60, 70, 40),
        (0.010, 0.07, 1.4391, 1.1209, 0.03113, 37, 2, 47, 53, 30),
        (0.010, 0.08, 1.3426, 1.0458, 0.03406, 31, 2, 38, 43, 24),
        (0.015, 0.03, 4.0796, 3.1776, 0.02166, 147, 5, 423, 612, 402),
        (0.015, 0.04, 2.8716, 2.2367, 0.02554, 88, 3, 188, 258, 163),
        (0.015, 0.05, 2.3307, 1.8153, 0.02917, 63, 3, 113, 149, 92),
        (0.015, 0.06, 2.0169, 1.5710, 0.03263, 49, 3, 79, 100, 61),
        (0.015, 0.07, 1.8089, 1.4089, 0.03596, 40, 2, 60, 74, 44),
        (0.02, 0.03, 6.9527, 5.4154, 0.02467, 220, 8, 1027, 1565, 1073),
        (0.02, 0.04, 4.0495, 3.1541, 0.02889, 110, 5, 314, 455, 300),
        (0.02, 0.05, 3.0509, 2.3763, 0.03282, 73, 4, 164, 228, 146),
        (0.02, 0.06, 2.5348, 1.9743, 0.03655, 55, 3, 106, 142, 89),
        (0.02, 0.07, 2.2146, 1.7250, 0.04012, 43, 3, 76, 99, 61),
        (0.02, 0.08, 1.9941, 1.5532, 0.04359, 36, 3, 58, 74, 45),
        (0.02, 0.09, 1.8315, 1.4265, 0.04696, 31, 2, 47, 58, 35),
        (0.02, 0.10, 1.7056, 1.3285, 0.05025, 27, 2, 39, 47, 28),
    ],
)
def test_characteristic_quantities(
    p1: float,
    p2: float,
    expected_h2: float,
    expected_h1: float,
    expected_s: float,
    expected_asn0: int,
    expected_asn1: int,
    expected_asn_p1: int,
    expected_asn_s: int,
    expected_asn_p2: int,
):
    # From: Quality Control And Industrial Statistics
    # by Duncan, J. Acheson
    # https://archive.org/details/in.ernet.dli.2015.214236/page/n9/mode/2up
    # Table R in the appendix, book page 890 / pdf 924

    alpha = 0.05
    beta = 0.1
    plan = BinomialSequentialSamplingPlan(p1, p2, alpha, beta)

    assert plan.h1 == pytest.approx(expected_h1, abs=0.01)
    assert plan.h2 == pytest.approx(expected_h2, abs=0.01)
    assert plan.s == pytest.approx(expected_s, abs=0.01)

    assert plan.average_sample_number(0) == pytest.approx(expected_asn0, abs=1)
    assert plan.average_sample_number(1) == pytest.approx(expected_asn1, abs=1)
    assert plan.average_sample_number(p1) == pytest.approx(expected_asn_p1, abs=1)
    assert plan.average_sample_number(p2) == pytest.approx(expected_asn_p2, abs=1)
    assert plan.average_sample_number(plan.s) == pytest.approx(expected_asn_s, abs=1)


@pytest.mark.parametrize(
    "n, a, r",
    [
        (2, None, 2),
        (3, None, 2),
        (4, None, 2),
        (5, None, 2),
        (6, None, 2),
        (7, None, 2),
        (8, None, 2),
        (9, None, 2),
        (10, None, 2),
        (11, None, 2),
        (12, None, 2),
        (13, None, 2),
        (14, None, 2),
        (15, None, 2),
        (16, None, 2),
        (17, None, 2),
        (18, None, 2),
        (19, None, 2),
        (20, None, 3),
        (21, None, 3),
        (22, None, 3),
        (23, None, 3),
        (24, None, 3),
        (25, None, 3),
        (26, None, 3),
        (27, None, 3),
        (28, None, 3),
        (29, None, 3),
        (30, None, 3),
        (31, 0, 3),
        (32, 0, 3),
        (33, 0, 3),
        (34, 0, 3),
        (35, 0, 3),
        (36, 0, 3),
        (37, 0, 3),
        (38, 0, 3),
        (39, 0, 3),
        (40, 0, 3),
        (41, 0, 3),
        (42, 0, 3),
        (43, 0, 3),
        (44, 0, 3),
        (45, 0, 3),
        (46, 0, 3),
        (47, 0, 3),
        (48, 0, 3),
        (49, 0, 4),
        (50, 0, 4),
        (51, 0, 4),
        (52, 0, 4),
        (53, 0, 4),
        (54, 0, 4),
        (55, 0, 4),
        (56, 0, 4),
        (57, 0, 4),
        (58, 0, 4),
        (59, 0, 4),
        (60, 0, 4),
        (61, 1, 4),
        (62, 1, 4),
        (63, 1, 4),
        (64, 1, 4),
        (65, 1, 4),
        (66, 1, 4),
        (67, 1, 4),
        (68, 1, 4),
        (69, 1, 4),
        (70, 1, 4),
    ],
)
def test_acceptance_and_rejection_numbers(n: int, a: int, r: int):
    # From: Quality Control And Industrial Statistics
    # by Duncan, J. Acheson
    # https://archive.org/details/in.ernet.dli.2015.214236/page/n9/mode/2up
    # Fig. 49 in the appendix, book page 154 / pdf 188

    alpha = 0.05
    beta = 0.1
    p1 = 0.01
    p2 = 0.08

    plan = BinomialSequentialSamplingPlan(p1, p2, alpha, beta)

    actual_a, actual_r = plan.compute_boundaries(n)

    assert actual_a == a
    assert actual_r == r
