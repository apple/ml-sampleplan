#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest

from sampleplan.confidence_interval import (
    binomial_proportion_interval_exact_midp,
    hypergeometric_proportion_interval_exact,
    sample_size_agresti_coull_binomial,
    sample_size_exact_binomial,
    sample_size_exact_mid_point_binomial,
)


@pytest.mark.parametrize(
    "confidence, num_successes, sample_size, expected_cl, expected_cu",
    [
        # Numbers come from https://rdrr.io/cran/binomSamSize/man/binom.midp.html
        (0.95, 0, 10, 0.000000000, 0.2588620),
        (0.95, 1, 10, 0.005001757, 0.4034747),
        (0.95, 2, 10, 0.034982476, 0.5194848),
        (0.95, 3, 10, 0.082626119, 0.6199217),
        (0.95, 4, 10, 0.142291370, 0.7088382),
        (0.95, 5, 10, 0.212008566, 0.7879914),
        (0.95, 6, 10, 0.291161788, 0.8577086),
        (0.95, 7, 10, 0.380078280, 0.9173739),
        (0.95, 8, 10, 0.480515200, 0.9650175),
        (0.95, 9, 10, 0.596525278, 0.9949982),
        (0.95, 10, 10, 0.741137963, 1.0000000),
        (0.9, 0, 5, 0.00000000, 0.3690451),
        (0.9, 1, 5, 0.02001793, 0.5969908),
        (0.9, 2, 5, 0.10573077, 0.7656125),
        (0.9, 3, 5, 0.23438755, 0.8942692),
        (0.9, 4, 5, 0.40300920, 0.9799821),
        (0.9, 5, 5, 0.63095487, 1.0000000),
    ],
)
def test_binomial_proportion_interval_exact_midp(
    confidence: float,
    num_successes: int,
    sample_size: int,
    expected_cl: float,
    expected_cu: float,
):
    cl, cu = binomial_proportion_interval_exact_midp(1 - confidence, sample_size, num_successes)

    assert cl == pytest.approx(expected_cl, abs=0.01)
    assert cu == pytest.approx(expected_cu, abs=0.01)


@pytest.mark.parametrize(
    "confidence, population_size, sample_size, num_successes, expected_cl, expected_cu",
    [
        # Numbers come from http://www.cluster-text.com/confidence_interval.php
        (0.95, 1000, 100, 42, 32.7, 51.7),
        (0.95, 500, 100, 42, 33.2, 51.2),
        (0.95, 200, 100, 42, 35.0, 49.0),
        (0.95, 1000, 200, 137, 62.3, 74.2),
        (0.99, 1000, 200, 137, 60.4, 75.8),
    ],
)
def test_hypergeometric_exact_interval(
    confidence: float,
    population_size: int,
    sample_size: int,
    num_successes: int,
    expected_cl: float,
    expected_cu: float,
):
    cl, cu = hypergeometric_proportion_interval_exact(confidence, population_size, sample_size, num_successes)

    assert cl == pytest.approx(expected_cl / 100.0, abs=0.01)
    assert cu == pytest.approx(expected_cu / 100.0, abs=0.01)


# Sample sizes


@pytest.mark.parametrize(
    "p, alpha, half_width, expected",
    [
        (0.12, 0.05, 0.025, 686),
        (0.5, 0.05, 0.05, 402),
    ],
)
def test_binomial_confidence_interval_exact(p: float, alpha: float, half_width: float, expected: int):
    # From https://sample-size.net/sample-size-conf-interval-proportion/
    actual = sample_size_exact_binomial(p, alpha, half_width)
    assert actual == expected


@pytest.mark.parametrize(
    "p, alpha, half_width, expected",
    [
        (0.15, 0.05, 0.02, 1228),
        (0.1, 0.05, 0.1, 41),
        (0.05, 0.05, 0.01, 1854),
    ],
)
def test_binomial_confidence_interval_agresti_coull(p: float, alpha: float, half_width: float, expected: int):
    actual = sample_size_agresti_coull_binomial(p, alpha, half_width)
    assert actual == expected


@pytest.mark.parametrize(
    "p, expected, alpha, half_width",
    [
        # Fosgate GT. Modified exact sample size for a binomial proportion with special
        # emphasis on diagnostic test parameter estimation.
        # Stat Med. 2005 Sep 30;24(18):2857-66. doi: 10.1002/sim.2146. PMID: 16134167.
        #
        # alpha = 0.1, w = 0.1
        (0.50, 68, 0.1, 0.1),
        (0.55, 68, 0.1, 0.1),
        (0.60, 65, 0.1, 0.1),
        (0.65, 62, 0.1, 0.1),
        (0.70, 59, 0.1, 0.1),
        (0.75, 52, 0.1, 0.1),
        (0.80, 45, 0.1, 0.1),
        (0.85, 39, 0.1, 0.1),
        (0.90, 29, 0.1, 0.1),
        # alpha = 0.1, w = 0.05
        (0.50, 270, 0.1, 0.05),
        (0.55, 268, 0.1, 0.05),
        (0.60, 260, 0.1, 0.05),
        (0.65, 248, 0.1, 0.05),
        (0.70, 229, 0.1, 0.05),
        (0.75, 204, 0.1, 0.05),
        (0.80, 175, 0.1, 0.05),
        (0.85, 140, 0.1, 0.05),
        (0.90, 100, 0.1, 0.05),
        # alpha = 0.05, w = 0.1
        (0.50, 96, 0.05, 0.1),
        (0.55, 95, 0.05, 0.1),
        (0.60, 92, 0.05, 0.1),
        (0.65, 88, 0.05, 0.1),
        (0.70, 80, 0.05, 0.1),
        (0.75, 72, 0.05, 0.1),
        (0.80, 64, 0.05, 0.1),
        (0.85, 53, 0.05, 0.1),
        # alpha = 0.05, w = 0.05
        (0.50, 384, 0.05, 0.05),
        (0.55, 380, 0.05, 0.05),
        (0.60, 369, 0.05, 0.05),
        (0.65, 351, 0.05, 0.05),
        (0.70, 323, 0.05, 0.05),
        (0.75, 288, 0.05, 0.05),
        (0.80, 249, 0.05, 0.05),
        (0.85, 200, 0.05, 0.05),
        (0.90, 148, 0.05, 0.05),
        # alpha = 0.01, w = 0.1
        (0.50, 165, 0.01, 0.1),
        (0.55, 162, 0.01, 0.1),
        (0.60, 160, 0.01, 0.1),
        (0.65, 151, 0.01, 0.1),
        (0.70, 140, 0.01, 0.1),
        (0.75, 128, 0.01, 0.1),
        (0.80, 115, 0.01, 0.1),
        (0.85, 99, 0.01, 0.1),
        (0.90, 80, 0.01, 0.1),
        # alpha = 0.01, w = 0.05
        (0.50, 662, 0.01, 0.05),
        (0.55, 655, 0.01, 0.05),
        (0.60, 635, 0.01, 0.05),
        (0.65, 605, 0.01, 0.05),
        (0.70, 560, 0.01, 0.05),
        (0.75, 500, 0.01, 0.05),
        (0.80, 430, 0.01, 0.05),
        (0.85, 353, 0.01, 0.05),
        (0.90, 260, 0.01, 0.05),
    ],
)
@pytest.mark.skip(reason="deprecated")
def test_sample_size_exact_mid_point_binomial(p: float, expected: int, alpha: float, half_width: float):
    actual = sample_size_exact_mid_point_binomial(p, alpha, half_width)
    assert actual == pytest.approx(expected, rel=0.1)


# TODO: Test exact not midP
