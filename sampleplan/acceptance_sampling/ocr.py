#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
import numpy.typing as npt
from numba import jit, objmode
from numpy.random import Generator

from sampleplan.acceptance_sampling import DoubleSamplingPlan, SingleSamplingPlan

SEED = 421234123


def sample_repeatedly(data: npt.NDArray[bool], repetitions: int, num_items: int, replace: bool, seed: int):
    rng = np.random.default_rng(seed)

    if replace:
        # binomial sampling
        sample = rng.choice(data, size=(repetitions, num_items), replace=True)
    else:
        # hypergeometric sampling
        sample = np.empty((repetitions, num_items), dtype=bool)
        for i in range(repetitions):
            sample[i, :] = rng.choice(data, size=num_items, replace=False)

    assert sample.shape == (repetitions, num_items)

    return sample


def compute_ocr_single_sampling(
    plan: SingleSamplingPlan, data: npt.NDArray[bool], repetitions: int, replace: bool
) -> float:
    sample = sample_repeatedly(data, repetitions, plan.n, replace, seed=SEED)

    counts = np.sum(sample, axis=1)
    decisions = counts <= plan.c

    assert decisions.size == repetitions

    return decisions.sum() / repetitions


def compute_ocr_double_sampling(
    plan: DoubleSamplingPlan, data: npt.NDArray[bool], repetitions: int, replace: bool
) -> float:
    decisions = np.empty(repetitions, dtype=bool)

    # We sample always both splits here, because otherwise with replacement, it would be tricky
    # to not sample things we sampled already when using two function calls
    sample = sample_repeatedly(data, repetitions, plan.n * 2, replace, seed=SEED)
    sample1, sample2 = np.hsplit(sample, 2)

    assert sample.shape == (repetitions, plan.n * 2)
    assert sample1.shape == (repetitions, plan.n)
    assert sample2.shape == (repetitions, plan.n)

    counts1 = np.sum(sample1, axis=1)
    assert counts1.size == repetitions

    counts2 = np.sum(sample2, axis=1)
    assert counts2.size == repetitions

    counts = counts1 + counts2
    assert np.allclose(counts, np.sum(sample, axis=1))

    decisions[counts1 <= plan.c1] = True
    decisions[counts1 > plan.c2] = False

    decisions[counts <= plan.c1] = True
    decisions[counts > plan.c2] = False

    assert decisions.size == repetitions

    return decisions.sum() / repetitions


@jit(nopython=True)
def compute_ocr_sequential_sampling_curtailed(
    lower_limits: npt.NDArray[float],
    upper_limits: npt.NDArray[float],
    cutoff_critical_value: int,
    data: npt.NDArray[bool],
    repetitions: int,
    replace: bool,
) -> float:
    result = []

    assert len(lower_limits) == len(upper_limits)
    n = len(lower_limits)

    for repetition in range(repetitions):
        num_errors = 0

        with objmode(samples="bool_[:]"):
            rng = np.random.default_rng(SEED + repetition)
            samples = rng.choice(data, data.size, replace=replace)

        for i in range(n):
            is_error = samples[i]

            if is_error:
                num_errors += 1

            if num_errors <= lower_limits[i]:
                result.append(True)
                break
            elif num_errors >= upper_limits[i]:
                result.append(False)
                break
        else:
            # Truncation, use single sampling
            if num_errors <= cutoff_critical_value:
                result.append(True)
            else:
                result.append(False)

    assert len(result) == repetitions

    return sum(result) / repetitions


@jit(nopython=True)
def compute_ocr_sequential_sampling_full(
    lower_limits: npt.NDArray[float],
    upper_limits: npt.NDArray[float],
    data: npt.NDArray[bool],
    repetitions: int,
    replace: bool,
) -> float:
    assert len(lower_limits) == len(upper_limits)

    limit = 10 * data.size if replace else data.size

    result = []
    for repetition in range(repetitions):
        num_errors = 0

        with objmode(samples="bool_[:]"):
            rng = np.random.default_rng(SEED + repetition)
            samples = rng.choice(data, data.size, replace=replace)

        for i in range(limit):
            is_error = samples[i]

            if is_error:
                num_errors += 1

            if num_errors <= lower_limits[i]:
                result.append(True)
                break
            elif num_errors >= upper_limits[i]:
                result.append(False)
                break
        else:
            # Just accept when no decision could be made
            result.append(True)

    assert len(result) == repetitions

    return sum(result) / repetitions
