#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import scipy.stats
from numba import njit
from scipy.optimize import root_scalar


@dataclass
class Boundaries:
    alpha: float
    beta: float
    lower_bound: float
    upper_bound: float
    log_lower_bound: float
    log_upper_bound: float


def compute_boundaries(alpha: float, beta: float) -> Boundaries:
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert alpha + beta <= 1

    lower_bound = beta / (1 - alpha)
    upper_bound = (1 - beta) / alpha
    log_lower_bound = math.log(lower_bound)
    log_upper_bound = math.log(upper_bound)

    return Boundaries(alpha, beta, lower_bound, upper_bound, log_lower_bound, log_upper_bound)


@dataclass
class SequentialSamplingPlanRegion:
    num_trials: int
    lower_limits: List[int]
    upper_limits: List[int]
    lower_limits_array: npt.NDArray[float] = None
    upper_limits_array: npt.NDArray[float] = None

    def __post_init__(self):
        assert len(self.lower_limits) == self.num_trials, (len(self.lower_limits), self.num_trials)
        assert len(self.upper_limits) == self.num_trials, (len(self.upper_limits), self.num_trials)

        self.lower_limits_array = np.array([x if x is not None else -np.inf for x in self.lower_limits])
        self.upper_limits_array = np.array([x if x is not None else np.inf for x in self.upper_limits])

        assert self.lower_limits_array.size == len(self.lower_limits)
        assert self.upper_limits_array.size == len(self.upper_limits)

    def pretty_print_limits(self):
        print("i\tcl\tcu")

        for i, (cl, cu) in enumerate(zip(self.lower_limits, self.upper_limits)):
            cl = cl if cl is not None else "*"
            cu = cu if cu is not None else "*"
            print(f"{i}\t{cl}\t{cu}")

        print()

    def plot_debug_plot(self):
        from matplotlib import pyplot as plt

        t = range(self.num_trials)

        plt.plot(
            t, [e if e is not None else np.nan for e in self.lower_limits], color="k", linestyle="--", linewidth=".75"
        )
        plt.plot(
            t, [e if e is not None else np.nan for e in self.upper_limits], color="k", linestyle="--", linewidth=".75"
        )

        for n in t:
            cl = self.lower_limits[n]
            cu = self.upper_limits[n]

            for d in range(n + 1):
                if cl is not None and d <= cl:
                    color = "b"
                elif cu is not None and d >= cu:
                    color = "r"
                else:
                    color = "silver"

                plt.plot([n], [d], marker="o", color=color)


@dataclass
class SequentialSamplingPlanProperties:
    num_trials: int
    num_defects: int
    region: SequentialSamplingPlanRegion
    P: npt.NDArray[float]
    A0: npt.NDArray[float]
    A1: npt.NDArray[float]

    average_sample_number: float

    def pretty_print_limits(self):
        self.region.pretty_print_limits()

    def pretty_print_probabilities(self, start_from: int = 0):
        print("i\tH0\t\t\tH1")

        for i, (h0, h1) in enumerate(zip(self.A0, self.A1)):
            if i < start_from:
                continue

            print(f"{i}\t{h0:.5f}\t\t{h1:.5f}")

    def plot_debug_plot(self):
        from matplotlib import pyplot as plt

        t = list(range(self.num_trials))

        self.region.plot_debug_plot()

        for n in t:
            for d in range(n + 1):
                p = self.P[n, d]
                plt.text(n + 0.15, d, f"{p:.5f}", fontsize="x-small")

        plt.grid()
        plt.xticks(np.arange(0, self.num_trials + 1, 1.0))
        plt.yticks(np.arange(0, self.num_trials + 1, 1.0))

        plt.xlim((0, self.num_trials))
        plt.ylim((0, self.num_defects))

        plt.tight_layout()
        plt.show()


class BinomialSequentialSamplingPlan:
    def __init__(self, p1: float, p2: float, alpha: float, beta: float):
        assert 0 <= p1 < p2 <= 1

        self._boundaries = compute_boundaries(alpha, beta)

        self._p1 = p1
        self._p2 = p2

        # From: Quality Control And Industrial Statistics
        # by Duncan, J. Acheson
        # https://archive.org/details/in.ernet.dli.2015.214236/page/n9/mode/2up
        # The formula itself can be found in Appendix (25) eq (7) on page 830 in the book/ page 864 in the PDF

        x = (1 - p1) / (1 - p2)
        denominator = (p2 / p1) * x
        log_denominator = math.log(denominator)

        # We use the same boundaries as Wald, they differ by the sign before h1; we leave h1 positive and move
        # the sign into the log boundaries, therefore inverting the fraction (- log (a/b) = log(b / a))
        self._h1 = -self._boundaries.log_lower_bound / log_denominator
        self._h2 = self._boundaries.log_upper_bound / log_denominator

        self._s = math.log(x) / log_denominator

    @property
    def h1(self) -> float:
        return self._h1

    @property
    def h2(self) -> float:
        return self._h2

    @property
    def s(self) -> float:
        return self._s

    def compute_region(self, num_trials: int):
        # From: Quality Control And Industrial Statistics
        # by Duncan, J. Acheson
        # Eq 26, page 155 in the book, 189 in the pdf

        lower_limits = []
        upper_limits = []

        for i in range(0, num_trials):
            lower_bound, upper_bound = self.compute_boundaries(i)
            lower_limits.append(lower_bound)
            upper_limits.append(upper_bound)

        return SequentialSamplingPlanRegion(num_trials=num_trials, lower_limits=lower_limits, upper_limits=upper_limits)

    def compute_boundaries(self, n: int) -> tuple[int, int]:
        lower_bound = -self.h1 + self.s * n
        upper_bound = self.h2 + self.s * n

        a = math.floor(lower_bound) if lower_bound >= 0 else None
        r = math.ceil(upper_bound) if upper_bound >= 0 else None

        return a, r

    def average_sample_number(self, p: float) -> float:
        # From: Quality Control And Industrial Statistics
        # by Duncan, J. Acheson
        # Eq 28 and 30, page 156 in the book, 190 in the pdf
        assert 0 <= p <= 1

        p1 = self._p1
        p2 = self._p2
        b = self._boundaries

        assert 0 <= p <= 1, p

        if np.isclose(p, 0):
            return self.h1 / self.s

        if np.isclose(p, 1):
            return self.h2 / (1 - self.s)

        if np.isclose(p, self.s):
            return (self.h1 * self.h2) / (self.s * (1 - self.s))

        def _fn(theta: float):
            # Lot fraction defective
            A = ((1 - p2) / (1 - p1)) ** theta
            B = (p2 / p1) ** theta
            p_prime = (1 - A) / (B - A)
            return p - p_prime

        # Lot fraction defective
        theta = root_scalar(_fn, x0=-1, x1=1).root

        # Probability of acceptance
        p_a_1 = b.upper_bound**theta
        p_a_2 = b.lower_bound**theta
        p_a = (p_a_1 - 1) / (p_a_1 - p_a_2)

        assert 0 <= p_a <= 1, p_a

        numerator = p_a * b.log_lower_bound + (1 - p_a) * b.log_upper_bound
        x = math.log(p2 / p1)
        y = math.log((1 - p2) / (1 - p1))
        denominator = p * x + (1 - p) * y

        asn = numerator / denominator
        asn = math.ceil(asn)

        return asn

    def average_sample_with_cutoff(self, p: float, cutoff: int) -> float:
        bounds = self.compute_region(cutoff)

        asn = self._average_sample_number_fast(cutoff, p, bounds.lower_limits_array, bounds.upper_limits_array)

        return asn[-1]

    @staticmethod
    @njit
    def _average_sample_number_fast(cutoff: int, p: float, lower_limits: np.ndarray, upper_limits: np.ndarray):
        # Implements "Aronian, 1968, Sequential Analysis, Direct Method" as described in
        # "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"
        # We adapt the part where P[n, x] and P[n, x-1] to using the binomial distribution

        # trials x defects
        P = np.zeros((cutoff + 1, cutoff + 1))

        P[0, 0] = 1  # P(x,D,0,N) = 1 if x = 0 else 0

        # For all trials
        for n in range(cutoff):
            cl = lower_limits[n]
            cu = upper_limits[n]

            # For all defects possibly seen so far
            for x in range(n + 1 + 1):
                a = 0
                b = 0

                if cl < x < cu:
                    a = P[n, x] * (1 - p)

                if cl < x - 1 < cu:
                    b = P[n, x - 1] * p

                P[n + 1, x] = a + b

        A0 = np.zeros(cutoff)  # Probability of Accepting H0 at step n
        A1 = np.zeros(cutoff)  # Probability of Accepting H1 at step n

        for n in range(cutoff):
            cl = lower_limits[n]
            cu = upper_limits[n]

            if np.isfinite(cl):
                cl = int(cl)
                a = P[n, cl]
                b = P[n, cl - 1] if cl - 1 >= 0 else 0

                A0[n] = a + b

            if np.isfinite(cu):
                cu = int(cu)

                a = P[n, cu]
                b = P[n, cu + 1] if cu + 1 <= cutoff else 0

                A1[n] = a + b

        asn = 0

        for n in range(1, cutoff):
            asn += n * (A0[n] + A1[n])

        return P, A0, A1, asn


class HypergeometricSequentialSamplingPlan:
    # Maps cutoff to region to avoid recomputing
    _regions_cache: Dict[int, SequentialSamplingPlanRegion] = {}

    def __init__(self, p1: float, p2: float, alpha: float, beta: float, lot_size: int):
        assert p1 < p2
        assert 0 <= p1 <= 1
        assert 0 <= p2 <= 1

        self._boundaries = compute_boundaries(alpha, beta)

        self._p1 = p1
        self._p2 = p2
        self._lot_size = lot_size

    def compute_lower_critical_limit(self, n: int, initial_guess: Optional[int] = None):
        # Implements equation 2.12 from "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

        if initial_guess is None:
            initial_guess = self._lot_size

        for x in range(min(initial_guess + 1, self._lot_size), -1, -1):
            ratio = self._compute_log_ratio(x, n)
            if ratio is None:
                continue

            if ratio < self._boundaries.log_lower_bound:
                return x

        return None

    def compute_upper_critical_limit(self, n: int, initial_guess: Optional[int] = None):
        # Implements equation 2.12 from "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

        if initial_guess is None:
            initial_guess = 0

        for x in range(max(initial_guess - 1, 0), self._lot_size + 1):
            ratio = self._compute_log_ratio(x, n)
            if ratio is None:
                continue

            if ratio > self._boundaries.log_upper_bound:
                return x

        return None

    def _compute_log_ratio(self, x: int, n: int) -> Optional[float]:
        a = scipy.stats.hypergeom.logpmf(x, self._lot_size, round(self._lot_size * self._p2), n)
        b = scipy.stats.hypergeom.logpmf(x, self._lot_size, round(self._lot_size * self._p1), n)

        if np.isinf(a):
            return None
        elif np.isinf(b):
            ratio = np.inf
        else:
            # log(a/b) = log(a) - log(b)
            ratio = a - b

        return ratio

    def compute_fixed_test_truncation(self) -> int:
        # Section 1.3 in "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

        # Avoid circular import
        from sampleplan.acceptance_sampling import SingleSamplingPlan

        plan = SingleSamplingPlan.hypergeometric(
            self._p1, self._p2, alpha=self._boundaries.alpha, beta=self._boundaries.beta, lot_size=self._lot_size
        )

        return plan.n

    def compute_truncated_wald_region(self, cutoff: Optional[int] = None) -> SequentialSamplingPlanRegion:
        if cutoff is None:
            cutoff = self._lot_size

        cache = HypergeometricSequentialSamplingPlan._regions_cache
        if cutoff in cache:
            return deepcopy(cache[cutoff])

        lower_limits = [None]
        upper_limits = [None]

        for n in range(1, cutoff + 1):
            cl = self.compute_lower_critical_limit(n, initial_guess=lower_limits[-1])
            cu = self.compute_upper_critical_limit(n, initial_guess=upper_limits[-1])

            if cl is None:
                cl = lower_limits[-1]

            if cu is None:
                cu = upper_limits[-1]

            lower_limits.append(cl)
            upper_limits.append(cu)

        assert len(lower_limits) == len(upper_limits)

        # Wedge truncation, we let upper limit be a horizontal line and adjust the lower limit
        # so that at the end, they only differ by one. The lower line then has a 45° angle
        upper_limits[-1] = upper_limits[-2]
        lower_limits[-1] = upper_limits[-1] - 1

        for i in range(1, cutoff + 1):
            # If there is a gap of size 2 or more, then we need to flatten it out and propagate
            # back the change we made until the max diff is at most 1
            diff = lower_limits[-i] - lower_limits[-i - 1]
            if diff > 1:
                lower_limits[-i - 1] += 1
            else:
                break

        region = SequentialSamplingPlanRegion(
            num_trials=cutoff + 1, lower_limits=lower_limits, upper_limits=upper_limits
        )

        cache[cutoff] = region

        return region

    def average_sample_number(self, d: int, cutoff: Optional[int] = None) -> SequentialSamplingPlanProperties:
        # Implements "Aronian, 1968, Sequential Analysis, Direct Method" as described in
        # "Meeker, Sequential Tests of the Hypergeometric Distribution, 1975"

        assert int(d) == d, "d is not a float, we need to pass number of defects, NOT the probability here!"

        if cutoff is None:
            cutoff = self._lot_size

        region = self.compute_truncated_wald_region(cutoff)

        num_trials = cutoff + 1

        P, A0, A1, asn = self._average_sample_number_fast(
            num_trials, self._lot_size, d, cutoff, region.lower_limits_array, region.upper_limits_array
        )

        props = SequentialSamplingPlanProperties(
            num_defects=d,
            num_trials=num_trials,
            region=region,
            P=P,
            A0=A0,
            A1=A1,
            average_sample_number=asn,
        )

        return props

    @staticmethod
    @njit
    def _average_sample_number_fast(
        num_trials: int, lot_size: int, d: int, cutoff: int, lower_limits: np.ndarray, upper_limits: np.ndarray
    ):
        # trials x defects
        P = np.zeros((num_trials, num_trials))

        P[0, 0] = 1  # P(x,D,0,N) = 1 if x = 0 else 0

        N = lot_size

        # For all trials
        for n in range(cutoff):
            cl = lower_limits[n]
            cu = upper_limits[n]

            # For all defects possibly seen so far
            for x in range(n + 1 + 1):
                a = 0
                b = 0

                if cl < x < cu:
                    a = P[n, x] * (N - n - d + x) / (N - n)

                if cl < x - 1 < cu:
                    b = P[n, x - 1] * (d - x + 1) / (N - n)

                P[n + 1, x] = a + b

        A0 = np.zeros(num_trials)  # Probability of Accepting H0 at step n
        A1 = np.zeros(num_trials)  # Probability of Accepting H1 at step n

        for n in range(num_trials):
            cl = lower_limits[n]
            cu = upper_limits[n]

            if np.isfinite(cl):
                cl = int(cl)
                a = P[n, cl]
                b = P[n, cl - 1] if cl - 1 >= 0 else 0

                A0[n] = a + b

            if np.isfinite(cu):
                cu = int(cu)

                a = P[n, cu]
                b = P[n, cu + 1] if cu + 1 <= num_trials else 0

                A1[n] = a + b

        asn = 0

        for n in range(1, cutoff + 1):
            asn += n * (A0[n] + A1[n])

        return P, A0, A1, asn
