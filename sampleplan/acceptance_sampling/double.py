#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sampleplan.acceptance_sampling.single import SingleSamplingPlan
from sampleplan.acceptance_sampling.util import (
    BinomialCdfGenerator,
    BinomialPmfGenerator,
    Cdf,
    HypergeometricCdfGenerator,
    HypergeometricPmfGenerator,
    Pmf,
)


@dataclass(frozen=True)
class DoubleSamplingPlan:
    p1: float
    p2: float
    alpha: float
    beta: float
    c1: int
    c2: int
    n: int
    r: int
    _cdf_generator: Callable[[float], Cdf]  # Returns the CDF for a given p
    _pmf_generator: Callable[[float], Pmf]  # Returns the PMF for a given p

    def average_sample_size(self, p: Optional[float] = None):
        if p is None:
            p = self.p1

        cdf = self._cdf_generator(p)

        return DoubleSamplingPlan._average_sample_size(cdf, self.c1, self.c2, self.n, self.n * self.r)

    def average_sample_size_curtailed(self, p: Optional[float] = None):
        if p is None:
            p = self.p1

        cdf = self._cdf_generator(p)
        pmf = self._pmf_generator(p)

        return DoubleSamplingPlan._average_sample_size_curtailed(p, cdf, pmf, self.c1, self.c2, self.n, self.n * self.r)

    @staticmethod
    def binomial(
        p1: float, p2: float, alpha: float, beta: float, r: int = 1, limit: int = 100_000
    ) -> "DoubleSamplingPlan":
        assert 0.0 < p1 < p2 < 1.0
        assert 0.0 < beta < 1 - alpha < 1.0

        cdf1 = BinomialCdfGenerator(p1)
        cdf2 = BinomialCdfGenerator(p2)

        pmf1 = BinomialPmfGenerator(p1)
        pmf2 = BinomialPmfGenerator(p2)

        single_plan = SingleSamplingPlan.binomial(p1, p2, alpha, beta, limit)

        c1, c2, n = DoubleSamplingPlan._compute_plan(single_plan, cdf1, cdf2, pmf1, pmf2, alpha, beta, r)

        return DoubleSamplingPlan(
            p1=p1,
            p2=p2,
            alpha=alpha,
            beta=beta,
            c1=c1,
            c2=c2,
            n=n,
            r=r,
            _cdf_generator=BinomialCdfGenerator,
            _pmf_generator=BinomialPmfGenerator,
        )

    @staticmethod
    def hypergeometric(
        p1: float, p2: float, alpha: float, beta: float, lot_size: int, r: int = 1
    ) -> "DoubleSamplingPlan":
        assert 0.0 < p1 < p2 < 1.0
        assert 0.0 < beta < 1 - alpha < 1.0

        cdf1 = HypergeometricCdfGenerator(p1, lot_size)
        cdf2 = HypergeometricCdfGenerator(p2, lot_size)

        pmf1 = HypergeometricPmfGenerator(p1, lot_size)
        pmf2 = HypergeometricPmfGenerator(p2, lot_size)

        single_plan = SingleSamplingPlan.hypergeometric(p1, p2, alpha, beta, lot_size)

        c1, c2, n = DoubleSamplingPlan._compute_plan(single_plan, cdf1, cdf2, pmf1, pmf2, alpha, beta, r)

        return DoubleSamplingPlan(
            p1=p1,
            p2=p2,
            alpha=alpha,
            beta=beta,
            c1=c1,
            c2=c2,
            n=n,
            r=r,
            _cdf_generator=lambda p: HypergeometricCdfGenerator(p, lot_size),
            _pmf_generator=lambda p: HypergeometricPmfGenerator(p, lot_size),
        )

    @staticmethod
    def _compute_plan(
        single_plan: SingleSamplingPlan,
        cdf1: Cdf,
        cdf2: Cdf,
        pmf1: Pmf,
        pmf2: Pmf,
        alpha: float,
        beta: float,
        r: int,
    ):
        # Implementation of Algorithm 1 of
        # Stijn Luca, Johan Vandercappellen & Johan Claes (2020): A web-based tool
        # to design and analyze single- and double-stage acceptance_sampling sampling plans, Quality Engineering,
        # 32:1, 58-74, DOI: 10.1080/08982112.2019.1641207
        # http://ndl.ethernet.edu.et/bitstream/123456789/30101/1/24..pdf
        # https://brb.nci.nih.gov/techreport/Optimal2-StageDesigns.pdf

        def _phi2(cdf: Cdf, pmf: Pmf, n1: int, n2: int, c1: int, c2: int) -> float:
            # Implementation of Equation 5
            pa_1 = cdf(c1, n1)
            pa_2 = 0

            for j in range(c1 + 1, c2 + 1):
                a = pmf(j, n1)
                b = cdf(c2 - j, n2)
                pa_2 += a * b

            result = pa_1 + pa_2
            return result

        c_single = single_plan.c
        n_single = single_plan.n

        c1 = np.zeros(c_single, dtype=int)
        c2 = np.zeros(c_single, dtype=int)
        n = np.zeros(c_single, dtype=int)

        for idx in range(c_single):
            c1[idx] = idx
            c2[idx] = max(c_single, c1[idx] + 1)
            n0 = max(c2[idx] + 1, n_single // (r + 1))

            alpha_hat = 1 - _phi2(cdf1, pmf1, n0, r * n0, c1[idx], c2[idx])
            beta_hat = _phi2(cdf2, pmf2, n0, r * n0, c1[idx], c2[idx])

            while beta_hat <= beta and alpha_hat > alpha:
                c2[idx] += 1
                alpha_hat = 1 - _phi2(cdf1, pmf1, n0, r * n0, c1[idx], c2[idx])
                beta_hat = _phi2(cdf2, pmf2, n0, r * n0, c1[idx], c2[idx])

            while beta_hat > beta:
                while alpha_hat <= alpha and beta_hat > beta:
                    alpha_hat = 1 - _phi2(cdf1, pmf1, n0, r * n0, c1[idx], c2[idx])
                    beta_hat = _phi2(cdf2, pmf2, n0, r * n0, c1[idx], c2[idx])
                    n0 += 1

                if alpha_hat > alpha:
                    c2[idx] += 1

                alpha_hat = 1 - _phi2(cdf1, pmf1, n0, r * n0, c1[idx], c2[idx])
                beta_hat = _phi2(cdf2, pmf2, n0, r * n0, c1[idx], c2[idx])

            n[idx] = n0

        # Equation 6
        average_sample_numbers = np.zeros(c_single)
        for i in range(c_single):
            average_sample_numbers[i] = DoubleSamplingPlan._average_sample_size(cdf1, c1[i], c2[i], n[i], n[i] * r)

        idx = np.argmin(average_sample_numbers)

        return c1[idx], c2[idx], n[idx] - 1

    @staticmethod
    def _average_sample_size(cdf: Cdf, c1: int, c2: int, n1: int, n2: int) -> float:
        # 15.7 from Introduction to Statistical Quality Control 6th Edition by Douglas C. Montgomery
        p1 = cdf(c1, n1) + (1 - cdf(c2, n1))
        return n1 + n2 * (1 - p1)

    @staticmethod
    def _average_sample_size_curtailed(p: float, cdf: Cdf, pmf: Pmf, c1: int, c2: int, n1: int, n2: int) -> float:
        # 15.8 from Introduction to Statistical Quality Control 6th Edition by Douglas C. Montgomery
        # The best explanation of this formula I could find was in
        # Quality Control And Industrial Statistics
        # by Duncan, J. Acheson
        # https://archive.org/details/in.ernet.dli.2015.214236/page/n9/mode/2up
        # has a PDF
        # The formula itself can be found in (25) on page 149/ page 183 in the PDF
        if np.isclose(p, 0.0):
            raise ValueError(f"p is too close to 0, was [{p}]")

        asn = n1
        for k in range(c1 + 1, c2 + 1):
            term3 = pmf(k, n1)
            term4 = cdf(c2 - k, n2)
            # Equal or more than c2 - k + 2 defects in n2 + 1 samples
            term5 = 1 - cdf(c2 - k + 2, n2 + 1) + pmf(c2 - k + 2, n2 + 1)
            term6 = (c2 - k + 1) / p
            term7 = n2 * term4 + term5 * term6
            term8 = term3 * term7

            asn += term8

        return asn
