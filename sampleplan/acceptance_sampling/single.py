#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass

from sampleplan.acceptance_sampling.util import (
    BinomialCdfGenerator,
    Cdf,
    HypergeometricCdfGenerator,
)


@dataclass(frozen=True)
class SingleSamplingPlan:
    p1: float
    p2: float
    alpha: float
    beta: float
    c: int
    n: int

    def average_sample_size(self):
        return self.n

    @staticmethod
    def binomial(p1: float, p2: float, alpha: float, beta: float, limit: int = 100_000) -> "SingleSamplingPlan":
        assert 0.0 < p1 < p2 < 1.0
        assert 0.0 < beta < 1 - alpha < 1.0

        cdf1 = BinomialCdfGenerator(p1)
        cdf2 = BinomialCdfGenerator(p2)

        c, n = SingleSamplingPlan._compute_plan(cdf1, cdf2, alpha, beta, limit)
        return SingleSamplingPlan(p1=p1, p2=p2, alpha=alpha, beta=beta, c=c, n=n)

    @staticmethod
    def hypergeometric(p1: float, p2: float, alpha: float, beta: float, lot_size: int) -> "SingleSamplingPlan":
        assert 0.0 < p1 < p2 < 1.0
        assert 0.0 < beta < 1 - alpha < 1.0

        cdf1 = HypergeometricCdfGenerator(p1, lot_size)
        cdf2 = HypergeometricCdfGenerator(p2, lot_size)

        c, n = SingleSamplingPlan._compute_plan(cdf1, cdf2, alpha, beta, lot_size)

        return SingleSamplingPlan(p1=p1, p2=p2, alpha=alpha, beta=beta, c=c, n=n)

    @staticmethod
    def _compute_plan(cdf1: Cdf, cdf2: Cdf, alpha: float, beta: float, limit: int):
        # https://personal.utdallas.edu/~metin/Ba3352/QualityAS.pdf
        # https://support.minitab.com/en-us/minitab/18/help-and-how-to/quality-and-process-improvement/acceptance-sampling/how-to/attributes-acceptance-sampling/methods-and-formulas/methods-and-formulas/
        # https://blog.minitab.com/en/understanding-statistics/how-to-perform-acceptance-sampling-by-attributes
        # https://github.com/cran/AcceptanceSampling

        c = 0
        n = 1

        actual_confidence = 0.0
        actual_beta = 1.0

        confidence = 1 - alpha

        while actual_beta > beta:
            while actual_confidence > confidence:
                n += 1
                actual_confidence = cdf1(c, n)
                actual_beta = cdf2(c, n)

                if actual_beta <= beta:
                    break

                if n > limit:
                    raise ValueError(f"Did not find sample plan for n < {limit}, check your inputs or increase 'limit'")

            if actual_beta <= beta and actual_confidence >= confidence:
                break

            c += 1
            actual_confidence = cdf1(c, n)
            actual_beta = cdf2(c, n)

        assert confidence >= 1 - alpha, (confidence, 1 - alpha)
        assert actual_beta <= beta, (actual_beta, beta)

        return c, n
