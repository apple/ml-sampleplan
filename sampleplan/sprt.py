#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from enum import Enum, auto
from typing import Callable, Optional, Sequence, Union

import numpy as np
import scipy.stats
from tqdm import trange


class Result(Enum):
    ACCEPT_H0 = auto()
    ACCEPT_H1 = auto()
    INDIFFERENT = auto()


class SPQRT:
    def __init__(
        self, alpha: float, beta: float, p0: float, p1: float, distribution: Union[str, Callable[[float, float], float]]
    ):
        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1
        assert p0 < p1
        assert 0 <= p0 <= 1
        assert 0 <= p1 <= 1

        self._alpha = alpha
        self._beta = beta
        self._p0 = p0
        self._p1 = p1

        if distribution == "bernoulli":
            distribution = scipy.stats.bernoulli.pmf
        elif isinstance(distribution, str):
            raise ValueError(f"Unknown distribution name: [{distribution}]")

        self._distribution = distribution

        self._lower_bound = self._beta / (1 - self._alpha)
        self._upper_bound = (1 - self._beta) / self._alpha
        self._log_lower_bound = math.log(self._lower_bound)
        self._log_upper_bound = math.log(self._upper_bound)

        self._data = []
        self._ratios = []
        self._logratios = []
        self._decisions = []

        self._decision: Optional[Result] = None
        self._decision_idx: Optional[int] = None

    def add(self, x: float) -> Result:
        self._data.append(x)

        likelihood_p0 = self._distribution(x, self._p0)
        likelihood_p1 = self._distribution(x, self._p1)
        last_term = self._ratios[-1] if len(self._ratios) else 1.0
        likelihood_ratio = last_term * likelihood_p1 / likelihood_p0
        self._ratios.append(likelihood_ratio)

        log_likelihood_p0 = math.log(likelihood_p0)
        log_likelihood_p1 = math.log(likelihood_p1)
        last_term = self._logratios[-1] if len(self._logratios) else 0.0
        log_likelihood_ratio = last_term + log_likelihood_p1 - log_likelihood_p0

        self._logratios.append(log_likelihood_ratio)

        if log_likelihood_ratio <= self._log_lower_bound:
            assert likelihood_ratio <= self._lower_bound
            decision = Result.ACCEPT_H0
        elif log_likelihood_ratio >= self._log_upper_bound:
            assert likelihood_ratio >= self._upper_bound
            decision = Result.ACCEPT_H1
        else:
            assert self._lower_bound <= likelihood_ratio <= self._upper_bound, (
                self._lower_bound,
                likelihood_ratio,
                self._upper_bound,
            )
            decision = Result.INDIFFERENT

        if decision != Result.INDIFFERENT and self._decision is None:
            self._decision = decision
            self._decision_idx = len(self._data) - 1

        self._decisions.append(decision)

        return decision

    def add_all(self, arr: Sequence[float]) -> Result:
        for x in arr:
            self.add(x)
        return self._decision

    def plot(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

        n = len(self._data)

        ax[0].plot(range(n), self._ratios, color="b")
        ax[0].axhline(y=self._lower_bound, xmin=0, xmax=n, color="g", label=r"$(1-\beta)/\alpha$")
        ax[0].axhline(y=self._upper_bound, xmin=0, xmax=n, color="r", label=r"$\beta/(1-\alpha)$")
        ax[0].set_title("Simulation of Wald's SPRT")
        ax[0].set_ylabel("LR")
        ax[0].legend(loc="best")

        ax[1].plot(range(n), self._logratios, color="b", linestyle="--")
        ax[1].axhline(y=self._log_lower_bound, xmin=0, xmax=n, color="g", label=r"$log((1-\beta)/\alpha)$")
        ax[1].axhline(y=self._log_upper_bound, xmin=0, xmax=n, color="r", label=r"$log(\beta/(1-\alpha))$")
        ax[1].set_ylabel("log(LR)")
        ax[1].set_xlabel("trials")
        ax[1].legend(loc="best")

        denom = np.log(self._p1 / (1 - self._p1)) - np.log(self._p0 / (1 - self._p0))
        slope = (np.log(1 - self._p0) - np.log(1 - self._p1)) / denom
        lower_intercept = self._log_lower_bound / denom
        upper_intercept = self._log_upper_bound / denom

        x = np.cumsum(self._data).tolist()
        ax[2].plot(x, color="b", linestyle="--")
        ax[2].axline(xy1=(1, lower_intercept), slope=slope, color="g")
        ax[2].axline(xy1=(1, upper_intercept), slope=slope, color="r")
        ax[2].set_ylabel("successes")
        ax[2].set_xlabel("trials")
        ax[2].legend(loc="best")

        if self._decision:
            for axis in ax:
                axis.axvline(x=self._decision_idx, linewidth=1, color="k", linestyle="dotted", alpha=0.8)


def _main_simple():
    from matplotlib import pyplot as plt

    n = 100
    alpha = 0.05
    beta = 0.2
    p0 = 0.01
    p1 = 0.03
    p = 0.1

    sprt = SPQRT(alpha=alpha, beta=beta, p0=p0, p1=p1, distribution="bernoulli")

    data = scipy.stats.bernoulli.rvs(p, size=n, random_state=91215)

    decision_found = False

    for i, x in enumerate(data):
        decision = sprt.add(x)
        if decision != Result.INDIFFERENT and not decision_found:
            print(decision)
            decision_found = True

    # _plotBinomialSPRT(1000, 0.06, 0.03, 0.05, .05, .2)

    sprt.plot()
    plt.show()


def _main_repeated(rotated: bool = True):
    from matplotlib import pyplot as plt

    N = 10

    n = 100
    alpha = 0.05
    beta = 0.2
    p0 = 0.0001
    p1 = 0.1
    p = 0.02

    scores = np.empty((N, n))
    datas = np.empty((N, n))
    decisions = np.full(N, n)

    for run in trange(N):
        sprt = SPQRT(alpha=alpha, beta=beta, p0=p0, p1=p1, distribution="bernoulli")
        data = scipy.stats.bernoulli.rvs(p, size=n, random_state=run)
        sprt.add_all(data)

        scores[run, :] = sprt._logratios
        datas[run, :] = sprt._data

        if sprt._decision:
            decisions[run] = sprt._decision_idx

    decision_idx = np.average(decisions)

    t = np.arange(1, n + 1)

    if rotated:
        avgs = np.average(datas, axis=0)
        stds = np.std(scores, axis=0)

        denom = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
        slope = (np.log(1 - p0) - np.log(1 - p1)) / denom
        lower_intercept = sprt._log_lower_bound / denom
        upper_intercept = sprt._log_upper_bound / denom

        x = np.cumsum(avgs).tolist()

        plt.plot(x, color="b", linestyle="--")
        plt.axline(xy1=(1, lower_intercept), slope=slope, color="g")
        plt.axline(xy1=(1, upper_intercept), slope=slope, color="r")
        plt.ylabel("successes")
        plt.xlabel("trials")
        plt.legend(loc="best")
        plt.axhline(y=0, xmin=0, xmax=n, color="k", linewidth=0.5)

        # plt.ylim(0, math.ceil(n * p))
    else:
        avgs = np.average(scores, axis=0)
        stds = np.std(scores, axis=0)
        plt.plot(t, avgs)

        plt.axhline(y=sprt._log_lower_bound, xmin=0, xmax=n, color="g", label="Accept $H_0: p = p_0 $")
        plt.axhline(y=sprt._log_upper_bound, xmin=0, xmax=n, color="r", label="Accept $H_1: p = p_1 $")
        plt.xlabel("trials")
        plt.ylabel("log(LR)")
        plt.legend(loc="best")

        for i in range(1):
            x = i + 1
            y_lower = avgs - x * stds
            y_upper = avgs + x * stds

            plt.fill_between(t, y_lower, y_upper, alpha=0.2 / x, color="tab:orange")

    plt.axvline(x=decision_idx, linewidth=1, color="k", linestyle="dotted", alpha=0.8)
    plt.suptitle("Wald's SPRT")
    plt.title(f"p = {p}, $p_0$ = {p0}, $p_1$ = {p1}, $\\alpha$ = {alpha}, $\\beta = {beta}$")

    plt.show()


if __name__ == "__main__":
    _main_repeated()
