from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib import ticker

from experiments.util import PATH_PLOTS, PATH_RESULTS
from sampleplan.acceptance_sampling import (
    HypergeometricSequentialSamplingPlan,
    SingleSamplingPlan,
)
from sampleplan.confidence_interval import hypergeometric_proportion_interval_exact


@dataclass
class DataPoint:
    experiment: str
    dataset_size: int
    sample_size: int
    error_pct: int
    confidence: float
    ci_half_width: float


DATASET_SIZES = [
    100,
    200,
    500,
    1000,
    2000,
    5000,
    7500,
    10000,
    # 15000,
    # 20000
]

ERROR_PERCENTAGES = [1, 2, 5, 8, 10, 15]

NUM_REPEATS = 1
CONFIDENCE_LEVEL = 0.95
DESIRED_HALF_WIDTH = 2
STEPSIZE = 100


def compute_static_baseline(pct_inspected: int, confidence: float) -> list[DataPoint]:
    result = []

    for dataset_size in range(100, 10000 + STEPSIZE, STEPSIZE):
        for error_pct in ERROR_PERCENTAGES:
            sample_size = int(dataset_size * pct_inspected / 100)
            num_errors = int(sample_size * error_pct / 100)

            low, high = hypergeometric_proportion_interval_exact(confidence, dataset_size, sample_size, num_errors)
            half_width = (high - low) / 2 * 100

            point = DataPoint(
                experiment=f"compute_static_baseline_{pct_inspected}",
                dataset_size=dataset_size,
                sample_size=sample_size,
                error_pct=error_pct,
                confidence=confidence,
                ci_half_width=half_width,
            )
            result.append(point)

    return result


def compute_sequential_sampling():
    alpha = 1 - CONFIDENCE_LEVEL
    beta = 0.2
    p_a = 0.03
    p_r = 0.05
    confidence = CONFIDENCE_LEVEL

    result = []

    stepsize = 250

    for dataset_size in tqdm.trange(100, 10000 + stepsize, stepsize):
        # For curtailing, we stop if we did not make a decision when reaching the Single Sampling Plan sample size
        ssp = SingleSamplingPlan.hypergeometric(p_a, p_r, alpha, beta, dataset_size)
        cutoff = ssp.average_sample_size()
        plan = HypergeometricSequentialSamplingPlan(p_a, p_r, alpha, beta, dataset_size)

        for error_pct in ERROR_PERCENTAGES:
            num_errors = int(dataset_size * error_pct / 100)

            props = plan.average_sample_number(num_errors, cutoff)
            sample_size = int(props.average_sample_number)

            try:
                low, high = hypergeometric_proportion_interval_exact(confidence, dataset_size, sample_size, num_errors)
                half_width = (high - low) / 2 * 100

            except Exception as e:
                print(e)
                continue

            point = DataPoint(
                experiment=f"sequential_sampling_alpha={alpha},beta={beta},p_a={p_a},p_r={p_r}",
                dataset_size=dataset_size,
                sample_size=sample_size,
                error_pct=error_pct,
                confidence=confidence,
                ci_half_width=half_width,
            )
            result.append(point)

    return result


def plot_baseline():
    points = compute_static_baseline(pct_inspected=20, confidence=CONFIDENCE_LEVEL)
    df = pd.DataFrame(points)

    # https://stackoverflow.com/a/31291567
    plt.figure()
    ax = sns.lineplot(data=df, x="dataset_size", y="ci_half_width", hue="error_pct")

    crossing_points = []

    for error_pct, group in df.groupby("error_pct"):
        group["next_ci_half_width"] = group["ci_half_width"].shift(-1)
        group["cross"] = (group["ci_half_width"] > DESIRED_HALF_WIDTH) & (
            group["next_ci_half_width"] <= DESIRED_HALF_WIDTH
        )

        crossed = group[group["cross"]]
        if len(crossed) == 0:
            continue

        p = crossed.iloc[0]
        x = p["dataset_size"]
        y = p["ci_half_width"]

        crossing_points.append((x, y))

        if error_pct != max(ERROR_PERCENTAGES):
            continue

        t = ax.text(
            x,
            y + 2,
            "Too Small  | Too Large",
            ha="center",
            va="center",
            size=12,
            bbox=dict(boxstyle="darrow,pad=0.3", fc="white", ec="steelblue", lw=2),
        )

    plt.scatter(x=[p[0] for p in crossing_points], y=[p[1] for p in crossing_points], s=15)
    plt.axhline(y=DESIRED_HALF_WIDTH, linewidth=1, linestyle="dotted")

    plt.title("20% Sampling")
    plt.xlabel("Dataset Size")
    plt.ylabel("Margin of Error in %")

    leg = ax.get_legend()
    leg.set_title("Error Rate %")

    ax.xaxis.set_major_formatter(ticker.EngFormatter())

    plt.savefig(PATH_PLOTS / "productionalize_20pct_baseline.pdf")
    plt.savefig(PATH_PLOTS / "productionalize_20pct_baseline.png")


def plot_sprt():
    plt.figure()
    PATH_RESULTS.mkdir(exist_ok=True, parents=True)
    p = PATH_RESULTS / "sprt.arrow"
    if p.is_file():
        df = pd.read_parquet(p)
    else:
        df = pd.DataFrame(compute_sequential_sampling())
        df.to_parquet(p)

    ax = sns.lineplot(data=df, x="dataset_size", y="ci_half_width", hue="error_pct")
    plt.savefig(PATH_PLOTS / "productionalize_sprt.pdf")


def main():
    plot_baseline()
    plot_sprt()


if __name__ == "__main__":
    main()
