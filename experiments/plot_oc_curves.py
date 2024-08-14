import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from experiments.util import PATH_PLOTS, PATH_RESULTS, QA_CONFIGS, Approach, QaConfig
from sampleplan.acceptance_sampling import (
    DoubleSamplingPlan,
    HypergeometricSequentialSamplingPlan,
    SingleSamplingPlan,
)
from sampleplan.acceptance_sampling.ocr import (
    compute_ocr_double_sampling,
    compute_ocr_sequential_sampling_curtailed,
    compute_ocr_sequential_sampling_full,
    compute_ocr_single_sampling,
)
from sampleplan.acceptance_sampling.sequential import BinomialSequentialSamplingPlan

DATASET_SIZES = [
    # 100,
    # 200,
    500,
    1000,
    # 2000,
    5000,
    # 7500,
    10000,
    # 15000,
    # 20000
]

ERROR_STEPSIZE = 0.25
SEED = 1000
REPETITIONS = 1000

PATH_OUT_STRICT = PATH_RESULTS / "ocr_strict.arrow"
PATH_OUT_RELAXED = PATH_RESULTS / "ocr_strict.arrow"


def simulate_ocr(approaches: list[Approach], dataset_sizes: list[int], qa_config: QaConfig):
    result = []

    for dataset_size in tqdm(dataset_sizes):
        plan_ssp_binomial = SingleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
        plan_ssp_hypergeometric = SingleSamplingPlan.hypergeometric(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
        )
        plan_dsp_binomial = DoubleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
        plan_dsp_hypergeometric = DoubleSamplingPlan.hypergeometric(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
        )

        plan_sprt_binomial = BinomialSequentialSamplingPlan(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
        borders_sprt_binomial_full = plan_sprt_binomial.compute_region(dataset_size)
        borders_sprt_binomial_curtailed = plan_sprt_binomial.compute_region(num_trials=plan_ssp_binomial.n)

        plan_sprt_hypergeometric = HypergeometricSequentialSamplingPlan(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, dataset_size
        )
        borders_sprt_hypergeometric_full = plan_sprt_hypergeometric.compute_truncated_wald_region()
        borders_sprt_hypergeometric_curtailed = plan_sprt_hypergeometric.compute_truncated_wald_region(
            cutoff=plan_ssp_hypergeometric.n
        )

        for d in np.arange(0, int(0.08 * dataset_size)):
            rng = np.random.default_rng(SEED + d)

            error_pct = d / dataset_size * 100
            data = np.zeros(dataset_size, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            rng.shuffle(data)

            for approach in approaches:
                if approach == Approach.SINGLE_SAMPLING_BINOMIAL:
                    p_acceptance = compute_ocr_single_sampling(plan_ssp_binomial, data, REPETITIONS, replace=True)
                elif approach == Approach.SINGLE_SAMPLING_HYPERGEOMETRIC:
                    p_acceptance = compute_ocr_single_sampling(
                        plan_ssp_hypergeometric, data, REPETITIONS, replace=False
                    )
                elif approach == Approach.DOUBLE_SAMPLING_BINOMIAL_FULL:
                    p_acceptance = compute_ocr_double_sampling(plan_dsp_binomial, data, REPETITIONS, replace=True)
                elif approach == Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL:
                    p_acceptance = compute_ocr_double_sampling(
                        plan_dsp_hypergeometric, data, REPETITIONS, replace=False
                    )

                elif approach == Approach.SPRT_BINOMIAL_FULL:
                    p_acceptance = compute_ocr_sequential_sampling_full(
                        borders_sprt_binomial_full.lower_limits_array,
                        borders_sprt_binomial_full.upper_limits_array,
                        data,
                        REPETITIONS,
                        replace=True,
                    )
                elif approach == Approach.SPRT_BINOMIAL_CURTAILED:
                    p_acceptance = compute_ocr_sequential_sampling_curtailed(
                        borders_sprt_binomial_curtailed.lower_limits_array,
                        borders_sprt_binomial_curtailed.upper_limits_array,
                        plan_ssp_binomial.c,
                        data,
                        REPETITIONS,
                        replace=True,
                    )
                elif approach == Approach.SPRT_HYPERGEOMETRIC_FULL:
                    p_acceptance = compute_ocr_sequential_sampling_full(
                        borders_sprt_hypergeometric_full.lower_limits_array,
                        borders_sprt_hypergeometric_full.upper_limits_array,
                        data,
                        REPETITIONS,
                        replace=False,
                    )
                elif approach == Approach.SPRT_HYPERGEOMETRIC_CURTAILED:
                    p_acceptance = compute_ocr_sequential_sampling_curtailed(
                        borders_sprt_hypergeometric_curtailed.lower_limits_array,
                        borders_sprt_hypergeometric_curtailed.upper_limits_array,
                        plan_ssp_hypergeometric.c,
                        data,
                        REPETITIONS,
                        replace=False,
                    )

                else:
                    raise RuntimeError(f"Unsupported approach: [{approach}]")

                entry = {
                    "approach": approach.name,
                    "p1": qa_config.p1,
                    "p2": qa_config.p2,
                    "alpha": qa_config.alpha,
                    "beta": qa_config.beta,
                    "qa_config_name": qa_config.name,
                    "dataset_size": dataset_size,
                    "error_pct": error_pct,
                    "p_acceptance": p_acceptance,
                    "distribution": approach.distribution,
                    "kind": approach.kind,
                }

                result.append(entry)

    return result


def plot_things(df: pd.DataFrame, qa_config: QaConfig):
    sns.color_palette("deep")

    g = sns.relplot(
        data=df,
        x="error_pct",
        y="p_acceptance",
        hue="distribution",
        col="dataset_size",
        row="kind",
        kind="line",
        height=3,
        aspect=1,
    )

    g.map(plt.axvline, x=qa_config.p1 * 100, ls="--", c="blue", linewidth=0.5)
    g.map(plt.axvline, x=qa_config.p2 * 100, ls="--", c="red", linewidth=0.5)
    plt.legend([], [], frameon=False)
    g._legend.remove()

    g.set_axis_labels("Error Rate %", "Acceptance Probability")
    g.set_titles(col_template="N = {col_name}", row_template="{row_name}")

    plt.tight_layout()

    plt.savefig(PATH_PLOTS / f"ocr_{qa_config.name}.pdf")


def main():
    approaches = [
        Approach.SINGLE_SAMPLING_BINOMIAL,
        Approach.SINGLE_SAMPLING_HYPERGEOMETRIC,
        Approach.DOUBLE_SAMPLING_BINOMIAL_FULL,
        Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL,
        Approach.SPRT_BINOMIAL_FULL,
        Approach.SPRT_HYPERGEOMETRIC_FULL,
        Approach.SPRT_BINOMIAL_CURTAILED,
        Approach.SPRT_HYPERGEOMETRIC_CURTAILED,
    ]

    for qa_config in QA_CONFIGS:
        data = simulate_ocr(approaches, DATASET_SIZES, qa_config)
        df = pd.DataFrame(data)
        plot_things(df, qa_config)


if __name__ == "__main__":
    main()
