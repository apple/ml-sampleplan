import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.util import PATH_RESULTS_ASNS, QA_CONFIGS, Approach
from sampleplan.acceptance_sampling import DoubleSamplingPlan, SingleSamplingPlan
from sampleplan.acceptance_sampling.sequential import (
    BinomialSequentialSamplingPlan,
    HypergeometricSequentialSamplingPlan,
)
from sampleplan.confidence_interval import (
    sample_size_exact_hypergeometric,
    sample_size_exact_mid_point_binomial,
)


def compute_asns(approaches: list[Approach], dataset_size: int, qa_config):
    results_x = []
    results_y = []
    results_labels = []
    results_config_names = []

    linewidth = 3

    print(f"QA config: {qa_config.name}")

    # We have a discrete distribution here, hence we sample for each defect and derive p from that
    error_numbers = np.arange(1, int(qa_config.asn_xlim * dataset_size + 1) + 1)
    error_percentages = error_numbers / dataset_size

    num_points = len(error_percentages)

    assert len(results_x) == len(results_y) == len(results_labels) == len(results_config_names), (
        len(results_x),
        len(results_y),
        len(results_labels),
        len(results_config_names),
    )

    asns = {}

    for approach in tqdm(approaches):
        assert len(results_x) == len(results_y) == len(results_labels), (
            len(results_x),
            len(results_y),
            len(results_labels),
        )

        if approach == Approach.CONFIDENCE_INTERVAL_BINOMIAL:
            nl = sample_size_exact_mid_point_binomial(
                qa_config.p1, qa_config.alpha, qa_config.confidence_interval_half_width
            )

            nu = sample_size_exact_mid_point_binomial(
                qa_config.p2, qa_config.alpha, qa_config.confidence_interval_half_width
            )

            cur_asns = np.array([nl for _ in error_percentages])
            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name + " p1"] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["CI binomial p1"] = cur_asns

            cur_asns = np.array([nu for _ in error_percentages])
            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name + " p2"] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["CI binomial p2"] = cur_asns
        elif approach == Approach.CONFIDENCE_INTERVAL_HYPERGEOMETRIC:
            # n = binomial_confidence_interval_sample_size(
            #     qa_config.p1, qa_config.alpha, qa_config.confidence_interval_half_width
            # )

            nl = sample_size_exact_hypergeometric(
                dataset_size, qa_config.p1, qa_config.alpha, qa_config.confidence_interval_half_width
            )
            nu = sample_size_exact_hypergeometric(
                dataset_size, qa_config.p2, qa_config.alpha, qa_config.confidence_interval_half_width
            )

            cur_asns = np.array([nl for _ in error_percentages])
            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name + " p1"] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["CI hypergeometric p1"] = cur_asns

            cur_asns = np.array([nu for _ in error_percentages])
            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name + " p2"] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["CI hypergeometric p2"] = cur_asns
        elif approach == Approach.SINGLE_SAMPLING_BINOMIAL:
            plan = SingleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
            cur_asns = np.array([plan.n for _ in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SSP binomial"] = cur_asns

            print(f"SSP_B: n={plan.n}, c={plan.c}")
        elif approach == Approach.SINGLE_SAMPLING_HYPERGEOMETRIC:
            plan = SingleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )
            cur_asns = np.array([plan.n for _ in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(cur_asns)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SSP hypergeometric"] = cur_asns

            print(f"SSP_H: n={plan.n}, c={plan.c}")
        elif approach == Approach.DOUBLE_SAMPLING_BINOMIAL_FULL:
            plan = DoubleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
            asns_f = np.array([plan.average_sample_size(p) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_f)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["DSP binomial full"] = asns_f

            print(f"DSP_B f: n={plan.n}, c1={plan.c1}, c2={plan.c2}")
        elif approach == Approach.DOUBLE_SAMPLING_BINOMIAL_CURTAILED:
            plan = DoubleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)
            asns_c = np.array([plan.average_sample_size_curtailed(p) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_c)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["DSP binomial curtailed"] = asns_c

            print(f"DSP_B c: n={plan.n}, c1={plan.c1}, c2={plan.c2}")
        elif approach == Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL:
            plan = DoubleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )
            asns_f = np.array([plan.average_sample_size(p) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_f)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["DSP hypergeometric full"] = asns_f

            print(f"DSP_H f: n={plan.n}, c1={plan.c1}, c2={plan.c2}")
        elif approach == Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_CURTAILED:
            plan = DoubleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )

            asns_c = np.array([plan.average_sample_size_curtailed(p) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_c)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["DSP hypergeometric curtailed"] = asns_c

            print(f"DSP_H c: n={plan.n}, c1={plan.c1}, c2={plan.c2}")
        elif approach == Approach.SPRT_BINOMIAL_FULL:
            plan = BinomialSequentialSamplingPlan(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

            asns_f = np.array([plan.average_sample_number(p) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_f)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SPRT binomial full"] = asns_f
        elif approach == Approach.SPRT_BINOMIAL_CURTAILED:
            ssp = SingleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

            plan = BinomialSequentialSamplingPlan(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

            cutoff = ssp.n * 3

            asns_c = np.array([plan.average_sample_with_cutoff(p, cutoff) for p in error_percentages])

            results_x.extend(error_percentages)
            results_y.extend(asns_c)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SPRT binomial curtailed"] = asns_c
        elif approach == Approach.SPRT_HYPERGEOMETRIC_FULL:
            plan = HypergeometricSequentialSamplingPlan(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, dataset_size
            )

            asns_f = np.array(
                [plan.average_sample_number(d, cutoff=dataset_size).average_sample_number for d in error_numbers]
            )

            results_x.extend(error_percentages)
            results_y.extend(asns_f)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SPRT hypergeometric full"] = asns_f
        elif approach == Approach.SPRT_HYPERGEOMETRIC_CURTAILED:
            plan = HypergeometricSequentialSamplingPlan(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, dataset_size
            )
            single_plan = SingleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )
            asns_c = np.array(
                [plan.average_sample_number(d, cutoff=single_plan.n).average_sample_number for d in error_numbers]
            )
            results_x.extend(error_percentages)
            results_y.extend(asns_c)
            results_labels.extend([approach.name] * num_points)
            results_config_names.extend([qa_config.name] * num_points)
            asns["SPRT hypergeometric curtailed"] = asns_c

        else:
            raise ValueError(f"Unsupported approach: [{approach}]")

        data = {
            "p": error_percentages * 100,
        }

        for name, asn in asns.items():
            assert len(error_percentages) == len(asn)
            data[name] = asn

        df = pd.DataFrame(data).convert_dtypes()
        df.to_csv(PATH_RESULTS_ASNS / f"raw_asns_{qa_config.name}_{dataset_size}.csv", index=False)

    distributions = []
    kinds = []

    for label in results_labels:
        # Remove p1/p2 from CI labels

        if "CONFIDENCE_INTERVAL" in label:
            p = label.split(" ")
            x = Approach[p[0]]
            d = x.distribution
            k = x.kind + " " + p[1]
        else:
            x = Approach[label]
            d = x.distribution
            k = x.kind

        distributions.append(d)
        kinds.append(k)

    data = {
        "qa_config": results_config_names,
        "label": results_labels,
        "p": results_x,
        "asn": results_y,
        "dataset_size": dataset_size,
        "kind": kinds,
        "distribution": distributions,
    }

    return pd.DataFrame(data)


def _main():
    PATH_RESULTS_ASNS.mkdir(exist_ok=True, parents=True)

    approaches = [
        # Approach.CONFIDENCE_INTERVAL_HYPERGEOMETRIC,
        Approach.SINGLE_SAMPLING_HYPERGEOMETRIC,
        Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL,
        Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_CURTAILED,
        Approach.SPRT_HYPERGEOMETRIC_FULL,
        Approach.SPRT_HYPERGEOMETRIC_CURTAILED,
        # Approach.CONFIDENCE_INTERVAL_BINOMIAL,
        Approach.SINGLE_SAMPLING_BINOMIAL,
        Approach.DOUBLE_SAMPLING_BINOMIAL_FULL,
        Approach.DOUBLE_SAMPLING_BINOMIAL_CURTAILED,
        Approach.SPRT_BINOMIAL_FULL,
        Approach.SPRT_BINOMIAL_CURTAILED,
    ]

    dfs = []

    for qa_config in QA_CONFIGS:
        for dataset_size in [500, 1000, 5000, 10_000]:
            df = compute_asns(approaches, dataset_size, qa_config)
            dfs.append(df)

    df = pd.concat(dfs).convert_dtypes()
    df.to_csv(PATH_RESULTS_ASNS / f"raw_asns.tsv", sep="\t", index=False)


if __name__ == "__main__":
    _main()
