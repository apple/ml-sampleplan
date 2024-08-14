import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import diskcache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
from seaborn import axes_style, plotting_context
from tabulate import tabulate
from tqdm import trange

from experiments.util import QA_CONFIGS, QaConfig
from sampleplan.acceptance_sampling import (
    DoubleSamplingPlan,
    HypergeometricSequentialSamplingPlan,
    SingleSamplingPlan,
)
from sampleplan.confidence_interval import (
    hypergeometric_proportion_interval_exact,
    sample_size_exact_hypergeometric,
)

P_REJECTIONS_CACHED = Path(__file__).parent.parent / "cache" / "rejections.json"
P_SAMPLE_SIZES_CACHED = Path(__file__).parent.parent / "cache" / "sample_sizes.json"
PATH_PLOTS = Path(__file__).parent.parent / "plots"

P_PLAN_CACHE = Path(__file__).parent.parent / "cache" / "plancache"


@dataclass(frozen=True)
class DatasetWithErrors:
    name: str
    num_errors: int
    num_instances: int

    @property
    def error_rate(self) -> float:
        return self.num_errors / self.num_instances * 100


@dataclass(frozen=True)
class Key:
    dataset: DatasetWithErrors
    qa_config: QaConfig
    plan: str


IMDB = DatasetWithErrors(name="IMDB", num_errors=499, num_instances=24799)
CONLL_2003 = DatasetWithErrors(name="CoNLL 2003", num_errors=217, num_instances=3380)
NUM_SIMULATIONS = 1000


def do_stuff():
    plancache = diskcache.Cache(str(P_PLAN_CACHE))

    rejections = {}
    sample_sizes = {}

    for dataset in [CONLL_2003, IMDB]:
        dataset_size = dataset.num_instances
        arr = np.zeros(dataset_size, dtype=bool)
        arr[: dataset.num_errors] = True

        rng = np.random.default_rng(seed=42)
        rng.shuffle(arr)

        assert np.sum(arr) == dataset.num_errors

        for qa_config in QA_CONFIGS:
            key_single = Key(dataset=dataset, qa_config=qa_config, plan="single")
            key_double_full = Key(dataset=dataset, qa_config=qa_config, plan="double full")
            key_double_curtailed = Key(dataset=dataset, qa_config=qa_config, plan="double curtailed")
            key_ss_full = Key(dataset=dataset, qa_config=qa_config, plan="SS Full")
            key_ss_curtailed = Key(dataset=dataset, qa_config=qa_config, plan="SS Curtailed")
            key_ci_low = Key(dataset=dataset, qa_config=qa_config, plan="ci_low")
            key_ci_high = Key(dataset=dataset, qa_config=qa_config, plan="ci_high")

            keys = [
                key_single,
                key_double_full,
                key_double_curtailed,
                key_ss_full,
                key_ss_curtailed,
                key_ci_low,
                key_ci_high,
            ]

            for key in keys:
                assert key not in rejections
                assert key not in sample_sizes
                rejections[key] = 0
                sample_sizes[key] = 0

            print(dataset.name, qa_config.name)

            single_sampling_plan = SingleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )
            double_sampling_plan = DoubleSamplingPlan.hypergeometric(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=dataset_size
            )
            sequential_sampling_plan = HypergeometricSequentialSamplingPlan(
                qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, dataset_size
            )

            if key_ss_full in plancache:
                sequential_sampling_region_full = plancache[key_ss_full]
            else:
                sequential_sampling_region_full = sequential_sampling_plan.compute_truncated_wald_region(
                    cutoff=dataset_size
                )
                plancache[key_ss_full] = sequential_sampling_region_full

            if key_ss_curtailed in plancache:
                sequential_sampling_region_curtailed = plancache[key_ss_curtailed]
            else:
                sequential_sampling_region_curtailed = sequential_sampling_plan.compute_truncated_wald_region(
                    cutoff=single_sampling_plan.n
                )
                plancache[key_ss_curtailed] = sequential_sampling_region_curtailed

            ci_sample_size_a = sample_size_exact_hypergeometric(
                dataset_size, qa_config.p1, qa_config.alpha, qa_config.confidence_interval_half_width
            )
            ci_sample_size_b = sample_size_exact_hypergeometric(
                dataset_size, qa_config.p2, qa_config.alpha, qa_config.confidence_interval_half_width
            )

            # key_single = Key(dataset=dataset.name, qa_config=qa_config.name, plan="single")

            sample_sizes[key_single] = single_sampling_plan.n
            sample_sizes[key_double_full] = double_sampling_plan.n * 2
            sample_sizes[key_double_curtailed] = []
            sample_sizes[key_ss_full] = []
            sample_sizes[key_ss_curtailed] = []
            sample_sizes[key_ci_low] = ci_sample_size_a
            sample_sizes[key_ci_high] = ci_sample_size_b

            for i in trange(NUM_SIMULATIONS):
                rng = np.random.default_rng(seed=i)

                # Single Sampling
                d_single_sampling = np.sum(rng.choice(arr, single_sampling_plan.n, replace=False))

                if d_single_sampling > single_sampling_plan.c:
                    rejections[key_single] += 1

                # Double Sampling
                sample_double_sampling1, sample_double_sampling2 = np.split(
                    rng.choice(arr, double_sampling_plan.n * 2, replace=False), 2
                )
                d1_double_sampling = np.sum(sample_double_sampling1)
                d2_double_sampling = np.sum(sample_double_sampling2)

                if d1_double_sampling > double_sampling_plan.c1:
                    d_double_sampling = d1_double_sampling + d2_double_sampling
                    if d_double_sampling > double_sampling_plan.c2:
                        rejections[key_double_full] += 1

                if d1_double_sampling > double_sampling_plan.c1:
                    d_double_sampling = d1_double_sampling
                    for i in range(double_sampling_plan.n):
                        if d_double_sampling > double_sampling_plan.c2:
                            rejections[key_double_curtailed] += 1
                            sample_sizes[key_double_curtailed].append(double_sampling_plan.n + i + 1)
                            break
                        else:
                            d_double_sampling += int(sample_double_sampling2[i])
                    else:
                        sample_sizes[key_double_curtailed].append(double_sampling_plan.n * 2)
                else:
                    sample_sizes[key_double_curtailed].append(double_sampling_plan.n * 2)

                # Sequential Sampling
                data_sequential_sampling = arr.copy()
                rng.shuffle(data_sequential_sampling)
                ss_reject, x = _simulate_sequential_sampling_plan_rejects(
                    data_sequential_sampling,
                    dataset_size,
                    np.array(sequential_sampling_region_full.lower_limits),
                    np.array(sequential_sampling_region_full.upper_limits),
                )
                if ss_reject:
                    rejections[key_ss_full] += 1
                sample_sizes[key_ss_full].append(x)

                ss_reject, x = _simulate_sequential_sampling_plan_rejects(
                    data_sequential_sampling,
                    dataset_size,
                    np.array(sequential_sampling_region_curtailed.lower_limits),
                    np.array(sequential_sampling_region_curtailed.upper_limits),
                )

                if ss_reject:
                    rejections[key_ss_curtailed] += 1
                sample_sizes[key_ss_curtailed].append(x)

                # Confidence Intervals
                d_ci_a = np.sum(rng.choice(arr, ci_sample_size_a, replace=False))
                d_ci_b = np.sum(rng.choice(arr, ci_sample_size_b, replace=False))

                ci_a_low, ci_a_high = hypergeometric_proportion_interval_exact(
                    1 - qa_config.alpha, dataset_size, ci_sample_size_a, d_ci_a
                )
                ci_b_low, ci_b_high = hypergeometric_proportion_interval_exact(
                    1 - qa_config.alpha, dataset_size, ci_sample_size_b, d_ci_b
                )

                e_a = d_ci_a / ci_sample_size_a
                e_b = d_ci_b / ci_sample_size_b

                if not (ci_a_low <= e_a <= ci_a_high):
                    rejections[key_ci_low] += 1

                if not (ci_b_low <= e_b <= ci_b_high):
                    rejections[key_ci_high] += 1

            for key in [key_double_curtailed, key_ss_full, key_ss_curtailed]:
                assert len(sample_sizes[key]) == NUM_SIMULATIONS, (key, len(sample_sizes[key]))
                sample_sizes[key] = np.mean(sample_sizes[key])

    r = []
    for k, v in rejections.items():
        e = asdict(k)
        e["n"] = v
        r.append(e)

    ss = []
    for k, v in sample_sizes.items():
        e = asdict(k)
        e["n"] = v
        ss.append(e)

    return r, ss


def _simulate_sequential_sampling_plan_rejects(
    data: np.ndarray, dataset_size: int, lower_limits: np.ndarray, upper_limits: np.ndarray
) -> Tuple[bool, int]:
    count = 0
    for i in range(dataset_size):
        x = data[i]
        count += x

        cl = lower_limits[i + 1]
        cu = upper_limits[i + 1]

        if cl is None:
            cl = -np.inf
        if cu is None:
            cu = np.inf

        if count <= cl:
            return False, i + 1

        if count >= cu:
            return True, i + 1

    return True, dataset_size


def plot_rejections(rejections: List[Dict[str, Any]]):
    sns.set_style("white")
    sns.set_context("paper")

    data = []

    for e in rejections:
        plan = e["plan"]
        plan_mapping = {
            "single": "SSP",
            "double full": "$DSP_F$",
            "double curtailed": "$DSP_C$",
            "SS Full": "$SPRT_F$",
            "SS Curtailed": "$SPRT_C$",
        }

        if plan not in plan_mapping:
            continue

        plan = plan_mapping[plan]

        e1 = {
            "dataset": e["dataset"]["name"],
            "plan": plan,
            "config": e["qa_config"]["name"],
            "n": (NUM_SIMULATIONS - e["n"]) / NUM_SIMULATIONS * 100,
            "decision": "accepted",
        }
        e2 = {
            "dataset": e["dataset"]["name"],
            "plan": plan,
            "config": e["qa_config"]["name"],
            "n": e["n"] / NUM_SIMULATIONS * 100,
            "decision": "rejected",
        }

        data.append(e1)
        data.append(e2)

    df = pd.DataFrame(data)

    p = (
        so.Plot(df, x="plan", y="n", color="decision")
        .label(x="Plan", y="%", color="")
        .theme(axes_style("white") | plotting_context("paper"))
        .facet(row="config", col="dataset")
        .add(so.Bars(), so.Stack(), legend=True)
        .layout(engine="tight")
    )

    p.save(PATH_PLOTS / "real_world.pdf")
    p.show()

    plt.show()


def tabulate_sample_sizes(sample_sizes: List[Dict[str, Any]]):
    df = pd.DataFrame(
        [
            {
                "dataset": x["dataset"]["name"],
                "config": x["qa_config"]["name"],
                "plan": x["plan"],
                "sample_size": x["n"],
            }
            for x in sample_sizes
        ]
    )
    print(tabulate(df, tablefmt="latex_booktabs"))


if __name__ == "__main__":
    rejections, sample_sizes = do_stuff()

    tabulate_sample_sizes(sample_sizes)

    plot_rejections(rejections)
