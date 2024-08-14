# \label{fig:sim_vs_analytical}

from pathlib import Path
from typing import Any, Dict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from experiments.util import LOT_SIZE, NUM_REPETITIONS, PATH_PLOTS, QA_CONFIGS, Approach
from sampleplan.acceptance_sampling import DoubleSamplingPlan, SingleSamplingPlan
from sampleplan.acceptance_sampling.sequential import (
    BinomialSequentialSamplingPlan,
    HypergeometricSequentialSamplingPlan,
)


def simulate_double_sampling_full_hypergeometric():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = DoubleSamplingPlan.hypergeometric(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=LOT_SIZE
        )

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        P = T / LOT_SIZE
        asns_analytical = np.array([plan.average_sample_size(p) for p in P])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        for d in T:
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng((d + 1) * (rep + 1))
                rng.shuffle(data)

                errors_in_first_batch = np.count_nonzero(data[: plan.n])

                # Accept after looking at n items
                if errors_in_first_batch <= plan.c1:
                    asns[d, rep] = plan.n
                # Rejecting after looking at n items
                elif errors_in_first_batch > plan.c2:
                    asns[d, rep] = plan.n
                else:
                    asns[d, rep] = plan.n * 2

        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL
        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_double_hypergeometric_full.pdf")


def simulate_double_sampling_full_binomial():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = DoubleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        P = T / LOT_SIZE
        asns_analytical = np.array([plan.average_sample_size(p) for p in P])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        for d in T:
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng((d + 1) * (rep + 1))
                rng.shuffle(data)

                errors_in_first_batch = np.count_nonzero(data[: plan.n])

                # Accept after looking at n items
                if errors_in_first_batch <= plan.c1:
                    asns[d, rep] = plan.n
                # Rejecting after looking at n items
                elif errors_in_first_batch > plan.c2:
                    asns[d, rep] = plan.n
                else:
                    asns[d, rep] = plan.n * 2

        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.DOUBLE_SAMPLING_BINOMIAL_FULL

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_double_binomial_full.pdf")


def simulate_double_sampling_curtailed_hypergeometric():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = DoubleSamplingPlan.hypergeometric(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, lot_size=LOT_SIZE
        )

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        P = T / LOT_SIZE
        P[0] = 0.0001
        asns_analytical = np.array([plan.average_sample_size_curtailed(p) for p in P])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        for d in T:
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng((d + 1) * (rep + 1))
                rng.shuffle(data)

                errors_in_first_batch = np.count_nonzero(data[: plan.n])

                # Accept after looking at n items
                if errors_in_first_batch <= plan.c1:
                    asns[d, rep] = plan.n
                    continue

                # Rejecting after looking at n items
                if errors_in_first_batch > plan.c2:
                    asns[d, rep] = plan.n
                    continue

                count = errors_in_first_batch
                for i in range(plan.n, plan.n * 2):
                    count += data[i]
                    if count > plan.c2:
                        asns[d, rep] = i
                        break
                else:
                    # Full inspection of second sample
                    asns[d, rep] = plan.n * 2

        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_CURTAILED

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_double_hypergeometric_curtailed.pdf")


def simulate_double_sampling_curtailed_binomial():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = DoubleSamplingPlan.binomial(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        P = T / LOT_SIZE
        P[0] = 0.0001
        asns_analytical = np.array([plan.average_sample_size_curtailed(p) for p in P])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        for d in tqdm(T):
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng((d + 1) * (rep + 1))
                rng.shuffle(data)
                sample = rng.choice(data, LOT_SIZE, replace=True)

                errors_in_first_batch = np.count_nonzero(data[: plan.n])

                # Accept after looking at n items
                if errors_in_first_batch <= plan.c1:
                    asns[d, rep] = plan.n
                    continue

                # Rejecting after looking at n items
                if errors_in_first_batch > plan.c2:
                    asns[d, rep] = plan.n
                    continue

                count = errors_in_first_batch
                for i in range(plan.n, plan.n * 2):
                    count += sample[i]
                    if count > plan.c2:
                        asns[d, rep] = i
                        break
                else:
                    # Full inspection of second sample
                    asns[d, rep] = plan.n * 2

        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.DOUBLE_SAMPLING_BINOMIAL_CURTAILED

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_double_binomial_curtailed.pdf")


def simulate_sequential_sampling_full_hypergeometric():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = HypergeometricSequentialSamplingPlan(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, LOT_SIZE
        )

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        asns_analytical = np.array([plan.average_sample_number(d, cutoff=LOT_SIZE).average_sample_number for d in T])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        region = plan.compute_truncated_wald_region(cutoff=LOT_SIZE)

        for d in T:
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng((d + 1) * (rep + 1))
                rng.shuffle(data)
                count = 0
                for i in range(LOT_SIZE):
                    x = data[i]
                    count += x

                    cl = region.lower_limits[i + 1]
                    cu = region.upper_limits[i + 1]

                    if cl is None:
                        cl = -np.inf
                    if cu is None:
                        cu = np.inf

                    if count <= cl or count >= cu:
                        asns[d, rep] = i
                        break

                else:
                    # If we did need to inspect the whole lot
                    asns[d, rep] = LOT_SIZE

        results[f"{qa_config.name}_config"] = qa_config
        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.DOUBLE_SAMPLING_HYPERGEOMETRIC_FULL

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_sequential_hypergeometric_full.pdf")


def simulate_sequential_sampling_full_binomial():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        plan = BinomialSequentialSamplingPlan(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

        cutoff = 10 * LOT_SIZE

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        asns_analytical = np.array([plan.average_sample_with_cutoff(d / LOT_SIZE, cutoff=cutoff) for d in T])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        region = plan.compute_region(cutoff)

        for d in tqdm(T):
            rng = np.random.default_rng(42)

            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                sample = rng.choice(data, cutoff, replace=True)

                count = 0
                for i in range(cutoff):
                    x = sample[i]
                    count += x

                    cl = region.lower_limits[i]
                    cu = region.upper_limits[i]

                    if cl is None:
                        cl = -np.inf
                    if cu is None:
                        cu = np.inf

                    if count <= cl or count >= cu:
                        asns[d, rep] = i
                        break

        results[f"{qa_config.name}_config"] = qa_config
        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.SPRT_BINOMIAL_FULL

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_sequential_binomial_full.pdf")


def simulate_sequential_sampling_curtailed_hypergeometric():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        fixed_size_test = SingleSamplingPlan.hypergeometric(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, LOT_SIZE
        )
        plan = HypergeometricSequentialSamplingPlan(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, LOT_SIZE
        )

        cutoff = fixed_size_test.n

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        asns_analytical = np.array([plan.average_sample_number(d, cutoff=cutoff).average_sample_number for d in T])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        region = plan.compute_truncated_wald_region(cutoff=cutoff)

        for d in T:
            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                rng = np.random.default_rng(42 + (d + 1) * (rep + 1))
                rng.shuffle(data)

                count = 0
                for i in range(cutoff):
                    x = data[i]
                    count += x

                    cl = region.lower_limits[i + 1]
                    cu = region.upper_limits[i + 1]

                    if cl is None:
                        cl = -np.inf
                    if cu is None:
                        cu = np.inf

                    if count <= cl or count >= cu:
                        asns[d, rep] = i
                        break
                else:
                    # If we did need to inspect as much as in the fixed test case
                    asns[d, rep] = cutoff

        results[f"{qa_config.name}_config"] = qa_config
        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.SPRT_HYPERGEOMETRIC_CURTAILED

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_sequential_hypergeometric_curtailed.pdf")


def simulate_sequential_sampling_curtailed_binomial():
    results = {}
    names = []

    for qa_config in QA_CONFIGS:
        fixed_size_test = SingleSamplingPlan.binomial(
            qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, LOT_SIZE
        )
        plan = BinomialSequentialSamplingPlan(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta)

        cutoff = fixed_size_test.n * 3

        T = np.arange(int(qa_config.asn_xlim * LOT_SIZE + 1) + 1)
        asns_analytical = np.array([plan.average_sample_with_cutoff(d / LOT_SIZE, cutoff=cutoff) for d in T])

        asns = np.zeros((len(T), NUM_REPETITIONS))

        region = plan.compute_region(cutoff)

        for d in tqdm(T):
            rng = np.random.default_rng(42)

            data = np.zeros(LOT_SIZE, dtype=bool)
            data[:d] = True
            assert np.count_nonzero(data) == d

            # Do the actual test
            for rep in range(NUM_REPETITIONS):
                sample = rng.choice(data, cutoff, replace=True)

                count = 0
                for i in range(cutoff):
                    x = sample[i]
                    count += x

                    cl = region.lower_limits[i]
                    cu = region.upper_limits[i]

                    if cl is None:
                        cl = -np.inf
                    if cu is None:
                        cu = np.inf

                    if count <= cl or count >= cu:
                        asns[d, rep] = i
                        break

        results[f"{qa_config.name}_config"] = qa_config
        results[f"{qa_config.name}_data"] = asns
        results[f"{qa_config.name}_analytical"] = asns_analytical
        results[f"{qa_config.name}_T"] = T
        results[f"{qa_config.name}_approach"] = Approach.SPRT_BINOMIAL_CURTAILED

        names.append(qa_config.name)

    results["names"] = np.array(names, dtype=object)

    plot_simulation_vs_analytical(results, PATH_PLOTS / "sim_sequential_binomial_curtailed.pdf")


def plot_simulation_vs_analytical(results: Dict[str, Any], p_out: Path):
    linewidth = 1.5
    names = results["names"]

    fig, axs = plt.subplots(2)
    for i, name in enumerate(names):
        data = results[f"{name}_data"]
        asns_analytical = results[f"{name}_analytical"]
        t = results[f"{name}_T"]

        averaged = np.mean(data, axis=1)

        ax = axs[i]
        ax.plot(t / LOT_SIZE, averaged, label="Simulation", linewidth=linewidth)
        ax.plot(t / LOT_SIZE, asns_analytical, label="Analytical", linewidth=linewidth)

        # plan = SingleSamplingPlan.hypergeometric(qa_config.p1, qa_config.p2, qa_config.alpha, qa_config.beta, LOT_SIZE)
        # ax.axhline(y=plan.n)

        y_min, y_max = ax.get_ylim()

        ax.text(x=0.0, y=y_max * 0.9, s=name, weight="bold")

        ax.set_xlabel("$p$")
        ax.set_ylabel("ASN")

    # plt.suptitle("Simulation and analytical solution for sequential sampling")
    # plt.legend()
    plt.tight_layout()
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.savefig(p_out)


if __name__ == "__main__":
    PATH_PLOTS.mkdir(exist_ok=True, parents=True)

    fs = [
        simulate_double_sampling_full_hypergeometric,
        simulate_double_sampling_curtailed_hypergeometric,
        simulate_sequential_sampling_full_hypergeometric,
        simulate_sequential_sampling_curtailed_hypergeometric,
        simulate_double_sampling_full_binomial,
        simulate_double_sampling_curtailed_binomial,
        simulate_sequential_sampling_full_binomial,
        simulate_sequential_sampling_curtailed_binomial,
    ]

    for i, f in enumerate(fs):
        f()
        print(f"{i+1}/{len(fs)}\t{f.__name__}")
