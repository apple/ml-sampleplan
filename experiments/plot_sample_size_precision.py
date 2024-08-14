import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from experiments.util import PATH_PLOTS
from sampleplan.confidence_interval import hypergeometric_proportion_interval_exact


def plot_single(num_instances: int, confidences: list[float], error_rate: float, plotname: str):
    plt.figure(figsize=(5, 3))
    x = list(range(1, num_instances + 1))

    errors = {}

    for confidence in sorted(confidences, key=lambda e: e):
        y = np.zeros_like(x, dtype=float)

        for sample_size in x:
            num_errors = round(error_rate * sample_size)

            try:
                cl, cu = hypergeometric_proportion_interval_exact(confidence, num_instances,
                                                                  sample_size, num_errors)
            except RuntimeError as e:
                print(e)
                continue

            half_width = (cu - cl) / 2

            if half_width == 0:
                continue

            y[sample_size] = half_width

            if sample_size in {100, 200} and confidence == 0.95:
                print(f"e={error_rate}, alpha={confidence}, n={sample_size}, h={half_width}")

            if sample_size > 400:
                break

        label = int(confidence * 100)
        errors[label] = y

    data = {
        "x": x,
    }

    for label, y in errors.items():
        # assert len(x) == len(y)
        data[label] = y * 100

    df = pd.DataFrame(data).convert_dtypes()
    df.to_csv(PATH_PLOTS / f"sample_size_precision_{int(error_rate * 100)}.csv", index=False)

    for label, y in errors.items():
        plt.plot(x, y, label=f"$\\alpha$={label}%")

    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(50))

    plt.xlabel("Sample size")
    plt.ylabel("Margin of error")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PATH_PLOTS / plotname)


def plot_stuff():
    # plot_single(num_instances=1000, confidences=[0.9, 0.95, 0.99], error_rate=0.05,
    #             plotname="sample_error_5.pdf")
    # plot_single(num_instances=1000, confidences=[0.9, 0.95, 0.99], error_rate=0.1,
    #             plotname="sample_error_10.pdf")
    plot_single(num_instances=1000, confidences=[0.9, 0.95, 0.99], error_rate=0.25,
                plotname="sample_error_25.pdf")

if __name__ == "__main__":
    plot_stuff()
