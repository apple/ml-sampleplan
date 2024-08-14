import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.util import PATH_PLOTS, PATH_RESULTS_ASNS, QA_CONFIGS


def plot_stuff():
    df = pd.read_csv(PATH_RESULTS_ASNS / "raw_asns.tsv", sep="\t")

    qa_configs = {qa_config.name: qa_config for qa_config in QA_CONFIGS}

    # CB_color_cycle = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"]
    # CB_color_cycle = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"]
    df["p"] = df["p"] * 100

    for qa_config_name, group in df.groupby("qa_config"):
        plt.figure()
        # plt.rc("axes", prop_cycle=(cycler("color", CB_color_cycle)))

        qa_config = qa_configs[qa_config_name]

        # plt.axvline(x=qa_config.p1, color="k", linestyle="dashed", linewidth=".75")
        # plt.axvline(x=qa_config.p2, color="k", linestyle="dashed", linewidth=".75")

        xmin, xmax, ymin, ymax = plt.axis()

        # plt.text(qa_config.p1 + 0.001, ymin + 5, "$p_a$")
        # plt.text(qa_config.p2 + 0.001, ymin + 5, "$p_r$")

        sns.color_palette("deep")

        plt.xlabel("Error rate $p$")
        plt.ylabel("Average sample number")

        g = sns.relplot(
            data=group,
            x="p",
            y="asn",
            hue="distribution",
            row="kind",
            col="dataset_size",
            kind="line",
            height=3,
            aspect=1,
            palette=["C1", "C0"],
        )
        g.map(plt.axvline, x=qa_config.p1 * 100, ls="--", c="blue", linewidth=0.5)
        g.map(plt.axvline, x=qa_config.p2 * 100, ls="--", c="red", linewidth=0.5)

        g.set_titles(col_template="N = {col_name}", row_template="{row_name}")

        g.set_ylabels("ASN")

        sns.despine()
        plt.legend([], [], frameon=False)
        g._legend.remove()

        plt.tight_layout()

        plt.savefig(PATH_PLOTS / f"p_vs_asn_{qa_config.name}_overall.pdf", bbox_inches="tight", pad_inches=0)
        print()


if __name__ == "__main__":
    plot_stuff()
