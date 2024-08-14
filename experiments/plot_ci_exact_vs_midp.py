from matplotlib import pyplot as plt
from tqdm import trange

from experiments.util import PATH_PLOTS
from sampleplan.confidence_interval import *

w = 0.02
alpha = 0.05
lot_size = 1000

binom_exact = []
binom_midp = []

hypergeom_exact = []
hypergeom_midp = []

diffs_binom = []
diffs_hypergeom = []

pct_decrease_hypergeom = []


t = []

fudge = 3

for i in trange(1 + fudge, lot_size // 2 + 1):
    p = i / lot_size

    n1 = sample_size_exact_hypergeometric(lot_size, p, alpha, w)
    n2 = sample_size_exact_mid_point_hypergeometric(lot_size, p, alpha, w)
    diff = n1 - n2
    decrease = (n1 - n2) / n1 * 100

    hypergeom_exact.append(n1)
    hypergeom_midp.append(n2)
    diffs_hypergeom.append(diff)
    pct_decrease_hypergeom.append(decrease)

    n1 = sample_size_exact_binomial(p, alpha, w)
    n2 = sample_size_exact_mid_point_binomial(p, alpha, w)
    diff = n1 - n2

    binom_exact.append(n1)
    binom_midp.append(n2)
    diffs_binom.append(diff)

    t.append(p)

plt.plot(t, binom_exact, label="Exact")
plt.plot(t, binom_midp, label="mid-P")
plt.legend()

plt.savefig(PATH_PLOTS / "ci_midp_exact_binom.pdf")


plt.figure()

plt.plot(t, hypergeom_exact, label="Exact")
plt.plot(t, hypergeom_midp, label="mid-P")
plt.legend()

plt.savefig(PATH_PLOTS / "ci_midp_exact_hypergeom.pdf")

plt.figure()

plt.plot(t, diffs_hypergeom, label="Hypergeom")
plt.plot(t, diffs_binom, label="Binom")
plt.legend()

plt.savefig(PATH_PLOTS / "ci_midp_exact_diff.pdf")

plt.figure(figsize=(8, 4), dpi=80)

plt.plot(t, diffs_hypergeom)
plt.xlabel("p")
plt.ylabel("$\Delta$ n")

plt.savefig(PATH_PLOTS / "ci_midp_exact_diff_hypergeom.pdf")

plt.figure(figsize=(8, 4), dpi=80)

plt.plot(t, diffs_hypergeom)
plt.xlabel("p")
plt.ylabel("% Decrease n")

plt.savefig(PATH_PLOTS / "ci_midp_exact_diff_hypergeom_dec.pdf")
