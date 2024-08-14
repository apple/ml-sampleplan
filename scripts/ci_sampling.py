"""
sample_size_exact_binomial does sampling with replacement. This is suitable for large datasets
of 10k samples or larger. In contrast, if the dataset had 1k samples or less, we should use the
hypergeometric estimation, which can be used also over the simulator:
https://sample-size.net/sample-size-conf-interval-proportion/

with replacement
n_binomial = sample_size_exact_binomial(p0, alpha, ci_half_width)

without replacement
n_hyper = sample_size_exact_hypergeometric(110000, p0, alpha, ci_half_width)
"""

from sampleplan.confidence_interval import sample_size_exact_binomial

n_binomial = sample_size_exact_binomial(p0=0.12, alpha=0.05, ci_half_width=0.05)
print(n_binomial)

n_binomial = sample_size_exact_binomial(p0=0.4, alpha=0.05, ci_half_width=0.05)
print(n_binomial)
