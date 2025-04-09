"""
Assumptions: 1. a sample is acceptable or unacceptable (has an error). 2. Samples must be
independent of each other.


sample_size_exact_binomial: for large enough datasets (e.g. >1k samples)

sample_size_exact_hypergeometric: sampling without replacement for small datasets <1k samples.
Limitation: quality should not be too close to 0 or 1. Also use the online simulator:
https://sample-size.net/sample-size-conf-interval-proportion/.
"""

from sampleplan.confidence_interval import sample_size_exact_binomial, \
    sample_size_exact_hypergeometric

n_binomial = sample_size_exact_binomial(p0=0.12, alpha=0.05, ci_half_width=0.05)
print(n_binomial)

n_hypergeometric = sample_size_exact_hypergeometric(lot_size=800,
                                                    p0=0.12,
                                                    alpha=0.05,
                                                    ci_half_width=0.05)
print(n_hypergeometric)
