from sampleplan.acceptance_sampling import SingleSamplingPlan

alpha = 0.05
beta = 0.2
p_a = 0.01
p_r = 0.05

plan = SingleSamplingPlan.binomial(p_a, p_r, alpha, beta)
print(f"Single Sampling Plan binomial: n={plan.n}, c={plan.c}")