import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
n = 16  # number of signal+noise samples (vector x)
m = 16  # number of noise-only reference samples (vector y)
S_i = np.ones(n)  # deterministic signal vector (assume constant)
sigma = 1.0       # noise standard deviation
b = 1.0           # signal amplitude (SNR ~ b^2 / sigma^2)

# 1. Generate samples
#np.random.seed(0)
y = np.random.normal(0, sigma, m)            # noise-only reference sample
signal = b * S_i                                 # pure signal
noise_for_x = np.random.normal(0, sigma, n)  # noise for signal+noise mixture
x = signal + noise_for_x                         # signal + noise sample

# 2. Compute ranks R_i: how many y_k values are less than each x_i
R = np.array([min(np.sum(xi > y), m - 1) for xi in x])

# 3. First moments of Gaussian order statistics for m = 16
# (from Table 1 of the paper)
mu_table = [-1.765, -1.285, -0.990, -0.763, -0.570, -0.396, -0.233, -0.077,
             0.077,  0.233,  0.396,  0.570,  0.763,  0.990,  1.285,  1.765]

# 4. Compute LO detection statistic: sum(S_i * a_m(R_i)) = sum(S_i * mu[R_i])
a_values = [mu_table[r] for r in R]
S_statistic = np.sum(S_i * a_values)

# 5. Approximate detection threshold V_p using Central Limit Theorem
# E{λ} = 0, Var{λ} ≈ sum(S_i^2) * Var(mu) ≈ sum(S_i^2) assuming Var ≈ 1
mu_lambda = 0
var_lambda = np.sum(S_i**2)
false_alarm_alpha = 0.01
z_alpha = norm.ppf(1 - false_alarm_alpha)
V_p = mu_lambda + np.sqrt(var_lambda) * z_alpha

# 6. Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y, 'o', label='Noise sample (y)', alpha=0.7)
plt.plot(signal, 'x-', label='Pure signal', alpha=0.7)
plt.plot(x, 's--', label='Signal + noise (x)', alpha=0.7)
plt.title("Signal + Noise vs. Reference Noise")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(R)
plt.title("Ranks R_i of x_i over noise y")
plt.xlabel("Sample index i")
plt.ylabel("Rank R_i (0 to m)")
plt.grid()

plt.tight_layout()
plt.show()

# 7. Print results
print(f"Rank vector R = {R}")
print(f"a_m(R_i) = {a_values}")
print(f"Detection statistic S = {S_statistic:.3f}")
print(f"Detection threshold V_p = {V_p:.3f}")
print(">>> DETECTED!" if S_statistic >= V_p else ">>> NOT DETECTED.")
