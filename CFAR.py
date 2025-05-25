import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# PARAMETERS
# ---------------------------
N = 128
window_size = 32
guard_len = 16
ref_len = 8
sigma = 1.0
signal_amp = 4.0
signal_width = 6
false_alarm_alpha = 0.01

np.random.seed(42)

# ---------------------------
# 1. Generate noise + signal
# ---------------------------
noise_only = np.random.normal(0, sigma, N)
signal_bumps = np.zeros(N)


def add_gaussian(center, width, amp, total_len):
    x = np.arange(total_len)
    return amp * np.exp(-0.5 * ((x - center) / (width / 2)) ** 2)


signal_bumps += add_gaussian(center=40, width=signal_width, amp=signal_amp, total_len=N)
signal_bumps += add_gaussian(center=90, width=signal_width, amp=signal_amp, total_len=N)

data = noise_only + signal_bumps

# ---------------------------
# 2. LO-Rank CFAR Detector
# ---------------------------
mu_table = [-1.765, -1.285, -0.990, -0.763, -0.570, -0.396, -0.233, -0.077,
            0.077, 0.233, 0.396, 0.570, 0.763, 0.990, 1.285, 1.765]

S_i = np.ones(guard_len)
LO_statistic = np.zeros(N)

for center in range(window_size // 2, N - window_size // 2):
    window = data[center - window_size // 2: center + window_size // 2]
    x = window[ref_len: ref_len + guard_len]
    y = np.concatenate([window[:ref_len], window[-ref_len:]])

    R = np.array([min(np.sum(xi > y), len(mu_table) - 1) for xi in x])
    a_vals = [mu_table[r] for r in R]
    stat = np.sum(S_i * a_vals)
    LO_statistic[center] = stat

# LO threshold
var_lambda = np.sum(S_i ** 2)
z_alpha = norm.ppf(1 - false_alarm_alpha)
threshold_lo = np.sqrt(var_lambda) * z_alpha

# ---------------------------
# 3. Standard CFAR (mean-based)
# ---------------------------
standard_cfar_stat = np.zeros(N)
threshold_std = np.zeros(N)
alpha_std = 3.5  # Can be tuned

# ---------------------------
# 4. OS-CFAR (median-based)
# ---------------------------
os_cfar_stat = np.zeros(N)
threshold_os = np.zeros(N)
alpha_os = 3  # Can be tuned

for center in range(window_size // 2, N - window_size // 2):
    window = data[center - window_size // 2: center + window_size // 2]
    x = window[ref_len: ref_len + guard_len]
    y = np.concatenate([window[:ref_len], window[-ref_len:]])

    # Standard CFAR (mean-based)
    mean_y = np.mean(y)
    std_val = np.mean(x)
    standard_cfar_stat[center] = std_val
    threshold_std[center] = alpha_std * mean_y

    # OS-CFAR (median-based)
    median_y = np.median(y)
    os_cfar_stat[center] = std_val
    threshold_os[center] = alpha_os * median_y

# ---------------------------
# 5. Plot everything
# ---------------------------
fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

axs[0].plot(noise_only, color='gray')
axs[0].set_title('Noise Only')
axs[0].grid()

axs[1].plot(signal_bumps, color='green')
axs[1].set_title('Injected Signal Bumps (Gaussian Targets)')
axs[1].grid()

axs[2].plot(data, label='Data (Signal + Noise)', color='black', linewidth=1)
axs[2].plot(LO_statistic, label='LO Statistic', color='blue')
axs[2].axhline(y=threshold_lo, color='red', linestyle='--', label='LO Threshold')
axs[2].set_title('LO CFAR Detection Output')
axs[2].legend()
axs[2].grid()

axs[3].plot(data, label='Data', color='black', linewidth=1)
axs[3].plot(standard_cfar_stat, label='Standard CFAR (mean)', color='orange')
axs[3].plot(threshold_std, linestyle='--', color='red', label='Threshold')
axs[3].set_title('Standard CFAR (mean-based)')
axs[3].legend()
axs[3].grid()

axs[4].plot(data, label='Data', color='black', linewidth=1)
axs[4].plot(os_cfar_stat, label='OS-CFAR (median)', color='purple')
axs[4].plot(threshold_os, linestyle='--', color='red', label='Threshold')
axs[4].set_title('OS-CFAR (median-based)')
axs[4].legend()
axs[4].grid()

plt.xlabel('Sample Index')
plt.tight_layout()
plt.show()
