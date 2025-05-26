import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# CFAR test runner
def run_cfar_analysis(noise_sigma=1.0, title_suffix=""):
    N = 128
    window_size = 32
    guard_len = 16
    ref_len = 8
    signal_amp = 4.0
    signal_width = 6
    false_alarm_alpha = 0.01

    noise_only = np.random.normal(0, noise_sigma, N)
    signal_bumps = np.zeros(N)

    def add_gaussian(center, width, amp, total_len):
        x = np.arange(total_len)
        return amp * np.exp(-0.5 * ((x - center) / (width / 2))**2)

    signal_bumps += add_gaussian(center=40, width=signal_width, amp=signal_amp, total_len=N)
    signal_bumps += add_gaussian(center=90, width=signal_width, amp=signal_amp, total_len=N)
    data = noise_only + signal_bumps

    mu_table = [-1.765, -1.285, -0.990, -0.763, -0.570, -0.396, -0.233, -0.077,
                0.077,  0.233,  0.396,  0.570,  0.763,  0.990,  1.285,  1.765]

    S_i = np.ones(guard_len)
    LO_statistic = np.zeros(N)
    standard_cfar_stat = np.zeros(N)
    threshold_std_mean = np.zeros(N)
    threshold_std_plus_std = np.zeros(N)
    alpha_std = 3.5
    beta_std = 1.5

    os_cfar_stat = np.zeros(N)
    threshold_os_median = np.zeros(N)
    threshold_os_plus_std = np.zeros(N)
    alpha_os = 1.8
    beta_os = 1.5

    for center in range(window_size // 2, N - window_size // 2):
        window = data[center - window_size // 2 : center + window_size // 2]
        x = window[ref_len : ref_len + guard_len]
        y = np.concatenate([window[:ref_len], window[-ref_len:]])

        # LO Detector (raw)
        R = np.array([min(np.sum(xi > y), len(mu_table) - 1) for xi in x])
        a_vals = [mu_table[r] for r in R]
        LO_statistic[center] = np.sum(S_i * a_vals)

        # Standard CFAR
        mean_y = np.mean(y)
        std_y = np.std(y)
        standard_cfar_stat[center] = np.mean(x)
        threshold_std_mean[center] = alpha_std * mean_y
        threshold_std_plus_std[center] = mean_y + beta_std * std_y

        # OS-CFAR
        median_y = np.median(y)
        os_cfar_stat[center] = np.mean(x)
        threshold_os_median[center] = alpha_os * median_y
        threshold_os_plus_std[center] = median_y + beta_os * std_y

    var_lambda = np.sum(S_i**2)
    z_alpha = norm.ppf(1 - false_alarm_alpha)
    threshold_lo = np.sqrt(var_lambda) * z_alpha

    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"CFAR Analysis - Noise σ = {noise_sigma} {title_suffix}", fontsize=14)

    axs[0].plot(noise_only, color='gray')
    axs[0].set_title('Noise Only')
    axs[0].grid()

    axs[1].plot(signal_bumps, color='green')
    axs[1].set_title('Injected Signal Bumps (Gaussian Targets)')
    axs[1].grid()

    axs[2].plot(data, label='Data', color='black', linewidth=1)
    axs[2].plot(LO_statistic, label='LO Statistic (raw)', color='blue')
    axs[2].axhline(y=threshold_lo, color='red', linestyle='--', label='LO Threshold')
    axs[2].set_title('LO CFAR Detection Output')
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[2].grid()

    axs[3].plot(data, label='Data', color='black', linewidth=1)
    axs[3].plot(standard_cfar_stat, label='Standard CFAR (mean)', color='orange')
    axs[3].plot(threshold_std_mean, linestyle='--', color='red', label='α·mean threshold')
    axs[3].plot(threshold_std_plus_std, linestyle='--', color='green', label='mean + β·std threshold')
    axs[3].set_title('Standard CFAR (mean-based)')
    axs[3].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[3].grid()

    axs[4].plot(data, label='Data', color='black', linewidth=1)
    axs[4].plot(os_cfar_stat, label='OS-CFAR (median)', color='purple')
    axs[4].plot(threshold_os_median, linestyle='--', color='red', label='α·median threshold')
    axs[4].plot(threshold_os_plus_std, linestyle='--', color='green', label='median + β·std threshold')
    axs[4].set_title('OS-CFAR (median-based)')
    axs[4].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[4].grid()

    plt.xlabel('Sample Index')
    plt.tight_layout()
    return fig

# Run all 3 plots and keep them open
fig1 = run_cfar_analysis(noise_sigma=0.5, title_suffix="(Low Noise)")
fig2 = run_cfar_analysis(noise_sigma=1.0, title_suffix="(Moderate Noise)")
fig3 = run_cfar_analysis(noise_sigma=2.0, title_suffix="(High Noise)")
plt.subplots_adjust(right=0.85)  # Leave space for legends
plt.show()
