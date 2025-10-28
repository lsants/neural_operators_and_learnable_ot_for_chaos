import numpy as np
import matplotlib.pyplot as plt

def plot_error_evolution(mse_per_time: np.ndarray, spectral_distance: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    timesteps = np.arange(len(mse_per_time))
    ax.plot(timesteps, mse_per_time, color='firebrick', linewidth=2)
    ax.fill_between(timesteps, mse_per_time, alpha=0.3, color='firebrick')

    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, ls='--', alpha=0.4)
    ax.set_yscale('log')
    
    mean_mse = np.mean(mse_per_time)
    final_mse = mse_per_time[-1]

    ax.axhline(mean_mse, color='firebrick', linestyle='--', alpha=0.7, label=f'Mean MSE: {mean_mse:.4f}')
    ax.text(0.02, 0.98, f'Final MSE: {final_mse:.4f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    
    return fig