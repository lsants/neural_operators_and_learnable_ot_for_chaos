import numpy as np
import matplotlib.pyplot as plt
from typing import Any

def plot_histogram_comparison(
    hist_true: tuple[np.ndarray, np.ndarray],
    hist_pred: tuple[np.ndarray, np.ndarray],
    **kwargs: Any,
):
    """
    Plot true vs predicted histograms for each component in separate subplots.

    Args:
        hist_true: Tuple of (bin_centers, hist_values) for true data, each of shape [dim, num_bins].
        hist_pred: Tuple of (bin_centers, hist_values) for predicted data, each of shape [dim, num_bins].
        comp_labels: List of component labels (passed via kwargs['comp_labels']).
        title: Optional overall plot title.

    Returns:
        matplotlib Figure object (useful for wandb logging or saving)
    """
    bin_centers_true, hist_values_true = hist_true
    bin_centers_pred, hist_values_pred = hist_pred

    dim = hist_true[0].shape[0]
    components = range(dim)

    # Dynamically choose colors from matplotlib tab10 (or similar)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(dim)]

    # Use provided component labels or default names
    comp_labels = kwargs.get("comp_labels", [f"Component {i+1}" for i in components])

    fig, axes = plt.subplots(1, dim, figsize=(4 * dim, 4))
    if dim == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(bin_centers_pred[i], hist_values_pred[i], "-", color=colors[i], lw=1.5, label="Predicted")
        ax.plot(bin_centers_true[i], hist_values_true[i], "--", color='k', lw=1, label="True")
        ax.set_xlabel(comp_labels[i])
        ax.set_ylabel("Density")
        ax.set_title(comp_labels[i])
        ax.legend()
        ax.grid(True, ls="--", alpha=0.6)

#     plt.suptitle(kwargs.get("title", "Histogram Comparison"), fontsize=16, y=0.93)
    plt.tight_layout()

    return fig

def plot_error_evolution(mse_per_time: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    timesteps = np.arange(len(mse_per_time))
    ax.plot(timesteps, mse_per_time, color='firebrick', linewidth=2)
    ax.fill_between(timesteps, mse_per_time, alpha=0.3, color='firebrick')

    ax.set_xlabel('Time Step', fontsize=11)
#     ax.set_ylabel('MSE', fontsize=11)
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