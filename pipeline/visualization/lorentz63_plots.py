from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Any

def plot_l63_full_single(trajectory: np.ndarray[tuple[int, int], Any], **kwargs):
    T = trajectory.shape[0]
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, hspace=0.3)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.plot(x, y, z, color='steelblue', lw=0.5)
    ax3d.scatter(x[0], y[0], z[0], color='red', marker='x', s=50, label='Start')
    ax3d.scatter(x[-1], y[-1], z[-1], color='blue', marker='o', s=100, label='End')
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    plotted_var = kwargs.get('plotted_variable', 'Trajectory')

    ax3d.legend()

    if 's_comp_labels' in kwargs:
        labels = kwargs['s_comp_labels']
    elif 'u_comp_labels' in kwargs:
        labels =  kwargs['u_comp_labels']
    else:
        labels = [r'comp_1(t)', r'comp_2(t)', r'comp_3(t)']

    colors = ['tab:cyan', 'tab:pink', 'tab:orange']

    for i, (coord, label, color) in enumerate(zip([x, y, z], labels, colors)):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(np.arange(T), coord, color=color, lw=1)
        ax.set_ylabel(label)
        ax.grid(True, ls='--', alpha=0.6)
        if i < 2:
            ax.tick_params(labelbottom=False)
        ax.set_xlabel("Time")

    fig.suptitle(f"{plotted_var} Visualization", fontsize=16, y=0.95)
    return fig

def plot_l63_full_double(
    traj1: np.ndarray[tuple[int, int], Any],
    traj2: np.ndarray[tuple[int, int], Any],
    **kwargs,
):
    """Plot two Lorenz-63 trajectories in 3D and component-wise 1D plots."""
    assert traj1.shape[1] == 3 and traj2.shape[1] == 3, "Each trajectory must be (T, 3)."

    T1, T2 = traj1.shape[0], traj2.shape[0]
    x1, y1, z1 = traj1[:, 0], traj1[:, 1], traj1[:, 2]
    x2, y2, z2 = traj2[:, 0], traj2[:, 1], traj2[:, 2]

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 2, hspace=0.3)

    plotted_var_1 = kwargs.get("plotted_variable_1", "Trajectory 1")
    plotted_var_2 = kwargs.get("plotted_variable_2", "Trajectory 2")

    # --- 3D phase-space plot ---
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.plot(x1, y1, z1, color="steelblue", lw=0.8, label=plotted_var_1)
    ax3d.plot(x2, y2, z2, color="darkorange", lw=0.8, label=plotted_var_2)

    ax3d.scatter(x1[0], y1[0], z1[0], color="red", marker="x", s=40)
    ax3d.scatter(x2[0], y2[0], z2[0], color="darkred", marker="x", s=40)
    ax3d.scatter(x1[-1], y1[-1], z1[-1], color="blue", marker="o", s=60)
    ax3d.scatter(x2[-1], y2[-1], z2[-1], color="navy", marker="o", s=60)

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax3d.legend()

    # --- Component-wise time series ---
    if "first" in kwargs and "second" in kwargs:
        labels_1 = kwargs["first"]
        labels_2 = kwargs["second"]
    else:
        labels_1 = ["comp_1(t)", "comp_2(t)", "comp_3(t)"]
        labels_2 = ["comp_1(t)", "comp_2(t)", "comp_3(t)"]

    colors1 = ["tab:cyan", "tab:pink", "tab:orange"]

    for i, (coord1, coord2, label_1, label_2, c1) in enumerate(
        zip([x1, y1, z1], [x2, y2, z2], labels_1, labels_2, colors1)
    ):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(np.arange(T1), coord1, color=c1, linestyle='-', lw=1, label=label_1)
        ax.plot(np.arange(T2), coord2, color=c1, linestyle='--', lw=1, alpha=0.8, label=label_2)
        ax.set_ylabel(f"Dim {i+1}")
        ax.grid(True, ls="--", alpha=0.6)
        if i < 2:
            ax.tick_params(labelbottom=False)
        ax.set_xlabel("Time")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{plotted_var_1} vs {plotted_var_2}", fontsize=16, y=0.95)

    return fig

def plot_summary_stats_comparison(
        s_true: np.ndarray, 
        s_pred: np.ndarray, 
        title: str
    ):
    """
    Plot true vs predicted summary statistics.
    
    Args:
        s_true: True summary statistics, shape [D,] or [T, D]
        s_pred: Predicted summary statistics, same shape as s_true
        title: Plot title
    
    Returns:
        matplotlib Figure object for wandb logging
    """
    if s_true.ndim == 1:
        s_true = s_true.reshape(-1, 1)
        s_pred = s_pred.reshape(-1, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), squeeze=False)
    ax = ax[0].flatten()
    
    indices = np.arange(len(s_true))
    
    ax[0].plot(indices, s_true, 'o-', color='steelblue', 
            label='True', markersize=4, linewidth=.5, alpha=0.8)
    ax[0].plot(indices, s_pred, 's--', color='coral', 
            label='Predicted', markersize=4, linewidth=.5, alpha=0.8)
    
    ax[0].set_ylabel(f'1D Statistic', fontsize=10)
    ax[0].grid(True, ls='--', alpha=0.4)
    ax[0].legend()
    
    mse = np.mean((s_true - s_pred)**2)
    spectral_distance = np.linalg.norm(np.fft.fft(s_true) - np.fft.fft(s_pred))
    ax[0].text(0.02, 0.9, f'Spectral Dist: {spectral_distance:.4f}', transform=ax[0].transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax[0].text(0.02, 0.95, f'MSE: {mse:.4e}', transform=ax[0].transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax[0].set_xlabel("Index", fontsize=10)
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig