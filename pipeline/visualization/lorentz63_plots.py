import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_trajectories_comparison(u_true: np.ndarray, 
                                    u_pred: np.ndarray, 
                                    title: str):
    """
    Plot true and predicted 3D trajectories side by side.
    
    Args:
        u_true: True trajectory, shape [T, 3]
        u_pred: Predicted trajectory, shape [T, 3]
        title: Plot title
    
    Returns:
        matplotlib Figure object for wandb logging
    """
    x_true, y_true, z_true = u_true[:, 0], u_true[:, 1], u_true[:, 2]
    x_pred, y_pred, z_pred = u_pred[:, 0], u_pred[:, 1], u_pred[:, 2]
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_true, y_true, z_true, color='steelblue', lw=0.8, label='True')
    ax1.scatter(x_true[0], y_true[0], z_true[0], color='red', marker='x', s=50, label='Start')
    ax1.scatter(x_true[-1], y_true[-1], z_true[-1], color='blue', marker='o', s=100, label='End')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("True Trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_pred, y_pred, z_pred, color='coral', lw=0.8, label='Predicted')
    ax2.scatter(x_pred[0], y_pred[0], z_pred[0], color='red', marker='x', s=50, label='Start')
    ax2.scatter(x_pred[-1], y_pred[-1], z_pred[-1], color='blue', marker='o', s=100, label='End')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Predicted Trajectory")
    ax2.legend()
    
    ax2.view_init(elev=ax1.elev, azim=ax1.azim)
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    
    return fig

def plot_1d_components_comparison(u_true: np.ndarray, 
                                  u_pred: np.ndarray, 
                                  dt: float, 
                                  title: str):
    """
    Plot true vs predicted trajectories for each component in separate rows.
    
    Args:
        u_true: True trajectory, shape [T, 3]
        u_pred: Predicted trajectory, shape [T, 3]
        dt: Time step for x-axis
        title: Plot title
    
    Returns:
        matplotlib Figure object for wandb logging
    """
    T = u_true.shape[0]
    time = np.arange(0, T * dt, dt)[:T]
    
    labels = ['x(t)', 'y(t)', 'z(t)']
    colors_true = ['tab:cyan', 'tab:pink', 'tab:orange']
    colors_pred = ['darkturquoise', 'deeppink', 'darkorange']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    for i, (label, color_true, color_pred) in enumerate(zip(labels, colors_true, colors_pred)):
        ax = axes[i]
        
        ax.plot(time, u_true[:, i], color=color_true, lw=1.5, label='True', alpha=0.8)
        
        ax.plot(time, u_pred[:, i], color=color_pred, lw=1.5, label='Predicted', alpha=0.8, linestyle='--')
        
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(loc='upper right')
        
        mse = np.mean((u_true[:, i] - u_pred[:, i])**2)
        spectral_distance = np.linalg.norm(np.fft.fft(u_true[:, i]) - np.fft.fft(u_pred[:, i]))
        
        ax.text(0.02, 0.9, f'Spectral Dist: {spectral_distance:.4f}', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.text(0.02, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    axes[-1].set_xlabel("Time", fontsize=11)
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
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