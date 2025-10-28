import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_l63(trajectory: np.ndarray):
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

    ax3d.set_title("Lorenz 63")
    ax3d.legend()
    
    labels = ['x(t)', 'y(t)', 'z(t)']
    colors = ['tab:cyan', 'tab:pink', 'tab:orange']
    for i, (coord, label, color) in enumerate(zip([x, y, z], labels, colors)):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(np.arange(0, 100, 0.01), coord, color=color, lw=1)
        ax.set_ylabel(label)
        ax.grid(True, ls='--', alpha=0.6)
        if i < 2:
            ax.tick_params(labelbottom=False)
        ax.set_xlabel("Time")

    plt.show()

if __name__ == "__main__":
    data = np.load("/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/data/lorenz63/3e39e796/train_data.npz")
    
    trajectory = data['traj_000000']
    plot_l63(trajectory)