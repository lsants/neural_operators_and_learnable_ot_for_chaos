import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Any
import numpy as np



if __name__ == "__main__":
    data = np.load("/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/data/lorenz63/3e39e796/train_data.npz")
    
    trajectory = data['traj_000000']
    plot_l63(trajectory)