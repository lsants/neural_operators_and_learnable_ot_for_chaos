import torch
import numpy as np
import matplotlib.pyplot as plt
from models.operator.mlp_timestepper import TimeStepperMLP
from models.summary.mlp_summary import MLPSummaryStats
from pipeline.visualization.lorentz63_plots import plot_l63_full_single, plot_l63_full_double
from pipeline.visualization import plot_histogram_comparison
from .metrics import compute_trajectory_errors
from pipeline.testing.metrics import get_histogram
from .eval_rollout import eval_rollout

path = '/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/outputs/lorenz63_20251029_210049_mlp_noisy_ca321833/ca321833/checkpoints/_epoch_1000.pt'

checkpoint = torch.load(path)
# We're cropping the trajectory window. Perhaps the model doesn't see the attractors during training?

# Load emulator model and summary model 
emulator = TimeStepperMLP(**checkpoint['emulator_config']).to(dtype=torch.float32)
summary = MLPSummaryStats(**checkpoint['summary_config']).to(dtype=torch.float32)
summary.load_state_dict(checkpoint['summary_state_dict'])
emulator.load_state_dict(checkpoint['model_state_dict'])

# Load data
dataset = np.load('data/lorenz63/3e39e796/train_data.npz')
param = torch.tensor(dataset['params_000000'][None, :], dtype=torch.float32)
T = 1000 # Compute Lyapunov time to do this correctly
u_true = torch.tensor(dataset['traj_000000'][None, :T, :], device='cpu', dtype=torch.float32)  # [B, T, D]

# Generate prediction
u_pred = eval_rollout(emulator, u_true, param)

# Compute summary statistics
s_true = summary(u_true).detach().numpy()
s_pred = summary(u_pred).detach().numpy()

u_true = u_true.detach().numpy()
u_pred = u_pred.detach().numpy()

# Compute histograms
k = int(np.ceil(np.sqrt(np.max([u_true.shape[1], u_pred.shape[1]]))))
hist_s_true = get_histogram(s_true, num_bins=k, density=True)
hist_s_pred = get_histogram(s_pred, num_bins=k, density=True)

hist_u_true = get_histogram(u_true, num_bins=k, density=True)
hist_u_pred = get_histogram(u_pred, num_bins=k, density=True)

# Compute Lyapunov Exponents (later)

errors = compute_trajectory_errors(u_true, u_pred)
mse = errors['mse']

# Errors for Latex table:
with open(f'results.txt', 'w') as f:
    f.writelines(f"Config: Need to add configs here\n")
    f.writelines(f"MSE: {mse}\n")
    f.writelines(f"Histogram Error: {errors['histogram_error']}\nEnergy Spec. Error: {errors['energy_spectrum']}\n")

# plot_visualizations(u_true, u_pred, errors)

u_label = r'u(t)'
u_comp_labels = [r'$x(t)$', r'$y(t)$', r'$z(t)$']
u_true_comp_labels = [r'$x(t)$_true', r'$y(t)$_true', r'$z(t)$_true']
u_pred_comp_labels = [r'$x(t)$_pred', r'$y(t)$_pred', r'$z(t)$_pred']

s_comp_labels = [f'$f_{i}(u(t))$' for i in range(1,s_true.shape[2]+1)]
s_true_comp_labels = [f"$f(u_{i}(t))_{{\\mathrm{{True}}}}$" for i in range(1, s_true.shape[2]+1)]
s_pred_comp_labels = [f"$f(u_{i}(t))_{{\\mathrm{{Pred}}}}$" for i in range(1, s_pred.shape[2]+1)]

print(s_pred_comp_labels, s_true_comp_labels)
s_label = r'f(u(t))'

fig_1 = plot_histogram_comparison(hist_s_true[0], hist_s_pred[0], comp_labels=s_comp_labels, title="Summary Histogram Comparison")
fig_2 = plot_histogram_comparison(hist_u_true[0], hist_u_pred[0], comp_labels=u_comp_labels, title="Trajectory Histogram Comparison")
fig_3 = plot_l63_full_single(u_true[0], plotted_variable=u_label, u_comp_labels=u_comp_labels)
fig_5 = plot_l63_full_double(u_pred[0], u_true[0], plotted_variable_1=u_label + " True", plotted_variable_2=u_label + " Predicted", first=u_pred_comp_labels, second=u_true_comp_labels)
if s_true.shape[2] == 3:
    fig_4 = plot_l63_full_single(s_true[0], plotted_variable=s_label, s_comp_labels=s_comp_labels)
    fig_6 = plot_l63_full_double(s_pred[0], s_true[0], plotted_variable_1=s_label + " True", plotted_variable_2=s_label + " Predicted", first=s_pred_comp_labels, second=s_true_comp_labels)
    fig_7 = plot_l63_full_double(s_true[0], u_true[0], plotted_variable_1=u_label, plotted_variable_2=s_label, first=u_comp_labels, second=s_comp_labels)
elif s_true.shape[2] == 1:
    pass

plt.show()