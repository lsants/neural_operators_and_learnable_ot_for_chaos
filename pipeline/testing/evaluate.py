import torch
import numpy as np
import matplotlib.pyplot as plt
from manim import config
from models.operator.mlp_timestepper import TimeStepperMLP
from models.summary.mlp_summary import MLPSummaryStats
from pipeline.visualization.lorentz63_plots import plot_l63_full_single, plot_l63_full_double
from pipeline.visualization import plot_histogram_comparison
from pipeline.testing.metrics import get_histogram
from pipeline.testing.eval_configs import EvaluateConfig
from pipeline.testing.metrics import compute_trajectory_errors
from pipeline.testing.eval_rollout import eval_rollout

def eval_exp(eval_config: dict[str, EvaluateConfig]):
    exp_config = eval_config['eval_config']
    checkpoint_path = exp_config.checkpoint_path
    dataset_path = exp_config.test_data_path
    eval_steps = exp_config.rollout_steps

    output_path = str(checkpoint_path).rsplit('/', 3)[0] + '/' + str(checkpoint_path).rsplit('/', 3)[1]
    checkpoint = torch.load(checkpoint_path)

    # Load emulator model and summary model 
    ot_or_noot = 'ot' if checkpoint['train_config']['ot_penalty'] > 0.0 else 'no_ot'
    noisy_or_clean = 'noisy' if checkpoint['train_config']['noise_level']  > 0.0 else 'clean'
    dtype = getattr(torch, exp_config.dtype) if exp_config.dtype == 'float32' else torch.float32
    device = torch.device(exp_config.device)

    emulator = TimeStepperMLP(**checkpoint['emulator_config']).to(dtype=dtype, device=device)

    summary_type = checkpoint['train_config']['summary_type'] if ot_or_noot == 'ot' else 'NA'

    if summary_type != 'NA':
        if summary_type == 'identity':
            from models.summary.identity_summary import IdentitySummaryStats
            summary = IdentitySummaryStats().to(dtype=dtype, device=device)
        elif summary_type == 'projection':
            from models.summary.projection_summary import ProjectionSummaryStats
            summary = ProjectionSummaryStats(**checkpoint['summary_config']).to(dtype=dtype, device=device)
        else:
            summary = MLPSummaryStats(**checkpoint['summary_config']).to(dtype=dtype, device=device)
    else:
            from models.summary.identity_summary import IdentitySummaryStats
            summary = IdentitySummaryStats().to(dtype=dtype, device=device)

    summary.load_state_dict(checkpoint['summary_state_dict'])
    emulator.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    dataset = np.load(dataset_path)
    param = torch.tensor(dataset['params_000000'][None, :],dtype=dtype, device=device)  # [1, P]

    T = eval_steps if eval_steps else dataset['traj_000000'].shape[0]

    u_true = torch.tensor(dataset['traj_000000'][None, :T, :], device=device, dtype=dtype)  # [B, T, D]

    noise = checkpoint['train_config']['noise_level'] * u_true.std(dim=1, keepdim=True) * torch.randn_like(u_true, device=device, dtype=dtype) # check this
    u_true = u_true + noise

    # Generate prediction
    u_pred = eval_rollout(emulator, u_true, param, rollout_steps=T)

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

    # Compute Lyapunov Exponents (TO DO)

    errors = compute_trajectory_errors(u_true, u_pred)
    mse = errors['mse']

    # Errors for Latex table:
    if summary_type == 'projection':
        state_index = checkpoint['train_config']['summary_config']['state']
    index_to_state = {0: 'x', 1: 'y', 2: 'z'} # for Lorenz63

    with open(f'{output_path}/results.txt', 'w') as f:
        f.writelines(f"Summary type: {summary_type}\n")
        if summary_type == 'projection':
            f.writelines(f"Projected state: {index_to_state[state_index]}\n")
        f.writelines(f"OT: {ot_or_noot}\n")
        f.writelines(f"Noise: {noisy_or_clean}\n")
        f.writelines(f"MSE: {mse}\n")
        f.writelines(f"Histogram Error: {errors['histogram_error']}\nEnergy Spec. Error: {errors['energy_spectrum']}\n")


    u_label = r'u(t)'
    u_comp_labels = [r'$x(t)$', r'$y(t)$', r'$z(t)$']
    u_true_comp_labels = [r'$x(t)$_true', r'$y(t)$_true', r'$z(t)$_true']
    u_pred_comp_labels = [r'$x(t)$_pred', r'$y(t)$_pred', r'$z(t)$_pred']

    s_comp_labels = [f'$f_{i}(u(t))$' for i in range(1,s_true.shape[2]+1)]
    if summary_type == 'projection':
        s_comp_labels = [[f'$f_{state_index + 1}(u(t))$']]
    s_true_comp_labels = [f"$f(u_{i}(t))_{{\\mathrm{{True}}}}$" for i in range(1, s_true.shape[2]+1)]
    s_pred_comp_labels = [f"$f(u_{i}(t))_{{\\mathrm{{Pred}}}}$" for i in range(1, s_pred.shape[2]+1)]


    s_label = r'f(u(t))'

    figures = []

    figures.append(plot_histogram_comparison(hist_s_true[0], hist_s_pred[0], comp_labels=s_comp_labels, title="Summary Histogram Comparison"))
    figures.append(plot_histogram_comparison(hist_u_true[0], hist_u_pred[0], comp_labels=u_comp_labels, title="Trajectory Histogram Comparison"))
    figures.append(plot_l63_full_single(u_true[0], plotted_variable=u_label, u_comp_labels=u_comp_labels))
    figures.append(plot_l63_full_double(u_pred[0], u_true[0], plotted_variable_1=u_label + " True", plotted_variable_2=u_label + " Predicted", first=u_pred_comp_labels, second=u_true_comp_labels))

    if s_true.shape[2] == 3:
        figures.append(plot_l63_full_single(s_true[0], plotted_variable=s_label, s_comp_labels=s_comp_labels))
        figures.append(plot_l63_full_double(s_pred[0], s_true[0], plotted_variable_1=s_label + " True", plotted_variable_2=s_label + " Predicted", first=s_pred_comp_labels, second=s_true_comp_labels))
        figures.append(plot_l63_full_double(s_true[0], u_true[0], plotted_variable_1=u_label, plotted_variable_2=s_label, first=u_comp_labels, second=s_comp_labels))

    # Save all figures
    for i, fig in enumerate(figures, 1):
        fig.savefig(f'{output_path}/figure_{i}.png', dpi=300, bbox_inches='tight')

    print("Evaluation Complete!")

    from pipeline.visualization.animations_ce import Lorenz63
    config['media_dir'] = output_path
    config['video_dir'] = output_path
    config['images_dir'] = output_path
    config['output_file'] = "lorenz63.mp4"
    config['pixel_height'] = 1080*2
    config['pixel_width'] = 1920*2
    config['frame_rate'] = 60

    scene = Lorenz63(u_true[0], u_pred[0], s_true[0][:, :, state_index] if summary_type == 'projection' else s_true[0])
    scene.render()

    # plt.show()

if __name__ == "__main__":
    dataset_path = 'data/lorenz63/9bda2e47/test_data.npz'
    checkpoint_path = '/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/outputs/lorenz63_no_ot_noisy/7d2bb0a0/checkpoints/_epoch_200.pt'
    dtype = 'float32'
    device = 'cpu'

    exp_info = {
        'eval_config': {
            'test_data_path': dataset_path,
            'checkpoint_path': checkpoint_path,
            'device': device,
            'dtype': dtype,
            'rollout_steps': 100,
            "crop_window_size": None,
            "noise_level": 0.1,
        }
    }
    exp_config = {
        'eval_config': EvaluateConfig(**exp_info['eval_config'])
    }

    eval_exp(exp_config)