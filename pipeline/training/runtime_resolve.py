import numpy
from pipeline.training.train_configs import TrainConfig, LossConfig

def resolve_runtime_config(exp_cfg: TrainConfig, loss_cfg: LossConfig) -> dict:
    sample = numpy.load(exp_cfg.train_data_path, allow_pickle=True)
    traj = sample["traj_000000"]
    param = sample["params_000000"]
    n_timesteps = exp_cfg.crop_window_size
    input_dim = traj.shape[1]
    param_dim = param.shape[0]

    return {
        "emulator_input_dim": input_dim + param_dim,
        "emulator_output_dim": input_dim,
        "n_samples": len(sample["ids"]),
        "summary_input_dim": input_dim * n_timesteps,
        "summary_output_dim": loss_cfg.num_summary_stats,
        "summary_embedding_dim": input_dim * loss_cfg.num_summary_stats * n_timesteps
    }