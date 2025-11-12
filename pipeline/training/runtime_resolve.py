from math import exp
import numpy
from pipeline.training.train_configs import TrainConfig

def resolve_runtime_config(exp_cfg: TrainConfig) -> None:
    """Ensures that emulator and summary stats configs are correct for model initialization."""
    sample = numpy.load(exp_cfg.train_data_path, allow_pickle=True)
    traj = sample["traj_000000"]
    param = sample["params_000000"]
    n_timesteps = exp_cfg.crop_window_size
    input_dim = traj.shape[1]
    param_dim = param.shape[0]

    exp_cfg.emulator_config["input_dim"] = input_dim + param_dim
    exp_cfg.emulator_config["output_dim"] = input_dim
    exp_cfg.summary_config["input_dim"] = input_dim
    exp_cfg.summary_config["output_dim"] = exp_cfg.summary_config.pop("num_summary_stats", None)