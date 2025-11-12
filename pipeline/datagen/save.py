import numpy as np
from pathlib import Path
import pickle
import json
from typing import Any
from datetime import datetime


def save_npz(
        data: list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        config,
        output_path: str | Path
):
    save_dict = {
        "ids": np.array([traj_id for traj_id, _, _ in data], dtype='str'),
        "config": np.void(pickle.dumps(config)),
    }

    for traj_id, traj, params in data:
        save_dict[f"traj_{traj_id}"] = traj
        save_dict[f"params_{traj_id}"] = params
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_dict)
    print(f"Saved {len(data)} trajectories -> {output_path}")
    metadata = {
        "experiment": config.experiment,
        "n_samples": config.n_samples,
        "t_start": config.t_start,
        "t_end": config.t_end,
        "subsample_stride": config.subsample_stride,
        "dt": config.dt,
        "n_timesteps": config.n_timesteps,
        "param_ranges": config.param_ranges,
        "n_dim": config.n_dim,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = output_path.with_suffix('.json')
    json_path.write_text(json.dumps(metadata, indent=4))
    print(f"Metadata -> {json_path}")
    