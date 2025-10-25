import torch
import numpy as np
from pathlib import Path
from config import DataGenConfig
from scipy.integrate import solve_ivp
from collections.abc import Callable
from dynamical_systems import IVP_MAP
from typing import Any

class DataGenerator:
    def __init__(self, config: DataGenConfig | str | dict):
        if isinstance(config, (str, Path)):
            config = DataGenConfig.from_json(config)
        elif isinstance(config, dict):
            config = DataGenConfig(**config)

        self.config = config
        self.rng = np.random.default_rng()
        self.ode_func: Callable = IVP_MAP[config.experiment]

    def sample_parameters(self) -> np.ndarray:
        params = []
        for _, (low, high) in self.config.param_ranges.items():
            sampled = self.rng.uniform(low, high)
            params.append(sampled)
        return np.stack(params, axis=-1)

    def _solve_ivp(self, params: np.ndarray):
        params = self.sample_parameters()
        t_span = (self.config.t_start, self.config.t_end)
        t_eval = np.arange(self.config.t_start, self.config.t_end, self.config.dt)
        init_conditions = self.rng.normal(0, 1, self.config.n_dim)

        solution = solve_ivp(
            fun=self.ode_func,
            t_span=t_span,
            y0=init_conditions,
            args=tuple(params),
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-9,
            atol=1e-12
        )
        step = self.config.trajectory_length
        trajectory = solution.y.T[::step, :]
        return trajectory
    
    def generate_one(self, idx: int) -> tuple[str, np.ndarray, np.ndarray]:
        params = self.sample_parameters()
        traj = self._solve_ivp(params)
        traj_id = f"{idx:06d}"
        return traj_id, traj, params
    
    def generate_dataset(self, progress: bool=True) -> list[tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        iterator = range(self.config.n_samples)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Generating trajectories")

        results = [self.generate_one(i) for i in iterator]
        return results
    
    def save_trajectory(self, trajectory: np.ndarray, parameters: np.ndarray):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        traj_id = len(list(output_dir.glob('*.npy')))