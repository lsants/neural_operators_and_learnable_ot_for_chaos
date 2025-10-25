import torch
import numpy as np
from torch.utils.data import Dataset
from .transforms import Transform

class DynamicalSystemDataset(Dataset):
    """
    - 'index': array of ids (n_samples)
    - 'traj_0', 'traj_1', ...: (time_steps, spatial_dim)_i, i=1,...,n_samples
    - 'param_0', 'param_1', ...: (n_params)_i, i=1,...,n_samples
    """
    def __init__(
            self, 
            npz_path: str, 
            device: str | torch.device,
            dtype: torch.dtype,
            transforms : list[Transform] | None = None,
    ):
        self.data = np.load(npz_path)
        self.ids = self.data['ids']
        self.transforms = transforms or []
        self.rng = torch.Generator()
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """ 
        Args:
            idx: Index of the trajectory
            
        Returns:
            tuple: (index, trajectory, parameters)
                - index: trajectory identifier
                - trajectory: tensor of shape (time_steps, spatial_dim)
                - parameters: tensor of shape (param_dim,)
        """
        traj_id = self.ids[idx]
        
        traj = self.data[f'traj_{traj_id}']
        params = self.data[f'params_{traj_id}']

        traj = torch.from_numpy(traj).to(dtype=self.dtype, device=self.device)
        params = torch.from_numpy(params).to(dtype=self.dtype, device=self.device)

        for t in self.transforms:
            traj = t(traj_id, traj, params)
        
        return traj_id, traj, params

def collate_fn(batch: list[tuple]) -> tuple:
    indices, trajs, params = zip(*batch)
    trajs_batch = torch.stack(trajs, dim=0)  # (batch_size, time_steps, spatial_dim)
    params_batch = torch.stack(params, dim=0)  # (batch_size, param_dim)
    
    return list(indices), trajs_batch, params_batch