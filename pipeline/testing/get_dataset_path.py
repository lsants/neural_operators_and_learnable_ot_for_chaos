import torch
def get_dataset_path(exp_path: str):
    checkpoint = torch.load(f"{exp_path}/checkpoints/best_model.pth")
    info = checkpoint['dataset_info']['name'].split('_')
    dataset_path = f"data/{info[0]}/{info[1]}/test_data.npz"

    return dataset_path
