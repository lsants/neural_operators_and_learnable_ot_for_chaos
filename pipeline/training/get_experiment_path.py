from pathlib import Path
def get_exp_path(train_path_str: str):
    modified_str = train_path_str.replace('data', 'outputs')
    new_path_str = modified_str.replace('train_outputs.npz', '')
    return Path(modified_str)