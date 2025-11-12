import re
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Any

def get_problem_name(train_data_path: str):
    match_problem_name = re.search(r"(?<=data\/)([^\/]+)", train_data_path)
    if match_problem_name is None:
        raise ValueError(f"Could not extract problem name from path: {train_data_path}")
    problem_name = match_problem_name.group(1)
    return problem_name

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_exp_name(problem_name: str, ot_type: str, clean_or_noisy: str) -> str:
    exp_name = f"{problem_name}_{ot_type}_{clean_or_noisy}"
    return exp_name

def get_exp_path(exp_name: str, timestamp) -> Path:
    hash_obj = hashlib.sha256()
    hash_obj.update(timestamp.encode())
    version = hash_obj.hexdigest()[:8]
    if version is None:
        raise ValueError(f"Could not extract version from experiment name: {exp_name}")
    exp_path = Path("outputs") / exp_name / version
    return exp_path

def get_dataset_config(train_data_path: str) -> dict[str, Any]:
    with open(train_data_path, 'rb') as f:
        dataset_config = json.load(f)
    return dataset_config