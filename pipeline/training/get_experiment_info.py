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

def get_exp_name(problem_name: str, ot_type: str, clean_or_noisy: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_obj = hashlib.sha256()
    hash_obj.update(timestamp.encode())
    version_hash = hash_obj.hexdigest()[:8]
    exp_name = f"{problem_name}_{timestamp}_{ot_type}_{clean_or_noisy}_{version_hash}"
    return exp_name

def get_exp_path(problem_name: str, exp_name: str) -> Path:
    version = re.search(r'[^_]*$', exp_name)
    if version is None:
        raise ValueError(f"Could not extract version from experiment name: {exp_name}")
    exp_path = Path("outputs") / exp_name / version.group(0)
    return exp_path

def get_dataset_config(train_data_path: str) -> dict[str, Any]:
    with open(train_data_path, 'rb') as f:
        dataset_config = json.load(f)
    return dataset_config