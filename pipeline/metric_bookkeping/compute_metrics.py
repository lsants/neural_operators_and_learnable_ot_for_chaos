import torch
import numpy as np

def compute_spectral_distance(u_true, u_hat, eps: float = 1e-8):
    U = np.fft.rfft(u_true, axis=1)
    U_hat = np.fft.rfft(u_hat, axis=1)
    P_true = np.abs(U) ** 2
    P_hat = np.abs(U_hat) ** 2
    spec_dist = np.abs(P_true - P_hat) / (P_true + eps)
    spec_dist_per_sample = np.mean(spec_dist, axis=1)

    return np.mean(spec_dist_per_sample).item()

def compute_trajectory_errors(u_true, u_hat):
    """
    Compute various error metrics for trajectory predictions.
    
    Args:
        u_true: True trajectory tensor, shape [batch, T, 3]
        u_hat: Predicted trajectory tensor, shape [batch, T, 3]
    
    Returns:
        Dictionary of error metrics
    """
    mse_per_time = np.mean((u_true - u_hat) ** 2, axis=(0, 2))  # [T]
    mse_per_component = np.mean((u_true - u_hat) ** 2, axis=(0, 1))  # [3]

    mse = np.mean((u_true - u_hat) ** 2).item()
    rmse = np.sqrt(np.mean((u_true - u_hat) ** 2)).item()
    spectral_distance = compute_spectral_distance(u_true, u_hat)
    mae = np.mean(np.abs(u_true - u_hat)).item()

    l2_error = np.mean(np.linalg.norm(u_true - u_hat, axis=-1) / (np.linalg.norm(u_true, axis=-1) + 1e-8)).item()
    
    max_error = np.max(np.abs(u_true - u_hat)).item()

    return {
        'mse': mse,
        'rmse': rmse,
        'spectral_distance': spectral_distance,
        'mae': mae,
        'l2_error': l2_error,
        'max_error': max_error,
        'mse_per_time': mse_per_time,
        'mse_per_component': mse_per_component,
    }

def compute_summary_errors(s_true, s_hat):
    """
    Compute error metrics for summary statistics.
    
    Args:
        s_true: True summary statistics tensor
        s_hat: Predicted summary statistics tensor
    
    Returns:
        Dictionary of error metrics
    """
    
    mse = np.mean((s_true - s_hat) ** 2).item()
    mae = np.mean(np.abs(s_true - s_hat)).item()
    spectral_distance = compute_spectral_distance(s_true, s_hat)
    l2_error = np.mean(np.linalg.norm(s_true - s_hat, axis=-1) / (np.linalg.norm(s_true, axis=-1) + 1e-8)).item()
    
    return {
        'summary_mse': mse,
        'summary_mae': mae,
        'summary_spectral_distance': spectral_distance,
        'summary_l2_error': l2_error,
    }
