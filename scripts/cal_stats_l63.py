import torch
import numpy as np

def cal_stats_l63(anchor_t: torch.Tensor, out_t: torch.Tensor):
    """
    Compute statistics for Lorenz63 system.
    
    Lorenz63 equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    
    Input shape: B x T x 3 (batch, time, [x,y,z])
    """
    def grad_t_centered(u):
        return 0.5 * (u[:, 2:] - u[:, :-2])          # (B, T-2, d)

    def stats_l63(u):
        x, y, z = u.unbind(-1)                       # (B, T,)

        g = grad_t_centered(u)                       # (B, T-2, d)
        x, y, z = x[:, 1:-1], y[:, 1:-1], z[:, 1:-1] # align with g

        stats = torch.stack([
            y - x,                                   # linear_x
            g[..., 0],                               # FDA with respect to x
            x,                                              
            x * z,                                   # nonlinear_y
            -y,                                      # linear_y
            g[..., 1],                               # FDA with respect to y
            y,
            x * y,                                   # nonlinear_z
            -z,                                      # linear_z
            g[..., 2],                               # FDA with respect to z
            z
        ], dim=-1)                                   # (B, T-2, 11)

        return stats

    with torch.no_grad():
        anchor_stats_flat = stats_l63(anchor_t)
    
    out_stats_flat = stats_l63(out_t)
    
    return (anchor_stats_flat, out_stats_flat)

def cal_stats_l1_score(anchor_t, out_t, anchor_param=[1], \
                    img_name='', folder_path = 'dist_plots', bins_test = 30, apply_gaussian = True, \
                    calculate_metric = 0, args = None, for_plot = False, for_stats = True, \
                    only_3d = False):
    def calculate_l1_score(hist_data_anchor, hist_data_predict):
        hist_data_anchor_normalized = hist_data_anchor[0].reshape(-1) / hist_data_anchor[0].reshape(-1).sum()
        hist_data_predic_normalized = hist_data_predict[0].reshape(-1) / hist_data_anchor[0].reshape(-1).sum()

        chi_score = abs(hist_data_anchor_normalized - hist_data_predic_normalized).sum() # + \
        return chi_score

    var = anchor_t
    var_k_1 = np.roll(var, 1, axis = 1)
    var_k_2 = np.roll(var, 2, axis = 1)
    var_k_p_1 = np.roll(var, -1, axis = 1)
    var_k_1, var_k_2, var_k_p_1 = torch.from_numpy(var_k_1), torch.from_numpy(var_k_2), torch.from_numpy(var_k_p_1)
    advection_stats = var_k_1 * (var_k_2 - var_k_p_1)

    ans = torch.from_numpy(var).permute(1, 0) # (d, T)
    grad_t = torch.gradient(ans, dim = 1)[0].permute(1, 0)
    mask = torch.ones_like(ans)
    mask[:, :1] = 0
    mask[:, -1:] = 0
    mask = mask.permute(1, 0)

    #########################out stats#########################################
    var_out = out_t
    var_k_1_out = np.roll(var_out, 1, axis = 1)
    var_k_2_out = np.roll(var_out, 2, axis = 1)
    var_k_p_1_out = np.roll(var_out, -1, axis = 1)
    var_k_1_out, var_k_2_out, var_k_p_1_out = torch.from_numpy(var_k_1_out), torch.from_numpy(var_k_2_out), torch.from_numpy(var_k_p_1_out)
    advection_stats_out = var_k_1_out * (var_k_2_out - var_k_p_1_out)

    ans_out = torch.from_numpy(var_out).permute(1, 0) # (d, T)
    grad_t_out = torch.gradient(ans_out, dim = 1)[0].permute(1, 0)
    mask = torch.ones_like(ans_out)
    mask[:, :1] = 0
    mask[:, -1:] = 0
    mask = mask.permute(1, 0)

    num_of_data = mask.sum()
    total_bins = np.sqrt(num_of_data)
    dim = 3
    bins_per_dim = np.floor(total_bins ** (1/dim))
    bins = int(bins_per_dim)

    anchor_stats = torch.stack([advection_stats.reshape(-1), grad_t.reshape(-1), torch.from_numpy(var.reshape(-1))]).permute(1, 0).numpy()
    hist_data = np.histogramdd(anchor_stats, bins = bins)
    out_stats = torch.stack([advection_stats_out.reshape(-1), grad_t_out.reshape(-1), torch.from_numpy(var_out.reshape(-1))]).permute(1, 0).numpy()
    hist_data_out = np.histogramdd(out_stats, bins = hist_data[-1])
    l1_score_3d = calculate_l1_score(hist_data, hist_data_out)

    return l1_score_3d
