import numpy as np
from scipy.integrate import solve_ivp

def lorenz63(t, x):
    ρ = 28
    σ = 10
    β = 8 / 3
    dxdt = σ * (x[1] - x[0])
    dydt = x[0] * (ρ - x[2]) - x[1]
    dzdt = x[0] * x[1] - β * x[2]
    return [dxdt, dydt, dzdt]

def generate_l63_data(F, N=3, T=200, dt=0.0005, initial_conditions = np.array([0]), t_res = 200):
    seed = int(F[-1])
    np.random.seed(seed)
    F = F[0]
    t_span = (0, T)
    if initial_conditions.sum() == 0:
        initial_conditions = np.random.normal(0, 1, N)
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(
        lorenz63,
        t_span,
        initial_conditions,
        # args=(F,),
        t_eval=t_eval,
        method='LSODA',
        rtol=1e-9,  # Relative tolerance
        atol=1e-12,  # Absolute tolerance
    )
    return sol.y.T[::t_res, :]
