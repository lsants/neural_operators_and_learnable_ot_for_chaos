import numpy as np

def lorenz63(t, y, sigma, beta, rho) -> np.ndarray:
    dydt = np.zeros(3)
    dydt[0] = sigma * (y[1] - y[0])
    dydt[1] = y[0] * (rho - y[2]) - y[1]
    dydt[2] = y[0] * y[1] - beta * y[2]
    return dydt

def lorenz96(t, y, F) -> np.ndarray:
    N = len(y)
    dydt = np.zeros(N)
    for i in range(N):
        dydt[i] = (y[(i + 1) % N] - y[(i - 2) % N]) * y[(i - 1) % N] - y[i] + F
    return dydt

IVP_MAP = {
    'lorenz63': lorenz63,
    'lorenz96': lorenz96
}