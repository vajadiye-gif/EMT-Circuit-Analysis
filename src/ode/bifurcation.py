"""
bifurcation.py
--------------
Quasi-static continuation sweeps over the input signal I
for the single-cell (ODE-only) EMT model.

Usage
-----
    from src.ode.bifurcation import run_hysteresis_sweeps
    data = run_hysteresis_sweeps(n_points=1000)
    # data keys: I_fwd, I_bwd, I_mid, res_fwd, res_bwd, res_mid
"""

import numpy as np
from scipy.integrate import odeint
from src.ode.emt_ode import ode_system


def sweep(I_values: np.ndarray, x0: np.ndarray,
          t_end: float = 7500, n_steps: int = 500) -> np.ndarray:
    """
    Sweep I values using the previous steady state as the next IC.

    Parameters
    ----------
    I_values : 1-D array of I values to step through
    x0       : length-7 initial condition; x0[6] is overwritten each step
    t_end    : integration time per I value (a.u.)
    n_steps  : ODE time grid points per run

    Returns
    -------
    results : (len(I_values), 7) steady-state concentrations
    """
    results = np.empty((len(I_values), 7))
    state   = x0.copy()
    t_span  = np.linspace(0, t_end, n_steps)

    for idx, I_val in enumerate(I_values):
        state[6] = I_val
        sol = odeint(ode_system, state, t_span)
        state = sol[-1].copy()
        results[idx] = state

    return results


def run_hysteresis_sweeps(n_points: int = 1000,
                          I_min: float = 20e3,
                          I_max: float = 120e3,
                          I_mid: float = 65e3) -> dict:
    """
    Run the three sweep arms that reveal tristability.

    Returns dict with keys:
        I_fwd, I_bwd, I_mid         — I-value arrays
        res_fwd, res_bwd, res_mid   — (n_points, 7) steady-state arrays
    """
    I_forward  = np.linspace(I_min, I_max, n_points)
    I_backward = np.linspace(I_max, I_min, n_points)
    I_mid_back = np.linspace(I_mid, I_min, n_points)

    x0_low = np.zeros(7)
    _ = ode_system(np.zeros(7), 0.0)   # Numba JIT warm-up

    print("Forward sweep (E→M)...")
    res_fwd = sweep(I_forward, x0_low)

    print("Backward sweep (M→E)...")
    res_bwd = sweep(I_backward, res_fwd[-1])

    print("Mid-backward sweep (hybrid branch)...")
    x0_mid = np.zeros(7)
    x0_mid[6] = I_mid
    sol_mid = odeint(ode_system, x0_mid, np.linspace(0, 7500, 500))
    res_mid = sweep(I_mid_back, sol_mid[-1])

    return dict(I_fwd=I_forward, I_bwd=I_backward, I_mid=I_mid_back,
                res_fwd=res_fwd, res_bwd=res_bwd, res_mid=res_mid)
