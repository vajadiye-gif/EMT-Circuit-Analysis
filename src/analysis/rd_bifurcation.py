"""
rd_bifurcation.py
-----------------
Full 2D RD quasi-static bifurcation sweep.

For each value of I the entire NxN grid is evolved to steady state
before recording the target cell's concentrations.  Three sweep arms
(forward, backward, mid-backward) reveal the tristability of the
spatially-extended system, mirroring the single-cell ODE sweeps in
src/ode/bifurcation.py.

Runtime on a 50×50 grid with T_END=500, DT=0.05, Numba parallel:
    ~2–5 s per run × 150 total runs ≈ 5–12 min on a typical Colab CPU.
    Reduce T_END or N_I to speed up exploratory runs.
"""

import numpy as np
import time

from src.rd.initialise import build_grid
from src.rd.simulate import simulate


def _run_rd_single(grid_ic, dt, steps, D, save_every):
    """Run one RD simulation and return the final spatial state."""
    frames, _ = simulate(grid_ic, dt, steps, D, save_every)
    return frames[-1]   # (nx, ny, ns)


def sweep_rd(I_values: np.ndarray, grid_ic: np.ndarray,
             row: int, col: int,
             dt: float, steps: int, D: np.ndarray,
             save_every: int, state_size: int):
    """
    Quasi-static RD sweep: previous steady-state field seeds next run.

    Parameters
    ----------
    I_values   : 1-D array of I values
    grid_ic    : (nx, ny, ns) initial spatial IC for the first step
    row, col   : target cell to record
    dt, steps  : time step and total steps per run
    D          : (ns,) diffusion coefficients
    save_every : passed to simulate() — use steps to save only final frame
    state_size : number of species (ns)

    Returns
    -------
    results    : (len(I_values), state_size) target-cell steady states
    grid       : final spatial state (for continuation into the next arm)
    """
    results = np.empty((len(I_values), state_size))
    grid = grid_ic.copy()

    for idx, I_val in enumerate(I_values):
        t0 = time.time()
        grid[:, :, 6] = I_val                              # pin I uniformly
        final_grid = _run_rd_single(grid, dt, steps, D, save_every)
        results[idx] = final_grid[row, col, :]
        grid = final_grid                                   # continue from SS

        if idx % 10 == 0 or idx == len(I_values) - 1:
            print(f"  [{idx+1:3d}/{len(I_values)}]  I={I_val/1e3:.1f}k  "
                  f"mZEB={results[idx,1]:.2e}  SNAIL={results[idx,3]:.2e}  "
                  f"({time.time()-t0:.1f}s)")

    return results, grid


def run_rd_hysteresis(grid_size: int, D: np.ndarray,
                      dt: float, t_end_ss: float, state_size: int,
                      I_min: float = 20e3, I_max: float = 120e3,
                      I_mid: float = 65e3, n_I: int = 50,
                      row: int = None, col: int = None) -> dict:
    """
    Three-arm RD bifurcation sweep.

    Returns dict with keys:
        I_fwd, I_bwd, I_mid          — I-value arrays
        res_fwd, res_bwd, res_mid    — (n_I, state_size) target-cell steady states
    """
    if row is None: row = grid_size // 2
    if col is None: col = grid_size // 2

    steps_ss = int(t_end_ss / dt)
    I_fwd      = np.linspace(I_min, I_max, n_I)
    I_bwd      = np.linspace(I_max, I_min, n_I)
    I_mid_back = np.linspace(I_mid, I_min, n_I)

    print(f"=== Forward sweep (E→M): I={I_min/1e3:.0f}k → {I_max/1e3:.0f}k ===")
    g0 = build_grid(grid_size, I_min)
    res_fwd, g_end_fwd = sweep_rd(I_fwd, g0, row, col,
                                   dt, steps_ss, D, steps_ss, state_size)

    print(f"\n=== Backward sweep (M→E): I={I_max/1e3:.0f}k → {I_min/1e3:.0f}k ===")
    res_bwd, _ = sweep_rd(I_bwd, g_end_fwd, row, col,
                           dt, steps_ss, D, steps_ss, state_size)

    print(f"\n=== Mid-backward sweep: I={I_mid/1e3:.0f}k → {I_min/1e3:.0f}k ===")
    g_mid = build_grid(grid_size, I_mid)
    g_mid = _run_rd_single(g_mid, dt, steps_ss, D, steps_ss)  # relax to SS first
    res_mid, _ = sweep_rd(I_mid_back, g_mid, row, col,
                           dt, steps_ss, D, steps_ss, state_size)

    return dict(I_fwd=I_fwd, I_bwd=I_bwd, I_mid=I_mid_back,
                res_fwd=res_fwd, res_bwd=res_bwd, res_mid=res_mid)
