"""
simulate.py
-----------
Operator-splitting (Lie-Trotter) reaction-diffusion time stepper.

Splitting scheme per timestep:
  1. Compute ∇²u  (Neumann Laplacian)
  2. Diffusion half-step:  u* = u + dt · D · ∇²u     (explicit Euler, per cell)
  3. Reaction full-step:   u' = RK4(u*, dt)            (handles stiff kinetics)

Stability requirement for the diffusion step:
    D · dt / dx² ≤ 0.5   (with dx = 1)
    → DT = 0.05, D_max = 10 ⟹ 10 · 0.05 = 0.5  (borderline stable)

Splitting error is O(dt) — acceptable here because spatial gradients and
reactions operate on well-separated scales.
"""

import numpy as np
from numba import njit, prange

from src.rd.laplacian import laplacian_neumann
from src.ode.emt_ode import ode_system

STATE_SIZE = 7   # number of chemical species


@njit
def cell_rhs(x: np.ndarray) -> np.ndarray:
    """ODE RHS without time argument, for use inside RK4."""
    return ode_system(x, 0.0)


@njit
def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """Classical RK4 step for a single grid cell."""
    k1 = cell_rhs(state)
    k2 = cell_rhs(state + 0.5 * dt * k1)
    k3 = cell_rhs(state + 0.5 * dt * k2)
    k4 = cell_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit(parallel=True, fastmath=True)
def simulate(grid: np.ndarray, dt: float, steps: int,
             D: np.ndarray, save_every: int):
    """
    Run the full 2D RD simulation.

    Parameters
    ----------
    grid       : (nx, ny, ns) initial condition — modified in-place
    dt         : timestep
    steps      : total number of timesteps
    D          : (ns,) diffusion coefficient per species
    save_every : save a snapshot every this many steps

    Returns
    -------
    frames : (n_frames, nx, ny, ns) saved concentration fields
    mass   : (n_frames,) total summed mass at each saved frame
    """
    nx, ny, ns = grid.shape
    lap      = np.zeros_like(grid)
    tmp      = np.zeros_like(grid)
    n_frames = steps // save_every + 1

    frames = np.zeros((n_frames, nx, ny, ns), dtype=np.float64)
    mass   = np.zeros(n_frames, dtype=np.float64)

    frames[0] = grid.copy()
    mass[0]   = np.sum(grid)
    f_idx     = 1

    for t in range(1, steps + 1):
        # Step 1: Laplacian of current grid
        laplacian_neumann(grid, lap)

        # Steps 2+3: diffusion then reaction, parallelised over rows
        for i in prange(nx):
            for j in range(ny):
                after_diff = np.empty(ns, dtype=np.float64)
                for k in range(ns):
                    after_diff[k] = grid[i, j, k] + dt * D[k] * lap[i, j, k]

                after_rxn = rk4_step(after_diff, dt)

                # Clamp: concentrations must be non-negative
                for k in range(ns):
                    tmp[i, j, k] = after_rxn[k] if after_rxn[k] > 0.0 else 0.0

        grid[:, :, :] = tmp[:, :, :]

        if t % save_every == 0:
            frames[f_idx] = grid.copy()
            mass[f_idx]   = np.sum(grid)
            f_idx += 1

    return frames, mass
