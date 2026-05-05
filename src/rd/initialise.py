"""
initialise.py
-------------
Grid initialisation helpers for the 2D RD simulation.

Default spatial layout:
  - Background  : E-state (low concentrations, uniform random)
  - Central patch (13×13 by default): M-state (high concentrations)
  - Channel k=6 : signal I, pinned uniformly across entire grid
"""

import numpy as np

STATE_SIZE = 7

# Canonical species index map — used by notebooks for readable indexing
SPECIES_MAP = {
    "miR200": 0,
    "mZEB":   1,
    "ZEB":    2,
    "SNAIL":  3,
    "mSNAIL": 4,
    "miR34":  5,
}


def build_grid(grid_size: int,
               I_val: float,
               high_min: float = 110e3,
               high_max: float = 120e3,
               low_min:  float = 20e3,
               low_max:  float = 50e3,
               half_patch: int = 6,
               seed: int = 42) -> np.ndarray:
    """
    Build a (grid_size, grid_size, STATE_SIZE) initial condition.

    Parameters
    ----------
    grid_size  : N for the N×N spatial grid
    I_val      : input signal I, uniform across the whole grid
    half_patch : half-width of the central M-state patch; patch = 2*half+1 cells
    seed       : random seed — use the same value across sweep arms for
                 identical spatial ICs (only I differs between runs)

    Returns
    -------
    g : (grid_size, grid_size, STATE_SIZE) float64 array
    """
    np.random.seed(seed)
    g = np.zeros((grid_size, grid_size, STATE_SIZE), dtype=np.float64)

    c = grid_size // 2  # grid centre

    for k in range(STATE_SIZE - 1):   # k = 0..5  (skip I channel)
        # Background: E-state
        g[:, :, k] = np.random.uniform(low_min, low_max, (grid_size, grid_size))
        # Central patch: M-state
        g[c - half_patch : c + half_patch + 1,
          c - half_patch : c + half_patch + 1, k] = np.random.uniform(
              high_min, high_max, (2*half_patch + 1, 2*half_patch + 1))

    # Signal I — uniform, constant
    g[:, :, 6] = I_val

    # Tiny symmetry-breaking noise
    g[:, :, :6] += np.random.rand(grid_size, grid_size, 6) * 0.5

    return g
