"""
laplacian.py
------------
Numba-parallelised 2D Laplacian with Neumann (zero-flux) boundary conditions,
implemented via the ghost-node method: u_ghost = u_interior ⟹ ∂u/∂n = 0.

5-point stencil interior:
    ∇²u[i,j] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] − 4·u[i,j]

Boundary rows/columns replace the missing neighbour with the adjacent interior
value (doubled), which is equivalent to a zero-gradient ghost node.
Corners apply the ghost-node rule in both directions simultaneously.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def laplacian_neumann(grid: np.ndarray, lap: np.ndarray) -> None:
    """
    Compute ∇²u in-place, writing results into `lap`.

    Parameters
    ----------
    grid : (nx, ny, ns) — concentration array
    lap  : (nx, ny, ns) — output array, overwritten in-place
    """
    nx, ny, ns = grid.shape

    # ── Interior: parallelise over rows, no boundary conflicts ───────
    for i in prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(ns):
                lap[i, j, k] = (
                    grid[i+1, j, k] + grid[i-1, j, k] +
                    grid[i, j+1, k] + grid[i, j-1, k] -
                    4.0 * grid[i, j, k]
                )

    # ── Edges + corners: parallelise over species ─────────────────────
    for k in prange(ns):
        # Left edge (i=0)
        for j in range(1, ny - 1):
            lap[0, j, k] = (2.0*grid[1, j, k] +
                            grid[0, j+1, k] + grid[0, j-1, k] -
                            4.0*grid[0, j, k])
        # Right edge (i=nx-1)
        for j in range(1, ny - 1):
            lap[nx-1, j, k] = (2.0*grid[nx-2, j, k] +
                               grid[nx-1, j+1, k] + grid[nx-1, j-1, k] -
                               4.0*grid[nx-1, j, k])
        # Bottom edge (j=0)
        for i in range(1, nx - 1):
            lap[i, 0, k] = (grid[i+1, 0, k] + grid[i-1, 0, k] +
                            2.0*grid[i, 1, k] - 4.0*grid[i, 0, k])
        # Top edge (j=ny-1)
        for i in range(1, nx - 1):
            lap[i, ny-1, k] = (grid[i+1, ny-1, k] + grid[i-1, ny-1, k] +
                               2.0*grid[i, ny-2, k] - 4.0*grid[i, ny-1, k])
        # Corners: ghost nodes in both directions
        lap[0,    0,    k] = (2.0*grid[1,    0,    k] + 2.0*grid[0,    1,    k] - 4.0*grid[0,    0,    k])
        lap[nx-1, 0,    k] = (2.0*grid[nx-2, 0,    k] + 2.0*grid[nx-1, 1,    k] - 4.0*grid[nx-1, 0,    k])
        lap[0,    ny-1, k] = (2.0*grid[1,    ny-1, k] + 2.0*grid[0,    ny-2, k] - 4.0*grid[0,    ny-1, k])
        lap[nx-1, ny-1, k] = (2.0*grid[nx-2, ny-1, k] + 2.0*grid[nx-1, ny-2, k] - 4.0*grid[nx-1, ny-1, k])
