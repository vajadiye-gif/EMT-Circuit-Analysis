"""
rmt_analysis.py
---------------
Random Matrix Theory analysis of the ZEB concentration field across
simulation frames.

For each frame the NxN concentration matrix is symmetrised as
    X_sym = (X + X.T) / 2
so all eigenvalues are real. The eigenvalue distribution is then
estimated via Gaussian KDE with a user-specified bandwidth.

Interpreting the spectrum:
  - Early frames: broad, roughly semicircular spectrum (disordered state)
  - As the M-state patch expands, rank-1 structure emerges → large positive
    outlier λ_max separates from the bulk
  - Negative λ region shrinks as the field becomes more uniform
"""

import numpy as np
from scipy.stats import gaussian_kde


def compute_eigenvalue_pdfs(frames: np.ndarray,
                             species_idx: int = 2,
                             bw_method: float = 0.15,
                             n_grid: int = 500):
    """
    Precompute eigenvalues and KDE PDFs for all saved frames.

    Parameters
    ----------
    frames      : (n_frames, nx, ny, ns) simulation output array
    species_idx : channel to analyse; default 2 = ZEB
    bw_method   : Gaussian KDE bandwidth (Scott's rule ≈ 0.12 for N=50)
    n_grid      : number of points in the eigenvalue evaluation grid

    Returns
    -------
    all_eigvals : (n_frames, N) sorted eigenvalue arrays
    x_grid      : (n_grid,)    evaluation grid, symmetric around 0
    all_kde     : (n_frames, n_grid) probability densities
    """
    conc = frames[:, :, :, species_idx]
    n_frames, N, M = conc.shape
    assert N == M, "Concentration slice must be square (nx == ny)"

    all_eigvals = np.zeros((n_frames, N))
    for t in range(n_frames):
        X = conc[t].astype(float)
        X_sym = (X + X.T) / 2          # symmetrise → real eigenvalues
        all_eigvals[t] = np.linalg.eigvalsh(X_sym)  # sorted ascending

    x_max  = np.abs(all_eigvals).max() * 1.05
    x_grid = np.linspace(-x_max, x_max, n_grid)

    all_kde = np.zeros((n_frames, n_grid))
    for t in range(n_frames):
        kde = gaussian_kde(all_eigvals[t], bw_method=bw_method)
        all_kde[t] = kde(x_grid)

    return all_eigvals, x_grid, all_kde
