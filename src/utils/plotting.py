"""
plotting.py
-----------
Shared Matplotlib helpers used across notebooks.
Call set_dark_theme() once at the top of any dark-themed notebook.
"""

import matplotlib.pyplot as plt
import numpy as np

DARK_BG    = '#0a0a14'
GRID_ALPHA = 0.3


def set_dark_theme() -> None:
    """Apply dark background rcParams to all subsequent Matplotlib figures."""
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    DARK_BG,
        'axes.edgecolor':    '#333333',
        'xtick.color':       'white',
        'ytick.color':       'white',
        'axes.labelcolor':   'white',
        'text.color':        'white',
        'legend.framealpha': 0.15,
        'legend.labelcolor': 'white',
        'grid.color':        '#333333',
    })


def plot_bifurcation(I_fwd, I_bwd, I_mid,
                     mzeb_fwd, mzeb_bwd, mzeb_mid,
                     snail_fwd, snail_bwd, snail_mid,
                     title_suffix: str = "",
                     save_path: str = None) -> None:
    """
    Two-panel bifurcation plot: mZEB vs I (left) and mZEB vs SNAIL (right).

    Parameters
    ----------
    I_fwd/bwd/mid  : I-value arrays for the three sweep arms
    mzeb_*/snail_* : corresponding steady-state arrays
    title_suffix   : appended to panel titles (e.g. '| RD 50×50')
    save_path      : if given, save figure to this path at 200 dpi
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(I_fwd / 1e3, mzeb_fwd, 'b.', ms=1, label='Forward (E→M)')
    ax1.plot(I_bwd / 1e3, mzeb_bwd, 'r.', ms=1, label='Backward (M→E)')
    ax1.plot(I_mid / 1e3, mzeb_mid, 'g.', ms=1, label='Mid-backward (hybrid)')
    ax1.set_xlabel('Input Signal I (×10³)', fontsize=12)
    ax1.set_ylabel('mZEB (steady state)',   fontsize=12)
    ax1.set_title(f'Bifurcation Diagram {title_suffix}', fontsize=13)
    ax1.legend(fontsize=9, markerscale=8)
    ax1.grid(True, alpha=GRID_ALPHA)

    ax2 = axes[1]
    ax2.plot(snail_fwd, mzeb_fwd, 'b.', ms=1.5, label='Forward (E→M)')
    ax2.plot(snail_bwd, mzeb_bwd, 'r.', ms=1.5, label='Backward (M→E)')
    ax2.plot(snail_mid, mzeb_mid, 'g.', ms=1.5, label='Mid-backward (hybrid)')
    ax2.set_xlabel('SNAIL (steady state)', fontsize=12)
    ax2.set_ylabel('mZEB (steady state)',  fontsize=12)
    ax2.set_title(f'mZEB vs SNAIL {title_suffix}', fontsize=13)
    ax2.legend(fontsize=9, markerscale=8)
    ax2.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
