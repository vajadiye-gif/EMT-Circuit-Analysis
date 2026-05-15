# EMT Computational Biophysics

> **Computational study of Epithelial–Mesenchymal Transition (EMT) via the SNAIL/ZEB/miR-200/miR-34 gene regulatory circuit — bifurcation analysis, 2D reaction-diffusion simulation, basin-of-attraction mapping, and Random Matrix Theory spectral analysis.**

---

## Table of Contents

1. [Overview](#overview)
2. [Scientific Background](#scientific-background)
3. [Repository Structure](#repository-structure)
4. [Installation & Setup](#installation--setup)
5. [Quick Start (Google Colab)](#quick-start-google-colab)
6. [Notebooks Guide](#notebooks-guide)
7. [Source Module Reference](#source-module-reference)
8. [Mathematical Formulation](#mathematical-formulation)
9. [Key Results & Figures](#key-results--figures)
10. [Parameter Reference](#parameter-reference)
11. [Numerical Methods & Stability](#numerical-methods--stability)
12. [Extending the Model](#extending-the-model)
13. [Citation & Acknowledgements](#citation--acknowledgements)

---

## Overview

This repository implements a full computational pipeline for studying **Epithelial–Mesenchymal Transition (EMT)** — a central mechanism in cancer metastasis and embryonic development. The regulatory network considered here is the minimal core circuit involving four key molecular players:

- **SNAIL** — transcription factor that drives EMT
- **ZEB** — transcription factor that maintains the mesenchymal state
- **miR-200** — microRNA that stabilises the epithelial state by repressing ZEB
- **miR-34** — microRNA that stabilises the epithelial state by repressing SNAIL

The circuit exhibits **tristability** at the single-cell ODE level (Epithelial, Mesenchymal, and Hybrid E/M steady states) and **bistability** in the spatially-extended 2D reaction-diffusion model, where diffusion destroys the Hybrid branch. This repository reproduces and analyses that distinction systematically.

**Key computational contributions:**

| Module | What it does |
|---|---|
| `src/ode/` | Numba-JIT 7-variable ODE with combinatorial miRNA–mRNA binding; quasi-static hysteresis sweeps |
| `src/rd/` | Operator-splitting RD time stepper on a 50×50 grid; Numba-parallel Neumann Laplacian |
| `src/analysis/` | Basin-of-attraction mapping; RD bifurcation sweep; Random Matrix Theory spectral analysis |
| `src/utils/` | Dark-theme Matplotlib helpers; reusable `ipywidgets` Play/Slider factory |

---

## Scientific Background

### The EMT Circuit

EMT is the process by which epithelial cells lose their polarity and cell–cell adhesion, acquiring a motile mesenchymal phenotype. The transition is governed by a double-negative feedback loop between transcription factors (SNAIL, ZEB) and microRNAs (miR-200, miR-34), forming a bistable switch.

```
miR-200 ──┤ ZEB ──┤ miR-200    (double-negative loop → bistability)
miR-34  ──┤ SNAIL ──┤ miR-34   (double-negative loop → bistability)
SNAIL ──→ ZEB                  (positive feed-forward)
```

<!-- 
  FIGURE PLACEHOLDER
  Replace with: results/figures/circuit_diagram.png
  Suggested: a clean circular diagram of the 4 nodes with arrow types (→ activation, ─┤ repression)
-->
![Circuit Diagram](https://github.com/vajadiye-gif/EMT-Circuit-Analysis/blob/main/results/figures/circuit_diagram.png)

### State Vector

The model tracks 7 quantities per cell:

| Index | Species | Biological role |
|---|---|---|
| 0 | miR-200 | miRNA repressing ZEB (6 binding sites) |
| 1 | mZEB | ZEB mRNA |
| 2 | ZEB | EMT master transcription factor |
| 3 | SNAIL | EMT master transcription factor |
| 4 | mSNAIL | SNAIL mRNA |
| 5 | miR-34 | miRNA repressing SNAIL (2 binding sites) |
| 6 | I | External input signal (TGF-β proxy, pinned constant) |

### Tristability vs Bistability

The ODE-only single-cell model admits **three coexisting steady states**:
- **E** (Epithelial): high miR-200/miR-34, low ZEB/SNAIL
- **M** (Mesenchymal): low miR-200/miR-34, high ZEB/SNAIL
- **Hybrid E/M**: intermediate concentrations of all species

When diffusion is added (2D RD model), the Hybrid branch is **destroyed by spatial mixing**. The spatially-extended system is bistable (E and M only), consistent with experimental observations that the Hybrid state is transient and spatially localised.

---

## Repository Structure

```
emt-computational-biophysics/
│
├── notebooks/                             # Jupyter notebooks — run in order
│   ├── 01_bifurcation_analysis.ipynb      # ODE tristability + hysteresis sweeps
│   ├── 02_mZEB_concentration_evolution.ipynb  # Basin of attraction mapping
│   ├── 03_reaction_diffusion_simulation.ipynb # 2D RD simulation + RD bifurcation
│   └── 04_rmt_eigenvalue_analysis.ipynb   # Random Matrix Theory spectral analysis
│
├── src/                                   # Importable Python source modules
│   ├── __init__.py
│   ├── ode/
│   │   ├── __init__.py
│   │   ├── emt_ode.py                    # Numba-JIT 7-variable ODE + Hill functions
│   │   └── bifurcation.py               # Quasi-static continuation (single-cell)
│   ├── rd/
│   │   ├── __init__.py
│   │   ├── laplacian.py                 # Parallel Neumann Laplacian (ghost-node)
│   │   ├── simulate.py                  # Operator-splitting RD time stepper
│   │   └── initialise.py               # Grid IC helpers (E/M patch)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── basin_of_attraction.py       # Attractor perturbation sweep
│   │   ├── rmt_analysis.py              # RMT eigenvalue PDF via Gaussian KDE
│   │   └── rd_bifurcation.py            # Full RD quasi-static bifurcation sweep
│   └── utils/
│       ├── __init__.py
│       ├── plotting.py                  # Dark-theme rcParams + bifurcation helpers
│       └── widgets.py                   # Play/IntSlider ipywidgets factory
│
├── results/
│   ├── figures/                          # Saved .png outputs (committed)
│   └── data/                             # .npy/.npz simulation frames (gitignored)
│
├── docs/
│   └── circuit_description.md            # Biology background + parameter table
│
├── requirements.txt
└── .gitignore
```

---

## Installation & Setup

### Prerequisites

- Python ≥ 3.9
- For local runs: a multi-core CPU is strongly recommended (the RD solver is Numba-parallel)
- For Colab: a standard T4/CPU runtime works; GPU is not used

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/emt-computational-biophysics.git
cd emt-computational-biophysics

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make src/ importable from notebooks (run once from repo root)
pip install -e .
```

`requirements.txt` covers:

```
numpy
scipy
matplotlib
numba
ipywidgets
```

> **Note on Numba:** The first call to any `@njit`-decorated function triggers JIT compilation — expect a 10–30 s overhead on the very first run. Subsequent calls use the cached binary.

### Google Colab Setup

Open any notebook in Colab, then run the setup cell at the top:

```python
# Mount Drive (optional — for saving results)
from google.colab import drive
drive.mount('/content/drive')

# Clone repo if not already present
import os
if not os.path.isdir('emt-computational-biophysics'):
    !git clone https://github.com/<your-username>/emt-computational-biophysics.git

os.chdir('emt-computational-biophysics')

# Install requirements
!pip install -q numba ipywidgets

# Enable custom widgets (required for Play button in Colab)
from google.colab import output
output.enable_custom_widget_manager()

# Add src/ to Python path
import sys
sys.path.insert(0, '.')
```

---

## Quick Start (Google Colab)

The fastest path to a result:

```python
# ── Single-cell bifurcation diagram ─────────────────────────────────
from src.ode.bifurcation import run_hysteresis_sweeps
from src.utils.plotting import set_dark_theme, plot_bifurcation

set_dark_theme()
data = run_hysteresis_sweeps(n_points=1000)

plot_bifurcation(
    data['I_fwd'], data['I_bwd'], data['I_mid'],
    data['res_fwd'][:, 1], data['res_bwd'][:, 1], data['res_mid'][:, 1],   # mZEB
    data['res_fwd'][:, 3], data['res_bwd'][:, 3], data['res_mid'][:, 3],   # SNAIL
    title_suffix="| ODE 7-variable",
    save_path="results/figures/bifurcation_ode.png"
)
```

Expected runtime: **~2–3 minutes** on Colab CPU.

```python
# ── 2D Reaction-Diffusion simulation ────────────────────────────────
import numpy as np
from src.rd.initialise import build_grid
from src.rd.simulate import simulate

DT         = 0.05
GRID_SIZE  = 50
STEPS      = 20_000
SAVE_EVERY = 200
D = np.array([0.1]*6 + [0.0])    # species 6 (I) does not diffuse

grid = build_grid(GRID_SIZE, I_val=65e3)
frames, mass = simulate(grid, DT, STEPS, D, SAVE_EVERY)

print(f"Saved {frames.shape[0]} frames, shape {frames.shape}")
```

Expected runtime: **~3–8 minutes** on Colab CPU (Numba-parallel).

---

## Notebooks Guide

Run the notebooks **in order** — later notebooks import variables computed in earlier ones (or re-run the relevant `src/` functions).

### `01_bifurcation_analysis.ipynb` — ODE Tristability

**What it does:**
- Runs three quasi-static hysteresis sweeps (forward E→M, backward M→E, mid-backward) over the TGF-β proxy signal I ∈ [20k, 120k]
- Plots the bifurcation diagram: mZEB steady state vs I, and mZEB vs SNAIL phase portrait
- Demonstrates the tristable regime and the two saddle-node bifurcation points

**Key parameters to tune:**

| Parameter | Default | Effect |
|---|---|---|
| `n_points` | 1000 | Sweep resolution; reduce to 200 for fast preview |
| `I_min / I_max` | 20k / 120k | Signal range |
| `I_mid` | 65k | Starting point for hybrid branch |
| `t_end` (per step) | 7500 | Integration time; increase if not reaching SS |

<!-- 
  FIGURE PLACEHOLDER
  Replace with: results/figures/bifurcation_ode.png
  Content: two-panel — mZEB vs I (left), mZEB vs SNAIL phase portrait (right)
  Three coloured branches: blue (E), red (M), green (Hybrid)
-->
![Tristability Portrait](https://github.com/vajadiye-gif/EMT-Circuit-Analysis/blob/main/results/figures/tristability_mZEB_vs_SNAIL.png)
---

### `02_mZEB_concentration_evolution.ipynb` — Basin of Attraction Mapping

**What it does:**
- Extracts the three attractor state vectors at a user-chosen I value
- Applies 35 multiplicative perturbation scales (×0.1 to ×3.0) to each attractor
- Integrates all 105 trajectories and tracks whether they return to their home basin
- Plots mZEB time series coloured by originating state, with a scatter plot of final values vs perturbation scale

**Interpretation:** The basin robustness plot reveals that:
- The **M basin** is widest — a 3× perturbation still returns to M
- The **E basin** is narrower — moderate perturbations suffice to escape
- The **Hybrid basin** is the smallest — it is easily destabilised

<!-- 
  FIGURE PLACEHOLDER
  Replace with: results/figures/basin_robustness.png
  Content: left panel = 105 mZEB trajectories (blue/red/green); right = scatter of final mZEB vs perturbation factor
-->
![Basin of Attraction](https://github.com/vajadiye-gif/EMT-Circuit-Analysis/blob/main/results/figures/mZEB_evolution.png)

---

### `03_reaction_diffusion_simulation.ipynb` — 2D Spatial EMT

**What it does:**
- Initialises a 50×50 grid with an M-state patch (13×13) embedded in an E-state background
- Runs the operator-splitting RD solver for ~20,000 steps (t = 1000 a.u.)
- Animates concentration heatmaps for all 6 dynamic species using ipywidgets Play
- Runs the full three-arm **RD bifurcation sweep** and overlays it on the ODE bifurcation to show that the Hybrid branch disappears
- Tracks total system mass over time to validate conservation

**Two key observations:**
1. The M-state patch **expands** over time for I in the bistable regime
2. The RD bifurcation diagram shows only **two branches** (E and M) — the Hybrid branch is absent

<!-- 
  FIGURE PLACEHOLDER
  Replace with: results/figures/rd_heatmap_sequence.png
  Content: 4–6 heatmap panels (e.g. ZEB concentration) at t=0, 250, 500, 750, 1000
  showing the M-patch expanding from the centre
-->
> 📷 *[RD heatmap time sequence — add `results/figures/rd_heatmap_sequence.png` here]*

<!-- 
  FIGURE PLACEHOLDER (optional)
  Replace with: results/figures/rd_bifurcation_comparison.png
  Content: overlay of ODE 3-branch diagram (faint) vs RD 2-branch diagram (bold)
-->
![RD Bistability](https://github.com/vajadiye-gif/EMT-Circuit-Analysis/blob/main/results/figures/full_RD_bistability.png)

---

### `04_rmt_eigenvalue_analysis.ipynb` — Random Matrix Theory Analysis

**What it does:**
- Treats the N×N ZEB concentration field at each saved frame as a (symmetrised) random matrix
- Computes eigenvalues via `numpy.linalg.eigvalsh` for all frames
- Estimates the eigenvalue PDF via Gaussian KDE
- Animates the spectral evolution — the distribution transitions from a broad semi-circular bulk (disordered early state) to a distribution with a large outlier eigenvalue λ_max as the M-patch organises spatial structure

**Physical meaning:** The emergence of a rank-1 outlier λ_max signals the development of long-range spatial correlations in the concentration field — a hallmark of the M-patch expanding coherently.

<!-- 
  FIGURE PLACEHOLDER
  Replace with: results/figures/rmt_eigenvalue_evolution.png
  Content: stacked KDE plots at 5–6 time points, or a 2D density plot (frame vs eigenvalue)
  showing the outlier separating from the bulk
-->
> 📷 *[RMT eigenvalue evolution — add `results/figures/rmt_eigenvalue_evolution.png` here]*

---

## Source Module Reference

### `src/ode/emt_ode.py`

Core ODE system. All functions are `@njit`-decorated for Numba JIT compilation.

```python
from src.ode.emt_ode import ode_system, hill_shifted

# Evaluate RHS at a state vector
x0 = np.zeros(7); x0[6] = 65e3    # I = 65k
dxdt = ode_system(x0, 0.0)        # returns shape-(7,) array
```

**`hill_shifted(val, threshold, n, leakage)`**
Inhibitory Hill function with a leakage floor — prevents species concentrations from reaching exactly zero:

```
H(val) = 1/(1 + (val/threshold)^n)
H_leaky = H + leakage * (1 - H)
```

**miRNA–mRNA binding model:**
Combinatorial binding across multiple sites is handled by summing over occupation states $i = 0, \ldots, n_{\text{sites}}$ weighted by $\binom{n}{i}$, giving effective degradation rates that depend on miRNA concentration non-linearly.

---

### `src/ode/bifurcation.py`

```python
from src.ode.bifurcation import run_hysteresis_sweeps, sweep

# Full three-arm sweep (recommended)
data = run_hysteresis_sweeps(n_points=500, I_min=20e3, I_max=120e3)

# Custom single sweep
I_vals = np.linspace(20e3, 120e3, 300)
x0 = np.zeros(7)
results = sweep(I_vals, x0, t_end=7500, n_steps=500)
# results shape: (300, 7)
```

---

### `src/rd/laplacian.py`

```python
from src.rd.laplacian import laplacian_neumann

# grid: (nx, ny, ns) float64
# lap:  (nx, ny, ns) float64, pre-allocated
laplacian_neumann(grid, lap)   # writes result in-place
```

**Boundary condition:** Neumann (zero-flux) implemented via the ghost-node method. The missing neighbour at each boundary cell is replaced by the nearest interior cell, enforcing $\partial u / \partial n = 0$ exactly.

**5-point stencil (interior):**
```
∇²u[i,j] = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] − 4·u[i,j]
```

---

### `src/rd/simulate.py`

```python
from src.rd.simulate import simulate
import numpy as np

D = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])  # I does not diffuse
frames, mass = simulate(grid, dt=0.05, steps=20000, D=D, save_every=200)
# frames: (101, 50, 50, 7)
# mass:   (101,)   — total concentration summed over grid and species
```

**Operator-splitting scheme (Lie-Trotter, per timestep):**
1. Compute $\nabla^2 u$ via `laplacian_neumann`
2. Diffusion explicit Euler half-step: $u^* = u + \Delta t \cdot D \cdot \nabla^2 u$
3. Reaction RK4 full step: $u' = \text{RK4}(u^*, \Delta t)$

---

### `src/rd/initialise.py`

```python
from src.rd.initialise import build_grid, SPECIES_MAP

# Build a 50×50 IC with I = 65k and a 13×13 M-patch at the centre
grid = build_grid(grid_size=50, I_val=65e3, half_patch=6, seed=42)
# grid shape: (50, 50, 7)

# Readable species indexing
zeb_idx = SPECIES_MAP['ZEB']   # → 2
```

---

### `src/analysis/basin_of_attraction.py`

```python
from src.analysis.basin_of_attraction import attractor_perturbation_sweep

trajectories, labels, perturbations, final_vals, t_span, attractor_info = \
    attractor_perturbation_sweep(
        I_fixed_k=65,              # I in units of 10³
        res_fwd=data['res_fwd'], I_forward=data['I_fwd'],
        res_bwd=data['res_bwd'], I_backward=data['I_bwd'],
        res_mid=data['res_mid'], I_mid_back=data['I_mid'],
        n_perturb=35,
        perturb_lo=0.1, perturb_hi=3.0,
    )
```

---

### `src/analysis/rmt_analysis.py`

```python
from src.analysis.rmt_analysis import compute_eigenvalue_pdfs

all_eigvals, x_grid, all_kde = compute_eigenvalue_pdfs(
    frames,
    species_idx=2,    # ZEB
    bw_method=0.15,
    n_grid=500
)
# all_eigvals: (n_frames, 50)
# all_kde:     (n_frames, 500)
```

---

### `src/analysis/rd_bifurcation.py`

```python
from src.analysis.rd_bifurcation import run_rd_hysteresis
import numpy as np

D = np.array([0.1]*6 + [0.0])
rd_data = run_rd_hysteresis(
    grid_size=50, D=D,
    dt=0.05, t_end_ss=500.0, state_size=7,
    I_min=20e3, I_max=120e3, I_mid=65e3,
    n_I=50
)
# Runtime: ~5–12 minutes on Colab CPU
```

---

### `src/utils/widgets.py`

```python
from src.utils.widgets import make_play_slider, make_species_toggle
from ipywidgets import interact
from IPython.display import display

play, slider = make_play_slider(n_frames=frames.shape[0], interval=150)

def show_frame(idx):
    # your plot function here
    pass

interact(show_frame, idx=slider)
display(play)
```

---

## Mathematical Formulation

### ODE System

The full system is:

$$
\dot{x}_i = f_i(x_0, \ldots, x_6), \quad i = 0, \ldots, 6
$$

**miR-200** (x₀):
$$\dot{x}_0 = g_{\text{miR200}} \cdot H(x_2, \theta_{200,Z}) \cdot H(x_3, \theta_{200,S}) - x_1 \cdot \gamma_{\text{miR200}}^{\text{eff}} - k_{\text{miR200}} x_0$$

**mZEB** (x₁):
$$\dot{x}_1 = g_{\text{mZEB}} \cdot H(x_2, \theta_{Z,Z}) \cdot H(x_3, \theta_{Z,S}) - x_1 \cdot \gamma_{\text{mZEB}}^{\text{eff}} - k_{\text{mZEB}} x_1$$

**ZEB** (x₂):
$$\dot{x}_2 = g_{\text{ZEB}} \cdot x_1 \cdot T_{\text{mZEB}}^{\text{eff}} - k_{\text{ZEB}} x_2$$

where $\gamma^{\text{eff}}$ and $T^{\text{eff}}$ are the effective degradation rates and translational efficiencies arising from the combinatorial miRNA–mRNA binding model (summed over all occupancy states of the binding sites). See `src/ode/emt_ode.py` for the complete expressions.

### Reaction-Diffusion System

$$
\frac{\partial \mathbf{u}}{\partial t} = \mathbf{D} \nabla^2 \mathbf{u} + \mathbf{f}(\mathbf{u})
$$

with **Neumann boundary conditions** $\partial u_k / \partial n = 0$ on all four boundaries (zero-flux — cells at the edge do not lose or gain material through the boundary). The signal channel ($k = 6$) has $D_6 = 0$ (pinned, no diffusion).

### Numerical Stability

The explicit Euler diffusion step requires:

$$
\frac{D_{\max} \cdot \Delta t}{\Delta x^2} \leq \frac{1}{2}
$$

With $\Delta x = 1$, $D_{\max} = 0.1$, the chosen $\Delta t = 0.05$ gives a CFL number of **0.005**, well within the stability bound.

---

## Key Results & Figures

<!--
  This section is a placeholder for figures from your simulation runs.
  Suggested figure set:

  1. results/figures/circuit_diagram.png
     — Hand-drawn or programmatic illustration of the 4-node regulatory circuit

  2. results/figures/bifurcation_ode.png
     — Two-panel: mZEB vs I (3 branches) | mZEB vs SNAIL phase portrait
     Generated by: notebook 01 or plot_bifurcation() in plotting.py

  3. results/figures/basin_robustness.png
     — 105 mZEB trajectories + scatter of final values vs perturbation scale
     Generated by: notebook 02

  4. results/figures/rd_ic.png
     — Initial condition heatmap showing the M-patch embedded in E-background
     Generated by: notebook 03

  5. results/figures/rd_heatmap_sequence.png
     — Time sequence of ZEB concentration heatmaps showing M-patch expansion
     Generated by: notebook 03

  6. results/figures/rd_bifurcation_comparison.png
     — ODE (3-branch) vs RD (2-branch) bifurcation overlay, showing Hybrid branch disappearance
     Generated by: notebook 03

  7. results/figures/rmt_eigenvalue_evolution.png
     — KDE eigenvalue PDFs at multiple timepoints; outlier emergence
     Generated by: notebook 04

  Replace each placeholder below with the corresponding Markdown image tag once figures are saved:
  ![caption](results/figures/filename.png)
-->

| Figure | Description | Notebook |
|---|---|---|
| `results/figures/bifurcation_ode.png` | ODE tristability — mZEB bifurcation diagram | 01 |
| `results/figures/basin_robustness.png` | Basin of attraction widths under perturbation | 02 |
| `results/figures/rd_heatmap_sequence.png` | ZEB field evolution on the 50×50 grid | 03 |
| `results/figures/rd_bifurcation_comparison.png` | ODE 3-branch vs RD 2-branch overlay | 03 |
| `results/figures/rmt_eigenvalue_evolution.png` | Eigenvalue PDF spectral transition | 04 |

---

## Parameter Reference

### Kinetic Parameters (ODE)

| Parameter | Value | Description |
|---|---|---|
| `g_miR200` | 2100 | miR-200 production rate |
| `g_mZEB` | 11 | mZEB production rate |
| `g_ZEB` | 100 | ZEB translation rate |
| `g_SNAIL` | 100 | SNAIL translation rate |
| `g_mSNAIL` | 90 | mSNAIL production rate |
| `g_miR34` | 1350 | miR-34 production rate |
| `k_miR200` | 0.05 | miR-200 degradation rate |
| `k_mZEB` | 0.5 | mZEB degradation rate |
| `k_ZEB` | 0.1 | ZEB degradation rate |
| `k_SNAIL` | 0.125 | SNAIL degradation rate |
| `k_miR34` | 0.05 | miR-34 degradation rate |

### Hill Thresholds

| Interaction | Threshold | Hill coefficient | Leakage |
|---|---|---|---|
| miR-200 → ZEB mRNA | 220k | 3 | 0.1 |
| miR-200 → SNAIL mRNA | 180k | 2 | 0.1 |
| ZEB → mZEB (auto) | 25k | 2 | 7.5 |
| SNAIL → mSNAIL (auto) | 200k | 1 | 0.1 |
| miR-34 → SNAIL mRNA | 300k | 1 | 0.1 |
| I → mSNAIL | 50k | 1 | 10.0 |

### RD Simulation Parameters

| Parameter | Value | Notes |
|---|---|---|
| `GRID_SIZE` | 50 | 50×50 spatial grid |
| `DT` | 0.05 | Timestep (explicit Euler diffusion) |
| `D` (species 0–5) | 0.1 | Equal diffusion for all molecular species |
| `D` (species 6, I) | 0.0 | Signal is spatially pinned |
| `half_patch` | 6 | M-patch half-width → 13×13 cells |
| `save_every` | 200 | Frame save interval (one frame per 10 a.u.) |

---

## Numerical Methods & Stability

### Why Operator Splitting?

The ODE kinetics (stiff, wide range of timescales) and the diffusion (explicit, stability-limited timestep) have different optimal solvers. Operator splitting lets each part use its ideal method:

- **Diffusion:** explicit Euler — cheap, first-order, but requires small $\Delta t$
- **Reaction:** RK4 — fourth-order accurate, handles the stiff Hill kinetics well per cell

The splitting error is $\mathcal{O}(\Delta t)$ (Lie-Trotter), acceptable here because the spatial gradients and reaction timescales are well-separated.

### Why Neumann Boundary Conditions?

Periodic BCs (used in, e.g., Gray-Scott models) are physically inappropriate for a finite tissue: they imply the domain wraps around. For cell diffusion in a bounded tissue, zero-flux boundaries (Neumann) correctly reflect the fact that molecular species cannot leave the tissue at the boundary.

### Numba Parallelisation

Both `laplacian_neumann` and the inner reaction-diffusion loop in `simulate` use `@njit(parallel=True)` with `prange`. On a 4-core Colab CPU this gives ~2–3× speedup over the serial implementation. The parallelisation is over spatial rows (interior Laplacian) and species channels (boundary cells), avoiding write conflicts.

---

## Extending the Model

Some natural extensions and starting points:

**1. Stochastic dynamics (Langevin SDE)**
```python
# Replace the deterministic ODE RHS with:
dx = f(x)*dt + sigma * np.sqrt(dt) * np.random.randn(7)
```
Adds intrinsic noise to explore how it widens or narrows the tristable region.

**2. Non-uniform diffusion / heterogeneous signal**
```python
# Build a spatially varying D field: (nx, ny, ns)
# Modify simulate() to use D[i,j,k] instead of a global D[k]
```

**3. Turing instability analysis**
Perform a linear stability analysis of the homogeneous steady state to find parameter regimes where diffusion-driven instability generates spatial patterns (requires $D_{\text{activator}} \ll D_{\text{inhibitor}}$).

**4. 3D geometry**
Replace the 2D Laplacian with a 3D version and initialise a spherical M-core in an E-medium — mimicking a tumour spheroid.

**5. Different boundary conditions**
Switch to **Dirichlet BCs** (fixed concentrations at the boundary, mimicking a maintained external signal gradient) by modifying `laplacian.py` to set boundary ghost nodes differently.

---

## Citation & Acknowledgements

If you use this code or the simulation framework in your research, please cite the original regulatory model:

> Lu, M., Jolly, M. K., Levine, H., Onuchic, J. N., & Ben-Jacob, E. (2013). MicroRNA-based regulation of epithelial–hybrid–mesenchymal fate determination. *PNAS*, 110(45), 18144–18149.

**Supervisor:** Ushasi Roy, IISER Pune  
**Project:** Semester project in computational biophysics, 2025–2026  
**Author:** Ved (BSMS Physics, IISER Pune)

---

<div align="center">
<sub>Built with NumPy · SciPy · Numba · Matplotlib · ipywidgets · Google Colab</sub>
</div>
