# EMT Computational Biophysics

Computational analysis of Epithelial–Mesenchymal Transition (EMT) using the
SNAIL/ZEB/miR200/miR34 core regulatory circuit.

## Repository structure

```
emt-computational-biophysics/
│
├── notebooks/                         # Jupyter notebooks — one per analysis module
│   ├── 01_bifurcation_analysis.ipynb
│   ├── 02_mZEB_concentration_evolution.ipynb
│   ├── 03_reaction_diffusion_simulation.ipynb
│   └── 04_rmt_eigenvalue_analysis.ipynb
│
├── src/                               # Importable Python source modules
│   ├── ode/
│   │   ├── emt_ode.py                # Numba-JIT 7-variable ODE + Hill functions
│   │   └── bifurcation.py           # Quasi-static continuation sweeps (single-cell)
│   ├── rd/
│   │   ├── laplacian.py             # Parallel Neumann Laplacian kernel
│   │   ├── simulate.py              # Operator-splitting RD time stepper
│   │   └── initialise.py           # Grid initialisation helpers (E/M patch IC)
│   ├── analysis/
│   │   ├── basin_of_attraction.py   # Multi-IC sweeps & basin mapping
│   │   ├── rmt_analysis.py          # RMT eigenvalue PDF via Gaussian KDE
│   │   └── rd_bifurcation.py        # Full RD quasi-static bifurcation sweep
│   └── utils/
│       ├── widgets.py               # Reusable Play/IntSlider ipywidgets factory
│       └── plotting.py              # Dark-theme rcParams & bifurcation plot helpers
│
├── results/
│   ├── figures/                      # Saved .png outputs (committed to repo)
│   └── data/                         # .npy/.npz simulation frames (gitignored — large)
│
├── docs/
│   └── circuit_description.md        # Biology background + parameter table
│
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install -r requirements.txt
```

Open the notebooks in order — each builds on shared `src/` modules.
