"""
basin_of_attraction.py
----------------------
Attractor perturbation sweep to map basin robustness.

Method:
  1. Pull the three known attractors (E, M, Hybrid) directly from the
     bifurcation sweep results at the chosen I value.
  2. Scale each attractor state vector by 35 perturbation factors (0.1–3.0).
  3. Integrate all 105 trajectories and check whether they return to their
     originating basin — testing basin robustness, not just basin existence.

This is distinct from a plain IC sweep: instead of scanning mZEB₀ from
scratch, we anchor around physically meaningful starting points (the
attractors themselves) and ask how far you can push before the trajectory
escapes to a different steady state.

Dependencies: run src/ode/bifurcation.py first to obtain
    res_fwd, res_bwd, res_mid, I_forward, I_backward, I_mid_back
"""

import numpy as np
from scipy.integrate import odeint
from src.ode.emt_ode import ode_system


def attractor_perturbation_sweep(
        I_fixed_k: float,
        res_fwd: np.ndarray, I_forward: np.ndarray,
        res_bwd: np.ndarray, I_backward: np.ndarray,
        res_mid: np.ndarray, I_mid_back: np.ndarray,
        n_perturb: int = 35,
        perturb_lo: float = 0.1,
        perturb_hi: float = 3.0,
        t_end: float = 15000,
        n_steps: int = 2000):
    """
    Perturb the three EMT attractors and integrate 105 trajectories.

    Parameters
    ----------
    I_fixed_k        : I value in units of 10³ (matches the widget slider)
    res_fwd/bwd/mid  : (n_I, 7) steady-state arrays from bifurcation sweeps
    I_forward/backward/mid_back : corresponding I-value arrays
    n_perturb        : number of perturbation scales per attractor
    perturb_lo/hi    : range of multiplicative perturbation factors
    t_end            : integration time per trajectory
    n_steps          : ODE time grid points

    Returns
    -------
    trajectories  : list of (n_steps,) mZEB time series — length 3*n_perturb
    labels        : np.ndarray of str ('E', 'M', 'Hybrid') per trajectory
    perturbations : (n_perturb,) perturbation scale array
    final_vals    : (3*n_perturb,) steady-state mZEB values
    t_span        : (n_steps,) time array
    attractor_info: list of (label, x0_base, color) — for downstream plotting
    """
    I_fixed = I_fixed_k * 1e3

    # ── Extract attractor state vectors at the closest sweep point ───
    x0_E      = res_fwd[np.argmin(np.abs(I_forward  - I_fixed))].copy(); x0_E[6]      = I_fixed
    x0_M      = res_bwd[np.argmin(np.abs(I_backward - I_fixed))].copy(); x0_M[6]      = I_fixed
    x0_hybrid = res_mid[np.argmin(np.abs(I_mid_back - I_fixed))].copy(); x0_hybrid[6] = I_fixed

    attractor_info = [
        ('E',      x0_E,      'royalblue'),
        ('M',      x0_M,      'tomato'),
        ('Hybrid', x0_hybrid, 'seagreen'),
    ]

    perturbations = np.linspace(perturb_lo, perturb_hi, n_perturb)
    t_span        = np.linspace(0, t_end, n_steps)

    trajectories = []
    labels_list  = []
    total        = len(attractor_info) * n_perturb
    count        = 0

    print(f"Integrating {total} trajectories at I = {I_fixed/1e3:.0f}×10³...")

    for label, x0_base, _ in attractor_info:
        for scale in perturbations:
            x0      = x0_base.copy()
            x0[:6] *= scale           # perturb all species simultaneously
            x0[6]   = I_fixed         # keep I pinned
            x0      = np.clip(x0, 0, None)   # enforce non-negativity

            sol = odeint(ode_system, x0, t_span, rtol=1e-8, atol=1e-10)
            trajectories.append(sol[:, 1])
            labels_list.append(label)
            count += 1
            if count % 10 == 0 or count == total:
                print(f"  [{count}/{total}] done")

    print("All integrations complete.")

    labels     = np.array(labels_list)
    final_vals = np.array([traj[-1] for traj in trajectories])

    print("\nBasin robustness summary:")
    for label, _, _ in attractor_info:
        mask   = labels == label
        median = np.median(final_vals[mask])
        spread = final_vals[mask].max() - final_vals[mask].min()
        print(f"  {label:6s} — median mZEB = {median:.2f},  spread = {spread:.4f}")

    return trajectories, labels, perturbations, final_vals, t_span, attractor_info
