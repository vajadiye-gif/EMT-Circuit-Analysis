"""
emt_ode.py
----------
Numba-JIT 7-variable ODE for the SNAIL/ZEB/miR200/miR34 EMT circuit.

State vector  x = [miR200, mZEB, ZEB, SNAIL, mSNAIL, miR34, I]

miRNA-mRNA binding uses the full combinatorial model (C6 for miR200/mZEB,
C2 for miR34/mSNAIL). Hill functions include a leakage floor.
"""

import numpy as np
from numba import njit

# ── Binomial coefficients for miRNA-mRNA site binding ────────────────
C6 = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])   # C(6,i)
C2 = np.array([1.0, 2.0, 1.0])                             # C(2,i)

L_arr       = np.array([1.0, 0.6, 0.3, 0.1, 0.05, 0.05, 0.05])
gamma_mRNA  = np.array([0.0, 0.04, 0.2, 1.0, 1.0, 1.0, 1.0])
gamma_miRNA = np.array([0.0, 0.005, 0.05, 0.5, 0.5, 0.5, 0.5])


@njit
def hill_shifted(val, threshold, n, leakage):
    """Inhibitory Hill function with a leakage floor."""
    H = 1.0 / (1.0 + (val / threshold) ** n)
    return H + leakage * (1.0 - H)


@njit
def ode_system(x, t):
    """
    Returns dx/dt for the 7-variable EMT ODE.
    Signature is (x, t) for compatibility with scipy.integrate.odeint.
    x[6] = I is treated as a constant (dI/dt = 0).
    """
    # ── Kinetic parameters ─────────────────────────────────────────
    g_miR34 = 1.35e3;  g_mSNAIL = 90.0;   g_SNAIL = 0.1e3
    g_miR200 = 2.1e3;  g_mZEB = 11.0;     g_ZEB = 0.1e3

    k_miR34 = 0.05;    k_mSNAIL = 0.5;    k_SNAIL = 0.125
    k_miR200 = 0.05;   k_mZEB = 0.5;      k_ZEB = 0.1

    t_miR34_SNAIL = 300e3;  t_mSNAIL_SNAIL = 200e3
    t_miR34_ZEB = 600e3;    t_miR34 = 10e3;    t_mSNAIL_I = 50e3
    t_miR200_ZEB = 220e3;   t_miR200_SNAIL = 180e3
    t_mZEB_ZEB = 25e3;      t_mZEB_SNAIL = 180e3;  t_miR200 = 10e3

    n_miR34_SNAIL = 1;  n_miR34_ZEB = 1;   n_mSNAIL_SNAIL = 1;  n_mSNAIL_I = 1
    n_miR200_ZEB = 3;   n_miR200_SNAIL = 2; n_mZEB_ZEB = 2;      n_mZEB_SNAIL = 2

    l_miR34_SNAIL = 0.1;   l_mSNAIL_SNAIL = 0.1;  l_miR34_ZEB = 0.2;  l_mSNAIL_I = 10.0
    l_miR200_ZEB = 0.1;    l_miR200_SNAIL = 0.1;   l_mZEB_ZEB = 7.5;   l_mZEB_SNAIL = 10.0

    # ── miR200-mZEB binding (6 sites) ─────────────────────────────
    degrad_miR200 = 0.0;  degrad_mZEB = 0.0;  trans_mZEB = 0.0
    ratio_miR200  = x[0] / t_miR200
    denom6        = (1.0 + ratio_miR200) ** 6
    for i in range(7):
        fac = ratio_miR200 ** i / denom6
        degrad_miR200 += gamma_miRNA[i] * C6[i] * i * fac
        degrad_mZEB   += gamma_mRNA[i]  * C6[i] * fac
        trans_mZEB    += L_arr[i]       * C6[i] * fac

    # ── miR34-mSNAIL binding (2 sites) ────────────────────────────
    degrad_miR34 = 0.0;  degrad_mSNAIL = 0.0;  trans_mSNAIL = 0.0
    ratio_miR34  = x[5] / t_miR34
    denom2       = (1.0 + ratio_miR34) ** 2
    for i in range(3):
        fac = ratio_miR34 ** i / denom2
        degrad_miR34  += gamma_miRNA[i] * C2[i] * i * fac
        degrad_mSNAIL += gamma_mRNA[i]  * C2[i] * fac
        trans_mSNAIL  += L_arr[i]       * C2[i] * fac

    # ── Hill functions ─────────────────────────────────────────────
    H_miR200_ZEB   = hill_shifted(x[2], t_miR200_ZEB,   n_miR200_ZEB,   l_miR200_ZEB)
    H_miR200_SNAIL = hill_shifted(x[3], t_miR200_SNAIL, n_miR200_SNAIL, l_miR200_SNAIL)
    H_mZEB_ZEB     = hill_shifted(x[2], t_mZEB_ZEB,     n_mZEB_ZEB,     l_mZEB_ZEB)
    H_mZEB_SNAIL   = hill_shifted(x[3], t_mZEB_SNAIL,   n_mZEB_SNAIL,   l_mZEB_SNAIL)
    H_miR34_SNAIL  = hill_shifted(x[3], t_miR34_SNAIL,  n_miR34_SNAIL,  l_miR34_SNAIL)
    H_miR34_ZEB    = hill_shifted(x[2], t_miR34_ZEB,    n_miR34_ZEB,    l_miR34_ZEB)
    H_mSNAIL_SNAIL = hill_shifted(x[3], t_mSNAIL_SNAIL, n_mSNAIL_SNAIL, l_mSNAIL_SNAIL)
    H_mSNAIL_I     = hill_shifted(x[6], t_mSNAIL_I,    n_mSNAIL_I,    l_mSNAIL_I)

    # ── ODEs ───────────────────────────────────────────────────────
    dxdt = np.zeros(7)
    dxdt[0] = g_miR200 * H_miR200_ZEB * H_miR200_SNAIL - x[1]*degrad_miR200 - k_miR200*x[0]
    dxdt[1] = g_mZEB * H_mZEB_ZEB * H_mZEB_SNAIL       - x[1]*degrad_mZEB   - k_mZEB*x[1]
    dxdt[2] = g_ZEB * x[1] * trans_mZEB                 - k_ZEB * x[2]
    dxdt[3] = g_SNAIL * x[4] * trans_mSNAIL             - k_SNAIL * x[3]
    dxdt[4] = g_mSNAIL * H_mSNAIL_I * H_mSNAIL_SNAIL  - x[4]*degrad_mSNAIL - k_mSNAIL*x[4]
    dxdt[5] = g_miR34 * H_miR34_ZEB * H_miR34_SNAIL    - x[4]*degrad_miR34  - k_miR34*x[5]
    dxdt[6] = 0.0
    return dxdt
