#  imports
from IPython.display import Markdown, display
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from time import time
try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# param
ODE_SOLVE_METHOD = 'RK45'  # 'Radau', 'BDF', 'LSODA', 'RK45', etc.

# --- Physical constants ---
h = 6.62607015e-34        # Planck constant, J·s
hbar = h / (2 * np.pi)    # Reduced Planck constant, J·s
k_B = 1.380649e-23        # Boltzmann constant, J/K
c = 299792458.0           # Speed of light, m/s
amu = 1.66053906660e-27   # Atomic mass unit, kg

# --- Transition parameters ---
f_S12_to_P12 = 377.107463380e12        # Hz
f_S12F1_to_S12 = 4.271676631815181e9  # Hz
f_S12F1_to_S12F2 = 6.834682610904290e9  # Hz
f_P12F1_to_P12 = 509.06e6           # Hz
f_p12F1_to_P12F2 = 814.5e6          # Hz

f_S12_to_P32 = 384.2304844685e12
# f_S12F1_to_S12 = 4.271676631815181e9
f_P32F0_to_P32 = 302.0738e6
f_P32F0_to_P32F1 = 72.2180e6
f_P32F1_to_P32F2 = 156.9470e6
f_P32F2_to_P32F3 = 266.6500e6

# note that D1x is the transition from S1/2 F=1 to P1/2 F'=x
# and D2x is the transition from S1/2 F=1 to P3/2 F'=x

# Central transition frequency: S1/2 F=1 → P1/2 F'=1
f_D11 = f_S12_to_P12 + f_S12F1_to_S12 - f_P12F1_to_P12
f_D12 = f_D11 + f_p12F1_to_P12F2
f_D20 = f_S12_to_P32 + f_S12F1_to_S12 - f_P32F0_to_P32
f_D21 = f_D20 + f_P32F0_to_P32F1
f_D22 = f_D21 + f_P32F1_to_P32F2
f_D23 = f_D22 + f_P32F2_to_P32F3

# all the f_d can be found using f_d = f_0 - delta_f_d
# delta_f_d = f_S12F1_to_S12F2

# Wavelength
lambda_D11 = c / f_D11
lambda_D12 = c / f_D12
lambda_D20 = c / f_D20
lambda_D21 = c / f_D21
lambda_D22 = c / f_D22
lambda_D23 = c / f_D23


# Linewidth (Γ_D1)
Gamma_D2 = 6.0666e6  # Hz
Gamma_D1 = 5.7500e6  # Hz

# S_ij coefficients for Einstein B calculation with ground i and excited j
S_coeff_D1_22 = 1/2
S_coeff_D1_21 = 1/2
S_coeff_D1_12 = 5/6
S_coeff_D1_11 = 1/6

S_coeff_D2_32 = 0
S_coeff_D2_23 = 7/10
S_coeff_D2_22 = 1/4
S_coeff_D2_21 = 1/20
S_coeff_D2_12 = 5/12
S_coeff_D2_11 = 5/12
S_coeff_D2_10 = 1/6
S_coeff_D2_01 = 0

# extra term from matching my derivation with einstein B convention, should be 3, but nonexistent (1) in the reference
S_CONVENTION_CORRECTION = 3 # 3

# Peak cross-section (sigma_0)
sigma_D11_0 = 3 * lambda_D11**2 / (2 * np.pi)
sigma_D12_0 = 3 * lambda_D12**2 / (2 * np.pi)

sigma_D20_0 = 3 * lambda_D20**2 / (2 * np.pi)
sigma_D21_0 = 3 * lambda_D21**2 / (2 * np.pi)
sigma_D22_0 = 3 * lambda_D22**2 / (2 * np.pi)
sigma_D23_0 = 3 * lambda_D23**2 / (2 * np.pi)

# Rubidium-87 mass
m_Rb87 = 86.909180527 * amu

# Temperature and Doppler parameter
T = 300.0  # K
u = np.sqrt(2 * k_B * T / m_Rb87)

# Rb vapor number density (low-pressure) (assumed)
T = 273.15 + 30 # K
if(T < 273.15 + 39.31): # is solid
    log10_P_Torr = -94.04826 -1961.258 / T - 0.03771687 * T + 42.57526 * np.log10(T)
else:
    log10_P_Torr = 15.88253 - 4529.635 / T + 5.8663e-4 * T - 2.99138 * np.log10(T)
P_Torr = 10**log10_P_Torr
P_Pa = P_Torr * 133.322368  # Pa
N = 0.2783 * P_Pa / (k_B * T)  # ideal gas law, m^-3
print(f"Rb-87 vapor number density N = {N/1e16:.2f} x10^16 m^-3 at T = {T:.2f} K")

# Saturation and pump intensities
I_sat_D11 = h * f_D11 * Gamma_D1 / (2 * sigma_D11_0)
I_sat_D12 = h * f_D12 * Gamma_D1 / (2 * sigma_D12_0)
I_pu_D1 = I_sat_D11  # set equal I_sat_D11
print(f"I_pu_D1 = {I_pu_D1/10:.2f} mW/cm^2 I_sat_D11 = {I_sat_D11/10:.2f} mW/cm^2")

I_sat_D20 = h * f_D20 * Gamma_D2 / (2 * sigma_D20_0)
I_sat_D21 = h * f_D21 * Gamma_D2 / (2 * sigma_D21_0)
I_sat_D22 = h * f_D22 * Gamma_D2 / (2 * sigma_D22_0)
I_sat_D23 = h * f_D23 * Gamma_D2 / (2 * sigma_D23_0)
I_pu_D2 = I_sat_D21  # set equal I_sat_D20
print(f"I_pu_D2 = {I_pu_D2/10:.2f} mW/cm^2 I_sat_D21 = {I_sat_D21/10:.2f} mW/cm^2")

# beam waist diameter
D = 2e-3  # 2 mm or 60 um
Gamma_tr = 1.13 * u/D  # transit-time broadening
print(f"Transit-time broadening Gamma_tr = {Gamma_tr/1e6:.2f} MHz")

# ------------------------------
# Part 1: Shared helper functions
# ------------------------------


# ------------------------------
# 1D Maxwell-Boltzmann PDF (projection on beam axis)
# ------------------------------
def mb_1d_pdf(v, u):
    """
    1D Maxwell-Boltzmann projection (normalized).
    v : array of velocities (m/s)
    u : most probable speed = sqrt(2 k_B T / m) (m/s)
    returns f_D(v; u)
    """
    return (1.0 / (u * np.sqrt(np.pi))) * np.exp(-(v**2) / (u**2))


# ------------------------------
# Lorentzian shapes
# ------------------------------
def lorentzian_unnormalized(delta, Gamma_Hz):
    """
    Unnormalized Lorentzian form used in some algebraic expressions:
    L(delta, Gamma) = (Gamma^2 / 4) / (delta^2 + Gamma^2 / 4)
    Here delta and Gamma_Hz are in Hz.
    """
    return ( (Gamma_Hz**2) / 4.0 ) / ( delta**2 + (Gamma_Hz**2) / 4.0 )

def lorentzian_normalized(delta, Gamma_Hz):
    """
    Normalized Lorentzian (area = 1). Use when the formula expects a lineshape g_H(δ,Γ):
    g_H(delta, Gamma) = (1/pi) * ( (Gamma/2) / (delta^2 + (Gamma/2)^2) )
    Input: delta (Hz), Gamma_Hz (Hz)
    Output: s/Hz (integrates to 1 over delta)
    """
    return (1.0 / pi) * ( (Gamma_Hz / 2.0) / (delta**2 + (Gamma_Hz / 2.0)**2) )

# ------------------------------
# Doppler convolution helper
# ------------------------------
def compute_velocity_integral(func_v, u, v_max_factor=7.0, n_v=20001):
    """
    Numerically integrate a function of velocity over the Maxwell 1D PDF using trapezoid.
    func_v : function or array-like of shape (n_v,) representing integrand at v values
             Note: integrand should already include the MB pdf factor if desired.
    u : thermal velocity parameter (sqrt(2 k_B T / m))
    v_max_factor : multiple of u to set integration bounds
    n_v : number of velocity grid points (prefer odd large numbers)
    Returns integral value (float) or array of integrals if func_v is 2D.
    """
    v_max = v_max_factor * u
    v = np.linspace(-v_max, v_max, n_v)
    dv = v[1] - v[0]

    # If func_v is a callable, evaluate it
    if callable(func_v):
        vals = func_v(v)
    else:
        vals = np.asarray(func_v)
        if vals.shape != v.shape:
            raise ValueError("func_v array shape must match velocity grid shape")

    integral = np.trapz(vals, v)
    return integral


# ------------------------------
# Einstein B coefficient from Eq. (C11)
# Bij = (1/3) c^3 π^2 / (ħ ω^3) * (2Jj+1)/(2Ji+1) * S_ij * Gamma_D (angular)
# (we implement the common algebraic form; the appendix notes ΓD in angular units)
# ------------------------------
def Bij_from_C11(omega_ij_rad, J_i, J_j, S_ij, GammaD_angular):
    """
    Compute the Einstein B_ij coefficient using Eq. (C11) form.
    Inputs:
      omega_ij_rad : transition angular frequency ω_ij (rad/s)
      J_i, J_j : total electronic J for initial and final (e.g., 1/2 or 3/2)
      S_ij : dimensionless line-strength factor (from ref; unitless)
      GammaD_angular : natural linewidth (angular) i.e., 2π * (Hz linewidth)
    Returns:
      B_ij in SI (m^3 / J / s) ???  (units consistent with I/c * gH)
    Note:
      Eq. (C11) in the paper shows Bij ∝ (1/3) c^3 π^2 / (ħ ω^3) * (2Jj+1)/(2Ji+1) * S_ij * ΓD.
      We implement that expression literally.
    """
    pref = (1.0/3.0) * (c**3) * (pi**2) / (hbar * (omega_ij_rad**3))
    factor = ((2.0 * J_j + 1.0) / (2.0 * J_i + 1.0))
    B_ij = pref * factor * S_ij * GammaD_angular * S_CONVENTION_CORRECTION
    return B_ij


# ------------------------------
# Pumping rates Rpr and Rpu (Eq. C6)
# R = Bij * (I / c) * gH( delta - k v ; Gamma )
# We work in Hz for detunings; Bij needs omega in rad/s; Gamma for gH (normalized) must be Hz
# ------------------------------
def R_rate_probe(Bij, I_probe_Wm2, delta_Hz, k_freq, v, Gamma_Hz):
    """
    Compute pumping rate Rpr for a velocity class v:
      Bij : Einstein B coefficient (from Bij_from_C11), units consistent
      I_probe_Wm2 : probe intensity (W/m^2)
      delta_Hz : scalar detuning (Hz) of probe from atomic rest resonance (probe freq - omega_ij/2pi)
      k_freq : 1/lambda (cycles per meter) such that Doppler shift in Hz is k_freq * v
      v : numpy array of velocities (m/s) or scalar
      Gamma_Hz : linewidth in Hz (FWHM)
    Returns: array of Rpr(v) (s^-1)
    """
    # Doppler-shifted detuning for each v (in Hz)
    delta_v = delta_Hz + k_freq * v
    # spectral energy density factor approximated by normalized Lorentzian lineshape gH (Hz^-1)
    gH_vals = lorentzian_normalized(delta_v, Gamma_Hz)  # integrates to 1 over Hz
    # intensity-to-energy-density:  ρ(ω) ≈ I / c  (since nearly monochromatic)
    rho = I_probe_Wm2 / c
    R = Bij * rho * gH_vals
    return R

def R_rate_pump(Bij, I_pump_Wm2, delta_pump_Hz, k_freq_pump, v, Gamma_Hz_pump):
    """
    Same as above but for pump (we keep separate function for clarity).
    """
    delta_v = delta_pump_Hz - k_freq_pump * v
    gH_vals = lorentzian_normalized(delta_v, Gamma_Hz_pump)
    rho = I_pump_Wm2 / c
    R = Bij * rho * gH_vals
    return R


# ------------------------------
# Decay mapping Γ_ji as in Eq. (C4)
# We'll implement a function to return the partial decay rates from excited states j (3..7)
# to ground states 1 and 2: Gamma_j1, Gamma_j2 ( in Hz).
# Appendix uses ΓD1,2 in angular units for Bij expression, but their numerical C4 uses fractions;
# those fraction multipliers apply to ΓD1,2 (we'll accept ΓD1_Hz input and return partials in Hz).
# ------------------------------
def partial_decay_rates_D1D2(GammaD1_Hz, GammaD2_Hz):
    """
    Return dictionary mapping excited index j -> (Gamma_j1_Hz, Gamma_j2_Hz).
    Following Eq. (C4a-C4e) using fractions (values are fractions of GammaD1 or GammaD2).
    Indices:
      j=3 -> P1/2 F'=1
      j=4 -> P1/2 F'=2
      j=5 -> P3/2 F'=0
      j=6 -> P3/2 F'=1
      j=7 -> P3/2 F'=2   (note: appendix labeling)
    Returns: dict {3: (Gamma31_Hz, Gamma32_Hz), ...}
    """
    rates = {}
    # D1 excited j=3 and j=4
    rates[3] = (GammaD1_Hz / 6.0, 5.0 * GammaD1_Hz / 6.0)   # Γ31, Γ32
    rates[4] = (GammaD1_Hz / 2.0, GammaD1_Hz / 2.0)         # Γ41, Γ42
    # D2 excited j=5..7
    rates[5] = (GammaD2_Hz, 0.0)                           # Γ51, Γ52
    rates[6] = (5.0 * GammaD2_Hz / 6.0, GammaD2_Hz / 6.0)  # Γ61, Γ62
    rates[7] = (GammaD2_Hz / 2.0, GammaD2_Hz / 2.0)        # Γ71, Γ72
    return rates


# ------------------------------
# Initial equilibrium populations N0_j(v) per Eq. (C5)
# N0_1(v) = (g1 / (g1+g2)) * N_v  where N_v is total density for that velocity v
# For a given total N_v (per v) this returns N0_j arrays.
# We'll assume g1 = 2*F1+1 = 3 (F=1) and g2 = 2*F2+1 = 5 (F=2)
# ------------------------------
def equilibrium_populations_per_v(N_v, v_array=None, g1=3, g2=5):
    """
    N_v: either scalar total density per v or an array matching v_array shape (atoms/m^3 in that velocity class)
    returns (N1_0, N2_0) arrays or scalars
    """
    N_v = np.asarray(N_v)
    frac1 = g1 / (g1 + g2)   # 3/8
    frac2 = g2 / (g1 + g2)   # 5/8
    N1_0 = frac1 * N_v
    N2_0 = frac2 * N_v
    return N1_0, N2_0


# ------------------------------
# ODE system (rate equations) builder
# We'll create a function that given laser parameters and velocity v returns an ODE function for dN/dt.
# State vector ordering: [N1, N2, N3, N4, N5, N6, N7] (all functions of v)
# Equations from (C1) (bichromatic) or (C13) (monochromatic) will be implemented in separate wrappers.
# Here we just provide a general helper for recombining terms.
# ------------------------------
def steady_state_solver(rhs_fun, N0, t_max, dt=None, rtol=1e-6, atol=1e-9):
    """
    Solve dN/dt = rhs_fun(t, N) to steady state.
    rhs_fun: callable (t, N) -> dN/dt
    N0: initial array
    t_max: simulation end time (s) - Appendix suggests t_end = 400 / GammaD2 (angular or Hz?)
    dt: optional fixed step for RK4 fallback
    Returns N(t_final)
    """
    if SCIPY_AVAILABLE:
        sol = solve_ivp(rhs_fun, (0.0, t_max), N0, method=ODE_SOLVE_METHOD, rtol=rtol, atol=atol, vectorized=False)
        if not sol.success:
            # Try a different solver
            sol = solve_ivp(rhs_fun, (0.0, t_max), N0, method='BDF', rtol=rtol, atol=atol)
        return sol.y[:, -1]
    else:
        # Simple RK4 integrator fallback (fixed dt)
        if dt is None:
            # set dt to small fraction of t_max
            dt = t_max / 2000.0
        N = N0.copy().astype(float)
        t = 0.0
        steps = int(np.ceil(t_max / dt))
        for _ in range(steps):
            k1 = rhs_fun(t, N)
            k2 = rhs_fun(t + dt/2.0, N + dt * k1 / 2.0)
            k3 = rhs_fun(t + dt/2.0, N + dt * k2 / 2.0)
            k4 = rhs_fun(t + dt, N + dt * k3)
            N = N + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            t += dt
        return N


# ------------------------------
# α(ω) integrand helper per Eq. (C12)
# α(ω) = (ℏ ω1j / c) ∫ dv f_D(v) g_H(δ1j - k1j v; ΓD1) * (N1(v) - (g_j/g1) * N_j(v))
# We will implement a function that given population arrays N1(v), Nj(v) returns the integrand and the α(ω).
# ------------------------------

def absorption_alpha_from_populations_v2(hbar, omega1j_rad, delta1j_Hz, k_freq, v_array, fD_v, Nj_pop, N1_pop, g_j, g1, GammaD1_Hz, B_1j):
    """
    Safer version: k_freq = 1/lambda (cycles/m); delta1j_Hz in Hz.
    Returns alpha (1/m).
    """
    # prefactor ℏ ω / c
    pref = (hbar * omega1j_rad) / c * B_1j

    # lineshape evaluated at (delta1j - k_freq * v) in Hz
    delta_v = delta1j_Hz + k_freq * v_array
    gH_vals = lorentzian_normalized(delta_v, GammaD1_Hz)   # units Hz^-1

    # integrand: fD(v) * gH( ... ) * (N1(v) - (g_j/g1) * N_j(v))
    # note: fD_v is omitted here because it's already included in N1_pop and Nj_pop
    integrand = gH_vals * (N1_pop - (g_j / g1) * Nj_pop)

    integral = np.trapz(integrand, v_array)   # integral over v (m/s), output units: (1/m/s) ??? combined pref fixes units
    alpha = pref * integral
    return alpha


def k_freq_from_lambda(lambda_m):
    return 1.0 / lambda_m


# ------------------------------
# Part 2: Monochromatic SAS (mSAS)
# ------------------------------.
# Requires: mb_1d_pdf, lorentzian_normalized, Bij_from_C11,
#           R_rate_probe, R_rate_pump, partial_decay_rates_D1D2,
#           equilibrium_populations_per_v, steady_state_solver,
#           absorption_alpha_from_populations_v2, k_freq_from_lambda
# ------------------------------

# --- User-provided / constants (should already exist in your constants cell) ---
# Example variable names expected from your constants:
#   f_D11, f_D12, f_D20, ...       (Hz)
#   lambda_D11, lambda_D12, ...   (m)
#   Gamma_D1_Hz, Gamma_D2_Hz      (Hz natural linewidth)  -- note suffix _Hz to be explicit
#   S_D11, S_D12, ...             (line strength fractions for scaling)
#   N_total (or N)                (m^-3) total atomic number density (for the isotope)
#   I_pr, I_pu                     (W/m^2) probe and pump intensities
#   T, m_Rb87, diameter, etc.
#
# If you used different names in your constants cell, adapt these names accordingly.
# For safety, we will try to read common names and fallback to previously used names in your session.

# Map common names 
GammaD1_Hz = Gamma_D1      
GammaD2_Hz = Gamma_D2

# Build partial decay rate mapping (Hz)
partial_rates = partial_decay_rates_D1D2(GammaD1_Hz, GammaD2_Hz)

# Electronic J values for D1 and D2 transitions
J_ground = 0.5
J_excited_D1 = 0.5   # P1/2
J_excited_D2 = 1.5   # P3/2

# degeneracies used in absorption prefactor
g1 = 2 * 1 + 1   # ground F=1 degeneracy = 3
g2 = 2 * 2 + 1   # ground F=2 degeneracy = 5

# --- helper: build Bij for all probe and pump transitions we need ---
# For mSAS we probe transitions j=3,4 (D1 P1/2 F'=1,2) from ground state 1 (F=1).
# We'll build Bij for (1<->3) and (1<->4) using Eq. C11. Use angular freq = 2π f.
def build_Bij_for_transitions(transitions_list):
    """
    transitions_list: list of dicts with keys:
       'f'   : transition frequency in Hz (omega/2pi)
       'lambda' : wavelength (m)
       'J_i', 'J_j' : electronic J initial and final
       'S_ij' : line strength factor S_{i->j} (dimensionless)
       'GammaD_Hz' : natural linewidth (Hz) appropriate for that line (D1 or D2)
    Returns: B_ij (float)
    """
    B_map = {}
    for tr in transitions_list:
        omega_rad = 2.0 * pi * tr['f']   # rad/s
        # appendix expects GammaD in angular units inside Bij pref:
        GammaD_angular = 2.0 * pi * tr['GammaD_Hz']
        B_map[tr['label']] = Bij_from_C11(omega_rad, tr['J_i'], tr['J_j'], tr['S_ij'], GammaD_angular)
    return B_map

# Define D1 probe transitions (j=3 and j=4)
transitions_D1 = [
    {'label': '1-3', 'f': f_D11, 'lambda': lambda_D11, 'J_i': J_ground, 'J_j': J_excited_D1, 'S_ij': S_coeff_D1_11, 'GammaD_Hz': GammaD1_Hz},
    {'label': '1-4', 'f': f_D12, 'lambda': lambda_D12, 'J_i': J_ground, 'J_j': J_excited_D1, 'S_ij': S_coeff_D1_12, 'GammaD_Hz': GammaD1_Hz},

    {'label': '3-1', 'f': f_D11, 'lambda': lambda_D11, 'J_i': J_excited_D1, 'J_j': J_ground, 'S_ij': S_coeff_D1_11, 'GammaD_Hz': GammaD1_Hz},
    {'label': '4-1', 'f': f_D12, 'lambda': lambda_D12, 'J_i': J_excited_D1, 'J_j': J_ground, 'S_ij': S_coeff_D1_21, 'GammaD_Hz': GammaD1_Hz},
]
B_map = build_Bij_for_transitions(transitions_D1)

# Also build Bij for pump transitions (D2 lines j=5..7) if needed (for mSAS pump and probe same source)
transitions_D2 = [
    {'label': '1-5', 'f': f_D20, 'lambda': lambda_D20, 'J_i': J_ground, 'J_j': J_excited_D2, 'S_ij': S_coeff_D2_10, 'GammaD_Hz': GammaD2_Hz},
    {'label': '1-6', 'f': f_D21, 'lambda': lambda_D21, 'J_i': J_ground, 'J_j': J_excited_D2, 'S_ij': S_coeff_D2_11, 'GammaD_Hz': GammaD2_Hz},
    {'label': '1-7', 'f': f_D22, 'lambda': lambda_D22, 'J_i': J_ground, 'J_j': J_excited_D2, 'S_ij': S_coeff_D2_12, 'GammaD_Hz': GammaD2_Hz},

    {'label': '5-1', 'f': f_D20, 'lambda': lambda_D20, 'J_i': J_excited_D2, 'J_j': J_ground, 'S_ij': S_coeff_D2_01, 'GammaD_Hz': GammaD2_Hz},
    {'label': '6-1', 'f': f_D21, 'lambda': lambda_D21, 'J_i': J_excited_D2, 'J_j': J_ground, 'S_ij': S_coeff_D2_11, 'GammaD_Hz': GammaD2_Hz},
    {'label': '7-1', 'f': f_D22, 'lambda': lambda_D22, 'J_i': J_excited_D2, 'J_j': J_ground, 'S_ij': S_coeff_D2_21, 'GammaD_Hz': GammaD2_Hz},
]
B_map_pump = build_Bij_for_transitions(transitions_D2)

# k_freq for each transition (cycles/m)
k_probe_13 = k_freq_from_lambda(lambda_D11)
k_probe_14 = k_freq_from_lambda(lambda_D12)
k_pump_13 = k_freq_from_lambda(lambda_D11)
k_pump_14 = k_freq_from_lambda(lambda_D12)
k_pump_15 = k_freq_from_lambda(lambda_D20)
k_pump_16 = k_freq_from_lambda(lambda_D21)
k_pump_17 = k_freq_from_lambda(lambda_D22)


# --- ODE RHS for monochromatic SAS (Eq. C13) ---
def rhs_mSAS_factory(v_i, probe_detuning_Hz, pump_detuning_Hz, I_pr_Wm2, I_pu_Wm2):
    """
    Return an RHS function dN/dt for a given velocity class v_i and the current laser detunings.
    - probe_detuning_Hz : scalar detuning of probe from 1->3 (and 1->4) resonance (Hz)
    - pump_detuning_Hz : for monochromatic SAS pump and probe come from same source -> same detuning
    - I_pr_Wm2, I_pu_Wm2 : intensities (W/m^2)
    The returned rhs(t, N) expects N as length-7 array [N1,N2,N3,N4,N5,N6,N7]
    All rates returned in s^-1.
    """
    # precompute total rates
    Rpr_13 = R_rate_probe(B_map['1-3'], I_pr_Wm2, probe_detuning_Hz - (f_D11 - f_D11), k_probe_13, v_i, GammaD1_Hz)
    Rpr_14 = R_rate_probe(B_map['1-4'], I_pr_Wm2, probe_detuning_Hz - (f_D12 - f_D11), k_probe_14, v_i, GammaD1_Hz)
    Rpr_31 = R_rate_probe(B_map['3-1'], I_pr_Wm2, (probe_detuning_Hz - (f_D11 - f_D11)), k_probe_13, v_i, GammaD1_Hz)
    Rpr_41 = R_rate_probe(B_map['4-1'], I_pr_Wm2, (probe_detuning_Hz - (f_D12 - f_D11)), k_probe_14, v_i, GammaD1_Hz)

    Rpu_13 = R_rate_pump(B_map['1-3'], I_pu_Wm2, pump_detuning_Hz - (f_D11 - f_D11), k_pump_13, v_i, GammaD1_Hz)
    Rpu_14 = R_rate_pump(B_map['1-4'], I_pu_Wm2, pump_detuning_Hz - (f_D12 - f_D11), k_pump_14, v_i, GammaD1_Hz)
    Rpu_31 = R_rate_pump(B_map['3-1'], I_pu_Wm2, (pump_detuning_Hz - (f_D11 - f_D11)), k_pump_13, v_i, GammaD1_Hz)
    Rpu_41 = R_rate_pump(B_map['4-1'], I_pu_Wm2, (pump_detuning_Hz - (f_D12 - f_D11)), k_pump_14, v_i, GammaD1_Hz)

    # Partial decay rates Γ_j1 and Γ_j2 in Hz
    rates = partial_rates  # mapping j->(Gamma_j1_Hz, Gamma_j2_Hz)

    def rhs(t, N):
        N1, N2, N3, N4, N5, N6, N7 = N

        # eq. (C13)
        R_1j = [0, 0, 0, 0, 0]  # index 0 unused
        R_1j[3] = Rpr_13 + Rpu_13
        R_1j[4] = Rpr_14 + Rpu_14
        R_j1 = [0, 0, 0, 0, 0]  # index 0 unused
        R_j1[3] = Rpr_31 + Rpu_31
        R_j1[4] = Rpr_41 + Rpu_41

        # spontaneous decays from excited to ground:
        Gamma31, Gamma32 = rates[3]
        Gamma41, Gamma42 = rates[4]
        Gamma51, Gamma52 = rates[5]
        Gamma61, Gamma62 = rates[6]
        Gamma71, Gamma72 = rates[7]

        # equilibrium populations for this velocity class (N_total_v will be passed externally; here we use placeholders)
        # The solver wrapper will set initial condition as N0 per v and also we need Gamma_tr and N0 to implement relaxation:
        # We'll capture these via closure variables later when creating rhs via factory wrapper. For now raise if not set.
        # To keep code readable, we'll assume closure will set N1_0 and N2_0 and Gamma_tr externally by rebinding.

        # Use placeholder names; actual values provided by outer wrapper through closure assignment
        N1_0_local = rhs.N1_0  # assigned by outer wrapper
        N2_0_local = rhs.N2_0
        Gamma_tr_local = rhs.Gamma_tr_local

        # Compose dN1/dt per Eq. (C13a)
        dN1dt = ((R_j1[3] * N3 - R_1j[3] * N1) + (R_j1[4] * N4 - R_1j[4] * N1)) \
                + ( Gamma31 * N3 + Gamma41 * N4 ) \
                + ( - Gamma_tr_local * (N1 - N1_0_local) )

        # dN2/dt : Eq (C13b)
        dN2dt = (Gamma32 * N3 + Gamma42 * N4) \
                - Gamma_tr_local * (N2 - N2_0_local)

        # dN3/dt Eq (C13c)
        dN3dt = - (R_j1[3] * N3 - R_1j[3] * N1) - (GammaD1_Hz + Gamma_tr_local) * N3

        # dN4/dt Eq (C13d)
        dN4dt = - (R_j1[4] * N4 - R_1j[4] * N1) - (GammaD1_Hz + Gamma_tr_local) * N4

        # dN5/dt Eq (C13e)
        dN5dt = 0

        # dN6/dt Eq (C13d for j=6)
        dN6dt = 0

        # dN7/dt Eq (C13e for j=7)
        dN7dt = 0
        return np.array([dN1dt, dN2dt, dN3dt, dN4dt, dN5dt, dN6dt, dN7dt])

    # Attach placeholders for N1_0, N2_0, Gamma_tr to rhs so outer wrapper can set them
    rhs.N1_0 = None
    rhs.N2_0 = None
    rhs.Gamma_tr_local = None
    return rhs


# --- Wrapper: compute alpha(omega) by looping velocities and solving steady state ---
def compute_mSAS_alpha(delta_array_Hz, N_total, I_pr_Wm2, I_pu_Wm2,
                       v_max_factor=7.0, n_v=20001, t_end_factor=400.0):
    """
    delta_array_Hz: 1D array of probe detunings (Hz) relative to f_D11 (the reference)
    N_total: total atomic density per m^3 (for the isotope)
    I_pr_Wm2, I_pu_Wm2: intensities
    Returns alpha_vs_detuning (1/m) array same length as delta_array_Hz
    """
    # build velocity grid and MB pdf
    v_max = v_max_factor * u
    v_grid = np.linspace(-v_max, v_max, n_v)
    fD_v = mb_1d_pdf(v_grid, u)  # normalized PDF

    # per-velocity total density N_v (atoms/m^3 per velocity class) -> N_total * fD_v (since fD_v normalized)
    N_v_array = N_total * fD_v

    # time to steady state: use t_end = t_end_factor / GammaD2_angular  (appendix suggests 400 / GammaD2)
    GammaD2_angular = 2.0 * pi * GammaD2_Hz
    t_end = t_end_factor / GammaD2_angular

    # equilibrium populations per v for initial condition
    N1_0_array, N2_0_array = equilibrium_populations_per_v(N_v_array, v_array=v_grid, g1=g1, g2=g2)

    # For each detuning in delta_array_Hz:
    #    for each velocity v:
    #        build rhs with v and detuning
    #        solve to steady-state -> collect N1, N3, N4
    #    compute alpha(delta) by integrating per Eq. C12
    #
    # This keeps the code straightforward (but slow). If you want optimization, we can reuse Jacobians / vectorize.

    alpha_array = np.zeros_like(delta_array_Hz)

    # precompute k_freq for probe transitions
    k13 = k_freq_from_lambda(lambda_D11)
    k14 = k_freq_from_lambda(lambda_D12)

    # loop over detunings
    for di, delta_hz in enumerate(delta_array_Hz):
        # For monochromatic SAS: pump detuning = probe detuning (scanning same source)
        probe_detuning_Hz = delta_hz
        pump_detuning_Hz = delta_hz

        # arrays to collect N1, N3, N4 as functions of v
        N1_v = np.zeros_like(v_grid)
        N3_v = np.zeros_like(v_grid)
        N4_v = np.zeros_like(v_grid)

        for idx, v_i in enumerate(v_grid):
            # create rhs for this velocity and detuning
            rhs_fun = rhs_mSAS_factory(v_i, probe_detuning_Hz, pump_detuning_Hz, I_pr_Wm2, I_pu_Wm2)

            # set closure values for equilibrium N0 & Gamma_tr
            rhs_fun.N1_0 = N1_0_array[idx]
            rhs_fun.N2_0 = N2_0_array[idx]
            rhs_fun.Gamma_tr_local = Gamma_tr

            # initial condition: start at equilibrium populations for this v
            N0_vec = np.zeros(7)
            N0_vec[0] = N1_0_array[idx]
            N0_vec[1] = N2_0_array[idx]
            # excited states start at near zero
            # integrate to steady state
            N_ss = steady_state_solver(rhs_fun, N0_vec, t_end)

            N1_v[idx] = N_ss[0]
            N3_v[idx] = N_ss[2]
            N4_v[idx] = N_ss[3]

        # Now compute alpha for probe transitions j=3 and j=4 and sum (Eq. C12 summation over j=3,4)
        # prefactor uses ℏ ω1j / c. We'll compute separately and sum.
        omega_13_rad = 2.0 * pi * f_D11
        omega_14_rad = 2.0 * pi * f_D12

        # alpha for j=3
        alpha_13 = absorption_alpha_from_populations_v2(hbar, omega_13_rad, probe_detuning_Hz  - (f_D11 - f_D11),
                                                       k13, v_grid, fD_v, N3_v, N1_v,
                                                       g_j=2*1+1, g1=g1, GammaD1_Hz=GammaD1_Hz, B_1j=B_map['1-3'])
        # alpha for j=4
        alpha_14 = absorption_alpha_from_populations_v2(hbar, omega_14_rad, probe_detuning_Hz - (f_D12 - f_D11),
                                                       k14, v_grid, fD_v, N4_v, N1_v,
                                                       g_j=2*2+1, g1=g1, GammaD1_Hz=GammaD1_Hz, B_1j=B_map['1-4'])
        alpha_array[di] = alpha_13 + alpha_14

        # (optional) print progress
        if di % max(1, len(delta_array_Hz)//10) == 0:
            print(f"mSAS: computed {di+1}/{len(delta_array_Hz)} detunings")

    return alpha_array
# ------------------------------
# Part 4: Final assembly + plotting (1x4 pump panels)
# ------------------------------
# ---------- User-tweakable parameters ----------
# Detuning grid (Hz) relative to f_D11 (probe reference)
delta_min_MHz = -500.0
delta_max_MHz = 1500.0
n_det = 401   # number of detuning points (increase for smoother curves)

# Velocity integration settings (these are heavy)
v_max_factor = 5.0
n_v = 2001    # use 2001 for testing; 20001 for full accuracy is slow

# Cell length for transmittance
L_cell = 0.075   # meters (75 mm)

# Use intensities (converted from mW/cm^2 as in your inputs)
I_pr_Wm2 = 0.03e-3 / 1e-4   # 0.03 mW/cm^2 -> W/m^2
I_pu_Wm2 = 1.3e-3  / 1e-4   # 1.3 mW/cm^2 -> W/m^2

# Which pump wavelengths to use (D2 F'=0..3)
pump_labels = [r"F' = 0", r"F' = 1", r"F' = 12", r"F' = 2"]
pump_freqs = [f_D20, f_D21, (f_D21+f_D22)/2, f_D22]   # Hz
pump_lambdas = [c/f for f in pump_freqs]  # m

# D1 probe wavelength (we probe S1/2 F=1 -> P1/2 F'=1 as reference)
lambda_probe_ref = lambda_D11
f_probe_ref = f_D11

# Which individual D1 contributions to optionally show
show_components = True

# ---------- build detuning array (Hz) ----------
delta_array_MHz = np.linspace(delta_min_MHz, delta_max_MHz, n_det)
# delta_array_MHz = np.array([750])  # single point test
delta_array_Hz = delta_array_MHz * 1e6

# ---------- compute the Doppler-averaged background mSAS alpha (slow) ----------
print("computing Doppler-averaged alpha (mSAS).")
print(f" detuning range: {delta_min_MHz} .. {delta_max_MHz} MHz, points = {n_det}")
print(f" velocity grid n_v = {n_v}, v_max_factor = {v_max_factor}")
t0 = time()
# compute_mSAS_alpha should be defined in Part 2
alpha_array = compute_mSAS_alpha(
    delta_array_Hz,
    N,                # total number density (m^-3) for the isotope 
    I_pr_Wm2,
    I_pu_Wm2,
    v_max_factor=v_max_factor,
    n_v=n_v,
    t_end_factor=400.0
)
t1 = time()
print(f" Done alpha0 calculation in {t1 - t0:.1f} s")
delta_array_MHz = delta_array_Hz / 1e6

# ------------------------------
# Save detuning and alpha to TXT
# ------------------------------
output_filename = "mSAS_alpha_output.txt"

# Stack into two columns: detuning (MHz) and alpha (1/m)
data_to_save = np.column_stack([delta_array_MHz, alpha_array])

np.savetxt(
    output_filename,
    data_to_save,
    fmt="%.6e",
    header="Detuning_MHz    Alpha_per_m",
    comments=""
)

print(f"Saved mSAS α(Δ) to {output_filename}")

# filename = "mSAS_alpha_output.txt"

# data = np.loadtxt(filename, skiprows=1)   # skip header line

# delta_array_MHz = data[:, 0]
# alpha_array = data[:, 1]

plt.figure(figsize=(8,5))
plt.plot(delta_array_MHz, alpha_array, label=r'$\alpha_{mSAS}$')
plt.xlabel('Detuning (MHz)')
plt.ylabel('Absorption coefficient α (1/m)')
plt.title('Monochromatic SAS absorption')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


