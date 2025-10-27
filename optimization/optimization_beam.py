# Minimizes mass of a rectangular cross-section cantilever beam
# under strength and Euler buckling constraints (slides pp.10–13).
# Units: geometry mm, force N, stress MPa (N/mm^2), E MPa, density kg/mm^3.

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import math

# -------- Optional SciPy (will fall back to refined grid if missing) --------
try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------------
# Problem data (edit as needed)
# ----------------------------
# Loads (kN → N)
F_V = 2.50e3 * 1e3     # vertical, N  (2.50E+03 kN)
F_H = 1.10e4 * 1e3     # axial,   N  (1.10E+04 kN)

# Material (Al alloy example)
E       = 70_000.0     # MPa (70 GPa)
sigma_u = 450.0        # MPa allowable/ultimate
rho     = 2.7e-6       # kg/mm^3  (2.7 t/m^3)

# Geometry
L   = 1000.0           # mm (beam length; adjust to your case)
phi = 1.5              # load/safety factor

# Design variable bounds (mm)
w_min, w_max = 1.0, 800.0
h_min, h_max = 1.0, 800.0

# ----------------------------
# Objective and constraints
# ----------------------------
def mass(w: float, h: float, l: float = L, density: float = rho) -> float:
    """Mass = density * volume; volume = w*h*l (mm^3)."""
    return density * w * h * l

def obj(x):
    w, h = x
    return mass(w, h)

def g1_strength(x):
    """Strength at clamp: sigma = (phi*F_V*L) / (W),  W = w*h^2/6.
       Return sigma - sigma_u  (<= 0 feasible)."""
    w, h = x
    if w <= 0 or h <= 0:
        return 1e9
    W = (w * h**2) / 6.0
    sigma = (phi * F_V * L) / W  # MPa (since N/mm^2)
    return sigma - sigma_u

def g2_buckling(x):
    """Euler buckling under axial: phi*F_H <= pi^2*E*I/le^2,
       I = h*w^3/12, le = 2*L. Return phi*F_H - Ncrit (<= 0 feasible)."""
    w, h = x
    if w <= 0 or h <= 0:
        return 1e9
    I  = (h * w**3) / 12.0         # mm^4
    le = 2.0 * L                   # mm
    Ncrit = (pi**2 * E * I) / (le**2)  # N
    return phi * F_H - Ncrit

# ----------------------------
# Analytic intersection (both constraints active)
# ----------------------------
def analytic_solution():
    """
    Solve:
      w h^2 = A = 6*phi*F_V*L / sigma_u
      h w^3 = B = 48*L^2*phi*F_H / (pi^2*E)
    → h = (A^3/B)^(1/5),  w = A/h^2
    """
    A = 6.0 * phi * F_V * L / sigma_u
    B = 48.0 * (L**2) * phi * F_H / (pi**2 * E)
    h_star = (A**3 / B)**(1.0/5.0)
    w_star = A / (h_star**2)
    return w_star, h_star

def project_to_bounds(w, h):
    return (
        min(max(w, w_min), w_max),
        min(max(h, h_min), h_max),
    )

# ----------------------------
# Solvers
# ----------------------------
def solve_with_scipy(x0=(50.0, 50.0)):
    cons = [
        {"type": "ineq", "fun": lambda x: -g1_strength(x)},  # g <= 0
        {"type": "ineq", "fun": lambda x: -g2_buckling(x)},
    ]
    bounds = [(w_min, w_max), (h_min, h_max)]
    res = minimize(
        obj, x0, method="SLSQP",
        bounds=bounds, constraints=cons,
        options=dict(maxiter=500, ftol=1e-12, disp=False)
    )
    return res

def solve_by_grid():
    """
    Two-stage refined grid search:
      1) coarse grid over bounds
      2) local refinement around best coarse point
    """
    # coarse
    ws = np.linspace(w_min, w_max, 300)
    hs = np.linspace(h_min, h_max, 300)
    best = None
    for w in ws:
        for h in hs:
            if g1_strength((w,h)) <= 0 and g2_buckling((w,h)) <= 0:
                val = obj((w,h))
                if best is None or val < best[0]:
                    best = (val, w, h)
    if best is None:
        return None

    # refine locally
    _, w0, h0 = best
    ws = np.linspace(max(w_min, w0-10), min(w_max, w0+10), 400)
    hs = np.linspace(max(h_min, h0-10), min(h_max, h0+10), 400)
    for w in ws:
        for h in hs:
            if g1_strength((w,h)) <= 0 and g2_buckling((w,h)) <= 0:
                val = obj((w,h))
                if val < best[0]:
                    best = (val, w, h)
    return best

# ----------------------------
# Plotting (no deprecated .collections)
# ----------------------------
def plot_feasible_and_contours(w_opt, h_opt, n=300, levels=15):
    from matplotlib.lines import Line2D

    ws = np.linspace(w_min, w_max, n)
    hs = np.linspace(h_min, h_max, n)
    W, H = np.meshgrid(ws, hs, indexing='xy')

    # Constraints field
    G1 = np.vectorize(lambda w,h: g1_strength((w,h)))(W, H)
    G2 = np.vectorize(lambda w,h: g2_buckling((w,h)))(W, H)
    feasible = (G1 <= 0) & (G2 <= 0)

    # Objective (mass) — proportional to w*h, so scale by (rho*L)
    Z = (rho * L) * (W * H)

    plt.figure(figsize=(8,6))
    # Feasible region shading
    plt.contourf(W, H, feasible.astype(float),
                 levels=[-0.5, 0.5, 1.5], alpha=0.15)

    # Constraint frontiers (g=0)
    c1 = plt.contour(W, H, G1, levels=[0.0], linewidths=2)
    c2 = plt.contour(W, H, G2, levels=[0.0], linewidths=2)

    # Objective contours
    CS = plt.contour(W, H, Z, levels=levels, linewidths=1)
    plt.clabel(CS, inline=True, fontsize=8)

    # Optimum marker
    plt.plot([w_opt], [h_opt], marker='o', markersize=8)
    plt.text(w_opt, h_opt, f"  w*={w_opt:.2f}, h*={h_opt:.2f}", va='bottom', ha='left')

    plt.xlabel('w  [mm]')
    plt.ylabel('h  [mm]')
    plt.title('Feasible domain & objective contours')

    # Legend via proxy handles (avoids deprecated attribute access)
    proxy_strength = Line2D([0], [0], linewidth=2)
    proxy_buckling = Line2D([0], [0], linewidth=2)
    plt.legend([proxy_strength, proxy_buckling],
               ['g1 = 0 (strength)', 'g2 = 0 (buckling)'],
               loc='best')

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Analytic intersection (unbounded) and projection to bounds
    w_eq, h_eq = analytic_solution()
    w_proj, h_proj = project_to_bounds(w_eq, h_eq)

    print(f"Unbounded analytic optimum (both constraints active): "
          f"w≈{w_eq:.3f} mm, h≈{h_eq:.3f} mm")
    if (abs(w_eq - w_proj) > 1e-9) or (abs(h_eq - h_proj) > 1e-9):
        print(f"Projected to bounds: w≈{w_proj:.3f} mm, h≈{h_proj:.3f} mm")

    # Numerical solve (prefers SciPy; falls back to refined grid)
    if SCIPY_OK:
        res = solve_with_scipy(x0=(max(w_min, min(w_proj, w_max)),
                                   max(h_min, min(h_proj, h_max))))
        if not res.success:
            print("SLSQP did not converge:", res.message)
            brute = solve_by_grid()
            if brute is None:
                raise RuntimeError("No feasible point found in grid.")
            m_opt, w_opt, h_opt = brute
        else:
            w_opt, h_opt = res.x
            m_opt = mass(w_opt, h_opt)
    else:
        print("SciPy not available; using a refined grid search.")
        brute = solve_by_grid()
        if brute is None:
            raise RuntimeError("No feasible point found in grid.")
        m_opt, w_opt, h_opt = brute

    # Report
    print(f"\nOptimal design (mm): w* = {w_opt:.3f}, h* = {h_opt:.3f}")
    print(f"Mass (kg) with L={L:.0f} mm: m* = {m_opt:.6f}")
    print(f"Strength margin g1 = {g1_strength((w_opt,h_opt)):.6f} MPa (<=0 ok)")
    print(f"Buckling  margin g2 = {g2_buckling((w_opt,h_opt)):.3f} N (<=0 ok)")

    # Plot
    plot_feasible_and_contours(w_opt, h_opt)
