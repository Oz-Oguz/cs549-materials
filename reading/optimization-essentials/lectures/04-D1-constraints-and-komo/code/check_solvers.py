"""Regression checks for the two Deck 05 solvers (added 2026-09-29, Stage C of planning/consolidation-2026-09-29).

  1. A rejected line-search candidate never replaces the current iterate, and a trial whose cost is not finite is
     rejected too (2b); the projected multiplier update can break the identity between inner stationarity and the
     Lagrangian's stationarity, so the outer residuals are measured rather than assumed (2c). newton_armijo on the follow-up audit's
     counterexample f = 1/2 1e14 z^2 with the model Hessian 1 (planning/audit-kin2opt-followup-2026-09-29/
     verify_deck05.py) must return z unchanged with the outcome 'stalled'; before 2026-09-29 it returned
     z = -89.9 with the cost multiplied by 8000.
  2. The same for komo_two_link.gauss_newton, on a residual whose Gauss-Newton direction cannot be shortened
     enough above the floor, and solve_komo must propagate the stall ('inner stalled') without moving z.
  3. The three Jacobians of the KOMO problem (smoothness, equalities, inequalities) against central differences
     at solution A, and the two objects the lecture keeps apart: the full feature Jacobian (104 x 40) and the
     inner residual Jacobian (46 x 40 at solution A).
  4. The reproduced numbers the slides quote: solution A's cost, active knots and KKT residuals.

Run:  python check_solvers.py      (about five seconds; exits nonzero on a failure)
"""
import contextlib
import io
import sys

import numpy as np

import constrained_ik as ci
import komo_two_link as km

ok = True


def check(cond, msg):
    global ok
    print(("  ok    " if cond else "  FAIL  ") + msg)
    ok = ok and bool(cond)


# 1. newton_armijo rejects the failed trial and keeps z
cost = lambda z: 0.5 * 1e14 * float(z @ z)
gradient = lambda z: 1e14 * z
model = lambda z: np.eye(len(z))
z0 = np.array([1.0])
z1, it, status = ci.newton_armijo(cost, gradient, model, z0, max_iter=1)
check(status == "stalled" and np.array_equal(z1, z0), f"newton_armijo on f = 1/2 1e14 z^2 with H_model = 1 from z = 1: outcome '{status}', z = {z1} (unchanged), cost {cost(z1):.3e}")

# 2. gauss_newton rejects a direction that no trial above the floor can shorten enough, and solve_komo propagates it
orig = km.al_residual
def stiff_residual(z, lam, kap, mu):
    # r = 1 + (z0 - 1) + 1e3 (1 - exp(-1e12 (z0 - 1)^2)): at z0 = 1 the residual is 1 with Jacobian 1, so the
    # Gauss-Newton direction is -1, but every trial from alpha = 1 down to the floor 1e-8 lands on the wall
    # (r >= 1.1 at alpha = 1e-8), so no trial above the floor passes the Armijo test
    u = z[0] - 1.0
    r = np.array([1.0 + u + 1e3 * (1.0 - np.exp(-1e12 * u * u))])
    J = np.zeros((1, km.N)); J[0, 0] = 1.0 + 1e3 * 2e12 * u * np.exp(-1e12 * u * u)
    return r, J
km.al_residual = stiff_residual
z0 = np.zeros(km.N); z0[0] = 1.0
z1, it, status = km.gauss_newton(z0, np.zeros(3 * km.T), np.zeros(4), km.mu_default())
check(status == "stalled" and np.array_equal(z1, z0), f"gauss_newton on a stiff residual: outcome '{status}' after {it} iteration(s), z unchanged = {np.array_equal(z1, z0)}")
with contextlib.redirect_stdout(io.StringIO()):
    z2, lam2, kap2 = km.solve_komo(z0)
check(km.LAST["status"] == "inner stalled" and np.array_equal(z2, z0), f"solve_komo propagates the stall: outcome '{km.LAST['status']}' after {km.LAST['outer']} outer iteration(s), z unchanged = {np.array_equal(z2, z0)}")
km.al_residual = orig

# 2b. a trial whose cost is not finite is rejected (a NaN cost compares false against the Armijo bound and was
#     accepted before 2026-09-29): the residual is finite at the start and undefined for a negative first coordinate
def domain_residual(z, lam, kap, mu):
    r = np.array([1.0 + z[0] if z[0] >= 0 else np.nan])
    J = np.zeros((1, km.N)); J[0, 0] = 1.0
    return r, J
km.al_residual = domain_residual
z0 = np.zeros(km.N)
z1, it, status = km.gauss_newton(z0, np.zeros(3 * km.T), np.zeros(4), km.mu_default(), max_iter=1)
check(status == "stalled" and np.array_equal(z1, z0) and np.isfinite(domain_residual(z1, None, None, None)[0]).all(),
      f"gauss_newton on a residual that is NaN for a negative first coordinate: outcome '{status}', z unchanged = {np.array_equal(z1, z0)}, residual at z finite")
km.al_residual = orig

# 2c. the projected multiplier update breaks the identity between inner stationarity and Lagrangian stationarity
#     (notes section 4.11): f = 1/2 (z + 2)^2, g = z <= 0, lambda_old = 1, mu = 1 -> inner minimiser z = -1, lambda_new = 0,
#     and |grad f + lambda_new grad g| = 1 at that point, so the outer residual must be measured, not assumed
lam_old, mu1 = 1.0, 1.0
F = lambda z: 0.5 * (z[0] + 2.0) ** 2 + (lam_old * z[0] + mu1 * z[0] ** 2 if (z[0] > 0 or lam_old > 0) else 0.0)
G = lambda z: np.array([(z[0] + 2.0) + (lam_old + 2 * mu1 * z[0] if (z[0] > 0 or lam_old > 0) else 0.0)])
H = lambda z: np.array([[1.0 + 2 * mu1]])
z_in, _, st_in = ci.newton_armijo(F, G, H, np.array([0.0]))
lam_new = max(lam_old + 2 * mu1 * z_in[0], 0.0)
stat_after = abs((z_in[0] + 2.0) + lam_new * 1.0)
check(st_in == "stationary" and abs(z_in[0] + 1.0) < 1e-6 and lam_new == 0.0 and abs(stat_after - 1.0) < 1e-6,
      f"projected update: inner minimiser z = {z_in[0]:.6f} ('{st_in}'), lambda_new = {lam_new}, Lagrangian stationarity residual after the update = {stat_after:.6f} (not small)")

# 3. Jacobians by central differences at solution A; the two Jacobian objects
zA = km.linear_init(km.ik_two_link(km.Y_STAR, True))
with contextlib.redirect_stdout(io.StringIO()):
    z, lam, kap = km.solve_komo(zA)
for label, fn in (("smoothness", km.acc_residuals), ("equality", km.eq_constraints), ("inequality", km.ineq_constraints)):
    _, J = fn(z)
    eps = 1e-6
    fd = np.column_stack([(fn(z + eps * d)[0] - fn(z - eps * d)[0]) / (2 * eps) for d in np.eye(km.N)])
    err = np.abs(J - fd).max() / max(1.0, np.abs(J).max())      # relative: RAI's control rows carry sqrt(dt)/dt^k (89.4 at T = 20)
    check(err < 1e-8, f"{label} Jacobian {J.shape} against central differences: max relative error {err:.1e}")
_, J_full = km.al_residual(zA, np.ones(3 * km.T), np.zeros(4), km.mu_default())
_, J_inner = km.al_residual(z, lam, kap, km.mu_default())
check(J_full.shape == (104, 40) and J_inner.shape == (46, 40), f"full feature Jacobian {J_full.shape}, inner residual Jacobian at solution A {J_inner.shape}")

# 4. the numbers the slides quote
st, comp = km.kkt_check(z, lam, kap)
active = sorted({int(i) // 3 + 1 for i in np.nonzero(lam > 1e-9)[0]})
check(abs(km.cost(z) - 15.1296) < 1e-3 and active == [12, 13] and st < 1e-9 * km.cost_scale() and comp < 1e-11 * km.cost_scale(),   # RAI's units (2026-10-21)
      f"solution A: cost {km.cost(z):.6f}, active knots {active}, KKT stationarity {st:.1e}, complementarity {comp:.1e}, outcome '{km.LAST['status']}'")
y, q2max, q0 = np.array([0.5, 0.2]), 60 * ci.DEG, np.array([30 * ci.DEG, 45 * ci.DEG])
with contextlib.redirect_stdout(io.StringIO()):
    q_al, lam_al, rows = ci.augmented_lagrangian(y, q2max, q0)
check(abs(lam_al[0] - 0.011918) < 1e-6 and abs(np.degrees(q_al[1]) - 60.0) < 1e-3, f"elbow limit by the augmented Lagrangian: lambda = {lam_al[0]:.6f}, q2 = {np.degrees(q_al[1]):.4f} deg, {len(rows)} outer iterations")

print("all checks passed" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
