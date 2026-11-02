"""The augmented Lagrangian on the outside and the damped Gauss-Newton step of Deck 04 on the inside (Deck 05,
sessions 4.1-4.2). The loop solves a nonlinear program whose cost is a plain sum of squares,

    minimize  f(z) = || r_f(z) ||^2          (RAI's convention: no 1/2)
    subject to  h(z) = 0,   g(z) <= 0,

with the multipliers kappa (equalities) and lambda (inequalities) and a fixed penalty weight mu:

    A(z) = f + kappa^T h + mu ||h||^2 + sum_i [g_i > 0 or lambda_i > 0] (lambda_i g_i + mu g_i^2)
    inner:  damped Gauss-Newton on A, written as 1/2 || r(z) ||^2 with the stacked residual
            r = ( sqrt(2) r_f ;  sqrt(2 mu) (h + kappa/(2 mu)) ;  sqrt(2 mu) (g_i + lambda_i/(2 mu)) for included i )
    outer:  kappa <- kappa + 2 mu h(z),    lambda_i <- max(lambda_i + 2 mu g_i(z), 0)

The outer loop stops on the three KKT residuals of the constrained problem (feasibility, Lagrangian stationarity,
complementarity), measured after the multiplier update. Deck 06's KOMO example
(lectures/04-D2-motion-planning/code/komo_two_link.py) supplies the two-link trajectory problem and calls this loop.

A problem is a `Problem`: residual(z) -> (r_f, J_f), eq(z) -> (h, J_h), ineq(z) -> (g, J_g), the number of
variables n, the Gauss-Newton damping rho, and tol_scale, the factor that puts the stationarity and complementarity
tolerances in the cost's units. The file has no print mode of its own; check_solvers.py tests it.
"""
import numpy as np


class Problem:
    def __init__(self, residual, eq, ineq, n, rho=1e-6, tol_scale=1.0):
        self.residual, self.eq, self.ineq = residual, eq, ineq
        self.n, self.rho, self.tol_scale = n, rho, tol_scale

    def cost(self, z):
        """The plain sum of squares of the cost rows (no 1/2)."""
        r, _ = self.residual(z)
        return float(r @ r)

    def grad_cost(self, z):
        r, J = self.residual(z)
        return 2.0 * J.T @ r


def al_residual(P, z, lam, kap, mu):
    """Stacked residual r(z) with 1/2||r||^2 = A(z) + const, and its Jacobian."""
    r_f, J_f = P.residual(z)
    h, J_h = P.eq(z)
    g, J_g = P.ineq(z)
    active = (g > 0) | (lam > 0)
    s = np.sqrt(2*mu)
    r = np.concatenate([np.sqrt(2.0)*r_f, s*(h + kap/(2*mu)), s*(g[active] + lam[active]/(2*mu))])   # 1/2||r||^2 = f + ...
    J = np.vstack([np.sqrt(2.0)*J_f, s*J_h, s*J_g[active]])
    return r, J


def gauss_newton(P, z, lam, kap, mu, max_iter=100, tol=1e-10, floor=1e-8, residual=al_residual):
    """The INNER solve: Deck 04's damped Gauss-Newton step on the stacked residual of A(z) with the multipliers
    lam, kap held fixed, Armijo backtracking restarted from alpha = 1 at every iteration (the included inequality rows
    are re-evaluated inside each trial, as KOMO does). Returns (z, iterations, status):
      'converged'  the accepted step is below tol: the inner loop's small-step stop (the inner residual carries the
                   multiplier shifts, so its norm does not vanish at the solution and Deck 04's relative gradient test
                   would be the wrong yardstick for it);
      'stalled'    the trial at the floor fails the Armijo test, or every trial's cost is not finite: the candidate
                   is REJECTED and z keeps its last accepted (finite) value, as in Deck 04's code;
      'limit'      max_iter reached.
    `residual` defaults to al_residual; check_solvers.py passes a stiff residual to test the rejection."""
    status = "limit"
    for it in range(max_iter):
        r, J = residual(P, z, lam, kap, mu)
        grad = J.T @ r
        delta = -np.linalg.solve(J.T @ J + P.rho*np.eye(P.n), grad)
        F0, alpha, slope = 0.5*r @ r, 1.0, float(grad @ delta)
        def F(a):
            r_new, _ = residual(P, z + a*delta, lam, kap, mu)
            return 0.5*r_new @ r_new
        ok = lambda a: np.isfinite(F(a)) and F(a) <= F0 + 0.01*a*slope     # a NaN cost compares false and must not be accepted
        while not ok(alpha) and alpha > floor:
            alpha *= 0.5
        if not ok(alpha):
            status = "stalled"                   # no finite, sufficiently decreasing trial above the floor: keep z
            break
        z = z + alpha*delta
        if np.linalg.norm(alpha*delta) < tol:
            status = "converged"
            break
    return z, it + 1, status


def solve(P, z0, mu, outer_max=30, verbose=True, residual=al_residual):
    """The OUTER loop: inner solve with the multipliers fixed, then the multiplier update. It stops on the KKT
    residuals of the constrained problem, measured with the UPDATED multipliers: feasibility (max|h|, max g+),
    Lagrangian stationarity |grad f + J_h^T kappa + J_g^T lambda| and complementarity max|lambda_i g_i|; the
    objective gradient |grad f| is not a sufficient test (it can be zero or not at a constrained optimum). The inner
    solve stops on its own rule (a small accepted step, see gauss_newton), and the three outer residuals are measured
    independently after the update: when no multiplier is clamped by the projection, the inner stationarity condition
    over the included rows coincides with the Lagrangian stationarity with the unprojected update, but a clamped row
    (lambda + 2 mu g < 0 -> lambda_new = 0) breaks that identity, which is why the outer residual is computed, not
    assumed. Returns (z, lam, kap, info) with info = {'status', 'outer', 'mu'}."""
    tol_stat, tol_comp = 1e-8*P.tol_scale, 1e-11*P.tol_scale
    h0, _ = P.eq(z0)
    g0, _ = P.ineq(z0)
    z, lam, kap = z0.copy(), np.zeros(len(g0)), np.zeros(len(h0))
    info = {"status": "limit", "outer": None, "mu": mu}
    if verbose:
        print("   outer  inner (iterations, outcome)   acc cost f   feasibility max|h|  max g+   Lagrangian stationarity   complementarity max|lambda g|   #active g   max lambda")
    for i in range(outer_max):
        z, n_inner, st_inner = gauss_newton(P, z, lam, kap, mu, residual=residual)
        h, J_h = P.eq(z)
        g, J_g = P.ineq(z)
        lam_new = np.maximum(lam + 2*mu*g, 0)
        kap_new = kap + 2*mu*h
        feas = max(np.abs(h).max(), max(g.max(), 0.0))
        stat = np.linalg.norm(P.grad_cost(z) + J_h.T @ kap_new + J_g.T @ lam_new)
        comp = np.abs(lam_new*g).max()
        if verbose:
            print(f"   {i:<6} {n_inner:<3} {st_inner:<24} {P.cost(z):.6f}   {np.abs(h).max():.1e}   {max(g.max(), 0):.1e}   {stat:.1e}                   {comp:.1e}                    {int((lam_new > 0).sum()):<10} {lam_new.max():.4f}")
        lam, kap = lam_new, kap_new
        if st_inner == "stalled":
            info["status"] = "inner stalled"
            if verbose:
                print("   the inner solve stalled: its candidate was rejected and the outer loop stops here")
            break
        if feas < 1e-9 and stat < tol_stat and comp < tol_comp:
            info["status"] = f"KKT residuals below tolerance (feasibility 1e-9, Lagrangian stationarity {tol_stat:.1e}, complementarity {tol_comp:.1e})"
            break
    info["outer"] = i + 1
    return z, lam, kap, info


def kkt_check(P, z, lam, kap):
    """Lagrangian stationarity and complementarity at (z, lam, kap)."""
    _, J_h = P.eq(z)
    g, J_g = P.ineq(z)
    stationarity = P.grad_cost(z) + J_h.T @ kap + J_g.T @ lam
    return np.linalg.norm(stationarity), np.abs(lam*g).max()
