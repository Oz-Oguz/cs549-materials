"""Session 3.1 worked example: gradient descent, step size, and Armijo backtracking on the
"elongated quadratic" f(z) = z1^2 + 10 z2^2 (the same function as the opt-explorer Unconstrained Opt. tab),
plus Newton's single step on it. Plain NumPy. Every number printed here appears in notes.md §5.

Run:  python descent.py          Expected runtime: well under a second.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

A = np.diag([1.0, 10.0])                     # f(z) = z^T A z = z1^2 + 10 z2^2
Z0 = np.array([3.0, 2.0])                    # the app's "far" start


def f(z):
    return float(z @ A @ z)


def grad(z):
    return 2.0 * A @ z


def hess(z):
    return 2.0 * A


def gd_fixed(z0, alpha, iters):
    z = z0.copy(); path = [z.copy()]
    for _ in range(iters):
        z = z - alpha * grad(z); path.append(z.copy())
    return np.array(path)


def armijo_backtracking(z, d, alpha0=1.0, shrink=0.5, c=0.01, verbose=True):
    """Backtracking line search on the Armijo condition f(z + a d) <= f(z) + c a grad^T d."""
    fz, slope = f(z), float(grad(z) @ d)
    a = alpha0
    while True:
        lhs, rhs = f(z + a * d), fz + c * a * slope
        if verbose:
            print(f"      try alpha = {a:8.5f}: f(z + alpha d) = {lhs:10.3f}  {'<=' if lhs <= rhs else '> '}  {rhs:8.3f}  -> {'accept' if lhs <= rhs else 'shrink'}")
        if lhs <= rhs:
            return a
        a *= shrink


def gd_armijo(z0, iters, alpha0=1.0, grow=1.2, verbose_first=True):
    """Backtrack when needed, grow the step 1.2x after each accepted step (no cap on alpha: the
    backtracking already limits it, and alpha has units, so a fixed cap suits no cost in general)."""
    z, a = z0.copy(), alpha0
    path, alphas = [z.copy()], []
    for k in range(iters):
        d = -grad(z)
        a = armijo_backtracking(z, d, alpha0=a, verbose=(verbose_first and k == 0))
        z = z + a * d; path.append(z.copy()); alphas.append(a)
        a = grow * a
    return np.array(path), np.array(alphas)


if __name__ == "__main__":
    lam = np.linalg.eigvalsh(hess(Z0))
    print("=== f(z) = z1^2 + 10 z2^2,  z0 =", Z0, " f(z0) =", f(Z0), "===")
    print("gradient at z0 =", grad(Z0), "  Hessian = diag", np.diag(hess(Z0)), f"  condition number = {lam[1]/lam[0]:.0f}")

    # (a) fixed step sizes: per-coordinate contraction factors 1 - alpha*lambda_i
    print("\n(a) fixed-step gradient descent, z^(i+1) = (I - alpha H) z^(i); factors (1 - 2 alpha, 1 - 20 alpha)")
    print("    first-step comparison from (3, 2):")
    for alpha in (0.05, 0.5):
        factors = 1.0 - alpha * lam
        first = gd_fixed(Z0, alpha, 1)[-1]
        print(f"    alpha = {alpha:.2f}: factors = {factors}, next point = {first}")
    for alpha in (0.02, 0.05, 0.08, 0.0909, 0.10, 0.11):
        fac = 1.0 - alpha * lam
        path = gd_fixed(Z0, alpha, 20)
        z20 = path[-1]
        note = ("diverges" if np.abs(fac).max() > 1 else "oscillates forever" if np.isclose(np.abs(fac).max(), 1.0, atol=1e-9) else "converges")
        print(f"    alpha = {alpha:6.4f}: factors = ({fac[0]:7.4f}, {fac[1]:7.4f})  |z_20| = {np.linalg.norm(z20):10.4f}  f(z_20) = {f(z20):12.4f}   {note}")
    a_opt = 2.0 / (lam[0] + lam[1])
    print(f"    best fixed step: alpha* = 2/(lambda_min + lambda_max) = {a_opt:.4f}, both factors +-{1 - a_opt*lam[0]:.4f} = (cond(H)-1)/(cond(H)+1)")
    print("    zeroing each coordinate in one step requires 1/lambda_1 = 0.5 and 1/lambda_2 = 0.05; no single alpha zeros both from this start")

    # (b) Armijo backtracking, first step in detail
    print("\n(b) Armijo backtracking at z0 with d = -grad f = ", -grad(Z0), ", alpha0 = 1, shrink 0.5, c = 0.01:")
    path, alphas = gd_armijo(Z0, 12)
    print("    accepted alphas over 12 steps:", alphas)
    print("    iterates z^(i):"); [print(f"      i = {k:2d}: z = {z}  f = {f(z):9.4f}") for k, z in enumerate(path[:6])]
    print(f"      ...  i = 12: z = {path[-1]}  f = {f(path[-1]):.4e}")

    # (c) Newton: one step on a quadratic
    zN = Z0 - np.linalg.solve(hess(Z0), grad(Z0))
    print("\n(c) Newton step z - H^{-1} grad f =", zN, " f =", f(zN), "  (one step, exactly, for any quadratic with A positive definite)")

    # (d) Newton on a non-convex function: it finds stationary points, not minima
    def f2(z): return z[0]**2 - z[1]**2
    def g2(z): return np.array([2*z[0], -2*z[1]])
    H2 = np.diag([2.0, -2.0])
    z = np.array([0.5, 1.0])
    zN2 = z - np.linalg.solve(H2, g2(z))
    dN = zN2 - z
    print("\n(d) f(z) = z1^2 - z2^2 (a saddle), z =", z, ": Newton goes to", zN2, "= the saddle point; f from", f2(z), "to", f2(zN2),
          f"; the Newton direction {dN} climbs: grad^T d = {g2(z) @ dN:+.2f}")
    for rho in (1.0, 3.0):
        d = -np.linalg.solve(H2 + rho*np.eye(2), g2(z))
        print(f"    regularised (H + rho I) with rho = {rho}: direction {d}, descent? grad^T d = {g2(z) @ d:.3f} ({'yes' if g2(z) @ d < 0 else 'no'}); positive definite once rho > 2")

    # (e) counts: how many GD steps for f < 1e-6 at the best fixed step, vs Newton's 1
    path = gd_fixed(Z0, a_opt, 200)
    k = next(i for i, zk in enumerate(path) if f(zk) < 1e-6)
    print(f"\n(a, continued) steps to f < 1e-6: gradient descent at alpha* needs {k}; Newton needs 1")
