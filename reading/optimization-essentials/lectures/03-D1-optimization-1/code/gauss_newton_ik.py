"""Sessions 3.1-3.2 worked examples: inverse kinematics of the two-link arm (l1 = 0.4 m, l2 = 0.3 m,
sessions 2.3-2.4) as the least-squares problem

    min_q f(q) = 1/2 |phi(q) - y*|^2,        grad f(q) = J(q)^T (phi(q) - y*),

solved by one gradient step (3.1) and by the damped Gauss-Newton step of Spring 2026 deck 03 p.56 (3.2):

    q^(i+1) = q^(i) + J#(q^(i)) (y* - phi(q^(i))),      J# = (J^T K J + D)^{-1} J^T K

with K = I and D = rho I. rho = 0 is plain Gauss-Newton (J^{-1} when J is square); rho > 0 is the
damped step (Levenberg-Marquardt with a fixed rho). Gradient descent with Armijo, exact Newton,
and the Armijo-protected Gauss-Newton direction are run for comparison.

Conventions used by every number printed here: the residual is r = phi(q) - y* (the essentials
write e = y* - phi = -r); the cost carries the 1/2; Armijo uses c = 0.01, halves from alpha = 1 and
REJECTS the step when the first trial at or below its floor (1e-6 unless stated) fails the test;
the iteration then stops where it is and reports "stalled". The floor is this implementation's
choice: a smaller trial can still pass the test (section (k) prints the trials around the floor),
so "stalled" describes this code's stopping rule, not a theorem about the direction. Gradient
descent grows alpha by 1.2 after each accepted step with no cap. Every run reports one of four
outcomes: "solved" (|phi - y*| < 1e-6), "stationary" (the residual is not small but passes the
course's relative gradient test |J^T r| <= 1e-8 |J|_F |r|, stated for K = I; it is not an angle to
the range of J and not MINPACK's columnwise-cosine criterion, and it can stop with task error left
along a weak direction; an absolute |grad f| test would fire early at a singular solution, where
J^T r shrinks faster than r, see (j)), "stalled" (the line search rejected the direction at its
floor), or "limit" (the iteration cap; "diverged" when a step is not finite or longer than 1e3 rad,
which the unprotected run of (k) reaches at iteration 364 if left to run). There is no small-step
stop: the |step| < 1e-6 values printed in (k) are observations. The Armijo search on the
Gauss-Newton direction restarts from alpha = 1 at every iteration (the natural step of a Newton-type
direction is 1); the grow-by-1.2 rule is gradient descent's. None of the outcomes certifies that no
q reaches y*: section (p) shows a reachable target where the standard gradient and Gauss-Newton steps are zero. The normal equations are solved as
written, (J^T J + rho I) delta = -J^T r, which squares the conditioning of J; a QR or SVD of J
(np.linalg.lstsq) avoids that, and section (k) shows the squared conditioning at a very small floor.
Every number printed here appears in notes.md section 5.

Run:  python gauss_newton_ik.py          Expected runtime: a few seconds.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
L1, L2 = 0.4, 0.3
d2r, r2d = np.deg2rad, np.degrees


def phi(q):
    q1, q2 = q
    return np.array([L1*np.cos(q1) + L2*np.cos(q1+q2), L1*np.sin(q1) + L2*np.sin(q1+q2)])


def jac(q):
    q1, q2 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1+q2), np.cos(q1+q2)
    return np.array([[-L1*s1 - L2*s12, -L2*s12], [L1*c1 + L2*c12, L2*c12]])


def cost(q, y):
    r = phi(q) - y
    return 0.5 * float(r @ r)


def grad(q, y):
    """grad f = J^T (phi - y*): the transpose rule of the essentials, section 6."""
    return jac(q).T @ (phi(q) - y)


def hess_exact(q, y):
    """Exact Hessian of f: J^T J plus the residual-weighted second derivatives of phi (notes eq. (11))."""
    q1, q2 = q
    c1, s1, c12, s12 = np.cos(q1), np.sin(q1), np.cos(q1+q2), np.sin(q1+q2)
    Hx = np.array([[-L1*c1 - L2*c12, -L2*c12], [-L2*c12, -L2*c12]])
    Hy = np.array([[-L1*s1 - L2*s12, -L2*s12], [-L2*s12, -L2*s12]])
    r = phi(q) - y
    J = jac(q)
    return J.T @ J + r[0]*Hx + r[1]*Hy


def fd_grad(q, y, h=1e-6):
    """Central-difference gradient, the check the kinematics notes promise for every gradient."""
    return np.array([(cost(q + h*e, y) - cost(q - h*e, y)) / (2*h) for e in np.eye(2)])


def armijo(q, d, y, c=0.01, floor=1e-6):
    """Backtracking on the Armijo test from alpha = 1, halving. Returns (alpha, ok). The last trial is
    the first alpha at or below the floor; ok=False means that trial failed, and the caller must not
    take the step. A smaller trial might pass: the floor is this implementation's stopping rule."""
    g = grad(q, y)
    a, c0, slope = 1.0, cost(q, y), float(g @ d)
    while cost(q + a*d, y) > c0 + c*a*slope and a > floor:
        a *= 0.5
    return a, cost(q + a*d, y) <= c0 + c*a*slope


def gn_step(q, y, rho):
    """One damped Gauss-Newton step; returns the new q and the step."""
    J = jac(q)
    Jsharp = np.linalg.solve(J.T @ J + rho*np.eye(2), J.T)        # (J^T J + rho I)^{-1} J^T
    step = Jsharp @ (y - phi(q))
    return q + step, step


STATUS_TEXT = {"solved": "solved: |phi - y*| < tol",
               "stationary": "stationary: the relative gradient test |J^T r| <= gtol |J|_F |r| passes with the residual not small (a least-squares compromise if it is a minimum, or a bad stationary point)",
               "stalled": "stalled: the line search rejected the direction at its floor",
               "diverged": "diverged: a step was not finite or longer than 1e3 rad",
               "limit": "limit: the iteration cap was reached"}


def stationary(q, y, gtol=1e-8):
    """The course's relative gradient test: |J^T r| <= gtol |J|_F |r| (Frobenius norm, K = I).
    Dimensionless, so scaling the cost does not change it, and it does not fire early at a singular
    solution where J^T r -> 0 faster than r. It is NOT an angle to the range of J and NOT MINPACK's
    columnwise cosine max_j |J_j^T r| / (|J_j| |r|): J = diag(1, 1e-9), r = (0, 1) passes at 1e-9
    although r is parallel to the second column, so the test can stop with task error left along
    a weak direction."""
    r = phi(q) - y
    return np.linalg.norm(jac(q).T @ r) <= gtol * np.linalg.norm(jac(q)) * np.linalg.norm(r)


def run_gn(q0, y, rho, iters=15, tol=1e-6, gtol=1e-8, linesearch=False, floor=1e-6):
    """Damped Gauss-Newton iteration. Returns (hist, status) with status one of 'solved', 'stationary',
    'stalled', 'diverged', 'limit' (see STATUS_TEXT). With linesearch=True the direction is kept and
    the step length is chosen by Armijo backtracking (restarted from alpha = 1 at every iteration) with
    the given floor; a direction whose trial at the floor fails is rejected and the run stops there
    ('stalled' at iteration hist[-1]['k'] + 1). gtol=0 disables the stationarity stop (used to count
    residual rises over a fixed number of iterations)."""
    q = q0.copy()
    hist = [dict(k=0, q=q.copy(), res=np.linalg.norm(phi(q) - y), grad=np.linalg.norm(grad(q, y)), step=None, alpha=None)]
    status = "limit"
    for k in range(1, iters + 1):
        q_new, step = gn_step(q, y, rho)
        alpha = None
        if linesearch:
            alpha, ok = armijo(q, step, y, floor=floor)
            if not ok:
                status = "stalled"
                break
            step = alpha * step
            q_new = q + step
        q = q_new
        res, gn = np.linalg.norm(phi(q) - y), np.linalg.norm(grad(q, y))
        hist.append(dict(k=k, q=q.copy(), res=res, grad=gn, step=np.linalg.norm(step), alpha=alpha))
        if not np.isfinite(res) or np.linalg.norm(step) > 1e3:
            status = "diverged"; break
        if res < tol:
            status = "solved"; break
        if gtol > 0 and stationary(q, y, gtol):
            status = "stationary"; break
    return hist, status


def run_lm(q0, y, rho0=0.05, iters=60, tol=1e-6, gtol=1e-8, max_trials=30):
    """Residual-based LM with actual/predicted cost reduction (slide 22).

    Returns (accepted_history, status, trial_evaluations). Each history entry
    after the initial state stores the damping USED and the gain ratio eta.
    A rejected trial leaves q unchanged and is recorded only in trials.

    The predicted reduction excludes the step penalty. Accept eta > 1e-4;
    double rho for eta < 0.25, halve it for eta > 0.75, otherwise retain it.
    These are explicit illustrative choices, not universal LM constants.
    The same solved/stationary tests as run_gn apply. Exhausting the retry
    budget or obtaining no positive predicted decrease reports 'stalled'.
    """
    if rho0 <= 0 or max_trials < 1:
        raise ValueError("LM needs positive initial damping and a trial budget")
    q, rho = q0.copy(), float(rho0)
    hist = [dict(k=0, q=q.copy(), res=np.linalg.norm(phi(q) - y),
                 grad=np.linalg.norm(grad(q, y)), step=None, rho=None, eta=None)]
    trials = []
    for k in range(1, iters + 1):
        if hist[-1]["res"] < tol:
            return hist, "solved", trials
        if gtol > 0 and stationary(q, y, gtol):
            return hist, "stationary", trials
        r, J = phi(q) - y, jac(q)
        f0 = 0.5 * float(r @ r)
        for _ in range(max_trials):
            rho_used = rho
            q_trial, step = gn_step(q, y, rho_used)
            predicted_residual = r + J @ step
            predicted = f0 - 0.5 * float(predicted_residual @ predicted_residual)
            actual = f0 - cost(q_trial, y)
            eta = actual / predicted if predicted > 0 else float("-inf")
            accepted = bool(np.isfinite(eta) and eta > 1e-4)
            trials.append(dict(k=k, q_base=q.copy(), q_trial=q_trial.copy(),
                               rho=rho_used, predicted=predicted, actual=actual,
                               eta=eta, accepted=accepted))
            if eta < 0.25 or not np.isfinite(eta):
                rho = min(2 * rho_used, 1e12)
            elif eta > 0.75:
                rho = max(0.5 * rho_used, 1e-12)
            if accepted:
                q = q_trial
                hist.append(dict(k=k, q=q.copy(), res=np.linalg.norm(phi(q) - y),
                                 grad=np.linalg.norm(grad(q, y)),
                                 step=np.linalg.norm(step), rho=rho_used, eta=eta))
                break
            if predicted <= 0:
                return hist, "stalled", trials
        else:
            return hist, "stalled", trials
    if hist[-1]["res"] < tol:
        return hist, "solved", trials
    if gtol > 0 and stationary(q, y, gtol):
        return hist, "stationary", trials
    return hist, "limit", trials


def run_gd(q0, y, iters=5000, tol=1e-6, cap=None):
    """Gradient descent on f with Armijo backtracking (c = 0.01, shrink 0.5, grow 1.2, no cap on alpha).
    cap=1.0 reproduces the old rule alpha <- min(1.2 alpha, 1), kept only to document its effect."""
    q, a = q0.copy(), 1.0
    for k in range(1, iters + 1):
        g = grad(q, y)
        d = -g
        while cost(q + a*d, y) > cost(q, y) + 0.01 * a * float(g @ d):
            a *= 0.5
        q = q + a*d
        a = 1.2*a if cap is None else min(1.2*a, cap)
        if np.linalg.norm(phi(q) - y) < tol:
            return k, q
    return iters, q


def run_newton(q0, y, iters=30, tol=1e-12):
    """Exact Newton on f (second derivatives of phi included): finds stationary points, not minima."""
    q = q0.copy()
    for k in range(1, iters + 1):
        q = q - np.linalg.solve(hess_exact(q, y), grad(q, y))
        if np.linalg.norm(grad(q, y)) < tol:
            break
    return k, q


def report(q, y):
    """The four quantities of an end point: residual, its excess over the best attainable residual,
    the tip's distance to the closest reachable point, and the gradient norm."""
    n = np.linalg.norm(y)
    r_min = max(0.0, n - (L1 + L2))
    p_c = (L1 + L2) * y / n if n > L1 + L2 else y
    res = np.linalg.norm(phi(q) - y)
    return (f"|phi - y*| = {res:.6f} m (excess over the best attainable {r_min:.6f}: {res - r_min:.1e} m), "
            f"tip {np.linalg.norm(phi(q) - p_c)*1000:.2f} mm from the closest reachable point, |grad f| = {np.linalg.norm(grad(q, y)):.1e}")


def table(hist, title):
    print(f"\n{title}")
    print("     i    q1 [deg]   q2 [deg]   phi(q)                  |phi - y*|   |step| [rad]   alpha")
    for h in hist:
        st = '' if h['step'] is None else f"{h['step']:8.4f}"
        al = '' if h['alpha'] is None else f"{h['alpha']:.2e}"
        print(f"    {h['k']:2d}   {r2d(h['q'][0]):8.3f}   {r2d(h['q'][1]):8.3f}   {phi(h['q'])}   {h['res']:10.2e}   {st:>8}   {al}")


if __name__ == "__main__":
    q0 = d2r([30.0, 45.0])
    y = np.array([0.5, 0.2])
    print("=== Two-link inverse kinematics, q0 = (30, 45) deg, phi(q0) =", phi(q0), "===")

    # (e) session 3.1: the gradient of the cost, checked, and one gradient step (Jacobian-transpose IK)
    print(f"\n(e) target y* = {y}, |y*| = {np.linalg.norm(y):.4f} (reachable: 0.1 <= |y*| <= 0.7); f(q0) = {cost(q0, y):.4f} m^2")
    g = grad(q0, y)
    print(f"    grad f(q0) = J^T (phi - y*) = {g} m^2/rad; central differences give {fd_grad(q0, y)} (max difference {np.abs(g - fd_grad(q0, y)).max():.1e})")
    a, ok = armijo(q0, -g, y)
    q1 = q0 - a*g
    print(f"    one gradient step with Armijo from alpha = 1: alpha = {a} accepted; step = {r2d(-a*g)} deg; tip {phi(q0)} -> {phi(q1)};"
          f" |phi - y*| {np.linalg.norm(phi(q0) - y):.4f} -> {np.linalg.norm(phi(q1) - y):.4f} m")
    _, gn = gn_step(q0, y, 0.0)
    ang = np.degrees(np.arccos(float(-g @ gn) / np.linalg.norm(g) / np.linalg.norm(gn)))
    print(f"    the Gauss-Newton step of 3.2 from the same point is {r2d(gn)} deg, {ang:.0f} deg away from -grad f")

    # (f) session 3.2: the first Gauss-Newton step by hand, then the table
    J0 = jac(q0)
    print("\n(f) first Gauss-Newton step by hand: J(q0) =\n", J0, "\n    y* - phi(q0) =", y - phi(q0))
    Js = np.linalg.solve(J0.T @ J0, J0.T)
    print("    J# = (J^T J)^{-1} J^T = J^{-1} (square case) =\n", Js, "\n    step = J# (y* - phi) =", Js @ (y - phi(q0)), "rad =", r2d(Js @ (y - phi(q0))), "deg")
    print(f"    Armijo on that step: f falls from {cost(q0, y):.4f} to {cost(q0 + Js @ (y - phi(q0)), y):.4f}, threshold {cost(q0, y) + 0.01*float(g @ (Js @ (y - phi(q0)))):.4f}: accepted at alpha = 1"
          f" (the linear model promised residual 0 and delivered {np.linalg.norm(phi(q0 + Js @ (y - phi(q0))) - y):.3f} m)")
    hist, _ = run_gn(q0, y, rho=0.0)
    table(hist, "    Gauss-Newton, rho = 0:")
    ratios = [b['res'] / a['res']**2 for a, b in zip(hist[1:], hist[2:])]
    print(f"    '{_}' in {hist[-1]['k']} iterations; residual exponents {[int(np.floor(np.log10(h['res']))) for h in hist]}: quadratic from step 2;"
          f" res_(i+1)/res_i^2 = {', '.join(f'{x:.2f}' for x in ratios)} for i = 1..{len(ratios)}: a constant times the square, not the square itself")

    # (g) damping: the first step, the iteration counts, and gradient descent on the same cost
    for rho in (0.05, 0.5):
        _, st = gn_step(q0, y, rho)
        h, _ = run_gn(q0, y, rho=rho, iters=400)
        print(f"\n(g) damped, rho = {rho}: first step {r2d(st)} deg, residual after it {np.linalg.norm(phi(q0 + st) - y):.4f} m"
              f" (the linear model predicted {np.linalg.norm(phi(q0) + J0 @ st - y):.4f} m left); converged in {h[-1]['k']} iterations to the same q = {r2d(h[-1]['q'])}")
    k_gd, q_gd = run_gd(q0, y)
    k_cap, _ = run_gd(q0, y, cap=1.0)
    print(f"    gradient descent with Armijo on the same cost: {k_gd} iterations to |phi - y*| < 1e-6, q = {r2d(q_gd)} deg"
          f" (with the old cap alpha <= 1 it needs {k_cap}: alpha carries units of rad^2/m^2 here and wants to be about 5)")
    rho_ad, q, hist_ad = 0.05, q0.copy(), []
    for k_ad in range(1, 30):                       # Simple cost-decrease adaptation; no prediction-ratio check
        while True:
            qn, _ = gn_step(q, y, rho_ad)
            if cost(qn, y) < cost(q, y):
                q = qn; rho_ad = max(rho_ad / 10, 1e-12); break
            rho_ad *= 10
            if rho_ad > 1e12:
                break
        hist_ad.append(np.linalg.norm(phi(q) - y))
        if hist_ad[-1] < 1e-6:
            break
    print(f"    simple cost-decrease adaptation from rho = 0.05: {k_ad} iterations, residuals {', '.join(f'{h:.1e}' for h in hist_ad)}: falling at every step")
    lm_hist, lm_status, lm_trials = run_lm(q0, y)
    lm_dampings = ", ".join(format(h["rho"], ".6g") for h in lm_hist[1:])
    lm_residuals = ", ".join(format(h["res"], ".2e") for h in lm_hist)
    print(f"    ratio-based adaptive LM (slide 27), rho starts at 0.05: {lm_hist[-1]['k']} accepted updates, "
          f"{sum(not t['accepted'] for t in lm_trials)} rejected trials, {lm_status}; "
          f"damping used {lm_dampings}; residuals {lm_residuals}")
    qs = d2r([-11.5171, 80.4057])
    ev = np.linalg.eigvalsh(jac(qs).T @ jac(qs))
    a_star = 2 / (ev[0] + ev[1])
    q = q0.copy()
    for k_fix in range(1, 5000):
        q = q - a_star * grad(q, y)
        if np.linalg.norm(phi(q) - y) < 1e-6:
            break
    ev0 = np.linalg.eigvalsh(J0.T @ J0)
    print(f"    near the solution the cost is a bowl with Hessian J^T J: eigenvalues {ev}, cond {ev[1]/ev[0]:.1f}; best fixed step alpha* = {a_star:.2f} needs {k_fix} iterations"
          f" (at q0, J^T J has eigenvalues {ev0} and cond {ev0[1]/ev0[0]:.1f} = 5.84^2, but the true Hessian there is indefinite: eigenvalues {np.linalg.eigvalsh(hess_exact(q0, y))})")

    # (h) the other solution: start elbow-down
    q0b = d2r([60.0, -45.0])
    hist, _ = run_gn(q0b, y, rho=0.0)
    print(f"\n(h) same target from q0 = (60, -45) deg: Gauss-Newton converges in {hist[-1]['k']} iterations to q = {r2d(hist[-1]['q'])} deg (the elbow-down solution). Local method: the start picks the solution.")

    # (i) exact Newton on the arm: a stationary point that is not a minimum
    kN, qN = run_newton(q0, y)
    print(f"\n(i) exact Newton (with the second derivatives of phi) from q0: {kN} steps to q = {r2d(qN)} deg, tip {phi(qN)}, |phi - y*| = {np.linalg.norm(phi(qN) - y):.4f} m:"
          f" the straight arm pointing at the target, past it, a saddle of f (Hessian eigenvalues {np.linalg.eigvalsh(hess_exact(qN, y))}); grad f = {grad(qN, y)}")
    yu_ = np.array([0.8, 0.1]); kb, qb = run_newton(q0, yu_)
    qb_w = (qb + np.pi) % (2*np.pi) - np.pi        # wrap to (-180, 180] deg for printing
    print(f"    exact Newton toward the unreachable (0.8, 0.1) from q0: {kb} steps to q = {r2d(qb_w)} deg, |phi - y*| = {np.linalg.norm(phi(qb) - yu_):.3f} m, Hessian eigenvalues {np.linalg.eigvalsh(hess_exact(qb, yu_))}: a folded-arm stationary point, far from the closest reachable point")
    q2gn = run_gn(q0, yu_, 0.0, iters=2)[0][-1]['q']; kc, qc = run_newton(q2gn, yu_)
    print(f"    exact Newton toward (0.8, 0.1) from Gauss-Newton's second iterate {r2d(q2gn)} deg: {kc} steps to q = {r2d(qc)} deg, |phi - y*| = {np.linalg.norm(phi(qc) - yu_):.6f} m, the closest reachable point; every such comparison must name its start")

    # (j) a target on the reach boundary: linear convergence
    yb = np.array([0.7, 0.0])
    print(f"\n(j) target y* = {yb}, |y*| = 0.7 (on the boundary)")
    hist, status = run_gn(q0, yb, rho=0.0, iters=12)
    table(hist, "    Gauss-Newton, rho = 0, full steps:")
    print(f"    -> '{status}' after {hist[-1]['k']} iterations; converges, but only LINEARLY: q2 halves each step and the residual, quadratic in q2 near the stretched arm, is divided by four."
          "\n       The solution is the stretched arm, where det J = l1 l2 sin q2 = 0. An absolute gradient test would stop this run early:"
          f" at iteration 9, |grad f| = {hist[9]['grad']:.1e} while |phi - y*| = {hist[9]['res']:.1e}, because J^T r shrinks faster than r at a singular solution; the relative gradient test lets it finish")
    hist, status = run_gn(q0, yb, rho=0.0, iters=40, linesearch=True)
    print(f"    the same direction with Armijo: every full step passes the test; '{status}' after {hist[-1]['k']} iterations, |phi - y*| = {hist[-1]['res']:.1e}")
    hist, _ = run_gn(q0, yb, rho=0.01, iters=60)
    print(f"    damped, rho = 0.01, full steps: after 60 iterations |phi - y*| = {hist[-1]['res']:.1e}, q2 = {r2d(hist[-1]['q'][1]):.2f} deg: at a singular solution the damping's own rate rho/(sigma_min^2 + rho) -> 1, so fixed damping is slow here")

    # (k) an unreachable target: the blow-up, and the two protections compared by four quantities
    yu = np.array([0.8, 0.1])
    print(f"\n(k) target y* = {yu}, |y*| = {np.linalg.norm(yu):.4f} (unreachable); best attainable residual {np.linalg.norm(yu) - 0.7:.6f} m at the closest reachable point {0.7*yu/np.linalg.norm(yu)}, the stretched arm at q = ({r2d(np.arctan2(yu[1], yu[0])):.3f}, 0) deg")
    hist, _ = run_gn(q0, yu, rho=0.0, iters=12)
    table(hist, "    Gauss-Newton, rho = 0, full steps:")
    print("    -> as the arm straightens, det J -> 0 and (J^T J)^{-1} blows up: a 50 rad step, then wandering; this 12-step run does not settle")
    hist, status = run_gn(q0, yu, rho=0.0, iters=400, linesearch=True)
    h_long, s_long = run_gn(q0, yu, rho=0.0, iters=400)
    print(f"    left to run, the unprotected iteration reports '{s_long}' at iteration {h_long[-1]['k']} (a step of {h_long[-1]['step']:.0f} rad)")
    q_c = np.array([np.arctan2(yu[1], yu[0]), 0.0]); q_1 = q_c + np.deg2rad([1.0, 0.0])
    print(f"    the residual is flat along the boundary: turning the stretched arm by 1 deg from q = (7.125, 0) moves the tip {np.linalg.norm(phi(q_1) - phi(q_c))*1000:.1f} mm and raises the residual by only {(np.linalg.norm(phi(q_1) - yu) - np.linalg.norm(phi(q_c) - yu))*1000:.2f} mm")
    hist, status = run_gn(q0, yu, rho=0.0, iters=400, linesearch=True)
    print(f"    Gauss-Newton direction + Armijo (floor 1e-6, failed trial rejected, alpha restarted from 1 each iteration): '{status}' at iteration {hist[-1]['k'] + 1}, at q = {r2d(hist[-1]['q'])} deg, phi = {phi(hist[-1]['q'])}")
    print(f"        {report(hist[-1]['q'], yu)}")
    q_st = hist[-1]['q']; _, d_st = gn_step(q_st, yu, 0.0); g_st = grad(q_st, yu)
    ang = np.degrees(np.arccos(float(-g_st @ d_st) / np.linalg.norm(g_st) / np.linalg.norm(d_st)))
    print(f"        -> not a stationary point: at the stall the Gauss-Newton direction is {ang:.2f} deg from downhill and {np.linalg.norm(d_st):.0f} rad long")
    print("        the trials around this implementation's floor (alpha halved from 1; the test is f(q + alpha d) - f(q) <= 0.01 alpha grad f^T d):")
    c0_st, slope_st = cost(q_st, yu), float(g_st @ d_st)
    for pw in (19, 20, 21, 22):
        a_t = 2.0 ** -pw; ch = cost(q_st + a_t*d_st, yu) - c0_st; bound = 0.01*a_t*slope_st
        print(f"            alpha = 2^-{pw} = {a_t:.2e} ({'the last trial at floor 1e-6' if pw == 20 else 'below the floor' if pw > 20 else 'above the floor'}): f changes by {ch:+.2e}, allowed at most {bound:+.2e}: {'passes' if ch <= bound else 'fails'}")
    print("        so the rejection is the floor's: a smaller trial passes, moving the joints by alpha |d| = 2^-21 * 1670 rad = 8e-4 rad for a cost change of -6e-10. Other floors, same direction rule:")
    for floor in (1e-8, 1e-12, 1e-15):
        try:
            h_f, s_f = run_gn(q0, yu, rho=0.0, iters=400, linesearch=True, floor=floor)
            at = h_f[-1]['k'] + 1 if s_f == 'stalled' else h_f[-1]['k']
            print(f"            floor {floor:.0e}: '{s_f}' at iteration {at}; {report(h_f[-1]['q'], yu)}")
        except np.linalg.LinAlgError as err:
            print(f"            floor {floor:.0e}: the normal matrix J^T J became numerically singular ({err}); forming J^T J squares the conditioning of J, which a QR or SVD of J avoids")
    print("        -> lowering the floor lets a few more tiny steps pass and leaves the tip 9.96 mm from the closest point: the poor progress is the direction's, the stopping event is the floor's")
    for rho in (0.001, 0.01, 0.1):
        hist, _ = run_gn(q0, yu, rho=rho, iters=100, gtol=0.0)
        settle = next((h['k'] for h in hist if h['step'] is not None and h['step'] < 1e-6), None)
        stat = next((h['k'] for h in hist if stationary(h['q'], yu)), None)
        rises = sum(1 for a, b in zip(hist, hist[1:]) if b['res'] > a['res'] + 1e-12)
        print(f"    damped, rho = {rho}, full steps, stationarity stop off: after 100 iterations q = {r2d(hist[-1]['q'])} deg, phi = {phi(hist[-1]['q'])}, det J = {np.linalg.det(jac(hist[-1]['q'])):.1e}; the residual rose {rises} times; |step| < 1e-6 from step {settle} (an observation: no small-step stop); the relative gradient test would pass from step {stat}")
        print(f"        {report(hist[-1]['q'], yu)}")
    print("        -> rho = 0.01 reaches the closest reachable point (grad f = 0, the least-squares compromise) and settles from step 24; rho = 0.1 also reaches it, settling from step 71; rho = 0.001 is too little damping for this target and wanders")
    print("    the same damped runs with the stationarity stop on (the runs the slide reports), measured at the iterate where they stop:")
    for rho in (0.01, 0.1):
        h_s, s_s = run_gn(q0, yu, rho=rho, iters=400)
        print(f"        rho = {rho}: '{s_s}' at step {h_s[-1]['k']}, |step| there {h_s[-1]['step']:.1e} rad; {report(h_s[-1]['q'], yu)}")
    rho_ad, q, settle, stat_ad = 1e-3, q0.copy(), None, None
    for k_ad in range(1, 61):
        while True:
            qn, st = gn_step(q, yu, rho_ad)
            if cost(qn, yu) < cost(q, yu):
                q = qn; rho_ad = max(rho_ad / 10, 1e-12); break
            rho_ad *= 10
            if rho_ad > 1e12:
                st = np.zeros(2); break
        if settle is None and np.linalg.norm(st) < 1e-6:
            settle = k_ad
        if stat_ad is None and stationary(q, yu):
            stat_ad = k_ad
    print(f"    adaptive damping from rho = 0.001 on the same target: |step| < 1e-6 from step {settle}; 'stationary' from step {stat_ad}; {report(q, yu)}")

    # (l) the set of fixed points does not depend on D; which one is reached can
    qa = run_gn(q0, y, 0.0)[0][-1]['q']
    qb = run_gn(q0, y, 0.5, iters=400)[0][-1]['q']
    same = diff = fail = 0; fail_by_rho = {0.0: 0, 0.5: 0}
    for a1 in np.linspace(-np.pi, np.pi, 35):
        for a2 in np.linspace(-np.pi, np.pi, 35):
            ends = []
            for rho in (0.0, 0.5):
                try:
                    h, _ = run_gn(np.array([a1, a2]), y, rho, iters=400)
                except np.linalg.LinAlgError:
                    h = [dict(res=1.0, q=None)]
                ends.append(h[-1]['q'] if h[-1]['res'] < 1e-6 else None)
                if ends[-1] is None:
                    fail_by_rho[rho] += 1
            if ends[0] is None or ends[1] is None:
                fail += 1
            elif np.linalg.norm(np.mod(ends[0] - ends[1] + np.pi, 2*np.pi) - np.pi) > 1e-3:
                diff += 1
            else:
                same += 1
    print(f"\n(l) fixed points with rho = 0 and rho = 0.5 from q0: {r2d(qa)} and {r2d(qb)} deg, difference {np.abs(qa - qb).max():.1e} rad: D changes the path, not this answer."
          f"\n    On a 35 x 35 grid of starts the two damping values reach the same solution from {same} starts, different solutions from {diff}, and one of them fails from {fail} (rho = 0 fails from {fail_by_rho[0.0]}, rho = 0.5 from {fail_by_rho[0.5]}):"
          "\n    D does not change the SET of fixed points (J# (y* - phi) = 0 iff J^T (y* - phi) = 0), but it can change which one the iteration reaches")

    # (m) what Gauss-Newton does not know: an elbow limit, and the obvious fix
    lim = d2r(60.0)
    q = q0.copy()
    for k in range(1, 200):
        qn, _ = gn_step(q, y, 0.0)
        qn[1] = np.clip(qn[1], -lim, lim)
        if np.linalg.norm(qn - q) < 1e-12:
            break
        q = qn
    print(f"\n(m) both solutions bend the elbow to |q2| = 80.4 deg. With an elbow stop at 60 deg, clipping q2 after each Gauss-Newton step stalls at q = {r2d(q)} deg,"
          f" tip {phi(q)}, |phi - y*| = {np.linalg.norm(phi(q) - y):.4f} m: a limit is a constraint, and Deck 05 finds a better allowed configuration")

    # (n) the Woodbury form of J# agrees with the normal-equation form
    K, D = np.eye(2), 0.05*np.eye(2)
    A = np.linalg.solve(J0.T @ K @ J0 + D, J0.T @ K)
    B = np.linalg.solve(D, J0.T) @ np.linalg.inv(J0 @ np.linalg.solve(D, J0.T) + np.linalg.inv(K))
    print(f"\n(n) Woodbury check at q0 with K = I, D = 0.05 I: max |(J^T K J + D)^-1 J^T K - D^-1 J^T (J D^-1 J^T + K^-1)^-1| = {np.abs(A - B).max():.1e}")

    # (p) a reachable target where the standard gradient and Gauss-Newton steps are zero: "stationary" is not "solved", and not "unreachable"
    q_s = np.zeros(2); y_s = np.array([0.5, 0.0])
    r_s = phi(q_s) - y_s; J_s = jac(q_s); g_s = grad(q_s, y_s)
    print(f"\n(p) stretched arm q = (0, 0) deg, tip {phi(q_s)}, reachable target y* = {y_s} (|y*| = 0.5, inside [0.1, 0.7]): r = phi - y* = {r_s}, J = {J_s.tolist()}, grad f = J^T r = {g_s}")
    print(f"    exact Hessian eigenvalues {np.linalg.eigvalsh(hess_exact(q_s, y_s))}: a saddle of f, not a minimum. Minimum-norm Gauss-Newton step (pseudo-inverse; J^T J is singular here) = {np.linalg.pinv(J_s) @ (y_s - phi(q_s))};"
          f" damped steps at rho = 1e-3, 1e-2, 1e-1: {', '.join(str(gn_step(q_s, y_s, rho)[1]) for rho in (1e-3, 1e-2, 1e-1))}; gradient step = 0")
    h_s, s_s = run_gn(q_s, y_s, rho=0.01)
    print(f"    run_gn from there reports '{s_s}' with |phi - y*| = {h_s[-1]['res']:.3f} m: a zero step is neither success nor proof that no q reaches y*")
    for start in ((0.0, 10.0), (0.0, -10.0), (0.0, 1.0)):
        h_b, s_b = run_gn(d2r(list(start)), y_s, rho=0.0, iters=40)
        q_w = (h_b[-1]['q'] + np.pi) % (2*np.pi) - np.pi                 # wrapped to (-180, 180] deg for printing
        print(f"    from q0 = {start} deg (a bend of {start[1]:.0f} deg) plain Gauss-Newton reports '{s_b}' after {h_b[-1]['k']} steps at q = {r2d(q_w)} deg (wrapped), |phi - y*| = {h_b[-1]['res']:.1e}, first step {h_b[1]['step']:.2f} rad")
    print("    -> the target has two solutions, (-36.87, 90) and (36.87, -90) deg; every one of these starts is near the singular line, so the first plain step is large (the lesson of (k)), and the protected runs are the ones to quote:")
    for label, kw in (("Armijo", dict(linesearch=True)), ("rho = 0.01", dict(rho=0.01))):
        h_t, s_t = run_gn(d2r([0.0, 10.0]), y_s, **({'rho': 0.0} | kw) if 'rho' not in kw else kw, iters=100)
        q_w = (h_t[-1]['q'] + np.pi) % (2*np.pi) - np.pi
        print(f"       from (0, 10) deg with {label}: '{s_t}' after {h_t[-1]['k']} steps at q = {r2d(q_w)} deg, first step {h_t[1]['step']:.2f} rad, largest step {max(h['step'] for h in h_t[1:]):.2f} rad")
