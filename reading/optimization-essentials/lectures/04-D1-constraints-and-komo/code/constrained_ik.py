"""Sessions 4.1-4.2 worked examples: constraints, KKT multipliers, and the three ways to solve a
constrained problem, all on the two-link arm of sessions 2.3-3.2 (l1 = 0.4 m, l2 = 0.3 m).

Cost convention of this deck:  f(q) = 1/2 ||phi(q) - y*||^2,  so that grad f = J^T (phi - y*).

(a) The disc problem   min 1/2 ||z - z_t||^2   s.t.  g(z) = ||z|| - 1 <= 0
    Multiplier by hand: lambda* = max(0, ||z_t|| - 1). Checked numerically here.
(b) Inverse kinematics with a two-sided elbow range: y* = (0.5, 0.2), -60 <= q2 <= 60 deg, written as
    g1 = q2 - 60 deg <= 0 and g2 = -60 deg - q2 <= 0. The unconstrained minima q2 = +80.41 deg and
    q2 = -80.41 deg each violate one limit, so the constrained problem has two local minima of equal
    cost: the upper limit active at (-3.48, 60) deg and the lower limit active at (47.09, -60) deg.
    With q2 pinned at a limit the tip moves on a circle of radius R; the best q1 points the tip along y*
    (closed form), and lambda follows from stationarity.
(c) Sensitivity: d f*/d q2max = -lambda*  (relaxing the limit by one degree buys lambda* * pi/180).
(d) The same problem, now with the two-sided elbow range -60 <= q2 <= 60 deg (two inequalities, only the
    upper one active), solved three ways: squared penalty (mu -> inf), log barrier (mu -> 0),
    augmented Lagrangian (mu fixed, lambda learned). Inner solver: Newton with the Gauss-Newton
    Hessian and Armijo backtracking (deck 04). Every table printed here appears in notes.md section 5.
(e) A target for which the limit is inactive: y* = (0.6, 0.15) -> the sign test gives lambda < 0, so lambda* = 0.
(f) Deck 04's clipped point (-16.40, 60) deg, where Gauss-Newton with q2 clipped after each step stalls: it
    fails stationarity along the free joint, so it is not a KKT point.
(g) KKT points that are not minima: the saddle (21.80, 0) deg of the unconstrained cost, and a constrained
    maximum on the upper limit; and a minimum where KKT fails (the stretched arm on the rail phi = (0.7, 0)).

(n) Inverse kinematics as a nonlinear program on two joints (Deck 05 slide 17): the posture cost
    1/2 ||q - q_ref||_D^2 with D = I and q_ref = q_0 = (30, 45) deg, and ONE task equality, the tip height
    h(q) = phi_2(q) - b = 0 with b = 0.2 m. Two joints and one equality leave a curve of feasible configurations;
    the answer is where the posture cost's circles touch it. Solved by Newton's method on the KKT system
    (q - q_ref + kappa grad h = 0, h = 0); the sensitivity d f*/d b = -kappa is checked at b = 0.21 m.

Solution names: this file names the two unconstrained solutions by the sign of q2, as the kinematics
reading does; Deck 04's 'elbow-up' is q2 > 0 and its 'elbow-down' is q2 < 0.

Run:  python constrained_ik.py          Expected runtime: about a second.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
L1, L2 = 0.4, 0.3
DEG = np.pi / 180


def phi(q):
    q1, q2 = q
    return np.array([L1*np.cos(q1) + L2*np.cos(q1+q2), L1*np.sin(q1) + L2*np.sin(q1+q2)])


def jac(q):
    q1, q2 = q
    s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1+q2), np.cos(q1+q2)
    return np.array([[-L1*s1 - L2*s12, -L2*s12], [L1*c1 + L2*c12, L2*c12]])


def f(q, y):        return 0.5 * float(np.sum((phi(q) - y)**2))
def grad_f(q, y):   return jac(q).T @ (phi(q) - y)
def hess_f_gn(q):   return jac(q).T @ jac(q)              # Gauss-Newton approximation


# ----------------------------------------------------------------------------- generic inner solver
def newton_armijo(F, G, H, z0, tol=1e-9, max_iter=200, c=0.01, floor=1e-12):
    """Newton's method on an UNCONSTRAINED inner objective (the penalty, barrier or augmented-Lagrangian
    objective of one outer iteration) with a positive-definite Hessian model and Armijo backtracking,
    restarted from alpha = 1 at every iteration (Deck 04). Returns (z, iterations, status):
      'stationary'  |grad| < tol: the inner objective's own gradient test, which is the right test here
                    because the inner problem has no constraints (the OUTER problem is judged by
                    feasibility, Lagrangian stationarity and complementarity instead, see the outer loops).
                    tol = 1e-9 because the Gauss-Newton Hessian model makes the last digits of the inner
                    gradient settle at 4e-11 to 5e-10 on these problems (rounding), so a tighter tolerance
                    ran to max_iter without changing a printed digit;
      'stalled'     the trial step at the floor fails the Armijo test: the candidate is REJECTED and z keeps
                    its last accepted value, as in Deck 04's gauss_newton_ik.py (before 2026-09-29 the
                    failed step was taken anyway);
      'limit'       max_iter reached.
    A small-step stop is not used."""
    z = np.array(z0, float)
    status, it = "limit", 0
    for it in range(max_iter):
        g = G(z)
        if np.linalg.norm(g) < tol:
            status = "stationary"
            break
        Hm = H(z)
        try:
            d = -np.linalg.solve(Hm + 1e-12*np.eye(len(z)), g)
        except np.linalg.LinAlgError:
            d = -g
        if g @ d >= 0:                       # not a descent direction: fall back to the gradient
            d = -g
        alpha, F0, slope = 1.0, F(z), float(g @ d)
        ok = lambda a: np.isfinite(F(z + a*d)) and F(z + a*d) <= F0 + c*a*slope
        while not ok(alpha) and alpha > floor:
            alpha *= 0.5
        if not ok(alpha):                    # the trial at the floor fails: reject it, keep z, report
            status = "stalled"
            break
        z = z + alpha*d
    return z, it, status


# ----------------------------------------------------------------------------- (a) the disc problem
def disc_problem(zt):
    """min 1/2||z - zt||^2 s.t. ||z|| - 1 <= 0, solved by hand and by a fine grid check."""
    r = np.linalg.norm(zt)
    if r <= 1:
        z, lam = zt.copy(), 0.0
    else:
        z, lam = zt / r, r - 1
    # KKT check
    grad_f_ = z - zt
    grad_g_ = z / max(np.linalg.norm(z), 1e-12)
    stat = grad_f_ + lam*grad_g_
    print(f"  z_t = {zt}  |z_t| = {r:.4f}  ->  z* = {z}  lambda* = {lam:.4f}   g(z*) = {np.linalg.norm(z)-1:+.4f}"
          f"   stationarity residual {np.linalg.norm(stat):.1e}   lambda*g = {lam*(np.linalg.norm(z)-1):+.1e}")


# ----------------------------------------------------------------------------- (b) IK with a joint limit
def constrained_ik_by_hand(y, q2max):
    """KKT solution with the limit active: q2 = q2max, q1 from a 1-D minimisation, lambda from stationarity."""
    # the tip radius is fixed by q2; the best q1 points the tip along y*
    R = np.sqrt(L1**2 + L2**2 + 2*L1*L2*np.cos(q2max))
    offset = np.arctan2(L2*np.sin(q2max), L1 + L2*np.cos(q2max))
    q1 = np.arctan2(y[1], y[0]) - offset
    q = np.array([q1, q2max])
    g = grad_f(q, y)
    lam = -g[1]                                 # stationarity: grad f + lambda * (0, 1) = 0
    return q, lam, R, g


def report_ik(y, q2max):
    rstar = np.linalg.norm(y)
    cos_q2 = (rstar**2 - L1**2 - L2**2) / (2*L1*L2)
    print(f"\n  target y* = {y}, |y*| = {rstar:.4f}; unconstrained IK: cos q2 = {cos_q2:.4f} -> q2 = +/-{np.degrees(np.arccos(cos_q2)):.2f} deg")
    q, lam, R, g = constrained_ik_by_hand(y, q2max)
    if lam <= 0:
        print(f"  sign test at q2 = {np.degrees(q2max):.0f} deg: lambda = -df/dq2 = {lam:+.5f} < 0, so the limit would have to pull; it is NOT active"
              f" (the unconstrained solution q2 = {np.degrees(np.arccos(cos_q2)):.2f} deg satisfies it): lambda* = 0, f* = 0")
        return None
    r = phi(q) - y
    print(f"  upper limit q2 <= {np.degrees(q2max):.0f} deg is active on the q2 > 0 branch. Reach at the limit R = {R:.4f} m > |y*|.")
    print(f"  q* = ({np.degrees(q[0]):.2f}, {np.degrees(q[1]):.2f}) deg,  phi(q*) = {phi(q)},  residual phi - y* = {r} (|r| = {np.linalg.norm(r):.4f} m)")
    print(f"  grad f(q*) = {g}   ->   df/dq1 = {g[0]:+.2e} (zero: free direction),  lambda* = -df/dq2 = {lam:.5f}")
    print(f"  f* = {f(q, y):.6f}")
    # finite-difference check of the stationarity claim: f along the limit is minimal at q1*
    for dq in (-0.01, 0.01):
        print(f"     f(q1* {dq:+.2f}, q2max) = {f(q + np.array([dq, 0]), y):.6f}  (>= f*)")
    return q, lam


def sensitivity(y, q2max, lam):
    eps = 1*DEG
    f_plus = f(constrained_ik_by_hand(y, q2max + eps)[0], y)
    f_0 = f(constrained_ik_by_hand(y, q2max)[0], y)
    print(f"\n  sensitivity: relax the limit by 1 deg: f* goes from {f_0:.6f} to {f_plus:.6f}, change {f_plus - f_0:+.6f};"
          f"  -lambda* * (1 deg in rad) = {-lam*eps:+.6f}")


# ----------------------------------------------------------------------------- (n) IK as an NLP with one task equality
def hess_tip_height(q):
    """Hessian of the tip height phi_2(q) = l1 sin q1 + l2 sin(q1 + q2)."""
    s1, s12 = np.sin(q[0]), np.sin(q[0] + q[1])
    return np.array([[-L1*s1 - L2*s12, -L2*s12], [-L2*s12, -L2*s12]])


def ik_nlp_height(q_ref, b, q0=None, tol=1e-13, max_iter=50):
    """min 1/2 ||q - q_ref||^2  s.t.  h(q) = phi_2(q) - b = 0, by Newton's method on the KKT system
    F(q, kappa) = (q - q_ref + kappa grad h, h) = 0, started at q0 (default q_ref) with kappa = 0.
    Returns (q*, kappa*, iterations)."""
    q = np.array(q_ref if q0 is None else q0, float); kappa = 0.0
    for it in range(max_iter):
        gh = jac(q)[1]; h = phi(q)[1] - b
        F = np.concatenate([q - q_ref + kappa*gh, [h]])
        if np.linalg.norm(F) < tol: break
        K = np.zeros((3, 3)); K[:2, :2] = np.eye(2) + kappa*hess_tip_height(q); K[:2, 2] = gh; K[2, :2] = gh
        d = np.linalg.solve(K, -F)
        q, kappa = q + d[:2], kappa + d[2]
    return q, kappa, it


# ----------------------------------------------------------------------------- (d) three methods
# The elbow range is two-sided, -q2max <= q2 <= q2max, written as two inequalities
#   g1(q) = q2 - q2max <= 0  (upper limit, active at the solution),  g2(q) = -q2max - q2 <= 0  (lower limit, inactive).
# All three methods see the same two constraints; only the upper one ends up doing work.
def g_all(q, q2max):   return np.array([q[1] - q2max, -q2max - q[1]])
GRAD_G = np.array([[0.0, 1.0], [0.0, -1.0]])          # rows: grad g1, grad g2


def penalty_method(y, q2max, q0, mus=(1, 10, 100, 1000)):
    print("\n  squared penalty  S = f + mu sum_i [g_i>0] g_i^2  (mu grows; inner Newton warm-started)")
    print("   mu      q1 (deg)  q2 (deg)  violation g1 (deg)   f          2 mu g1 (= lambda estimate)   inner (iterations, outcome)")
    q = np.array(q0, float); rows = []
    for mu in mus:
        act = lambda z: (g_all(z, q2max) > 0).astype(float)
        F = lambda z: f(z, y) + mu*np.sum(act(z)*g_all(z, q2max)**2)
        G = lambda z: grad_f(z, y) + 2*mu*GRAD_G.T @ (act(z)*g_all(z, q2max))
        H = lambda z: hess_f_gn(z) + 2*mu*GRAD_G.T @ np.diag(act(z)) @ GRAD_G
        q, n_it, st = newton_armijo(F, G, H, q)
        g1 = g_all(q, q2max)[0]
        print(f"   {mu:<7} {np.degrees(q[0]):8.3f}  {np.degrees(q[1]):8.3f}  {np.degrees(g1):+15.4f}   {f(q, y):.6f}   {2*mu*max(g1,0):.5f}                   {n_it}, {st}")
        rows.append((mu, q.copy(), g1, 2*mu*max(g1, 0)))
    return q, rows


def barrier_method(y, q2max, q0, mus=(0.01, 1e-3, 1e-4, 1e-5, 1e-6)):
    print("\n  log barrier  B = f - mu sum_i log(-g_i)  (mu shrinks; start must be strictly feasible)")
    print("   mu      q1 (deg)  q2 (deg)  slack -g1 (deg)   f          -mu/g1 (= lambda estimate)   inner (iterations, outcome)")
    q = np.array(q0, float); rows = []
    assert np.all(g_all(q, q2max) < 0), "log barrier needs a strictly feasible start"
    for mu in mus:
        F = lambda z: f(z, y) - mu*np.sum(np.log(-g_all(z, q2max))) if np.all(g_all(z, q2max) < 0) else np.inf
        G = lambda z: grad_f(z, y) - mu*GRAD_G.T @ (1.0/g_all(z, q2max))
        H = lambda z: hess_f_gn(z) + mu*GRAD_G.T @ np.diag(1.0/g_all(z, q2max)**2) @ GRAD_G
        q, n_it, st = newton_armijo(F, G, H, q)
        g1 = g_all(q, q2max)[0]
        print(f"   {mu:<7} {np.degrees(q[0]):8.3f}  {np.degrees(q[1]):8.3f}  {np.degrees(-g1):14.4f}   {f(q, y):.6f}   {-mu/g1:.5f}                     {n_it}, {st}")
        rows.append((mu, q.copy(), g1, -mu/g1))
    return q, rows


def augmented_lagrangian(y, q2max, q0, mu=1.0, outer=6):
    print(f"\n  augmented Lagrangian  A = f + sum_i lambda_i g_i + mu sum_i [g_i>0 or lambda_i>0] g_i^2,  mu = {mu} fixed,"
          "  lambda_i <- max(lambda_i + 2 mu g_i, 0)")
    print("   outer  lambda1 (before)  q1 (deg)  q2 (deg)  g1 (deg)       lambda1 (after)  lambda2   inner            | outer residuals with the updated multipliers: feasibility max g+   stationarity |grad f + sum lambda_i grad g_i|   complementarity max|lambda_i g_i|   (and |grad f| itself, NOT a test)")
    q, lam = np.array(q0, float), np.zeros(2)
    rows = []
    for i in range(outer):
        act = lambda z: ((g_all(z, q2max) > 0) | (lam > 0)).astype(float)
        F = lambda z: f(z, y) + lam @ g_all(z, q2max) + mu*np.sum(act(z)*g_all(z, q2max)**2)
        G = lambda z: grad_f(z, y) + GRAD_G.T @ lam + 2*mu*GRAD_G.T @ (act(z)*g_all(z, q2max))
        H = lambda z: hess_f_gn(z) + 2*mu*GRAD_G.T @ np.diag(act(z)) @ GRAD_G
        q, n_it, st = newton_armijo(F, G, H, q)
        g = g_all(q, q2max)
        lam_new = np.maximum(lam + 2*mu*g, 0)
        # the OUTER stopping test is the KKT residual with the updated multipliers, measured here and not assumed:
        # the inner solve stops on its own gradient test, and a clamped multiplier (lambda + 2 mu g < 0 -> 0) breaks
        # the identity between inner stationarity and Lagrangian stationarity with the projected update
        feas = max(g.max(), 0.0)
        stat = np.linalg.norm(grad_f(q, y) + GRAD_G.T @ lam_new)
        comp = np.abs(lam_new*g).max()
        print(f"   {i:<6} {lam[0]:15.6f}   {np.degrees(q[0]):8.3f}  {np.degrees(q[1]):8.3f}  {np.degrees(g[0]):+10.5f}   {lam_new[0]:.6f}         {lam_new[1]:.1f}       {n_it}, {st:<10}   | {feas:.1e}   {stat:.1e}   {comp:.1e}   (|grad f| = {np.linalg.norm(grad_f(q, y)):.4f})")
        rows.append((i, lam.copy(), q.copy(), g.copy(), lam_new.copy(), (feas, stat, comp)))
        lam = lam_new
        if st == "stalled":
            print("   the inner solve stalled: its candidate was rejected and the outer loop stops here")
            break
        if feas < 1e-9 and stat < 1e-8 and comp < 1e-11:        # the same triplet as komo_two_link.solve_komo
            print(f"   outer stop at iteration {i}: feasibility, Lagrangian stationarity and complementarity are all below tolerance 1e-9 / 1e-8 / 1e-11 (|grad f| = {np.linalg.norm(grad_f(q, y)):.4f} here, and it is not a test: it can be zero or not at a constrained optimum)")
            break
    return q, lam, rows


if __name__ == "__main__":
    print("=== (a) the disc problem: min 1/2|z - z_t|^2  s.t.  |z| - 1 <= 0 ===")
    for zt in ([2.0, 0.0], [1.2, 0.9], [0.5, 0.3]):
        disc_problem(np.array(zt))

    print("\n=== (a') the rail h(z) = (z1 + z2 - 1)/sqrt(2) = 0 with the same cost: kappa* = h(z_t), z* = z_t - kappa* grad h ===")
    n_h = np.array([1.0, 1.0]) / np.sqrt(2)
    for zt in ([1.6, 0.6], [-0.3, -0.4]):
        zt = np.array(zt); h = float(n_h @ zt) - 1/np.sqrt(2); z = zt - h*n_h
        print(f"  z_t = {zt}: kappa* = h(z_t) = {h:+.4f}, z* = {z}, h(z*) = {float(n_h @ z) - 1/np.sqrt(2):+.1e}, stationarity |z* - z_t + kappa* grad h| = {np.linalg.norm(z - zt + h*n_h):.1e}")
    print("\n=== (b) inverse kinematics of the two-link arm with the elbow range -60 <= q2 <= 60 deg ===")
    y, q2max = np.array([0.5, 0.2]), 60*DEG
    print("  g1 = q2 - 60 deg <= 0 (upper limit), g2 = -60 deg - q2 <= 0 (lower limit); both unconstrained solutions violate one of them")
    q_star, lam_star = report_ik(y, q2max)
    print("\n  the mirror solution (q2 < 0 branch, lower limit g2 active) has the same cost:")
    q_m, lam_m, R_m, g_m = constrained_ik_by_hand(y, -q2max)
    print(f"  q = ({np.degrees(q_m[0]):.2f}, {np.degrees(q_m[1]):.2f}) deg, f = {f(q_m, y):.6f}, lambda_2* = +df/dq2 = {g_m[1]:.5f} (grad g2 = (0, -1))")

    print("\n=== (c) sensitivity: the multiplier is the price of the constraint ===")
    sensitivity(y, q2max, lam_star)

    print("\n=== (d) the same problem solved three ways, from q0 = (30, 45) deg ===")
    q0 = np.array([30*DEG, 45*DEG])
    penalty_method(y, q2max, q0)
    barrier_method(y, q2max, q0)
    augmented_lagrangian(y, q2max, q0)
    print(f"\n  exact KKT solution for comparison: q* = ({np.degrees(q_star[0]):.3f}, {np.degrees(q_star[1]):.3f}) deg, lambda* = {lam_star:.6f}")

    print("\n=== (e) a target for which the limit is inactive: y* = (0.6, 0.15) ===")
    report_ik(np.array([0.6, 0.15]), q2max)

    print("\n=== (f) Deck 04's clipped point: Gauss-Newton with q2 clipped to the range after each step ===")
    q = np.array([30*DEG, 45*DEG])
    for it in range(200):
        J = jac(q); qn = q + np.linalg.solve(J.T @ J, J.T @ (y - phi(q))); qn[1] = np.clip(qn[1], -q2max, q2max)
        if np.linalg.norm(qn - q) < 1e-12: break
        q = qn
    gq = grad_f(q, y)
    print(f"  stalls at q = ({np.degrees(q[0]):.2f}, {np.degrees(q[1]):.2f}) deg after {it} steps, f = {f(q, y):.4f} = {f(q, y)/f(q_star, y):.2f} f*;"
          f" grad f = {gq}: df/dq1 = {gq[0]:+.4f} is not zero, so stationarity fails along the free joint and this is not a KKT point")

    print("\n=== (g) KKT points that are not minima, and a minimum where KKT fails ===")
    q_s = np.array([np.arctan2(y[1], y[0]), 0.0]); gs = grad_f(q_s, y)
    print(f"  the straight arm pointing past the target, q = ({np.degrees(q_s[0]):.2f}, 0) deg: grad f = {gs} (a KKT point with no active limit),"
          f" f = {f(q_s, y):.5f} = {f(q_s, y)/f(q_star, y):.2f} f*: a saddle of the unconstrained cost, not a minimum")
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda q1: -f(np.array([q1, q2max]), y), bounds=(2.5, 3.5), method="bounded")
    q_x = np.array([res.x, q2max]); gx = grad_f(q_x, y)
    print(f"  on the upper limit at q = ({np.degrees(q_x[0]):.2f}, 60) deg: df/dq1 = {gx[0]:+.1e}, lambda = -df/dq2 = {-gx[1]:+.4f} > 0: every KKT condition holds,"
          f" and f = {f(q_x, y):.5f} is the MAXIMUM of f along the limit")
    L_shift = f(q_star - np.array([0.0, 2*np.pi]), y) + lam_star*((q_star[1] - 2*np.pi) - q2max)
    print(f"  the Lagrangian f + lambda* (q2 - 60 deg) has no global minimum: at q* it is {f(q_star, y):.5f}, at q* with q2 shifted by -360 deg (the same arm pose) it is {L_shift:.4f}, and every further shift lowers it by 2 pi lambda* = {2*np.pi*lam_star:.4f}")
    Js = jac(np.zeros(2))
    print(f"  the stretched arm at q = (0, 0) with the rail phi(q) = (0.7, 0): J = {Js.tolist()} has rank {np.linalg.matrix_rank(Js)},"
          f" so the two rail gradients are parallel; the point is the only feasible one (a minimum) and no kappa balances a gradient outside their span: KKT needs independent active gradients")

    print("\n=== (n) IK as a nonlinear program on two joints: min 1/2|q - q_ref|^2  s.t.  tip height phi_2(q) = 0.2 m ===")
    q_ref, b = np.array([30*DEG, 45*DEG]), 0.2
    q_n, kap_n, it_n = ik_nlp_height(q_ref, b)
    gh = jac(q_n)[1]; f_n = 0.5*float(np.sum((q_n - q_ref)**2))
    print(f"  the reference posture q_ref = (30, 45) deg puts the tip at {phi(q_ref)}: the equality is violated by h(q_ref) = {phi(q_ref)[1] - b:+.4f} m")
    print(f"  answer after {it_n} Newton steps on the KKT system: q* = ({np.degrees(q_n[0]):.2f}, {np.degrees(q_n[1]):.2f}) deg, tip at {phi(q_n)},"
          f" h(q*) = {phi(q_n)[1] - b:.1e} m, f* = {f_n:.5f} rad^2, kappa* = {kap_n:.4f} rad^2/m")
    print(f"  stationarity |q* - q_ref + kappa* grad h| = {np.linalg.norm(q_n - q_ref + kap_n*gh):.1e}; grad f(q*) = q* - q_ref = {q_n - q_ref} rad"
          f" is not zero, it is parallel to grad h = {gh} m/rad")
    q_b, _, _ = ik_nlp_height(q_ref, b + 0.01, q0=q_n); f_b = 0.5*float(np.sum((q_b - q_ref)**2))
    print(f"  raising the height to 0.21 m: q* = ({np.degrees(q_b[0]):.2f}, {np.degrees(q_b[1]):.2f}) deg; predicted change -kappa* 0.01 = {-0.01*kap_n:+.5f},"
          f" computed {f_b - f_n:+.5f} rad^2")
    best = min((0.5*float(np.sum((np.array([q1, q2]) - q_ref)**2)), q1, q2)
               for q1 in np.linspace(-np.pi, np.pi, 3601)
               for v in [(b - L1*np.sin(q1))/L2] if abs(v) <= 1
               for a in (np.arcsin(v), np.pi - np.arcsin(v))
               for q2 in [(a - q1 + np.pi) % (2*np.pi) - np.pi])
    print(f"  a scan of the whole feasible curve (q1 on a 0.1 deg grid) finds its closest point at ({np.degrees(best[1]):.1f}, {np.degrees(best[2]):.1f}) deg with cost {best[0]:.5f}: the same answer")

