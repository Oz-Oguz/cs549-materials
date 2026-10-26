"""Session 4.3 worked example (Deck 06; source retained in the Deck 05 folder): k-order trajectory optimization (the KOMO formulation) of the two-link arm
(l1 = 0.4 m, l2 = 0.3 m) around a disc obstacle, solved with the augmented Lagrangian on the outside and the
damped Gauss-Newton step of deck 04 on the inside. It is the mathematics behind `ry.KOMO`, written out in RAI's
convention for time and costs (rai commit ccc36b55, the robotic 0.4.1 wheel; measured in the container on 2026-10-21,
planning/komo-rai-convention-2026-10-21/), so that its numbers equal those of
    komo = ry.KOMO(C, phases=1, slicesPerPhase=T, kOrder=k, enableCollisions=...)
    komo.addControlObjective([], k, 1.)                                           # the cost
    komo.addObjective([1.], <tip position>, [], ry.OT.eq, [1.], [y*])             # reach the target
    komo.addObjective([1.], ry.FS.jointState, [], ry.OT.eq, [1.], order=1)        # arrive at rest
    komo.addObjective([], <negative distance of three points to the disc>, [], ry.OT.ineq, [1.])
code/komo_two_link_rai.py builds exactly this KOMO and compares.

Time (RAI's constructor fixes the duration of a phase at 1): the motion lasts one time unit, so the time step between
knots is dt = 1/T (RAI's tau; 0.05 for T = 20). The start q_0 is fixed and repeated as RAI's prefix (q_{-1} := q_0),
so the first step from the start is penalized, not forced to zero.

Problem (T = 20 knots, q_0 fixed, k = 2):
    variables      q_1, ..., q_T  in R^2                                  (z in R^{2T})
    cost           f(z) = sum_{t=1..T} || r_t ||^2,   r_t = sqrt(dt) (q_{t-2} - 2 q_{t-1} + q_t) / dt^2   (RAI's sos: no 1/2)
                   so f = dt^{-3} sum ||q_{t-2} - 2 q_{t-1} + q_t||^2, which approximates the integral of ||qddot||^2 over the motion
                   (k = 1: r_t = sqrt(dt) (q_t - q_{t-1}) / dt, f approximates the integral of ||qdot||^2)
    equalities     h(z) = ( phi(q_T) - y* ,  (q_T - q_{T-1}) / dt ) = 0     (reach the target, arrive at rest: zero final velocity)
    inequalities   g_{t,p}(z) = r_o - || p(q_t) - c || <= 0                   (three points p on the arm stay out of the disc)

Augmented Lagrangian (Toussaint's variant, RAI's Lagrangian; RAI's NLP_Solver also grows mu by 5 per outer
iteration from 1 to 1e4, while the course keeps it fixed):
    A(z) = f + kappa^T h + mu ||h||^2 + sum_i [g_i > 0 or lambda_i > 0] (lambda_i g_i + mu g_i^2),   mu fixed
    inner:  damped Gauss-Newton on A written as 1/2 || r(z) ||^2 with the stacked residual
            r = ( sqrt(2) r_t ;  sqrt(2 mu) (h + kappa/(2 mu)) ;  sqrt(2 mu) (g_i + lambda_i/(2 mu)) for active i )
    outer:  kappa <- kappa + 2 mu h(z),    lambda_i <- max(lambda_i + 2 mu g_i(z), 0)

Two initial guesses (straight-line interpolation in joint space to the two IK solutions of the last knot, named by
the sign of q2 as the kinematics reading and Deck 04 do: q2 = +80.41 deg is Deck 04's 'elbow-up', q2 = -80.41 deg
its 'elbow-down') lead to two different local optima: the fog analogy of deck 04 on a trajectory. Before
2026-10-21 this script used raw knot differences (dt = 1) and a 1/2 on the cost; the motions are unchanged by the
switch (a positive factor on the only soft term), while the printed costs and multipliers are RAI's.

Run:   python komo_two_link.py            prints both solutions, the multipliers, the KKT check, the conditioning of the
                                          Gauss-Newton matrix and the between-knot clearance   (about two seconds)
       python komo_two_link.py --drake    additionally hands the identical program to Drake + SNOPT, checks KKT at SNOPT's
                                          point with least-squares multipliers, and restarts the augmented Lagrangian there
       python komo_two_link.py --k1       replaces the acceleration cost by a velocity cost (k = 1): same route, different timing
       python komo_two_link.py --sweep    T = 10, 20, 40 (dt = 1/T): f is nearly constant, and the between-knot clearance changes
"""
import sys
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)
L1, L2 = 0.4, 0.3
DEG = np.pi / 180
T = 20
Q0 = np.array([30*DEG, 45*DEG])
Y_STAR = np.array([0.5, 0.2])
OBS_C, OBS_R = np.array([0.49, 0.35]), 0.05
MU_RAW = 10.0         # the penalty weight of the earlier raw-difference version (cost 1/2 sum ||raw differences||^2)
RHO = 1e-6            # Gauss-Newton damping
N = 2*T               # number of decision variables


def dt():
    """RAI's time step tau for ry.KOMO(C, phases=1, slicesPerPhase=T, ...): the motion lasts one time unit."""
    return 1.0 / T


def cost_scale():
    """RAI's cost over the earlier raw-difference cost for the same motion: sum ||sqrt(dt) D^k q / dt^k||^2 against
    1/2 sum ||D^k q||^2, i.e. 2 dt^(1 - 2k) (16000 for k = 2, 40 for k = 1, at T = 20)."""
    return 2.0 * dt()**(1 - 2*K_ORDER)


def mu_default():
    """The course keeps mu fixed (RAI's NLP_Solver grows it from 1 by a factor 5 up to 1e4). It is the earlier weight
    10 times cost_scale(): 1.6e5 for k = 2 and 400 for k = 1 at T = 20. That keeps the earlier balance between the cost
    and the penalty on the target and obstacle rows, so the solutions are the earlier ones in RAI's units. The rest row
    is now a velocity, (q_T - q_{T-1})/dt, so the same mu penalizes it 1/dt^2 = 400 times more strongly than before,
    and the outer iteration counts differ from the raw-difference version (guess A: 5 instead of 6)."""
    return MU_RAW * cost_scale()
K_ORDER = 1 if "--k1" in sys.argv else 2     # 2: acceleration cost (the lecture's problem); 1: velocity cost, for comparison


# ----------------------------------------------------------------------------- kinematics of three points on the arm
def p_tip(q):    return np.array([L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]), L1*np.sin(q[0]) + L2*np.sin(q[0]+q[1])])
def p_elbow(q):  return np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
def p_mid2(q):   return np.array([L1*np.cos(q[0]) + 0.5*L2*np.cos(q[0]+q[1]), L1*np.sin(q[0]) + 0.5*L2*np.sin(q[0]+q[1])])


def J_point(q, l2frac):
    """Jacobian of L1 R(q1) e1 + l2frac L2 R(q1+q2) e1."""
    s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q[0]+q[1]), np.cos(q[0]+q[1])
    a = l2frac*L2
    return np.array([[-L1*s1 - a*s12, -a*s12], [L1*c1 + a*c12, a*c12]])


POINTS = [("tip", p_tip, 1.0), ("elbow", p_elbow, 0.0), ("mid of link 2", p_mid2, 0.5)]


def link_clearance(q):
    """A DIAGNOSTIC, not a constraint of the program: the distance from the disc's surface to the nearer of the two
    whole links (line segments base-elbow and elbow-tip), negative inside. The program constrains only the three
    sampled points above, so a solution of the three-point model can still have a link crossing the disc."""
    c, best = OBS_C, np.inf
    for a, b in ((np.zeros(2), p_elbow(q)), (p_elbow(q), p_tip(q))):
        d = b - a
        s = np.clip((c - a) @ d / (d @ d), 0.0, 1.0)
        best = min(best, np.linalg.norm(c - (a + s*d)))
    return best - OBS_R


def knots(z):
    """q_0 ... q_T as a (T+1) x 2 array."""
    return np.vstack([Q0, np.asarray(z).reshape(T, 2)])


# ----------------------------------------------------------------------------- the three residual groups and their Jacobians
def acc_residuals(z):
    """The smoothness residuals in RAI's addControlObjective convention: sqrt(dt) (q_{t-2} - 2 q_{t-1} + q_t) / dt^2
    (k = 2, with q_{-1} := q_0 as RAI's prefix, so the first one is sqrt(dt) (q_1 - q_0) / dt^2: a start from rest), or
    sqrt(dt) (q_t - q_{t-1}) / dt when K_ORDER = 1. The cost is the plain sum of their squares."""
    Q = knots(z)
    c = np.sqrt(dt()) / dt()**K_ORDER
    r, J = np.zeros(2*T), np.zeros((2*T, N))
    for t in range(1, T+1):
        if K_ORDER == 2:
            qm2 = Q[t-2] if t >= 2 else Q0
            r[2*(t-1):2*t] = qm2 - 2*Q[t-1] + Q[t]
            terms = ((t, 1.0), (t-1, -2.0), (t-2, 1.0))
        else:
            r[2*(t-1):2*t] = Q[t] - Q[t-1]
            terms = ((t, 1.0), (t-1, -1.0))
        for idx, coef in terms:
            if idx >= 1:                         # only free knots have columns
                J[2*(t-1):2*t, 2*(idx-1):2*idx] += coef*np.eye(2)
    return c*r, c*J


def eq_constraints(z):
    Q = knots(z)
    h = np.concatenate([p_tip(Q[T]) - Y_STAR, (Q[T] - Q[T-1]) / dt()])     # RAI's order-1 jointState: a velocity
    J = np.zeros((4, N))
    J[0:2, 2*(T-1):2*T] = J_point(Q[T], 1.0)
    J[2:4, 2*(T-1):2*T] = np.eye(2) / dt()
    J[2:4, 2*(T-2):2*(T-1)] = -np.eye(2) / dt()
    return h, J


def ineq_constraints(z):
    Q = knots(z)
    g, J = np.zeros(3*T), np.zeros((3*T, N))
    for t in range(1, T+1):
        for p, (_, p_fn, frac) in enumerate(POINTS):      # p indexes the three sample points; k is the Markov order
            d = p_fn(Q[t]) - OBS_C
            dist = np.linalg.norm(d)
            g[3*(t-1)+p] = OBS_R - dist
            J[3*(t-1)+p, 2*(t-1):2*t] = -(d/dist) @ J_point(Q[t], frac)
    return g, J


def cost(z):
    """RAI's sos cost: the plain sum of squares of the control rows (no 1/2)."""
    r, _ = acc_residuals(z)
    return float(r @ r)


def grad_cost(z):
    r, J = acc_residuals(z)
    return 2.0 * J.T @ r


# ----------------------------------------------------------------------------- augmented Lagrangian with a Gauss-Newton inner loop
def al_residual(z, lam, kap, mu):
    """Stacked residual r(z) with 1/2||r||^2 = A(z) + const, and its Jacobian."""
    r_acc, J_acc = acc_residuals(z)
    h, J_h = eq_constraints(z)
    g, J_g = ineq_constraints(z)
    active = (g > 0) | (lam > 0)
    s = np.sqrt(2*mu)
    r = np.concatenate([np.sqrt(2.0)*r_acc, s*(h + kap/(2*mu)), s*(g[active] + lam[active]/(2*mu))])   # 1/2||r||^2 = f + ...
    J = np.vstack([np.sqrt(2.0)*J_acc, s*J_h, s*J_g[active]])
    return r, J


def gauss_newton(z, lam, kap, mu, max_iter=100, tol=1e-10, floor=1e-8):
    """The INNER solve: Deck 04's damped Gauss-Newton step on the stacked residual of A(z) with the multipliers
    lam, kap held fixed, Armijo backtracking restarted from alpha = 1 at every iteration (the active set is
    re-evaluated inside each trial, as KOMO does). Returns (z, iterations, status):
      'converged'  the accepted step is below tol: the inner loop's small-step stop, ON here (the inner residual
                   carries the multiplier shifts, so its norm does not vanish at the solution and Deck 04's
                   relative gradient test would be the wrong yardstick for it);
      'stalled'    the trial at the floor fails the Armijo test, or every trial's cost is not finite: the candidate
                   is REJECTED and z keeps its last accepted (finite) value, as in Deck 04's code (before
                   2026-09-29 a failed step was taken anyway, and a NaN cost compared false and was accepted);
      'limit'      max_iter reached."""
    status = "limit"
    for it in range(max_iter):
        r, J = al_residual(z, lam, kap, mu)
        grad = J.T @ r
        delta = -np.linalg.solve(J.T @ J + RHO*np.eye(N), grad)
        F0, alpha, slope = 0.5*r @ r, 1.0, float(grad @ delta)
        def F(a):
            r_new, _ = al_residual(z + a*delta, lam, kap, mu)
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


LAST = {"status": None, "outer": None, "mu": None}      # outcome of the last solve_komo call, for describe()


def solve_komo(z0, mu=None, outer_max=30, verbose=True):
    """The OUTER loop (13): inner solve with the multipliers fixed, then the multiplier update. It stops on the
    KKT residuals of the constrained problem, measured with the UPDATED multipliers: feasibility (max|h|, max g+),
    Lagrangian stationarity |grad f + J_h^T kappa + J_g^T lambda| and complementarity max|lambda_i g_i|; the
    objective gradient |grad f| is not a sufficient test here (it can be zero or not at a constrained optimum; in
    this problem it stays at about 4.3e+02 along path A, in RAI's cost units). The inner solve stops on its own rule (a small accepted step, see
    gauss_newton), not on a gradient tolerance, and the three outer residuals are measured independently after the
    update: when no multiplier is clamped by the projection, the inner stationarity condition over the included
    rows coincides with the Lagrangian stationarity with the unprojected update, but a clamped row (lambda + 2 mu g
    < 0 -> lambda_new = 0) breaks that identity, which is why the outer residual is computed, not assumed."""
    mu = mu_default() if mu is None else mu
    tol_stat, tol_comp = 1e-8*cost_scale(), 1e-11*cost_scale()     # the earlier tolerances, in RAI's cost units
    z, lam, kap = z0.copy(), np.zeros(3*T), np.zeros(4)
    LAST["status"] = "limit"; LAST["mu"] = mu
    if verbose:
        print("   outer  inner (iterations, outcome)   acc cost f   feasibility max|h|  max g+   Lagrangian stationarity   complementarity max|lambda g|   #active g   max lambda")
    for i in range(outer_max):                       # outer iterations are i; k is the Markov order
        z, n_inner, st_inner = gauss_newton(z, lam, kap, mu)
        h, J_h = eq_constraints(z)
        g, J_g = ineq_constraints(z)
        lam_new = np.maximum(lam + 2*mu*g, 0)
        kap_new = kap + 2*mu*h
        feas = max(np.abs(h).max(), max(g.max(), 0.0))
        stat = np.linalg.norm(grad_cost(z) + J_h.T @ kap_new + J_g.T @ lam_new)
        comp = np.abs(lam_new*g).max()
        if verbose:
            print(f"   {i:<6} {n_inner:<3} {st_inner:<24} {cost(z):.6f}   {np.abs(h).max():.1e}   {max(g.max(), 0):.1e}   {stat:.1e}                   {comp:.1e}                    {int((lam_new > 0).sum()):<10} {lam_new.max():.4f}")
        lam, kap = lam_new, kap_new
        if st_inner == "stalled":
            LAST["status"] = "inner stalled"
            if verbose:
                print("   the inner solve stalled: its candidate was rejected and the outer loop stops here")
            break
        if feas < 1e-9 and stat < tol_stat and comp < tol_comp:
            LAST["status"] = f"KKT residuals below tolerance (feasibility 1e-9, Lagrangian stationarity {tol_stat:.1e}, complementarity {tol_comp:.1e})"
            break
    LAST["outer"] = i + 1
    return z, lam, kap


def kkt_check(z, lam, kap):
    _, J_h = eq_constraints(z)
    g, J_g = ineq_constraints(z)
    stationarity = grad_cost(z) + J_h.T @ kap + J_g.T @ lam
    return np.linalg.norm(stationarity), np.abs(lam*g).max()


# ----------------------------------------------------------------------------- initial guesses: straight lines in joint space
def ik_two_link(y, q2_positive=True):
    """The two IK solutions of a reachable target, named by the sign of q2 (q2 > 0 is Deck 04's elbow-up)."""
    c2 = (y @ y - L1**2 - L2**2) / (2*L1*L2)
    q2 = np.arccos(c2) * (1 if q2_positive else -1)
    q1 = np.arctan2(y[1], y[0]) - np.arctan2(L2*np.sin(q2), L1 + L2*np.cos(q2))
    return np.array([q1, q2])


def linear_init(q_goal):
    return np.array([Q0 + (t/T)*(q_goal - Q0) for t in range(1, T+1)]).ravel()


def describe(name, z, lam, kap):
    Q = knots(z)
    g, _ = ineq_constraints(z)
    print(f"\n  --- {name} ---")
    print(f"  q_T = ({np.degrees(Q[T][0]):.2f}, {np.degrees(Q[T][1]):.2f}) deg,  phi(q_T) = {p_tip(Q[T])},  {'acceleration' if K_ORDER == 2 else 'velocity'} cost f = {cost(z):.6f}")
    print(f"  kappa (target x, y; rest x, y) = {kap}")
    active_knots = sorted({int(i)//3 + 1 for i in np.nonzero(lam > 1e-9)[0]})
    print(f"  obstacle constraint active (lambda > 0) at knots {active_knots}; points: "
          + ", ".join(f"t={i//3+1} {POINTS[i%3][0]} lambda={lam[i]:.4f}" for i in np.nonzero(lam > 1e-9)[0]))
    print(f"  min distance of the three points to the obstacle surface over the path: {(-g).min():+.2e} m (>= 0 means clear)")
    lc = np.array([link_clearance(Q[t]) for t in range(1, T+1)])
    lc_between = min(link_clearance((1-s)*Q[t] + s*Q[t+1]) for t in range(1, T) for s in np.linspace(0, 1, 101))
    print(f"  whole-link clearance (a diagnostic; the program constrains only the three points): at the knots {lc.min():+.2e} m"
          f" (knot {int(lc.argmin()) + 1}), between knots {lc_between:+.2e} m; knots within 1e-6 m of the disc or inside: {[int(t) + 1 for t in np.nonzero(lc < 1e-6)[0]]}"
          + (f"  -> a link crosses the disc at knots {[int(t) + 1 for t in np.nonzero(lc < -1e-6)[0]]} although the three points are clear" if lc.min() < -1e-6 else ""))
    st, comp = kkt_check(z, lam, kap)
    print(f"  outcome of the outer loop: {LAST['status']} after {LAST['outer']} outer iterations")
    print(f"  KKT check: || grad f + J_h^T kappa + J_g^T lambda || = {st:.1e},   max |lambda_i g_i| = {comp:.1e}"
          f"   (the objective gradient alone, |grad f| = {np.linalg.norm(grad_cost(z)):.1e}, is not small and is not a test)")
    tip = np.array([p_tip(q) for q in Q])
    print("  tip path (every 4th knot):", " ".join(f"({x:.3f},{y:.3f})" for x, y in tip[::4]))
    print("  joint steps |q_t - q_{t-1}| (deg):", " ".join(f"{np.degrees(np.linalg.norm(Q[t]-Q[t-1])):.2f}" for t in range(1, T+1)))
    # between the knots: the constraint holds at the knots only; interpolate linearly in joint space and look
    worst, where = 0.0, None
    for t in range(1, T):
        for s in np.linspace(0, 1, 101):
            q = (1-s)*Q[t] + s*Q[t+1]
            for _, p_fn, _ in POINTS:
                pen = OBS_R - np.linalg.norm(p_fn(q) - OBS_C)
                if pen > worst: worst, where = pen, (t, t+1)
    print(f"  three points between knots (linear interpolation in joint space): deepest cut into the disc {worst*1000:.2f} mm" + (f", between knots {where[0]} and {where[1]}" if where else ""))
    # conditioning of the Gauss-Newton matrix at the solution (Deck 04 slide 9's trajectory row)
    r, J = al_residual(z, lam, kap, mu_default()); e = np.linalg.eigvalsh(J.T @ J); _, Ja = acc_residuals(z); ea = np.linalg.eigvalsh(Ja.T @ Ja)
    print(f"  conditioning: cond(J^T J) of the augmented-Lagrangian residual = {e[-1]/e[0]:.2e}; of the smoothness rows alone {ea[-1]/ea[0]:.2e}")
    return Q


def drake_cross_check(z_init, z_al, lam_al, kap_al):
    from pydrake.all import MathematicalProgram, SnoptSolver
    prog = MathematicalProgram()
    z = prog.NewContinuousVariables(N, "z")
    # the control residual is linear in z: r = A z + b  ->  f = ||A z + b||^2 (RAI's sos, no 1/2)
    _, A = acc_residuals(np.zeros(N)); b, _ = acc_residuals(np.zeros(N))
    prog.Add2NormSquaredCost(A, -b, z)                                      # adds ||A z - (-b)||^2

    def h_fn(zz):
        Q = [Q0] + [zz[2*i:2*i+2] for i in range(T)]
        qT, qTm1 = Q[T], Q[T-1]
        tip = [L1*np.cos(qT[0]) + L2*np.cos(qT[0]+qT[1]) - Y_STAR[0], L1*np.sin(qT[0]) + L2*np.sin(qT[0]+qT[1]) - Y_STAR[1]]
        return np.array(tip + [(qT[0]-qTm1[0])/dt(), (qT[1]-qTm1[1])/dt()])

    def g_fn(zz):
        out = []
        for i in range(T):
            q = zz[2*i:2*i+2]
            for _, _, frac in POINTS:
                px = L1*np.cos(q[0]) + frac*L2*np.cos(q[0]+q[1]) - OBS_C[0]
                py = L1*np.sin(q[0]) + frac*L2*np.sin(q[0]+q[1]) - OBS_C[1]
                out.append(OBS_R - np.sqrt(px*px + py*py))
        return np.array(out)

    bh = prog.AddConstraint(h_fn, np.zeros(4), np.zeros(4), z)
    bg = prog.AddConstraint(g_fn, -np.inf*np.ones(3*T), np.zeros(3*T), z)
    prog.SetInitialGuess(z, z_init)
    result = SnoptSolver().Solve(prog)
    zd = result.GetSolution(z)
    print(f"\n  Drake + SNOPT from the same initial guess: success = {result.is_success()},  acceleration cost = {cost(zd):.6f}  (AL: {cost(z_al):.6f})")
    print(f"  max |q_drake - q_AL| over all knots = {np.degrees(np.abs(zd - z_al).max()):.4f} deg")
    lam_d = result.GetDualSolution(bg); kap_d = result.GetDualSolution(bh)
    act_al = np.nonzero(lam_al > 1e-9)[0]; act_d = np.nonzero(np.abs(lam_d) > 1e-9)[0]
    print(f"  SNOPT duals of the obstacle constraints: nonzero at indices {[int(i) for i in act_d]} (AL: {[int(i) for i in act_al]})")
    for i in act_al:
        print(f"     i={i}: AL lambda = {lam_al[i]:.5f}   SNOPT dual = {lam_d[i]:+.5f}")
    print(f"  SNOPT duals of the equalities = {kap_d}   (AL kappa = {kap_al})")
    # KKT at SNOPT's point, with multipliers fitted by least squares on its active set (no sign convention needed)
    _, J_h = eq_constraints(zd); g_d, J_g = ineq_constraints(zd)
    act = np.nonzero(g_d > -1e-6)[0]
    M = np.vstack([J_h, J_g[act]]).T
    mult, *_ = np.linalg.lstsq(M, -grad_cost(zd), rcond=None)
    st = np.linalg.norm(grad_cost(zd) + M @ mult)
    print(f"  KKT at SNOPT's point: active obstacle rows {[int(i) for i in act]} (g > -1e-6), least-squares multipliers lambda = {mult[4:]}, kappa = {mult[:4]},"
          f" stationarity residual {st:.1e}, max violation {max(g_d.max(), 0):.1e}, |h| = {np.abs(eq_constraints(zd)[0]).max():.1e}")
    # restart the augmented Lagrangian from SNOPT's point with those multipliers: does it stay?
    lam0 = np.zeros(3*T); lam0[act] = np.maximum(mult[4:], 0)
    z_r, lam_r, kap_r = z_al.copy(), lam0, mult[:4]
    z_r = zd.copy()
    for _ in range(5):
        z_r, _, _ = gauss_newton(z_r, lam_r, kap_r, mu_default())
        h_r, _ = eq_constraints(z_r); g_r, _ = ineq_constraints(z_r)
        lam_r = np.maximum(lam_r + 2*mu_default()*g_r, 0); kap_r = kap_r + 2*mu_default()*h_r
    print(f"  augmented Lagrangian restarted at SNOPT's point: after 5 outer iterations max |q - q_SNOPT| = {np.degrees(np.abs(z_r - zd).max()):.4f} deg,"
          f" cost {cost(z_r):.6f}, KKT residual {kkt_check(z_r, lam_r, kap_r)[0]:.1e}: " + ("it stays, so this is a second KKT point of the same program" if np.degrees(np.abs(zd - z_al).max()) > 0.1 else "the two solvers converged to the same candidate in this comparison (evidence for one KKT point here, not a proof of uniqueness)"))
    return zd


def sweep():
    """T = 10, 20, 40 from guess A with dt = 1/T: RAI's scaling makes f approximate the integral of the squared
    acceleration, so it stays nearly constant, and the between-knot clearance changes."""
    global T, N
    for Tn in (10, 20, 40):
        T, N = Tn, 2*Tn
        z0 = linear_init(ik_two_link(Y_STAR, True))
        z, lam, kap = solve_komo(z0, verbose=False)
        Q = knots(z); worst = 0.0
        for t in range(1, T):
            for s in np.linspace(0, 1, 101):
                q = (1-s)*Q[t] + s*Q[t+1]
                worst = max(worst, max(OBS_R - np.linalg.norm(p_fn(q) - OBS_C) for _, p_fn, _ in POINTS))
        print(f"  T = {Tn:3d}: dt = {dt():.4f}, f = {cost(z):.4f}, kappa = {kap}, deepest between-knot cut {worst*1000:.2f} mm, active multipliers at knots {sorted({int(i)//3+1 for i in np.nonzero(lam > 1e-9)[0]})}")
    T, N = 20, 40


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        print("=== the same problem at T = 10, 20, 40 (guess A), dt = 1/T as in ry.KOMO(C, 1, T, 2, ...): f approximates the integral of the squared acceleration ===")
        sweep(); sys.exit(0)
    print(f"=== KOMO on the two-link arm: T = {T}, k = {K_ORDER} ({'acceleration' if K_ORDER == 2 else 'velocity'} cost), obstacle disc at {OBS_C} r = {OBS_R}, target {Y_STAR}, q_0 = (30, 45) deg ===")
    print(f"  RAI's timing: ry.KOMO(C, phases=1, slicesPerPhase={T}, kOrder={K_ORDER}): the motion lasts one time unit, dt = {dt():.4f};"
          f" cost = sum of squared rows sqrt(dt) D^{K_ORDER} q / dt^{K_ORDER} (no 1/2), {cost_scale():.0f} times the earlier 1/2-sum of raw differences;"
          f" fixed mu = {mu_default():.4g} (= 10 x {cost_scale():.0f})")
    q_pos, q_neg = ik_two_link(Y_STAR, True), ik_two_link(Y_STAR, False)
    print(f"  IK solutions for the last knot: q2 > 0 ({np.degrees(q_pos[0]):.2f}, {np.degrees(q_pos[1]):.2f}) deg,"
          f" q2 < 0 ({np.degrees(q_neg[0]):.2f}, {np.degrees(q_neg[1]):.2f}) deg")
    z_init_A, z_init_B = linear_init(q_pos), linear_init(q_neg)
    sols = {}
    for name, z_init in (("A: straight line to the q2 > 0 solution", z_init_A), ("B: straight line to the q2 < 0 solution", z_init_B)):
        g0, _ = ineq_constraints(z_init)
        print(f"\n  initial guess {name}: {'acceleration' if K_ORDER == 2 else 'velocity'} cost {cost(z_init):.6f}, deepest obstacle penetration {max(g0.max(),0):.4f} m")
        z, lam, kap = solve_komo(z_init)
        sols[name[0]] = (z, lam, kap)
        Q = describe(f"solution from {name}", z, lam, kap)
        if "--drake" in sys.argv:
            drake_cross_check(z_init, z, lam, kap)
    # the sparsity of the stacked Jacobian: each row touches at most k + 1 consecutive knots, so J^T J is block-banded
    r, J = al_residual(z_init_A, np.ones(3*T), np.zeros(4), mu_default())
    nz = np.count_nonzero(J)
    print(f"\n  the full feature Jacobian (every smoothness, equality and inequality row, unscaled sparsity pattern): {J.shape[0]} x {J.shape[1]}, {nz} nonzeros of {J.size} ({100*nz/J.size:.1f} %); each smoothness row touches at most {K_ORDER + 1} knots")
    for key in ("A", "B"):
        zs, ls, ks = sols[key]
        r_in, J_in = al_residual(zs, ls, ks, mu_default())
        print(f"  the inner residual Jacobian at solution {key} (the {2*T} smoothness rows times sqrt(2), so that 1/2 ||r||^2 holds RAI's cost without 1/2; the 4 equality rows and the {int(((ineq_constraints(zs)[0] > 0) | (ls > 0)).sum())} included inequality rows multiplied by sqrt(2 mu) = {np.sqrt(2*mu_default()):.4g}; the multiplier shifts kappa/(2 mu) and lambda/(2 mu) sit in the residual values, not in the rows): {J_in.shape[0]} x {J_in.shape[1]}")
