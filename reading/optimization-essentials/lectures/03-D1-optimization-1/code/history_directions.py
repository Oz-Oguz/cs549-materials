"""Deck 04 slides 33 and 34: how L-BFGS and nonlinear conjugate gradient reuse history, on the explorer's Rosenbrock
valley f(z) = (1 - z1)^2 + 100 (z2 - z1^2)^2 from its "Classic start" z^(0) = (-1.2, 1.0).

The traces reproduce the optimization explorer's own line search (modules/opt-explorer/src/concepts/scalarOptimizers.ts):
exact line minimization by doubling then bisection on the directional slope, phi'(alpha) = 0, started from the slider's
default alpha = 0.1. The CG rule is the explorer's: Polak-Ribiere with beta clamped at 0 and a restart to steepest
descent when the direction is not downhill. L-BFGS keeps the last m = 3 (move, gradient-change) pairs with the usual
scalar initialization gamma = s^T y / y^T y and the two-loop recursion, as the explorer's L-BFGS button does, and slide 33
shows the direction proposed after three accepted updates (its Exact line-search option reproduces the trace). The explorer's BFGS (full inverse update from the
identity, strong-Wolfe line search) is ported too, so that the slide-32 demonstration's numbers can be checked.

Run:  python history_directions.py     (plain NumPy; prints every number used on the slides and in notes.md section 3.5,
      and asserts: every stored pair has positive curvature s^T y > 0, every proposed direction is downhill, the L-BFGS
      direction differs visibly from the negative gradient, the CG direction is the sum of its two drawn parts, and the
      optional quadratic example of the earlier version still satisfies its checks)
"""
import numpy as np

Z0 = np.array([-1.2, 1.0])           # the explorer's "Classic start"
ALPHA0 = 0.1                         # the explorer's default step-size slider, the line search's first trial
M_PAIRS = 3                          # L-BFGS memory for the illustration


def f(z):
    return float((1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2)


def grad(z):
    return np.array([-2 * (1 - z[0]) + 200 * (z[1] - z[0] ** 2) * (-2 * z[0]), 200 * (z[1] - z[0] ** 2)])


def hess(z):
    return np.array([[2 - 400 * z[1] + 1200 * z[0] ** 2, -400 * z[0]], [-400 * z[0], 200.0]])


# --- the explorer's line searches, ported line by line ---------------------------------------------------------------

def exact_line_search(z, d, alpha_init=ALPHA0, tol=1e-10):
    """scalarOptimizers.ts exactLineSearch: double alpha until phi'(alpha) > 0, then bisect on phi'(alpha) = 0."""
    dphi = lambda a: float(grad(z + a * d) @ d)
    if dphi(0.0) >= 0:
        return alpha_init * 0.01
    hi = alpha_init
    for _ in range(30):
        if dphi(hi) > 0:
            break
        hi *= 2
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        dm = dphi(mid)
        if abs(dm) < tol or (hi - lo) < 1e-14:
            return mid
        if dm < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wolfe_line_search(z, d, alpha_init=ALPHA0, strong=True, c1=1e-4, c2=0.5, max_iter=25):
    """scalarOptimizers.ts selectLineSearchAlpha for the wolfe and strong-wolfe modes (Nocedal-Wright 3.5 and 3.6)."""
    phi = lambda a: f(z + a * d)
    dphi = lambda a: float(grad(z + a * d) @ d)
    phi0, dphi0 = phi(0.0), dphi(0.0)
    if dphi0 >= 0:
        return alpha_init * 0.01
    curvature_ok = (lambda da: abs(da) <= c2 * abs(dphi0)) if strong else (lambda da: da >= c2 * dphi0)

    def zoom(a_lo, a_hi):
        lo, hi = a_lo, a_hi
        for _ in range(max_iter):
            aj = (lo + hi) / 2
            phi_j = phi(aj)
            if phi_j > phi0 + c1 * aj * dphi0 or phi_j >= phi(lo):
                hi = aj
            else:
                dj = dphi(aj)
                if curvature_ok(dj):
                    return aj
                if dj * (hi - lo) >= 0:
                    hi = lo
                lo = aj
            if abs(hi - lo) < 1e-12:
                break
        return (lo + hi) / 2

    alpha_prev, phi_prev, alpha = 0.0, phi0, alpha_init
    alpha_max = max(alpha_init * 64, 2.0)
    for i in range(max_iter):
        phi_a = phi(alpha)
        if phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_prev):
            return zoom(alpha_prev, alpha)
        dphi_a = dphi(alpha)
        if curvature_ok(dphi_a):
            return alpha
        if dphi_a >= 0:
            return zoom(alpha, alpha_prev)
        alpha_prev, phi_prev = alpha, phi_a
        alpha = min(alpha * 2, alpha_max)
        if alpha >= alpha_max:
            break
    return alpha


# --- the three direction rules -----------------------------------------------------------------------------------------

def cg_direction(g_new, g_old, d_prev):
    """The explorer's cgStep rule: Polak-Ribiere beta clamped at 0; restart to -g if the result is not downhill."""
    if g_old is None or float(g_old @ g_old) < 1e-30:
        return -g_new, 0.0, "first step: steepest descent"
    beta_pr = float(g_new @ (g_new - g_old)) / float(g_old @ g_old)
    beta = max(0.0, beta_pr)
    d = -g_new + beta * d_prev
    note = "" if beta == beta_pr else f"beta clamped to 0 (beta_PR = {beta_pr:.3f}): restart"
    if float(g_new @ d) >= 0:
        return -g_new, 0.0, "not downhill: restart to steepest descent"
    return d, beta, note


def lbfgs_direction(g, pairs):
    """The two-loop recursion over the stored (s, y) pairs, oldest first, with H0 = gamma I from the newest pair."""
    if not pairs:
        return -g, None
    q = g.copy(); alphas = []
    for s, y in reversed(pairs):                      # newest first
        rho = 1.0 / float(y @ s); a = rho * float(s @ q); alphas.append(a); q = q - a * y
    s_n, y_n = pairs[-1]
    gamma = float(s_n @ y_n) / float(y_n @ y_n)
    r = gamma * q
    for (s, y), a in zip(pairs, reversed(alphas)):    # oldest first
        rho = 1.0 / float(y @ s); b = rho * float(y @ r); r = r + (a - b) * s
    return -r, gamma


def one_pair_rule(g, s, y):
    """Slide 33's one-pair rule: after an exact line search (s^T g = 0) the two-loop recursion with the newest pair reduces to
    d = gamma (-g + (y^T g / y^T s) s): the inverse curvature gamma along the move rescales the negative gradient, and
    the s term turns it toward the previous move by an amount that the gradient change sets."""
    gamma = float(s @ y) / float(y @ y)
    return gamma * (-g + float(y @ g) / float(y @ s) * s), gamma, float(y @ g) / float(y @ s)


def bfgs_inverse_update(H, s, y):
    """The explorer's bfgsStep update (I - rho s y^T) H (I - rho y s^T) + rho s s^T, reset to I when s^T y <= 1e-10."""
    sy = float(s @ y)
    if sy <= 1e-10:
        return np.eye(2)
    rho = 1.0 / sy; I = np.eye(2)
    return (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)


# --- traces ----------------------------------------------------------------------------------------------------------

def trace_cg(n_updates=3):
    """n accepted CG updates with the explorer's exact line search; returns the points, gradients, directions, betas,
    step lengths and the direction proposed at the end (with its two drawn parts)."""
    z = Z0.copy(); g_old = None; d_prev = None
    pts, gs, ds, betas, alphas, notes = [z.copy()], [grad(z)], [], [], [], []
    for _ in range(n_updates):
        g = grad(z); d, beta, note = cg_direction(g, g_old, d_prev)
        a = exact_line_search(z, d)
        ds.append(d); betas.append(beta); alphas.append(a); notes.append(note)
        g_old, d_prev = g, d
        z = z + a * d; pts.append(z.copy()); gs.append(grad(z))
    g = grad(z); d_next, beta_next, note_next = cg_direction(g, g_old, d_prev)
    return dict(points=np.array(pts), grads=np.array(gs), directions=np.array(ds), betas=betas, alphas=alphas, notes=notes,
                d_next=d_next, beta_next=beta_next, note_next=note_next, parts=(-g, beta_next * d_prev), d_prev=d_prev)


def trace_lbfgs(n_updates=3, m=M_PAIRS):
    """n accepted L-BFGS updates (memory m, two-loop recursion, exact line search); the direction proposed at the end."""
    z = Z0.copy(); pairs = []
    pts, gs, ds, gammas, alphas = [z.copy()], [grad(z)], [], [], []
    for _ in range(n_updates):
        g = grad(z); d, gamma = lbfgs_direction(g, pairs)
        if float(g @ d) >= 0:                           # the usual safeguard; never triggers on this trace (asserted)
            d, gamma = -g, None
        a = exact_line_search(z, d)
        ds.append(d); gammas.append(gamma); alphas.append(a)
        z_new = z + a * d; s, y = z_new - z, grad(z_new) - g
        pairs.append((s, y)); pairs = pairs[-m:]
        z = z_new; pts.append(z.copy()); gs.append(grad(z))
    d_next, gamma_next = lbfgs_direction(grad(z), pairs)
    return dict(points=np.array(pts), grads=np.array(gs), directions=np.array(ds), gammas=gammas, alphas=alphas,
                pairs=pairs, d_next=d_next, gamma_next=gamma_next)


def trace_bfgs(n_updates=3, line_search="strong-wolfe"):
    """The explorer's BFGS from the identity with its strong-Wolfe (or wolfe) line search: slide 32's demo and, after
    the first update, slide 33's figure. Returns the points, the stored pairs, the inverse estimate and the direction
    proposed at the end, with the one-pair expansion d = -g + a y + (b - a) s of the two-loop form (H0 = I)."""
    z = Z0.copy(); H = np.eye(2); pts, alphas, notes, pairs, dirs = [z.copy()], [], [], [], []
    for _ in range(n_updates):
        g = grad(z); d = -H @ g; reset = False
        if float(g @ d) >= 0:
            d, reset = -g, True
        a = wolfe_line_search(z, d, strong=(line_search == "strong-wolfe"))
        z_new = z + a * d; s, y = z_new - z, grad(z_new) - g
        H = np.eye(2) if reset else bfgs_inverse_update(H, s, y)
        notes.append("reset" if reset else ""); alphas.append(a); pairs.append((s, y)); dirs.append(d); z = z_new; pts.append(z.copy())
    g = grad(z); d_next = -H @ g
    s, y = pairs[-1]; rho = 1.0 / float(y @ s)
    a_coef = rho * float(s @ g); b_coef = rho * float(y @ (g - a_coef * y))       # the two-loop scalars with H0 = I, one pair
    return dict(points=np.array(pts), alphas=alphas, notes=notes, pairs=pairs, directions=dirs, H=H, d_next=d_next,
                expansion=dict(a=a_coef, b=b_coef))


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(float(u @ v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def check(cg, lb, bf=None):
    if bf is not None:                                           # slide 33: the explorer's BFGS after its first update
        s, y = bf["pairs"][-1]; g = grad(bf["points"][-1]); d = bf["d_next"]
        assert float(s @ y) > 0, "the BFGS pair has non-positive curvature"
        assert float(g @ d) < 0, "the BFGS direction is not downhill"
        assert angle_deg(d, -g) > 10, "the BFGS direction is not visibly distinct from the negative gradient"
        assert np.allclose(bf["H"] @ y, s, atol=1e-9), "the BFGS estimate does not satisfy the secant condition"
        e = bf["expansion"]
        assert np.allclose(-g + e["a"] * y + (e["b"] - e["a"]) * s, d, atol=1e-9), "the one-pair expansion disagrees with -H g"
    for s, y in lb["pairs"]:
        assert float(s @ y) > 0, "a stored pair has non-positive curvature"
    g = grad(lb["points"][-1])
    assert float(g @ lb["d_next"]) < 0, "the L-BFGS direction is not downhill"
    assert angle_deg(lb["d_next"], -g) > 10, "the L-BFGS direction is not visibly distinct from the negative gradient"
    g = grad(cg["points"][-1])
    assert float(g @ cg["d_next"]) < 0, "the CG direction is not downhill"
    assert np.allclose(sum(cg["parts"]), cg["d_next"], atol=1e-12), "the CG direction is not the sum of its two parts"
    assert cg["beta_next"] > 0 and cg["note_next"] == "", "the CG step shown restarted; the illustration needs beta > 0"
    for d, a, z in zip(cg["directions"], cg["alphas"], cg["points"][:-1]):
        assert abs(float(grad(z + a * d) @ d)) < 1e-6, "an accepted CG update is not a line minimum"
    # the one-pair rule (exact line search) equals the two-loop recursion with the newest pair, and its bracket is the CG direction
    s, y = lb["pairs"][-1]; g = grad(lb["points"][-1])
    d_rule, gamma, c = one_pair_rule(g, s, y)
    assert np.allclose(d_rule, lbfgs_direction(g, [(s, y)])[0], atol=1e-12), "the one-pair rule disagrees with the recursion"
    assert np.allclose(-g + c * s, cg["d_next"], atol=1e-6), "the one-pair bracket is not the CG direction"


# --- the earlier quadratic example (optional reference in notes section 3.5) ------------------------------------------

def quadratic_example():
    """f = z1^2 + 10 z2^2 from (-2.5, 0.8): one exact line minimum, then the one-pair L-BFGS and the CG directions.
    Both point at the minimizer there (each is conjugate to the previous move after an exact line minimum, which in two
    dimensions fixes the direction); the Rosenbrock slides replaced it because that coincidence hid the methods' difference."""
    A = np.diag([1.0, 10.0]); H = 2 * A; gq = lambda z: 2 * A @ z
    z0 = np.array([-2.5, 0.8]); g0 = gq(z0); d0 = -g0
    a = float(g0 @ g0) / float(d0 @ H @ d0); z1 = z0 + a * d0; g1 = gq(z1)
    s, y = z1 - z0, g1 - g0
    d_l, gamma = lbfgs_direction(g1, [(s, y)])
    d_c, beta, _ = cg_direction(g1, g0, d0)
    assert abs(float(d_l @ H @ d0)) < 1e-9 and abs(float(d_c @ H @ d0)) < 1e-9
    assert abs(d_l[0] * z1[1] - d_l[1] * z1[0]) < 1e-9 and abs(d_c[0] * z1[1] - d_c[1] * z1[0]) < 1e-9
    return dict(alpha=a, z1=z1, g1=g1, s=s, y=y, gamma=gamma, d_lbfgs=d_l, beta=beta, d_cg=d_c)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    cg, lb, bf, bf1 = trace_cg(), trace_lbfgs(), trace_bfgs(), trace_bfgs(1)
    check(cg, lb, bf1)
    print("Rosenbrock f = (1 - z1)^2 + 100 (z2 - z1^2)^2 from the Classic start", Z0, " f =", f(Z0), " grad =", grad(Z0))
    print("\noptional reference (slide 32's demo): the explorer's BFGS (identity start, strong-Wolfe line search) after its first accepted update:")
    s1, y1 = bf1["pairs"][-1]; g1 = grad(bf1["points"][-1]); d1 = bf1["d_next"]; e = bf1["expansion"]
    print(f"  update 1: steepest descent, alpha {bf1['alphas'][0]:.5f} (strong Wolfe)  ->  z_new = {bf1['points'][-1]}  f = {f(bf1['points'][-1]):.4f}  grad = {g1}")
    print(f"  pair: s = {s1}  y = {y1}  s.y = {float(s1 @ y1):.1f} > 0;  H1 = {bf1['H'].tolist()}  (H1 y = {bf1['H'] @ y1} = s)")
    print(f"  -g_new = {-g1}  |g_new| = {np.linalg.norm(g1):.2f};  BFGS direction -H1 g_new = {d1}  |d| = {np.linalg.norm(d1):.2f};  angle from -g_new = {angle_deg(d1, -g1):.1f} deg;  slope {float(g1 @ d1):.1f} (downhill)")
    print(f"  one-pair expansion: d = -g_new + a y + (b - a) s with a = {e['a']:.4f}, b = {e['b']:.4f}  ->  -g + {e['a']:.3f} y + {e['b'] - e['a']:.3f} s")
    print("\nslide 33: L-BFGS (m = 3, two-loop recursion, exact line search), three accepted updates:")
    for i in range(3):
        print(f"  update {i + 1}: direction {lb['directions'][i]}  alpha {lb['alphas'][i]:.5f}  ->  z = {lb['points'][i + 1]}  f = {f(lb['points'][i + 1]):.5f}  grad = {lb['grads'][i + 1]}")
    for k, (s, y) in enumerate(lb["pairs"], 1):
        print(f"  stored pair {k}: s = {s}  y = {y}  s.y = {float(s @ y):.4f}  (H s at the new point = {hess(lb['points'][k]) @ s})")
    g = grad(lb["points"][-1]); d = lb["d_next"]
    print(f"  at z^(3) = {lb['points'][-1]}: -grad = {-g}  |grad| = {np.linalg.norm(g):.4f};  gamma = {lb['gamma_next']:.5f};  L-BFGS direction = {d}  |d| = {np.linalg.norm(d):.4f}")
    print(f"  angle between -grad and the L-BFGS direction = {angle_deg(d, -g):.1f} deg;  slope grad.d = {float(g @ d):.4f} (downhill)")
    s, y = lb["pairs"][-1]; d_rule, gamma1, c = one_pair_rule(g, s, y)
    print(f"  slide 33's one-pair rule: gamma = {gamma1:.5f}, y.g / y.s = {c:.1f}, d = gamma (-g + {c:.1f} s) = {d_rule}  (newest pair only; same direction, {np.linalg.norm(d_rule) / np.linalg.norm(d):.1f} times longer than with three pairs)")
    print(f"  the exact Newton direction there, for reference = {-np.linalg.solve(hess(lb['points'][-1]), g)}")
    print("\nnonlinear CG (Polak-Ribiere, clamped, exact line search), three accepted updates:")
    for i in range(3):
        print(f"  update {i + 1}: beta {cg['betas'][i]:.5f} direction {cg['directions'][i]}  alpha {cg['alphas'][i]:.5f}  ->  z = {cg['points'][i + 1]}  f = {f(cg['points'][i + 1]):.5f}  grad = {cg['grads'][i + 1]}" + (f"  [{cg['notes'][i]}]" if cg['notes'][i] else ""))
    g = grad(cg["points"][-1]); mg, bd = cg["parts"]
    print(f"  at z^(3) = {cg['points'][-1]}: -g_new = {mg};  previous direction d_prev = {cg['d_prev']}  (previous step = alpha d_prev with alpha {cg['alphas'][-1]:.5f});  beta = {cg['beta_next']:.5f}")
    print(f"  beta d_prev = {bd};  d_new = -g_new + beta d_prev = {cg['d_next']};  angle between -g_new and d_new = {angle_deg(cg['d_next'], mg):.1f} deg;  slope g.d = {float(g @ cg['d_next']):.4f} (downhill)")
    print("\nthe explorer's BFGS (identity start, strong-Wolfe line search), three steps for the slide-32 demonstration:")
    for i in range(3):
        print(f"  step {i + 1}: alpha {bf['alphas'][i]:.5f} -> z = {bf['points'][i + 1]}  f = {f(bf['points'][i + 1]):.5f}" + (f"  [{bf['notes'][i]}]" if bf['notes'][i] else ""))
    q = quadratic_example()
    print(f"\noptional quadratic reference (notes 3.5): alpha* {q['alpha']:.5f}, z1 {q['z1']}, gamma {q['gamma']:.5f}, L-BFGS {q['d_lbfgs']}, beta {q['beta']:.5f}, CG {q['d_cg']}: both point at the minimizer")
    print("all checks passed")
