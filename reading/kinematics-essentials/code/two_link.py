"""Sessions 2.3–2.4 worked example: a two-link planar arm, l1 = 0.4 m, l2 = 0.3 m, q = (30°, 45°).

Forward kinematics as a product of planar homogeneous transforms (3×3; the 3-D version is
4×4 with the same structure), the Jacobian by differentiation and by finite differences,
its determinant and singular values, and the singular configurations. Plain NumPy so it
runs anywhere. Every number printed here appears in notes.md §5 and on the slides.

Run:  python two_link.py          Expected runtime: well under a second.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

L1, L2 = 0.4, 0.3                       # link lengths [m]
Q_DEG = np.array([30.0, 45.0])          # the worked configuration


def rot2(theta):
    """Planar homogeneous rotation about the origin by theta radians (3×3)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def trans2(x, y):
    """Planar homogeneous translation by (x, y) (3×3)."""
    return np.array([[1.0, 0.0, x],
                     [0.0, 1.0, y],
                     [0.0, 0.0, 1.0]])


def fk_transforms(q):
    """The four factors of ^W X^E(q) and their running products (notes §4.4, eq. 10)."""
    q1, q2 = q
    factors = [("R(q1)", rot2(q1)), ("T(l1,0)", trans2(L1, 0.0)), ("R(q2)", rot2(q2)), ("T(l2,0)", trans2(L2, 0.0))]
    X = np.eye(3)
    running = []
    for name, F in factors:
        X = X @ F
        running.append((name, X.copy()))
    return factors, running, X


def fk(q):
    """End-effector position by the closed form (notes eq. 11)."""
    q1, q2 = q
    return np.array([L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
                     L1 * np.sin(q1) + L2 * np.sin(q1 + q2)])


def jacobian(q):
    """Analytic 2×2 position Jacobian (notes eq. 12)."""
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12],
                     [ L1 * c1 + L2 * c12,  L2 * c12]])


def jacobian_fd(q, h=1e-6):
    """Central finite differences, one joint at a time (notes §5 (d))."""
    J = np.zeros((2, 2))
    for i in range(2):
        e = np.zeros(2); e[i] = h
        J[:, i] = (fk(q + e) - fk(q - e)) / (2 * h)
    return J


if __name__ == "__main__":
    q = np.deg2rad(Q_DEG)
    print(f"=== Two-link planar arm: l1 = {L1} m, l2 = {L2} m, q = {Q_DEG} deg ===")

    # ---- 2.3: forward kinematics as a product of transforms
    factors, running, X_WE = fk_transforms(q)
    print("\n(a) ^W X^E = R(q1) T(l1,0) R(q2) T(l2,0); running products:")
    for stage, (name, X) in enumerate(running, 1):
        print(f"  stage {stage}, after {name:9s}: construction origin ({X[0,2]:.4f}, {X[1,2]:.4f}), heading {np.degrees(np.arctan2(X[1,0], X[0,0])):6.1f} deg")
    print("  ^W X^E =\n", X_WE)
    p = fk(q)
    print(f"(b) closed form: x = l1 cos q1 + l2 cos(q1+q2) = {L1*np.cos(q[0]):.4f} + {L2*np.cos(q.sum()):.4f} = {p[0]:.4f}")
    print(f"                 y = l1 sin q1 + l2 sin(q1+q2) = {L1*np.sin(q[0]):.4f} + {L2*np.sin(q.sum()):.4f} = {p[1]:.4f}")
    assert np.allclose(X_WE[:2, 2], p)
    print(f"    heading of the end-effector frame: q1 + q2 = {np.degrees(q.sum()):.0f} deg")

    # ---- inverse of a transform: check X^-1 = [[R^T, -R^T p],[0,1]]
    R, t = X_WE[:2, :2], X_WE[:2, 2]
    X_inv = np.eye(3); X_inv[:2, :2] = R.T; X_inv[:2, 2] = -R.T @ t
    print("(c) ^E X^W = (^W X^E)^-1 via [R^T, -R^T p]; check X X^-1 = I:", np.allclose(X_WE @ X_inv, np.eye(3)))
    print("    the base seen from the end-effector: ^E p^W =", X_inv[:2, 2])

    # ---- 2.4: the Jacobian
    J = jacobian(q)
    print("\n(d) J(q) analytic =\n", J)
    print("    J(q) central finite differences (h = 1e-6) =\n", jacobian_fd(q))
    print("    max |analytic - FD| =", f"{np.abs(J - jacobian_fd(q)).max():.2e}")
    print("    column 1 = (-y, x) =", np.array([-p[1], p[0]]), "  (base joint turns the whole arm about W)")
    print("    column 2 = l2 (-sin(q1+q2), cos(q1+q2)) =", L2 * np.array([-np.sin(q.sum()), np.cos(q.sum())]))
    # ---- differential forward kinematics: known joint angular velocities give tip velocity
    qdot_forward = np.array([0.01, 0.0])                       # rad/s; same components as slide31's displacement, different units
    tip_velocity_forward = J @ qdot_forward
    print("    differential FK: q_dot =", qdot_forward, "rad/s -> tip velocity =", tip_velocity_forward, "m/s")
    print("    tip velocity in mm/s =", 1000 * tip_velocity_forward, "; multiply J, do not invert it")
    det = np.linalg.det(J)
    print(f"(e) det J = {det:.4f};  l1 l2 sin q2 = {L1*L2*np.sin(q[1]):.4f}")
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"    singular values = {sv}  (semi-axes of the manipulability ellipse, in m per rad)")

    # ---- first-order prediction vs actual motion
    dq = np.deg2rad([5.0, 0.0])
    print(f"\n(f) move joint 1 by 5 deg: actual tip motion = {fk(q + dq) - p},  J dq = {J @ dq}")
    dq = np.deg2rad([30.0, 0.0])
    print(f"    move joint 1 by 30 deg: actual = {fk(q + dq) - p},  J dq = {J @ dq}  (first order only)")

    # ---- singular configurations
    print("\n(g) singularities: det J = l1 l2 sin q2 = 0  <=>  q2 = 0 (stretched) or q2 = 180 deg (folded)")
    for q2d in (0.0, 180.0, 10.0, 45.0, 90.0):
        qq = np.deg2rad([30.0, q2d]); Jq = jacobian(qq)
        svq = np.linalg.svd(Jq, compute_uv=False)
        print(f"    q2 = {q2d:5.1f} deg: det J = {np.linalg.det(Jq):7.4f}, sigma = {svq}, rank = {np.linalg.matrix_rank(Jq, tol=1e-9)}")
    qs = np.deg2rad([30.0, 0.0]); Js = jacobian(qs)
    radial = fk(qs) / np.linalg.norm(fk(qs))
    print("    at q2 = 0 the columns are parallel:", Js[:, 0] / np.linalg.norm(Js[:, 0]), Js[:, 1] / np.linalg.norm(Js[:, 1]))
    print("    and both are perpendicular to the radial direction", radial, ": J^T radial =", Js.T @ radial)

    # ---- differential inverse kinematics in the square, non-singular case: q_dot = J^-1 xdot
    xdot = np.array([0.0, 0.1])                                # desired tip velocity: 0.1 m/s straight up
    print("\n(i) differential IK, square case: q_dot = J^-1 xdot for xdot = (0, 0.1) m/s")
    for q2d in (45.0, 10.0, 1.0):
        qq = np.deg2rad([30.0, q2d]); Jq = jacobian(qq)
        qd = np.linalg.solve(Jq, xdot)
        print(f"    q2 = {q2d:4.0f} deg: det J = {np.linalg.det(Jq):.4f}, q_dot = {qd} rad/s, |q_dot| = {np.linalg.norm(qd):.2f}")
    print("    this upward request has a component along the arm, so it costs ever larger joint rates as the arm straightens: the inverse blows up at det J = 0")

    # ---- horizontal-only task: multiple answers, one minimum-norm choice, and zero-task motion
    J_x = J[0:1, :]
    request_x = np.array([0.1])                              # m/s; vertical velocity is unspecified
    rates_A = np.array([request_x[0] / J_x[0, 0], 0.0])
    rates_B = np.array([0.0, request_x[0] / J_x[0, 1]])
    rates_min = np.linalg.pinv(J_x) @ request_x
    extra_rates = rates_B - rates_A
    print("\n(j) horizontal-only differential IK: keep q = (30, 45) deg and request x_dot = 0.1 m/s")
    print("    J_x =", J_x, "m/rad; y_dot is unspecified (not constrained to zero)")
    squared_gain = float((J_x @ J_x.T)[0, 0])
    one_row_pinv = J_x.T / squared_gain
    np.testing.assert_allclose(one_row_pinv, np.linalg.pinv(J_x), atol=1e-14)
    print("    for this nonzero row: J_x^+ = J_x^T / sum(J_x entries squared)")
    print(f"    squared-entry sum = {squared_gain:.8f}; J_x^+ = {one_row_pinv.ravel()} rad/m")
    print("    using slide 41's rounded entries: squared-entry sum = 0.32388808, J_x^+ = (-1.5123, -0.8948) rad/m")
    for name, rates in (("A", rates_A), ("B", rates_B), ("minimum norm", rates_min)):
        tip_velocity = J @ rates
        print(f"    {name}: q_dot = {rates} rad/s, tip (x_dot, y_dot) = {tip_velocity} m/s, |q_dot| = {np.linalg.norm(rates):.4f} rad/s")
        np.testing.assert_allclose(J_x @ rates, request_x, atol=1e-14)
    np.testing.assert_allclose(J_x @ extra_rates, [0.0], atol=1e-14)
    np.testing.assert_allclose(rates_A + extra_rates, rates_B, atol=1e-14)
    np.testing.assert_allclose(rates_min @ extra_rates, 0.0, atol=1e-14)
    assert np.linalg.norm(rates_min) < min(np.linalg.norm(rates_A), np.linalg.norm(rates_B))
    print("    extra coordinated rates B - A =", extra_rates, "rad/s")
    print("    their task effect J_x (B - A) =", J_x @ extra_rates, "m/s; their full tip velocity =", J @ extra_rates, "m/s")
    print("    this null motion preserves horizontal velocity only; vertical velocity can change")

    # ---- the angular part: a third task row
    print("\n(h) full planar Jacobian (heading, x, y) is 3x2: [1, 1] above the rows of J, angular first; 3 task quantities, 2 joints")
    J3 = np.vstack([[1.0, 1.0], J])
    print(J3, "\n    rank =", np.linalg.matrix_rank(J3), "< 3: the three motion components cannot be set independently")
    heading_fixed_rates = np.array([1.0, -1.0])
    print("    q_dot =", heading_fixed_rates, "rad/s gives (heading rate [rad/s], tip velocity [m/s]) =", J3 @ heading_fixed_rates)
    print("    at this pose, a still tip requires q_dot =", np.linalg.solve(J, np.zeros(2)), "rad/s, so its heading rate is also zero")

    # ---- (k) the closing bridge: a static position target, and the local joint-change problem it poses
    y_star = np.array([0.5, 0.2])                              # position target [m], the same one Deck 04 uses
    e = y_star - p                                             # remaining displacement, not a velocity
    print(f"\n(k) closing bridge: position target y* = {y_star} m, tip phi(q) = {p} m")
    print(f"    remaining displacement e = y* - phi(q) = {e} m, |e| = {np.linalg.norm(e):.4f} m (no time step anywhere here)")
    dq = np.linalg.solve(J, e)                                 # the local model J dq ~ e; J is square and invertible here
    q_new = q + dq
    print(f"    local model J dq ~ e; J is square and invertible (det J = {np.linalg.det(J):.4f}), so dq = J^-1 e = {dq} rad = {np.degrees(dq)} deg")
    print(f"    q + dq = {np.degrees(q_new)} deg, phi(q + dq) = {fk(q_new)} m: the linear model promised 0 m left, the arm has {np.linalg.norm(y_star - fk(q_new)):.4f} m left")
    print("    a 74 deg elbow change is far outside the range where (f) trusted the model; Deck 04 derives which step to trust, for any J, and repeats it (backup slide)")
    q_s = np.zeros(2); y_s = np.array([0.5, 0.0]); e_s = y_s - fk(q_s); J_s = jacobian(q_s)
    print(f"    the zero-step caution: stretched arm q = (0, 0) deg, tip {fk(q_s)}, reachable target y* = {y_s} (|y*| = {np.linalg.norm(y_s):.1f} is inside [0.1, 0.7]):")
    print(f"        e = {e_s}, J = {J_s.tolist()}, J^T e = {J_s.T @ e_s}: no first-order motion along the arm, so every step built from J and e is zero here, although bending the elbow reaches the target")
