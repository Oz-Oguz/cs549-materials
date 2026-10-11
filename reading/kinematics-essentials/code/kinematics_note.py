"""Every number quoted in the week-2 reading `lectures/02-reading-kinematic-foundations/notes.md`,
printed by one script, plus the reading's two new figures.

The reading is the self-contained kinematics chapter behind decks 02 and 03 (sessions 2.1-2.4).
Its numbers are frozen: they already appear in Deck 03's notes and slides, in Deck 04's notes, and
in the `two-link-arm` module's default view. This script reproduces all of them and asserts each
one, so a single run checks the chapter. Sections below are the reading's section numbers.

  section 2  ^W X^E as a 3x3 planar pose, its inverse, and the one 4x4 instance (the mug on the tray)
  section 3  the elbow position: one link of the kinematic tree, evaluated
  section 4  forward kinematics as a product of four factors, with the running product after each
  section 5  the reach annulus, closed-form two-link inverse kinematics, and the 2 / 1 / 0 solution cases
  section 6  the Jacobian: the two columns as arrows, the finite-difference check, det J, the full SVD,
             and how far the first-order prediction misses at 5 degrees and at 30 degrees
  section 7  differential inverse kinematics: q_dot = J^-1 x_dot, the null space of a one-dimensional
             task, J^+ = V Sigma^+ U^T by hand, and Gauss-Newton IK from two starting configurations
  section 8  singularities: the q2 sweep 45, 10, 1, 0 degrees under one command, the folded pose, and
             what damping does to each singular direction
  Panda      the 3-D example in Drake: the ready pose, the all-zero pose, and the near-singular pose

It also writes two figures into ../fig/ (pass --no-figures to skip them):

  fig/kinematic_tree.png   the arm-plus-camera tree, with the path "up to the root, down the other
                           branch" that computes ^G X^O highlighted
  fig/nullspace_line.png   the joint-rate plane at q = (30 deg, 45 deg) for the one-dimensional task
                           "the tip's x-velocity is 0.1 m/s": the line of solutions, the null
                           direction, and the minimum-norm point

Run (from anywhere; every path is built from __file__, never from the working directory, because
scripts/check_all.sh step 2 runs each lectures/*/code/*.py with the working directory set to its
own folder):

    .venv/bin/python lectures/02-reading-kinematic-foundations/code/kinematics_note.py
    .venv/bin/python lectures/02-reading-kinematic-foundations/code/kinematics_note.py --no-figures

Imports rather than reimplements: `rot2`, `trans2`, `fk_transforms`, `fk`, `jacobian`,
`jacobian_fd`, `L1`, `L2`, `Q_DEG` from `lectures/02-D2-transforms-fk-jacobian/code/two_link.py`
(whose module-level `np.set_printoptions(precision=4, suppress=True)` is inherited here on purpose,
so both scripts print the same digits), and optionally `build_Panda`, `Panda_numbers`, `Q_BENT`,
`Q_HOME`, `Q_NEAR` from `drake/01-fk-jacobian/fk_jacobian_demo.py`. The Drake import is guarded: a
machine without pydrake still prints and checks every two-link number, says why the 3-D example was
skipped (the demonstration folder is missing, pydrake is not installed, or another import failed),
and exits 0. The helper folders are found from this file's location in both layouts: the repository
checkout, and the student package, whose code/ folder keeps two_link.py beside this file.

Exit code 0 and the final line "all frozen numbers reproduced" mean every asserted number matched
to the precision the reading prints. A mismatch prints the label, what was computed and what was
expected, and exits 1.

Tested 2026-09-21 with numpy 2.5.2 and drake 1.46.0 on macOS arm64: 0.3 s with --no-figures, 0.6 s
with the figures and the Panda, well inside check_all.sh's budget. Matplotlib 3.11.1 draws the
figures on the Agg backend, so no display is needed.

Note for `scripts/check_all.sh`: step 2 runs this file (its glob skips only `*make_figures*`), so a
health check redraws both PNGs. That is intended — the reading's rule is that one script produces
both the numbers and the figures — but pass --no-figures if a run must leave `fig/` untouched.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")          # keep stderr empty: check_all.sh step 2 judges the exit code, and a quiet log reads better

# --- paths, built from this file's location and never from the working directory -------------------
_HERE = Path(__file__).resolve().parent


def _find_up(*parts):
    """The nearest folder at or above this file that holds parts/..., or None. In the repository that is
    the root (lectures/02-D2-.../code, drake/01-fk-jacobian); in the student package it is the package
    root (drake/01-fk-jacobian). An unreadable ancestor is skipped. The nearest match wins."""
    for base in (_HERE, *_HERE.parents):
        try:
            if base.joinpath(*parts).is_dir():
                return base.joinpath(*parts)
        except OSError:
            continue
    return None


# the two-link helper: the copy beside this file (the student package) wins over the Deck 03 folder
_TWO_LINK_DIR = _HERE if (_HERE / "two_link.py").is_file() else (_find_up("lectures", "02-D2-transforms-fk-jacobian", "code") or _HERE)
_DRAKE_DIR = _find_up("drake", "01-fk-jacobian")
_FIG_DIR = _HERE.parent / "fig"
for _d in (_TWO_LINK_DIR, _DRAKE_DIR):
    if _d is not None and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# two_link.py calls np.set_printoptions(precision=4, suppress=True) at module level; we inherit it.
from two_link import L1, L2, Q_DEG, fk, fk_transforms, jacobian, jacobian_fd, rot2, trans2  # noqa: E402

try:
    from fk_jacobian_demo import Panda_numbers, Q_BENT, Q_HOME, Q_NEAR, build_Panda
    HAVE_DRAKE = True
    DRAKE_NOTE = ""
except ImportError as _e:
    HAVE_DRAKE = False
    if _DRAKE_DIR is None:
        DRAKE_NOTE = "drake/01-fk-jacobian/fk_jacobian_demo.py was not found next to this folder"
    elif getattr(_e, "name", "") == "pydrake":
        DRAKE_NOTE = "the pydrake package is not installed"
    else:
        DRAKE_NOTE = f"importing fk_jacobian_demo failed ({_e})"


# --- the checker ----------------------------------------------------------------------------------
FAILURES = []


def check(label, got, want, tol):
    """Record whether a computed quantity matches the number the reading prints.

    `tol` is half a unit in the last digit the reading shows, looser where the reading itself says
    "about", and one full unit in the last digit at the four places marked TRUNCATED below, where the
    frozen literal was obtained from a 4-decimal-rounded J rather than from the exact one and so
    differs from the exact value in its last digit by a little over half a unit. Nothing is silently
    accepted: every failure is listed at the end and the script exits 1.
    """
    g, w = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if g.shape != w.shape or not np.all(np.abs(g - w) <= tol):
        FAILURES.append((label, np.array2string(g, precision=6), np.array2string(w, precision=6), tol))
        return False
    return True


def head(title):
    print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")


# --- planar and spatial helpers the reading needs and two_link.py does not carry -------------------
def rot_z(theta):
    """A 4x4 homogeneous rotation about the world z axis by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0, 0.0],
                     [s, c, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def pose4(R, p):
    """The 4x4 pose [[R, p], [0, 1]] from a 3x3 rotation and a 3-vector."""
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, p
    return X


def invert4(X):
    """The inverse of a 4x4 pose, by the structure [[R^T, -R^T p], [0, 1]] rather than by a solver."""
    R, p = X[:3, :3], X[:3, 3]
    return pose4(R.T, -R.T @ p)


def ik_two_link(y, tol=1e-9):
    """Closed-form inverse kinematics of the two-link planar arm by the law of cosines.

    The elbow triangle has sides l1, l2 and r = |y|, so the interior angle at the elbow fixes
    cos q2 = (r^2 - l1^2 - l2^2) / (2 l1 l2). Each admissible sign of q2 gives one solution, and q1
    then follows by subtracting, from the direction of the target, the angle the second link opens
    up at the shoulder:

        q1 = atan2(y*_y, y*_x) - atan2(l2 sin q2, l1 + l2 cos q2).

    Returns a list of configurations in radians: two when the target is strictly inside the annulus,
    one on either boundary circle (where sin q2 = 0 and the two solutions coincide), none outside it.
    """
    y = np.asarray(y, dtype=float)
    r2 = float(y @ y)
    c2 = (r2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c2) > 1.0 + tol:
        return []                                             # outside the annulus: no solution
    if abs(c2) >= 1.0 - tol:                                  # on a boundary circle: exactly one
        q2 = 0.0 if c2 > 0 else np.pi
        q1 = np.arctan2(y[1], y[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        return [np.array([q1, q2])]
    out = []
    for sign in (+1.0, -1.0):
        q2 = sign * np.arccos(c2)
        q1 = np.arctan2(y[1], y[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        out.append(np.array([q1, q2]))
    return out


def elbow(q):
    """The elbow (joint 2) position in the world frame."""
    return np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])


def svd_fixed(J):
    """The SVD of J with a fixed sign convention, so that the reading's exact signs print.

    The singular values are unique; a singular *vector pair* is not. Negating u_i and v_i together
    leaves J = U Sigma V^T untouched, so the sign of a pair is a convention and nothing geometric
    depends on it: u_i and -u_i span the same task direction, v_i and -v_i the same joint direction.
    The convention chosen here is "the first component of each right singular vector is positive",
    which is what LAPACK happens to return for this J and which pins the printed signs on any
    machine.
    """
    U, s, Vt = np.linalg.svd(J)
    V = Vt.T.copy()
    U = U.copy()
    for i in range(V.shape[1]):
        if V[0, i] < 0.0:
            V[:, i] *= -1.0
            if i < U.shape[1]:
                U[:, i] *= -1.0
    return U, s, V


def gauss_newton_ik(q0, y_star, steps=5):
    """Gauss-Newton IK: q <- q + J^+(q) (y* - phi(q)), the differential-IK step repeated.

    Returns the history [(k, q, residual, step norm)]. J is square and full rank along both paths
    used in the reading, so J^+ = J^-1 there; `pinv` is used anyway because it is the step the
    reading writes, and it is the step that still means something when J is not square. This is the
    same iteration as `run_gn(..., rho=0.0)` in lectures/03-D1-optimization-1/code/gauss_newton_ik.py,
    with J^+ in place of (J^T J)^-1 J^T.
    """
    q = np.asarray(q0, dtype=float).copy()
    hist = [(0, q.copy(), float(np.linalg.norm(fk(q) - y_star)), None)]
    for k in range(1, steps + 1):
        step = np.linalg.pinv(jacobian(q)) @ (y_star - fk(q))
        q = q + step
        hist.append((k, q.copy(), float(np.linalg.norm(fk(q) - y_star)), float(np.linalg.norm(step))))
    return hist


# --- section 2: frames, poses and rigid transformations -------------------------------------------
def section_2():
    head("Section 2  Frames, poses and rigid transformations")
    q = np.deg2rad(Q_DEG)
    _, _, X_WE = fk_transforms(q)
    print(f"(a) the planar pose of the end-effector at q = ({Q_DEG[0]:.0f} deg, {Q_DEG[1]:.0f} deg), as a 3x3:")
    print("    ^W X^E =\n", X_WE)
    check("2(a) ^W X^E (3x3)", X_WE, [[0.2588, -0.9659, 0.4241],
                                     [0.9659, 0.2588, 0.4898],
                                     [0.0, 0.0, 1.0]], 5e-5)

    R, p = X_WE[:2, :2], X_WE[:2, 2]
    X_EW = np.eye(3)
    X_EW[:2, :2], X_EW[:2, 2] = R.T, -R.T @ p
    print("\n(b) the same pose read backwards: ^E X^W = (^W X^E)^-1 = [[R^T, -R^T p], [0, 1]]")
    print("    ^E X^W =\n", X_EW)
    print(f"    ^W X^E (^W X^E)^-1 = I: {np.allclose(X_WE @ X_EW, np.eye(3))}")
    print("    the base of the arm, seen from the end-effector:  ^E p^W =", X_EW[:2, 2])
    print("    the same point as ^W p^W = (0, 0): a point has one location and as many coordinate")
    print("    triples as there are frames to express it in.")
    check("2(b) ^E p^W", X_EW[:2, 2], [-0.5828, 0.2828], 5e-5)

    # the elbow (joint 2's origin), seen from the end-effector: the reading's second
    # "same point, different numbers" check. It must come out as (-l2, 0): the elbow sits one
    # link length straight behind the tip along the end-effector's own x axis.
    _, running, _ = fk_transforms(q)
    p_W_elbow = running[1][1][:2, 2]                      # after R(q1) T(l1, 0): the elbow in W
    p_E_elbow = R.T @ (p_W_elbow - p)                     # eq. (8) of the reading
    print("    the elbow, seen from the end-effector:  ^E p^elbow = R^T (^W p^elbow - ^W p^E) =", p_E_elbow)
    print(f"    i.e. (-l2, 0) = ({-L2:.4f}, 0): one link length straight behind the tip along E's x axis.")
    check("2(b) ^E p^elbow", p_E_elbow, [-0.3000, 0.0000], 5e-5)

    # The one 4x4 instance in the whole reading: a mug standing on a tray, the tray standing in a
    # room. The tray is 0.75 m up, 0.40 m out along x and 0.20 m along y, turned 30 degrees about the
    # vertical; the mug sits 15 cm along the tray's own x axis, 5 cm along its y, 5 cm above its
    # surface, and is turned back by 30 degrees so that it ends up square with the room.
    p_WT, p_TM = np.array([0.40, 0.20, 0.75]), np.array([0.15, 0.05, 0.05])
    R_WT, R_TM = rot_z(np.deg2rad(30.0))[:3, :3], rot_z(np.deg2rad(-30.0))[:3, :3]
    X_WT, X_TM = pose4(R_WT, p_WT), pose4(R_TM, p_TM)
    print("\n(c) the one 4x4 instance: a mug on a tray, the tray in a room.")
    print("    ^W p^T = (0.40, 0.20, 0.75) m, ^W R^T = R_z(30 deg)")
    print("    ^T p^M = (0.15, 0.05, 0.05) m, ^T R^M = R_z(-30 deg)")
    print("    ^W X^T =\n", X_WT)
    print("    ^T X^M =\n", X_TM)
    X_WM = X_WT @ X_TM
    print("    ^W X^M = ^W X^T ^T X^M  (the inner letter T cancels) =\n", X_WM)
    print("    last column, i.e. ^W p^M =", X_WM[:3, 3], " m")
    check("2(c) ^W p^M", X_WM[:3, 3], [0.5049, 0.3183, 0.80], 5e-5)

    rot_only = R_WT @ p_TM
    print("\n    the two halves of that last column, separately:")
    print("     ^W R^T ^T p^M =", rot_only, "  (the mug's offset, turned into the room's axes)")
    print("     ^W p^T        =", p_WT, "  (the tray's own location)")
    print("     their sum     =", rot_only + p_WT, "= ^W p^M: a location transforms by R and then p,")
    print("                     whereas a direction would transform by R alone.")
    check("2(c) ^W R^T ^T p^M", rot_only, [0.1049, 0.1183, 0.05], 5e-5)

    print("\n    the rotation half: ^W R^M = ^W R^T ^T R^M =\n", X_WM[:3, :3])
    print(f"    which is the identity to {np.abs(X_WM[:3, :3] - np.eye(3)).max():.1e}: the two 30-degree turns cancel,")
    print("    so the mug is square with the room even though the tray is not.")
    check("2(c) ^W R^M = I", X_WM[:3, :3], np.eye(3), 1e-12)

    print("\n    and the inverse of a 4x4 by structure, not by a solver:")
    print("    (^W X^M)^-1 = ^M X^W =\n", invert4(X_WM))
    print(f"    ^W X^M ^M X^W = I: {np.allclose(X_WM @ invert4(X_WM), np.eye(4))}")
    print("    The planar arm above is the 3x3 case of exactly this object.")

    print("\n(d) the planar round trip of the reading's section 3: frame B at (0.40, 0.20) m turned 30 deg,")
    print("    the point P at (0.15, 0.05) m in B's axes.")
    R2 = rot_z(np.deg2rad(30.0))[:2, :2]
    p_AB = np.array([0.40, 0.20]); p_BP = np.array([0.15, 0.05])
    p_AP = p_AB + R2 @ p_BP
    back = R2.T @ (p_AP - p_AB)
    X3 = np.block([[R2, p_AB[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
    X3_inv = np.block([[R2.T, -(R2.T @ p_AB)[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
    wrong = X3.T @ np.append(p_AP, 1.0)
    print(f"    ^A p^P = p + R b = {p_AP} m  (equation (3))")
    print(f"    back: R^T (^A p^P - p) = {back} m  (equation (5): subtract the offset, rotate back)")
    print(f"    the inverse pose's blocks: the rotation block R^T and the position block -R^T p = {-(R2.T @ p_AB)} m; X X^-1 = I: {np.allclose(X3 @ X3_inv, np.eye(3))}")
    print(f"    the plain transpose of X applied to (^A p^P, 1) gives {wrong}: it does not recover b, and")
    print("    its last row is not (0, 0, 1), so the transposed matrix is not the inverse pose.")
    check("2(d) round trip ^A p^P", p_AP, [0.5049, 0.3183], 5e-5)
    check("2(d) round trip back to B", back, [0.15, 0.05], 1e-9)
    check("2(d) -R^T p", -(R2.T @ p_AB), [-0.4464, 0.0268], 5e-5)
    check("2(d) plain transpose", wrong, [0.5964, 0.0232, 1.2656], 5e-5)


# --- section 3: the robot as a tree of transforms --------------------------------------------------
def section_3():
    head("Section 3  The robot as a tree of transforms")
    q = np.deg2rad(Q_DEG)
    print("The arm branch has one path from the end-effector to the root: E <- L2 <- L1 <- W. Stopping")
    print("that walk one link early evaluates the elbow instead of the tip.")
    X_WL1 = rot2(q[0]) @ trans2(L1, 0.0)
    print("\n    ^W X^L2 = R(q1) T(l1, 0) =\n", X_WL1)
    print("    elbow position, i.e. joint 2's origin in the world:", X_WL1[:2, 2])
    print("    by hand: (l1 cos q1, l1 sin q1) =", elbow(q))
    check("3 elbow position", X_WL1[:2, 2], [0.3464, 0.2000], 5e-5)
    check("3 elbow by hand", elbow(q), [0.3464, 0.2000], 5e-5)
    print("\n    A camera bolted to the world is a second branch off the same root. Reaching the object")
    print("    it sees, from the gripper, means walking up to the root and down the other branch:")
    print("      ^G X^O = (^W X^G)^-1 ^W X^C ^C X^O")
    print("    and the inner letters cancel: G-W, W-C, C-O. See fig/kinematic_tree.png.")


# --- section 4: forward kinematics as a differentiable function -----------------------------------
def section_4():
    head("Section 4  Forward kinematics as a differentiable function")
    q = np.deg2rad(Q_DEG)
    factors, running, X_WE = fk_transforms(q)
    print("^W X^E(q) = R(q1) T(l1, 0) R(q2) T(l2, 0): four factors, each depending on at most one joint.")
    for (name, F), (_, X) in zip(factors, running):
        print(f"\n  factor {name} =\n{F}")
        print(f"  running product after {name} =\n{X}")
        print(f"    its origin is at ({X[0, 2]:.4f}, {X[1, 2]:.4f}) and its x axis points "
              f"{np.degrees(np.arctan2(X[1, 0], X[0, 0])):.0f} deg from the world x axis")
    p = fk(q)
    heading = np.degrees(q.sum())
    print("\n  the closed form, as the last column of the product:")
    print(f"    x = l1 cos q1 + l2 cos(q1 + q2) = {L1 * np.cos(q[0]):.4f} + {L2 * np.cos(q.sum()):.4f} = {p[0]:.4f} m")
    print(f"    y = l1 sin q1 + l2 sin(q1 + q2) = {L1 * np.sin(q[0]):.4f} + {L2 * np.sin(q.sum()):.4f} = {p[1]:.4f} m")
    print(f"    heading = q1 + q2 = {heading:.0f} deg")
    check("4 tip position", p, [0.4241, 0.4898], 5e-5)
    check("4 tip from the product", X_WE[:2, 2], [0.4241, 0.4898], 5e-5)
    check("4 heading [deg]", heading, 75.0, 5e-4)


# --- section 5: the inverse problem, stated honestly ----------------------------------------------
def section_5():
    head("Section 5  The inverse problem, stated honestly")
    print(f"The workspace is the annulus |l1 - l2| <= |y*| <= l1 + l2, that is "
          f"{abs(L1 - L2):.1f} m <= |y*| <= {L1 + L2:.1f} m.")
    check("5 annulus radii", [abs(L1 - L2), L1 + L2], [0.1, 0.7], 5e-12)

    print("\nClosed-form IK by the law of cosines (see ik_two_link above). Three targets, one per case.")
    expected = {
        (0.5, 0.2): [(-11.517, 80.406, (0.392, -0.080)), (55.120, -80.406, (0.229, 0.328))],
        (0.7, 0.0): [(0.000, 0.000, (0.400, 0.000))],
        (0.8, 0.1): [],
    }
    for target, want in expected.items():
        y = np.array(target)
        sols = ik_two_link(y)
        print(f"\n  target y* = ({y[0]:.1f}, {y[1]:.1f}) m, |y*| = {np.linalg.norm(y):.3f} m: "
              f"{len(sols)} solution(s)")
        if not sols:
            print(f"    |y*| = {np.linalg.norm(y):.3f} m > {L1 + L2:.1f} m: outside the annulus, so cos q2")
            print(f"    would have to be {(y @ y - L1**2 - L2**2) / (2*L1*L2):.3f}, which no angle delivers.")
            print("    Not 'the solver failed': there is nothing to find. The nearest reachable point exists;")
            print("    a zero residual does not.")
        for i, (s, (w_q1, w_q2, w_elbow)) in enumerate(zip(sols, want), 1):
            e = elbow(s)
            print(f"    solution {i}: q = ({np.degrees(s[0]):.3f} deg, {np.degrees(s[1]):.3f} deg), "
                  f"sign of q2 = {'+' if s[1] > 1e-9 else ('-' if s[1] < -1e-9 else '0')}")
            print(f"                 phi(q) = {fk(s)} m (back on the target), elbow at "
                  f"({e[0]:.3f}, {e[1]:.3f}) m, det J = {np.linalg.det(jacobian(s)):.4f}")
            check(f"5 target {target} solution {i} q1 [deg]", np.degrees(s[0]), w_q1, 5e-4)
            check(f"5 target {target} solution {i} q2 [deg]", np.degrees(s[1]), w_q2, 5e-4)
            check(f"5 target {target} solution {i} elbow", e, w_elbow, 5e-4)
            check(f"5 target {target} solution {i} reaches y*", fk(s), y, 1e-9)
        check(f"5 target {target} solution count", len(sols), len(want), 0)
        if target == (0.5, 0.2):
            print("    The two are told apart by the sign of q2, not by 'up' and 'down': the first has its")
            print("    elbow below the line from the base to the target, the second above it.")
            print("    NOTE on a frozen number: the second solution's q1 is 55.120 deg. The value 55.117 deg")
            print("    that circulates in the Pass-1 planning brief is a typo; 55.120 is what")
            print("    lectures/03-D1-optimization-1/code/gauss_newton_ik.py and Deck 04 notes section 5(e) print,")
            print("    and it is the value symmetry requires: the two solutions straddle atan2(0.2, 0.5) =")
            print(f"    {np.degrees(np.arctan2(0.2, 0.5)):.3f} deg by {np.degrees(np.arctan2(0.2, 0.5)) + 11.517:.3f} deg either way.")
        if target == (0.7, 0.0):
            print("    On the outer circle the two solutions have merged into one: cos q2 = 1 exactly, so the")
            print("    arm is stretched straight and det J = 0. Section 5 has walked into section 8 already.")


# --- section 6: the Jacobian ------------------------------------------------------------------------
def section_6():
    head("Section 6  The Jacobian")
    q = np.deg2rad(Q_DEG)
    p, J = fk(q), jacobian(q)

    print("(a) Turn one joint at a time and draw the arrow the tip follows.")
    for i, name in ((0, "joint 1 alone (the base joint turns the whole arm about W)"),
                    (1, "joint 2 alone (the elbow swings only the second link)")):
        col = J[:, i]
        mag = float(np.linalg.norm(col))
        print(f"    {name}:")
        print(f"      direction and size: {col} m per rad/s, magnitude {mag:.4f} m/rad "
              f"= {100 * mag * np.pi / 180:.2f} cm/deg")
    print("    column 1 is exactly (-y, x) =", np.array([-p[1], p[0]]), ": perpendicular to the arm,")
    print("    as a rotation about the world origin must be.")
    print("    column 2 is l2 (-sin(q1 + q2), cos(q1 + q2)) =", L2 * np.array([-np.sin(q.sum()), np.cos(q.sum())]))
    check("6(a) column 1", J[:, 0], [-0.4898, 0.4241], 5e-5)
    check("6(a) |column 1| [m/rad]", np.linalg.norm(J[:, 0]), 0.648, 5e-4)
    check("6(a) |column 1| [cm/deg]", 100 * np.linalg.norm(J[:, 0]) * np.pi / 180, 1.13, 5e-3)
    check("6(a) column 2", J[:, 1], [-0.2898, 0.0776], 5e-5)
    check("6(a) |column 2| [m/rad]", np.linalg.norm(J[:, 1]), 0.3, 5e-12)

    print("\n(b) That table of two arrows, written side by side, is a matrix, and by the chain rule it is")
    print("    the derivative of the forward kinematics:")
    print("    J(q) = d phi / d q =\n", J)
    check("6(b) J", J, [[-0.4898, -0.2898], [0.4241, 0.0776]], 5e-5)

    J_fd = jacobian_fd(q)
    print("\n(c) the check that costs nothing: central finite differences, h = 1e-6")
    print("    J by finite differences =\n", J_fd)
    print(f"    max |analytic - finite difference| = {np.abs(J - J_fd).max():.2e}: floating-point round-off in the")
    print("    difference quotient at h = 1e-6 (the central difference's own truncation error, of order h^2, is")
    print("    far smaller). The hand derivation is right.")
    check("6(c) finite-difference agreement", np.abs(J - J_fd).max() < 1e-10, True, 0)

    det = float(np.linalg.det(J))
    print(f"\n(d) det J = {det:.4f} = l1 l2 sin q2 = {L1:.1f} * {L2:.1f} * sin 45 deg = {L1 * L2 * np.sin(q[1]):.4f}")
    check("6(d) det J", det, 0.0849, 5e-5)
    check("6(d) det J = l1 l2 sin q2", det, L1 * L2 * np.sin(q[1]), 1e-12)

    U, s, V = svd_fixed(J)
    print(f"\n(e) the full SVD, both sides: J = U Sigma V^T")
    print(f"    sigma = {s}, ratio sigma_max / sigma_min = cond(J) = {s[0] / s[1]:.2f}")
    print("    U (task-space directions; the semi-axes of the manipulability ellipse) =\n", U)
    print(f"      u1 = {U[:, 0]}   (the direction the tip moves fastest, {s[0]:.4f} m/s per rad/s)")
    print(f"      u2 = {U[:, 1]}   (the direction it moves slowest, {s[1]:.4f} m/s per rad/s)")
    print("    V (joint-space directions) =\n", V)
    print(f"      v1 = {V[:, 0]}   (the joint-rate direction that buys u1)")
    print(f"      v2 = {V[:, 1]}   (the joint-rate direction that buys only u2)")
    print("    The sign of a singular-vector pair is a convention: negating u_i and v_i together")
    print("    leaves J unchanged. svd_fixed pins it by making each v_i's first component positive.")
    print(f"    NOTE on two frozen numbers: the exact ratio is {s[0] / s[1]:.4f}, which rounds to 5.84, and v1's")
    print(f"    first component is {V[0, 0]:.6f}, which rounds to 0.9181. The frozen 5.83 and 0.9182 are what")
    print("    the 4-decimal-rounded J gives (5.8332 and 0.918164). The reading should print 5.84 and")
    print("    0.9182 only if it says which matrix it used; safest is 5.84 and 0.9181, from the exact J.")
    check("6(e) sigma", s, [0.7037, 0.1206], 5e-5)
    # TRUNCATED: the exact ratio is 5.8356, which rounds to 5.84; the frozen 5.83 is the ratio of the
    # singular values of the 4-decimal-rounded J (5.8332). One unit in the last digit.
    check("6(e) cond(J)", s[0] / s[1], 5.83, 1e-2)
    check("6(e) u1", U[:, 0], [-0.8022, 0.5970], 5e-5)
    check("6(e) u2", U[:, 1], [0.5970, 0.8022], 5e-5)
    # TRUNCATED: the exact first component is 0.918149, which rounds to 0.9181; the frozen 0.9182 is
    # what the 4-decimal-rounded J gives (0.918164). One unit in the last digit.
    check("6(e) v1", V[:, 0], [0.9182, 0.3962], 1e-4)
    check("6(e) v2", V[:, 1], [0.3962, -0.9182], 1e-4)
    check("6(e) U Sigma V^T = J", U @ np.diag(s) @ V.T, J, 1e-12)

    print("\n(f) J is a derivative, so it is exact only in the limit. How wrong is it at a finite step?")
    for deg in (5.0, 30.0):
        dq = np.deg2rad([deg, 0.0])
        actual, predicted = fk(q + dq) - p, J @ dq
        err = float(np.linalg.norm(actual - predicted))
        print(f"    turn joint 1 by {deg:.0f} deg: actual displacement {actual} m, J dq = {predicted} m")
        print(f"      error {err:.4f} m = {1000 * err:.2f} mm")
        if deg == 5.0:
            check("6(f) 5 deg actual", actual, [-0.0443, 0.0351], 5e-5)
            check("6(f) 5 deg J dq", predicted, [-0.0427, 0.0370], 5e-5)
            check("6(f) 5 deg error [mm]", 1000 * err, 2.47, 5e-3)
        else:
            print(f"      which is {100 * err:.2f} cm, printed as 8.8 cm in the reading")
            check("6(f) 30 deg error [cm]", 100 * err, 8.8, 5e-2)
    print("    Six times the step, thirty-six times the error: the error is second order in dq.")

    # (g) the full planar Jacobian, task (heading, x, y), angular block on top as the reading's (34)
    J_full = np.vstack([[1.0, 1.0], J])
    print("\n(g) the full planar Jacobian for the task (heading, x, y), angular row on top, eq. (34):")
    print("    J_full =\n", J_full)
    print(f"    rank J_full = {np.linalg.matrix_rank(J_full)} of 3: three task quantities, two joints,")
    print("    so an arbitrary (heading rate, x-velocity, y-velocity) cannot be produced.")
    check("6(g) J_full rows", J_full, [[1.0, 1.0], [-0.4898, -0.2898], [0.4241, 0.0776]], 5e-5)
    check("6(g) rank J_full", np.linalg.matrix_rank(J_full), 2, 0)


# --- section 7: differential inverse kinematics ----------------------------------------------------
def section_7():
    head("Section 7  Differential inverse kinematics, and the iterative IK it implies")
    q = np.deg2rad(Q_DEG)
    J = jacobian(q)
    x_dot = np.array([0.0, 0.1])

    print("(a) the square, full-rank case: q_dot = J^-1 x_dot for x_dot = (0, 0.1) m/s, 10 cm/s straight up")
    q_dot = np.linalg.solve(J, x_dot)
    print(f"    q_dot = {q_dot} rad/s, |q_dot| = {np.linalg.norm(q_dot):.4f} rad/s")
    print(f"    check: J q_dot = {J @ q_dot} m/s")
    print("    Joint 2 turns backwards while joint 1 turns forwards: lifting the tip straight up is not")
    print("    'raise both joints'.")
    check("7(a) q_dot", q_dot, [0.3415, -0.5772], 5e-5)
    check("7(a) |q_dot|", np.linalg.norm(q_dot), 0.67, 5e-3)
    check("7(a) J q_dot = x_dot", J @ q_dot, x_dot, 1e-12)
    dt = 0.01
    step = dt * q_dot
    actual = fk(q + step) - fk(q)
    model_err = np.linalg.norm(actual - dt * x_dot)
    print(f"    one integrated command over dt = {dt} s: dq = {step} rad = {np.degrees(step)} deg,")
    print(f"    actual tip increment {actual} m against the predicted {dt * x_dot} m: model error {model_err:.1e} m,")
    print(f"    about 3 micrometers for a step of at most {np.degrees(np.abs(step)).max():.2f} deg per joint.")
    check("7(a) integrated dq", step, [0.003415, -0.005772], 5e-7)
    check("7(a) actual tip increment", actual, [-0.000002, 0.000998], 5e-7)
    check("7(a) integration model error [m]", model_err, 3.0e-6, 2e-7)
    check("7(a) largest joint move [deg]", np.degrees(np.abs(step)).max(), 0.33, 5e-3)

    print("\n(b) J^+ = V Sigma^+ U^T, computed by hand from the SVD of section 6(e):")
    U, s, V = svd_fixed(J)
    S_plus = np.diag(1.0 / s)
    J_plus = V @ S_plus @ U.T
    print("    Sigma^+ = diag(1/sigma) =\n", S_plus)
    print("    J^+ = V Sigma^+ U^T =\n", J_plus)
    print("    np.linalg.pinv(J) =\n", np.linalg.pinv(J))
    print("    np.linalg.inv(J)  =\n", np.linalg.inv(J))
    print(f"    max difference between the three: "
          f"{max(np.abs(J_plus - np.linalg.pinv(J)).max(), np.abs(J_plus - np.linalg.inv(J)).max()):.1e}")
    print("    Square and full rank, so the pseudo-inverse IS the inverse. The pseudo-inverse earns its")
    print("    name only when J is not square or not full rank.")
    check("7(b) J^+ = pinv(J)", J_plus, np.linalg.pinv(J), 1e-12)
    check("7(b) J^+ = inv(J)", J_plus, np.linalg.inv(J), 1e-12)

    print("\n(c) the null space, made visible by asking for less: the one-dimensional task 'the tip's")
    print("    x-coordinate', so m = 1 and n = 2.")
    J_x = J[0, :]
    print(f"    J_x = {J_x} (the first row of J), a 1x2 matrix. Its rank is 1, so its null")
    print("    space has dimension n - rank = 2 - 1 = 1.")
    n_dir = np.array([-J_x[1], J_x[0]])
    n_dir = n_dir / np.linalg.norm(n_dir)
    if n_dir[0] < 0:
        n_dir = -n_dir                         # convention: point the null direction into q_dot_1 > 0
    print(f"    unit null direction v2 = {n_dir} rad/s (J_x's second right singular vector), and J_x . v2 = {float(J_x @ n_dir):.1e}")
    print("    Move the joints along v2 and the tip's x-coordinate does not change at first order at all.")
    print(f"    NOTE on a frozen number: the second component is {n_dir[1]:.6f}, which rounds to -0.8606.")
    print("    The -0.8607 in the Pass-1 planning brief is a rounding slip; the reading should print -0.8606.")
    # TRUNCATED: the exact second component is -0.860647, which rounds to -0.8606, and the
    # 4-decimal-rounded J gives -0.860640, so the frozen -0.8607 is a rounding slip either way.
    # One unit in the last digit; the reading should print -0.8606.
    check("7(c) null direction", n_dir, [0.5092, -0.8606], 5e-5)
    check("7(c) J_x . n = 0", float(J_x @ n_dir), 0.0, 1e-15)

    x_dot_1d = 0.1
    q_dot_min = np.linalg.pinv(J_x.reshape(1, 2)) @ np.array([x_dot_1d])
    print(f"\n    J_x^+ x_dot for x_dot = {x_dot_1d} m/s: q_dot = {q_dot_min} rad/s, "
          f"|q_dot| = {np.linalg.norm(q_dot_min):.4f} rad/s")
    print(f"    check: J_x q_dot = {float(J_x @ q_dot_min):.4f} m/s, as asked")
    print("    Every q_dot on the line {q_dot : J_x q_dot = 0.1} does the task; J^+ returns the one")
    print("    nearest the origin, which is the point where the line meets the perpendicular from the")
    print("    origin. Two examples of the others, same task velocity, more joint speed:")
    for t in (+0.35, -0.40):
        other = q_dot_min + t * n_dir
        print(f"      q_dot = {other} rad/s: |q_dot| = {np.linalg.norm(other):.4f} rad/s, "
              f"J_x q_dot = {float(J_x @ other):.4f} m/s")
    print("    See fig/nullspace_line.png. The general solution is q_dot = J^+ x_dot + (I - J^+ J) z.")
    P = np.eye(2) - np.linalg.pinv(J_x.reshape(1, 2)) @ J_x.reshape(1, 2)
    print("    I - J^+ J =\n", P, "   which projects onto that line's direction:")
    print(f"      (I - J^+ J) (1, 0) = {P @ np.array([1.0, 0.0])}, parallel to n: "
          f"{np.allclose(np.cross(np.append(P @ np.array([1.0, 0.0]), 0.0), np.append(n_dir, 0.0)), 0)}")
    check("7(c) minimum-norm point", q_dot_min, [-0.1512, -0.0895], 5e-5)
    check("7(c) minimum-norm point achieves the task", float(J_x @ q_dot_min), x_dot_1d, 1e-15)
    check("7(c) minimum norm is smallest", np.linalg.norm(q_dot_min) <= min(
        np.linalg.norm(q_dot_min + t * n_dir) for t in (0.35, -0.40)), True, 0)

    print("\n(d) one joint-position update from the local model (the reading's section 7, Figure 4):")
    y_star = np.array([0.5, 0.2])
    e = y_star - fk(q)
    dq = np.linalg.solve(J, e)
    q_new = q + dq
    e_new = y_star - fk(q_new)
    print(f"    e = y* - phi(q) = {e} m, |e| = {np.linalg.norm(e):.6f} m  (target minus tip)")
    print(f"    J dq = e with det J = {np.linalg.det(J):.4f}: dq = {dq} rad = {np.degrees(dq)} deg")
    print(f"    q + dq = {np.degrees(q_new)} deg, phi(q + dq) = {fk(q_new)} m")
    print(f"    e_new = y* - phi(q + dq) = {e_new} m, |e_new| = {np.linalg.norm(e_new):.6f} m: the linear model")
    print("    promised zero; 0.17 m remains, because a 74 deg elbow turn is far outside the small-change range.")
    check("7(d) e", e, [0.075944, -0.289778], 5e-7)
    check("7(d) |e|", np.linalg.norm(e), 0.2996, 5e-5)
    check("7(d) dq [rad]", dq, [-0.920116, 1.293088], 5e-7)
    check("7(d) dq [deg]", np.degrees(dq), [-52.72, 74.09], 5e-3)
    check("7(d) q + dq [deg]", np.degrees(q_new), [-22.72, 119.09], 5e-3)
    check("7(d) phi(q + dq)", fk(q_new), [0.335682, 0.143665], 5e-7)
    check("7(d) e_new", e_new, [0.164318, 0.056335], 5e-7)
    check("7(d) |e_new|", np.linalg.norm(e_new), 0.1737, 5e-5)

    print("\n(e) self-check numbers for the reading's section 10: the midpoint of the two horizontal-only")
    print("    answers A and B, and an angular velocity from two world axes.")
    qA = np.array([x_dot_1d / J_x[0], 0.0]); qB = np.array([0.0, x_dot_1d / J_x[1]])
    mid = 0.5 * (qA + qB)
    print(f"    A = {qA} rad/s, B = {qB} rad/s, midpoint = {mid} rad/s, |midpoint| = {np.linalg.norm(mid):.4f} rad/s")
    print(f"    J_x midpoint = {float(J_x @ mid):.4f} m/s (the request), vertical tip velocity {float(J[1] @ mid):.4f} m/s")
    axes = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).T
    omega = axes @ np.array([0.3, -0.1])
    print(f"    two hinges with world axes (0,0,1) and (0,1,0) at 0.3 and -0.1 rad/s: omega = {omega} rad/s, |omega| = {np.linalg.norm(omega):.4f} rad/s")
    check("7(e) midpoint", mid, [-0.1021, -0.1725], 5e-5)
    check("7(e) midpoint does the task", float(J_x @ mid), 0.1, 1e-12)
    check("7(e) midpoint vertical velocity", float(J[1] @ mid), -0.0567, 5e-5)
    check("7(e) |midpoint|", np.linalg.norm(mid), 0.2005, 5e-5)
    check("7(e) omega", omega, [0.0, -0.1, 0.3], 1e-12)

    print("\n(f) repeat the step and it becomes an IK algorithm: q <- q + J^+(q) (y* - phi(q)).")
    print("    This is Gauss-Newton on the cost of section 5, min_q |phi(q) - y*|^2: linearize, solve,")
    print("    move, linearise again.")
    y_star = np.array([0.5, 0.2])
    want_res = [3.00e-1, 1.74e-1, 3.59e-2, 1.52e-3, 3.96e-6, 2.03e-11]
    hist = gauss_newton_ik(np.deg2rad([30.0, 45.0]), y_star, steps=5)
    print(f"    from q^(0) = (30 deg, 45 deg) to y* = ({y_star[0]}, {y_star[1]}):")
    print("      k    q1 [deg]    q2 [deg]   |phi(q) - y*| [m]   |step| [rad]")
    for k, qk, res, st in hist:
        print(f"     {k:2d}   {np.degrees(qk[0]):9.3f}   {np.degrees(qk[1]):9.3f}        {res:9.2e}   "
              f"{'' if st is None else f'{st:9.2e}'}".rstrip())
        check(f"7(d) residual at k = {k}", res, want_res[k], 0.005 * want_res[k])
    print("      From k = 1 the residual's exponents run -1, -2, -3, -6, -11: each one roughly doubles")
    print("      the last, which is what quadratic convergence looks like when you read only the")
    print("      exponent. Four steps put the tip within 4e-6 m (k = 4), five within 2e-11 m (k = 5),")
    print("      machine precision on this problem.")
    check("7(d) endpoint q [deg]", np.degrees(hist[-1][1]), [-11.517, 80.406], 5e-4)

    hist_b = gauss_newton_ik(np.deg2rad([60.0, -45.0]), y_star, steps=5)
    k_conv = next(k for k, _, res, _ in hist_b if res < 1e-6)
    print(f"\n    from q^(0) = (60 deg, -45 deg), the same target:")
    print("      k    q1 [deg]    q2 [deg]   |phi(q) - y*| [m]   |step| [rad]")
    for k, qk, res, st in hist_b:
        print(f"     {k:2d}   {np.degrees(qk[0]):9.3f}   {np.degrees(qk[1]):9.3f}        {res:9.2e}   "
              f"{'' if st is None else f'{st:9.2e}'}".rstrip())
    print(f"      |phi - y*| < 1e-6 m after {k_conv} iterations, at q = "
          f"({np.degrees(hist_b[k_conv][1][0]):.2f} deg, {np.degrees(hist_b[k_conv][1][1]):.2f} deg):")
    print("      the OTHER of section 5's two solutions. The method is local, so the starting")
    print("      configuration, not the target, picks which solution you get.")
    check("7(d) second start: iterations to 1e-6", k_conv, 4, 0)
    check("7(d) second start: endpoint q [deg]", np.degrees(hist_b[k_conv][1]), [55.12, -80.41], 5e-3)


# --- section 8: singularities as rank deficiency ---------------------------------------------------
def section_8():
    head("Section 8  Singularities as rank deficiency")
    x_dot = np.array([0.0, 0.1])
    print("The same command x_dot = (0, 0.1) m/s, at q1 = 30 deg, as the elbow straightens.")
    print("\n   q2 [deg]      det J       sigma_1    sigma_2   rank              q_dot [rad/s]      |q_dot|")
    want_det = {45.0: 0.0849, 10.0: 0.0208, 1.0: 0.0021, 0.0: 0.0}
    want_norm = {45.0: 0.67, 10.0: 2.10, 1.0: 18.47}
    for q2d in (45.0, 10.0, 1.0, 0.0):
        qq = np.deg2rad([30.0, q2d])
        Jq = jacobian(qq)
        det = float(np.linalg.det(Jq))
        s = np.linalg.svd(Jq, compute_uv=False)
        rank = int(np.linalg.matrix_rank(Jq, tol=1e-9))
        if rank == 2:
            qd = np.linalg.solve(Jq, x_dot)
            print(f"   {q2d:6.0f}    {det:9.4f}    {s[0]:9.4f}  {s[1]:9.4f}      {rank}   "
                  f"{np.array2string(qd, precision=4, suppress_small=True):>22s}   {np.linalg.norm(qd):8.2f}")
            check(f"8 |q_dot| at q2 = {q2d} deg", np.linalg.norm(qd), want_norm[q2d], 5e-3)
        else:
            print(f"   {q2d:6.0f}    {det:9.4f}    {s[0]:9.4f}  {s[1]:9.4f}      {rank}   "
                  f"{'J is singular: no exact solution for this request (the least-squares answer is in (a))':>22s}")
        check(f"8 det J at q2 = {q2d} deg", det, want_det[q2d], 5e-5)
    print("   Ten degrees from the singularity the command already costs 2.1 rad/s; one degree from it,")
    print("   18.5 rad/s, for the same 10 cm/s. det J did not warn you: it only shrank.")
    det_zero = float(np.linalg.det(jacobian(np.deg2rad([30.0, 0.0]))))
    print(f"   The det J printed as 0.0000 in the last row is actually {det_zero:.1e}: floating-point")
    print("   arithmetic almost never lands on exactly zero, which is why the rank test above needs a")
    print("   tolerance (1e-9 here).")

    q0 = np.deg2rad([30.0, 0.0])
    J0 = jacobian(q0)
    U, s, V = svd_fixed(J0)
    print(f"\n(a) at q2 = 0 exactly: rank J = {int(np.linalg.matrix_rank(J0, tol=1e-9))}, sigma = {s}")
    print(f"    J =\n{J0}")
    print("    its two columns are parallel:")
    print("      column 1 / |column 1| =", J0[:, 0] / np.linalg.norm(J0[:, 0]))
    print("      column 2 / |column 2| =", J0[:, 1] / np.linalg.norm(J0[:, 1]))
    u1 = U[:, 0]
    print(f"    so the tip can move at first order only along +/- u1 = +/- {u1}")
    print("    (the sign is a convention), which is perpendicular to the stretched arm: the radial")
    print("    direction is lost.")
    achieved_component = float(u1 @ x_dot)
    print(f"\n    Of the command (0, 0.1) m/s: the component along u1 is {achieved_component:.4f} m/s, and the")
    print(f"    remaining {np.linalg.norm(x_dot - achieved_component * u1):.4f} m/s lies along the lost radial direction.")
    check("8(a) rank at q2 = 0", int(np.linalg.matrix_rank(J0, tol=1e-9)), 1, 0)
    check("8(a) reachable direction u1", u1, [-0.5, 0.866025], 5e-5)
    check("8(a) achieved component [m/s]", achieved_component, 0.0866, 5e-5)
    check("8(a) lost component [m/s]", np.linalg.norm(x_dot - achieved_component * u1), 0.05, 5e-5)

    qd = np.linalg.pinv(J0) @ x_dot
    achieved, lost = J0 @ qd, x_dot - J0 @ qd
    print(f"\n    The least-norm least-squares answer, which pinv returns without complaint:")
    print(f"      q_dot = {qd} rad/s, |q_dot| = {np.linalg.norm(qd):.4f} rad/s -- finite, and NOT zero")
    print(f"      achieved x_dot = J q_dot = {achieved} m/s")
    print(f"      lost      = x_dot - J q_dot = {lost} m/s, |lost| = {np.linalg.norm(lost):.4f} m/s")
    print("    So pinv does the reachable part of the command, silently drops the rest, and reports")
    print("    nothing. That silence is the failure mode: the arm moves, the task does not finish.")
    check("8(a) least-norm q_dot", qd, [0.1045, 0.0448], 5e-5)
    check("8(a) achieved x_dot", achieved, [-0.0433, 0.0750], 5e-5)
    check("8(a) lost x_dot", lost, [0.0433, 0.0250], 5e-5)

    q180 = np.deg2rad([30.0, 180.0])
    J180 = jacobian(q180)
    s180 = np.linalg.svd(J180, compute_uv=False)
    print(f"\n(b) the other root of det J = l1 l2 sin q2 = 0: the folded pose q2 = 180 deg")
    print(f"    det J = {np.linalg.det(J180):.4f}, sigma = {s180}, rank = {int(np.linalg.matrix_rank(J180, tol=1e-9))}")
    print(f"    Same rank loss, different sigma_1: {s180[0]:.4f} against {s[0]:.4f} stretched. The tip is")
    print(f"    only {np.linalg.norm(fk(q180)):.1f} m from the base here, so the surviving gearing is smaller.")
    check("8(b) det J at q2 = 180 deg", np.linalg.det(J180), 0.0, 5e-5)
    check("8(b) sigma at q2 = 180 deg", s180, [0.3162, 0.0], 5e-5)

    rho = 1e-2
    print(f"\n(c) damping. The step (J^T J + rho I)^-1 J^T acts on singular direction i by the gain")
    print(f"    sigma / (sigma^2 + rho) instead of 1 / sigma. With rho = {rho:g}:")
    print("      sigma      1/sigma    sigma/(sigma^2 + rho)")
    want_gain = {0.7037: 1.3929, 0.1206: 4.9136, 0.01: 0.9901, 0.001: 0.1000}
    for sigma in (0.7037, 0.1206, 0.01, 0.001):
        gain = sigma / (sigma ** 2 + rho)
        print(f"     {sigma:7.4f}   {1 / sigma:9.1f}   {gain:20.4f}")
        check(f"8(c) gain at sigma = {sigma}", gain, want_gain[sigma], 5e-5)
    sigma_star = np.sqrt(rho)
    print(f"    The gain is largest at sigma = sqrt(rho) = {sigma_star:.1f}, where it equals "
          f"1 / (2 sqrt(rho)) = {1 / (2 * sigma_star):.1f},")
    print("    and it falls back to zero as sigma does. So damping bounds the gain at 1 / (2 sqrt(rho)) rad/s of")
    print("    joint speed per m/s requested along that direction, and keeps the requested joint speed finite")
    print("    at the price of tracking the command only approximately. That is a soft tradeoff, not a hard")
    print("    speed limit: a joint-speed limit needs an explicit constraint.")
    check("8(c) max gain 1/(2 sqrt(rho))", 1 / (2 * sigma_star), 5.0, 5e-12)
    check("8(c) argmax sqrt(rho)", sigma_star, 0.1, 5e-12)
    check("8(c) gain at sqrt(rho) is the max", sigma_star / (sigma_star ** 2 + rho), 1 / (2 * sigma_star), 1e-12)

    print("\n    the same two poses as above, undamped against damped, for x_dot = (0, 0.1) m/s:")
    for q2d in (1.0, 0.0):
        qq = np.deg2rad([30.0, q2d])
        Jq = jacobian(qq)
        damped = np.linalg.solve(Jq.T @ Jq + rho * np.eye(2), Jq.T @ x_dot)
        try:
            undamped = np.linalg.solve(Jq, x_dot)
            undamped_str = (f"{np.array2string(undamped, precision=4)} rad/s, "
                            f"|q_dot| = {np.linalg.norm(undamped):.3e} rad/s")
        except np.linalg.LinAlgError:
            undamped_str = "LinAlgError: J is exactly singular, so J^-1 does not exist"
        print(f"      q2 = {q2d:4.0f} deg: J^-1 x_dot = {undamped_str}")
        print(f"                     (J^T J + rho I)^-1 J^T x_dot = {damped} rad/s, "
              f"|q_dot| = {np.linalg.norm(damped):.4f} rad/s")
        print(f"                     achieved x_dot = {Jq @ damped} m/s, lost "
              f"{np.linalg.norm(x_dot - Jq @ damped):.4f} m/s")
        check(f"8(c) damped step at q2 = {q2d} deg is finite", bool(np.all(np.isfinite(damped))), True, 0)
        check(f"8(c) damped step at q2 = {q2d} deg is small", np.linalg.norm(damped) < 1.0, True, 0)
    print("    At q2 = 1 deg the inverse exists and asks for 18.5 rad/s; the damped step asks for 0.11.")
    print("    At q2 = 0 the inverse has no business existing, and note what numpy does: because det J")
    print(f"    is {det_zero:.1e} rather than exactly 0, `solve` does not raise. It returns an answer of")
    print("    order 1e15 rad/s, which is meaningless, and says nothing. Silent nonsense is worse than an")
    print("    exception, and it is why a differential-IK loop is written with a damped step or a")
    print("    singularity test rather than with an inverse. The damped step is unbothered at both")
    print("    poses: nothing in its formula needs to know whether it is at a singularity.")

    # (d) Lynch & Park's pair (printed 173): unit links at (0, pi/4) and (0, 3 pi/4). Same |det J|,
    # different sigma_min: the determinant scales area; sigma_min is the weakest local gain.
    def J_unit(q1, q2):
        s1, c1, s12, c12 = np.sin(q1), np.cos(q1), np.sin(q1 + q2), np.cos(q1 + q2)
        return np.array([[-s1 - s12, -s12], [c1 + c12, c12]])
    print("\n(d) Lynch & Park's pair, unit links, q = (0, pi/4) and (0, 3 pi/4) (their printed page 173):")
    sig_pair = []
    for q2 in (np.pi / 4, 3 * np.pi / 4):
        Jp = J_unit(0.0, q2); sig = np.linalg.svd(Jp, compute_uv=False); sig_pair.append(sig)
        print(f"    q2 = {np.degrees(q2):.0f} deg: |det J| = {abs(np.linalg.det(Jp)):.4f}, sigma = ({sig[0]:.3f}, {sig[1]:.3f}), cond(J) = {sig[0] / sig[1]:.2f}")
    print("    Same area, but the first ellipse is about twice as thin: sigma_min measures the weakest local gain.")
    check("8(d) L&P |det J| both", [abs(np.linalg.det(J_unit(0.0, np.pi / 4))), abs(np.linalg.det(J_unit(0.0, 3 * np.pi / 4)))], [0.7071, 0.7071], 5e-5)
    check("8(d) L&P sigma at pi/4", sig_pair[0], [2.073, 0.341], 5e-4)
    check("8(d) L&P sigma at 3 pi/4", sig_pair[1], [1.073, 0.659], 5e-4)
    check("8(d) L&P cond ratio", [sig_pair[0][0] / sig_pair[0][1], sig_pair[1][0] / sig_pair[1][1]], [6.08, 1.63], 5e-3)


# --- the 3-D example: the Franka Panda in Drake ----------------------------------------------------
def section_panda():
    head("The 3-D example  Franka Panda in Drake: the same ideas with 7 joints and a 6x7 Jacobian")
    if not HAVE_DRAKE:
        print(f"The Panda section is skipped: {DRAKE_NOTE}. Every two-link number above was printed and")
        print("checked without it.")
        if DRAKE_NOTE.startswith("the pydrake package"):
            print("To run this part, install the drake package; the repository's SETUP.md or the package README")
            print("says how.")
        elif DRAKE_NOTE.startswith("drake/01-fk-jacobian"):
            print("To run this part, keep the drake/ folder where it came: beside code/ in the student package,")
            print("at the repository root in the checkout.")
        return
    _, plant, pc, _, Panda, ee = build_Panda()

    p_bent, J_bent, rank_bent, smin_bent = Panda_numbers(plant, pc, Panda, ee, Q_BENT)
    _, s_bent, Vt_bent = np.linalg.svd(J_bent)
    print("(a) the ready pose, q =", Q_BENT)
    print("    flange position ^W p^E =", p_bent, "m")
    print(f"    rank J = {rank_bent} of 6 (J is 6x7), sigma_min = {smin_bent:.4f}")
    print("    singular values =", s_bent)
    print(f"    null-space dimension = n - rank J = 7 - {rank_bent} = {7 - rank_bent}: one whole direction of joint")
    print("    motion that the flange does not see. On the two-link arm that was a line in a plane;")
    print("    here it is a line in seven dimensions, and it is the elbow swinging round.")
    # J is 6x7, so V is 7x7 and its last column spans the one-dimensional null space.
    null_vec = Vt_bent[-1, :]
    print("    unit null-space vector (its overall sign is a convention) =\n     ", null_vec)
    print(f"    |J n| = {np.linalg.norm(J_bent @ null_vec):.2e} (zero to numerical precision), |n| = "
          f"{np.linalg.norm(null_vec):.4f}")
    check("Panda ready flange", p_bent, [0.307, 0.0, 0.590], 5e-4)
    check("Panda ready rank", rank_bent, 6, 0)
    check("Panda ready sigma_min", smin_bent, 0.2244, 5e-5)
    check("Panda ready null dimension", 7 - rank_bent, 1, 0)
    check("Panda ready null vector is in the null space", np.linalg.norm(J_bent @ null_vec), 0.0, 1e-12)

    p_home, J_home, rank_home, smin_home = Panda_numbers(plant, pc, Panda, ee, Q_HOME)
    print("\n(b) the all-zero configuration, q =", Q_HOME, "(the arm straight up)")
    print("    flange position ^W p^E =", p_home, "m")
    print(f"    rank J = {rank_home} of 6, sigma_min = {smin_home:.4f}")
    print("    singular values =", np.linalg.svd(J_home, compute_uv=False))
    print("    the omega_x row of J (rotation of the flange about the world x axis) =", J_home[0, :])
    print(f"    every entry of it is zero to {np.abs(J_home[0, :]).max():.1e}: the seven axis directions span only a")
    print("    plane here (joints 1, 3, 5 and 7 about z; joints 2, 4 and 6 about y), so no combination of joint speeds turns the flange")
    print("    about x. This is rank deficiency you can see in the geometry, not only in a number.")
    check("Panda zero flange", p_home, [0.088, 0.0, 0.926], 5e-4)
    check("Panda zero rank", rank_home, 5, 0)
    check("Panda zero sigma_min", smin_home, 0.0, 1e-12)
    check("Panda zero omega_x row", J_home[0, :], np.zeros(7), 1e-15)

    p_near, J_near, rank_near, smin_near = Panda_numbers(plant, pc, Panda, ee, Q_NEAR)
    print("\n(c) the near-singular pose, q =", Q_NEAR)
    print("    flange position ^W p^E =", p_near, "m")
    print(f"    rank J at tolerance 1e-6 = {rank_near} of 6, which says 'not singular'")
    print(f"    sigma_min = {smin_near:.6f} = {smin_near:.1e}: the rank has not dropped, but the weakest local")
    print("    gain is tiny, so this is weak motion authority, not a rank loss")
    print("    singular values =", np.linalg.svd(J_near, compute_uv=False))
    print(f"    0.1 / sigma_min = {0.1 / smin_near:.0f}: moving the flange at 0.1 units per second along the")
    print("    weakest task direction costs about 230 rad/s of joint speed, which no real arm will do.")
    print("    Rank counts independent task directions; the printed estimate uses a tolerance.")
    print("    sigma_min is the weakest local gain; it predicts the cost of motion along that weak direction.")
    check("Panda near rank at tol 1e-6", rank_near, 6, 0)
    check("Panda near sigma_min", smin_near, 4.3e-4, 5e-5)
    check("Panda near 0.1/sigma_min [rad/s]", 0.1 / smin_near, 230.0, 5.0)


# --- the two figures --------------------------------------------------------------------------------
MAROON, TEAL, GREY = "#9b1b4b", "#1f6f8b", "#777777"


def figure_kinematic_tree(path):
    """The arm-plus-camera tree, with the path from the gripper to the object highlighted.

    Two colours carry meaning and nothing else does: maroon for the part of the walk that goes UP to
    the root, where each edge is traversed backwards and therefore contributes an inverse, and teal
    for the part that goes DOWN the camera branch, where each edge contributes itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)

    # The arm branch is the top row, the camera branch hangs off the same root. Nodes are drawn as
    # text with a rounded box, so each box is exactly as wide as its own label.
    nodes = {"W": (0.00, 0.00), "L1": (1.70, 0.00), "L2": (3.40, 0.00), "E": (5.10, 0.00),
             "C": (0.00, -1.50), "O": (1.70, -1.50)}
    labels = {"W": "W\nworld (root)", "L1": "L1\nlink 1", "L2": "L2\nlink 2", "E": "E = G\ngripper",
              "C": "C\ncamera", "O": "O\nobject"}
    arm = ("W", "L1", "L2", "E")
    edges = [("W", "L1", MAROON, r"$R(q_1)\,T(l_1,0)$"),
             ("L1", "L2", MAROON, r"$R(q_2)\,T(l_2,0)$"),
             ("L2", "E", MAROON, r"$^{L2}X^{E}$, fixed"),
             ("W", "C", TEAL, r"$^{W}X^{C}$, fixed"),
             ("C", "O", TEAL, r"$^{C}X^{O}$, measured")]

    for a, b, colour, text in edges:
        (xa, ya), (xb, yb) = nodes[a], nodes[b]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.0, shrinkA=30, shrinkB=30))
        if ya != yb:                                   # the one vertical edge: label to its right
            ax.text(xa + 0.14, (ya + yb) / 2, text, color=colour, fontsize=9, ha="left", va="center")
        elif a in arm:                                 # arm edges: label below the top row
            ax.text((xa + xb) / 2, ya - 0.36, text, color=colour, fontsize=9, ha="center", va="top")
        else:                                          # the camera edge: label above the bottom row
            ax.text((xa + xb) / 2, ya + 0.36, text, color=colour, fontsize=9, ha="center", va="bottom")

    for name, (x, y) in nodes.items():
        colour = MAROON if name in arm else TEAL
        ax.text(x, y, labels[name], fontsize=8.5, ha="center", va="center", color="#222", zorder=4,
                bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor=colour, linewidth=1.6))

    # The highlighted walk: back up the arm branch to the root, then down the camera branch.
    ax.annotate("", xy=(0.10, 0.72), xytext=(5.00, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=MAROON, lw=1.4, ls=(0, (5, 3))))
    ax.text(2.55, 1.15, "up to the root: each arm edge is walked backwards,\nso each of its factors is inverted",
            color=MAROON, fontsize=9, ha="center", va="center")
    ax.text(0.85, -2.05, "down the other branch: each edge is walked forwards", color=TEAL, fontsize=9,
            ha="center", va="center")
    ax.text(4.35, -1.05, r"$^{G}X^{O} \;=\; (^{W}X^{G})^{-1}\;\;{}^{W}X^{C}\;\;{}^{C}X^{O}$",
            fontsize=12, ha="center", va="center", color="#222222")
    ax.text(4.35, -1.75, r"inner letters cancel:  $G\!-\!W$,  $W\!-\!C$,  $C\!-\!O$",
            fontsize=9.5, ha="center", va="center", color="#222222")
    ax.text(2.60, 1.95, "The robot is a tree: links are the nodes, joints are the edges, every edge is one transform,\n"
                        "and only the two arm edges depend on a joint value. A camera is another branch off the root.",
            fontsize=9, ha="center", va="center", color=GREY)
    ax.set_xlim(-0.55, 6.37)          # span 6.92 by 4.45, the figure's own 7.0 : 4.5 ratio, so
    ax.set_ylim(-2.30, 2.15)          # "aspect equal" wastes almost no canvas
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_nullspace_line(path):
    """The joint-rate plane for the one-dimensional task 'the tip's x-velocity is 0.1 m/s'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.deg2rad(Q_DEG)
    J_x = jacobian(q)[0, :]
    n_dir = np.array([-J_x[1], J_x[0]])
    n_dir = n_dir / np.linalg.norm(n_dir)
    if n_dir[0] < 0:
        n_dir = -n_dir
    q_min = np.linalg.pinv(J_x.reshape(1, 2)) @ np.array([0.1])

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)
    ax.axhline(0.0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0.0, color="#cccccc", lw=0.8, zorder=0)
    ax.grid(True, color="#f2f2f2", zorder=0)

    line = q_min[:, None] + n_dir[:, None] * np.array([-0.52, 0.44])
    ax.plot(line[0], line[1], color=MAROON, lw=2.2, zorder=2,
            label=r"$J_x \dot q = 0.1$ m/s: the solutions")
    nl = n_dir[:, None] * np.array([-0.22, 0.30])
    ax.plot(nl[0], nl[1], color=TEAL, lw=1.8, ls=(0, (6, 4)), zorder=2,
            label=f"null direction $\\hat v_2 = ({n_dir[0]:.4f}, {n_dir[1]:.4f})$")
    ax.text(0.19, -0.245, r"$J_x n = 0$: move along $n$ and" "\n" r"the tip's $x$ does not change",
            fontsize=8.5, color=TEAL, ha="left", va="center")

    # the perpendicular from the origin, which is what makes the minimum-norm point the minimum-norm one
    ax.plot([0.0, q_min[0]], [0.0, q_min[1]], color="#222222", lw=1.3, ls=(0, (2, 2)), zorder=3)
    ax.plot(0.0, 0.0, "o", color="#222222", ms=5, zorder=4)
    ax.text(0.03, 0.015, "origin: the arm stands still,\nso the task is not done",
            fontsize=8.5, color="#222222", ha="left", va="bottom")

    for t_off, dx, dy, va in ((+0.35, 0.048, 0.005, "center"), (-0.40, 0.035, 0.020, "bottom")):
        other = q_min + t_off * n_dir
        ax.plot([0.0, other[0]], [0.0, other[1]], color=GREY, lw=0.8, ls=(0, (1, 3)), zorder=1)
        ax.plot(*other, "o", color=GREY, ms=7, zorder=4)
        ax.text(other[0] + dx, other[1] + dy,
                f"same $\\dot x$, more joint speed\n$|\\dot q| = {np.linalg.norm(other):.4f}$ rad/s",
                fontsize=8.5, color=GREY, ha="left", va=va)

    ax.plot(*q_min, "o", color=MAROON, ms=9, mec="white", mew=1.4, zorder=5)
    ax.plot([-0.300, q_min[0] - 0.012], [-0.252, q_min[1] - 0.014], color=MAROON, lw=0.8,
            ls=(0, (1, 3)), zorder=1)
    ax.text(-0.60, -0.265,
            f"$J_x^{{+}}\\dot x = ({q_min[0]:.4f},\\ {q_min[1]:.4f})$\n"
            f"$|\\dot q| = {np.linalg.norm(q_min):.4f}$ rad/s, minimum norm\n"
            "(the perpendicular from the origin)",
            fontsize=8.5, color=MAROON, ha="left", va="top")

    ax.set_xlim(-0.62, 0.86)
    ax.set_ylim(-0.52, 0.40)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\dot q_1$  [rad/s]")
    ax.set_ylabel(r"$\dot q_2$  [rad/s]")
    ax.set_title("One task, a line of answers: the tip's $x$-velocity is 0.1 m/s at "
                 r"$q = (30°, 45°)$," "\n"
                 r"where $J_x = (-0.4898,\ -0.2898)$ m per rad", fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.96)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures():
    head("Figures")
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    tree, null = _FIG_DIR / "kinematic_tree.png", _FIG_DIR / "nullspace_line.png"
    figure_kinematic_tree(tree)
    figure_nullspace_line(null)
    for p in (tree, null):
        print(f"  wrote {p}")


# --- main --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-figures", action="store_true",
                    help="print and check the numbers, but do not redraw ../fig/*.png")
    args = ap.parse_args()

    print("Kinematic foundations for robot learning: every frozen number in notes.md, reprinted.")
    print(f"Two-link planar arm: l1 = {L1} m, l2 = {L2} m, worked configuration q = "
          f"({Q_DEG[0]:.0f} deg, {Q_DEG[1]:.0f} deg).")
    print("pydrake available: the 3-D Panda example runs below." if HAVE_DRAKE
          else f"The 3-D Panda example is skipped: {DRAKE_NOTE}.")

    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    section_8()
    section_panda()
    if not args.no_figures:
        make_figures()

    head("Frozen-number check")
    if FAILURES:
        print(f"{len(FAILURES)} number(s) did NOT reproduce:")
        for label, got, want, tol in FAILURES:
            print(f"  {label}: computed {got}, expected {want} (tolerance {tol})")
        return 1
    print("Every number the reading quotes was recomputed here and matched to the printed precision.")
    print("The Panda section " + ("ran." if HAVE_DRAKE else f"was skipped ({DRAKE_NOTE})."))
    print("all frozen numbers reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
