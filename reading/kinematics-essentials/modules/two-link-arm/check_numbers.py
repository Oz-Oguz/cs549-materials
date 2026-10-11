"""Every number the two-link-arm module shows in its default view, in its URL presets, and at
each of the four preset buttons, including the folded pose q2 = 180 degrees, which has no URL.

Run:  .venv/bin/python modules/two-link-arm/check_numbers.py

The forward kinematics and the Jacobian are imported from the lecture's own code, so this
script and `index.html` can disagree only if one of them is wrong:

    lectures/02-D2-transforms-fk-jacobian/code/two_link.py   fk(q), jacobian(q)

Part 1 (the original view, q = (30°, 45°)) is printed first because the slides quote it; parts
2 and 3 are the sections added below it. The joint rates at a singular configuration are the
least-norm solution q̇ = J⁺ v, computed here by numpy.linalg.lstsq and in the module by an
explicit 2 × 2 singular value decomposition, which is the comparison that matters.

Conventions shared with the module: |v| = 0.1 m/s; "radial" is the unit vector from the base to
the tip and "tangential" is that vector turned by +90°; every printed direction is a unit vector
whose sign is fixed by asking for a non-negative y component, so the same line is always spelled
the same way. Notation follows notation.md: q, φ(q), J = ∂φ/∂q, σ the singular values, J⁺ the
pseudo-inverse.
"""
import importlib.util
import itertools
import pathlib

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):      # a Windows code page must not stop the printout
    sys.stdout.reconfigure(errors="replace")

HERE = pathlib.Path(__file__).resolve().parent
LECTURE = HERE.parents[1] / "lectures" / "02-D2-transforms-fk-jacobian" / "code" / "two_link.py"
if not LECTURE.exists():                    # the student package keeps the helper at code/two_link.py
    LECTURE = HERE.parents[1] / "code" / "two_link.py"
spec = importlib.util.spec_from_file_location("two_link", LECTURE)
two_link = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_link)
fk, jacobian, L1, L2 = two_link.fk, two_link.jacobian, two_link.L1, two_link.L2

VMAG = 0.1                                  # the commanded tip speed [m/s], fixed in the module
np.set_printoptions(precision=4, suppress=True)


def upward(u):
    """Fix the sign of a direction by asking for a non-negative y component."""
    u = np.asarray(u, dtype=float)
    if u[1] < -1e-12 or (abs(u[1]) <= 1e-12 and u[0] < 0):
        return -u
    return u


def directions(q):
    """The radial (base to tip) and tangential unit vectors at the configuration q."""
    p = fk(q)
    r = p / np.linalg.norm(p)
    return r, np.array([-r[1], r[0]])


def wrap180(deg):
    """Fold an angle into (-180, 180], which is how the module prints every heading."""
    d = float(deg)
    while d > 180.0:
        d -= 360.0
    while d <= -180.0:
        d += 360.0
    return d


def f3(x):
    return f"{0.0 if abs(x) < 5e-4 else x:.3f}"


def f4(x):
    return f"{0.0 if abs(x) < 5e-5 else x:.4f}"


def velocity_block(q2_deg, mode, q1_deg=30.0):
    """One state of part 2: the joint rates for a 0.1 m/s command in the chosen direction."""
    q = np.deg2rad([q1_deg, q2_deg])
    J, p = jacobian(q), fk(q)
    r, t = directions(q)
    d = r if mode == "radial" else t
    ang = np.degrees(np.arctan2(d[1], d[0]))
    v = VMAG * d
    U, S, Vt = np.linalg.svd(J)
    singular = abs(np.linalg.det(J)) < 1e-9
    qd, *_ = np.linalg.lstsq(J, v, rcond=None)
    reach = J @ qd
    lost = v - reach
    print(f"  ?q1={q1_deg:.0f}&q2={q2_deg:.0f}&v={mode}")
    print(f"    direction {ang:.1f} deg   ẋ = ({f4(v[0])}, {f4(v[1])}) m/s")
    if not singular:
        print(f"    q̇ = J⁻¹ ẋ = ({f3(qd[0])}, {f3(qd[1])}) rad/s,  |q̇| = {f3(np.linalg.norm(qd))} rad/s"
              f"   (J q̇ = v exactly, residual {np.linalg.norm(lost):.1e})")
    else:
        rd, ld = upward(U[:, 0]), upward(lost / np.linalg.norm(lost)) if np.linalg.norm(lost) > 1e-9 else upward(U[:, 1])
        print(f"    det J = 0; least-norm q̇ = J⁺ ẋ = ({f3(qd[0])}, {f3(qd[1])}) rad/s,  |q̇| = {f3(np.linalg.norm(qd))} rad/s")
        print(f"    reachable part   ({f4(reach[0])}, {f4(reach[1])}) m/s = {f3(np.linalg.norm(reach))} of {f3(VMAG)} m/s,"
              f" along ({f3(rd[0])}, {f3(rd[1])})")
        print(f"    unreachable part ({f4(lost[0])}, {f4(lost[1])}) m/s = {f3(np.linalg.norm(lost))} of {f3(VMAG)} m/s,"
              f" along ({f3(ld[0])}, {f3(ld[1])})")
    uA, uB = upward(U[:, 0]), upward(U[:, 1])
    print(f"    cheapest {f3(VMAG / S[0])} rad/s along ±({f3(uA[0])}, {f3(uA[1])})   σ_max = {f3(S[0])}")
    if S[1] > 1e-9:
        print(f"    dearest  {f3(VMAG / S[1])} rad/s along ±({f3(uB[0])}, {f3(uB[1])})   σ_min = {f3(S[1])}"
              f"   ratio {f3(S[0] / S[1])}")
    else:
        print(f"    dearest  unreachable along ±({f3(uB[0])}, {f3(uB[1])})   σ_min = 0   ratio infinite")


def pose(th, p):
    return float(th), np.asarray(p, dtype=float)


def compose(A, B):
    """Compose two planar poses, which is the product of their 3 × 3 homogeneous transforms."""
    thA, pA = A
    thB, pB = B
    c, s = np.cos(thA), np.sin(thA)
    return pose(thA + thB, pA + np.array([c * pB[0] - s * pB[1], s * pB[0] + c * pB[1]]))


def chain(factors):
    X = pose(0.0, [0.0, 0.0])
    run = [X]
    for F in factors:
        X = compose(X, F)
        run.append(X)
    return run                                   # run[2] is the elbow, run[4] the tip


def order_block(q1_deg=30.0, q2_deg=45.0):
    """Part 3: the correct product and the two mis-ordered ones, at the same configuration."""
    q1, q2 = np.deg2rad([q1_deg, q2_deg])
    R1, T1, R2, T2 = pose(q1, [0, 0]), pose(0, [L1, 0]), pose(q2, [0, 0]), pose(0, [L2, 0])
    runs = {
        "correct  R(q1) T(l1,0) R(q2) T(l2,0)": chain([R1, T1, R2, T2]),
        "ghost A  T(l1,0) R(q1) T(l2,0) R(q2)": chain([T1, R1, T2, R2]),
        "ghost B  R(q1) T(l1,0) T(l2,0) R(q2)": chain([R1, T1, T2, R2]),
    }
    ref = runs["correct  R(q1) T(l1,0) R(q2) T(l2,0)"]
    for name, run in runs.items():
        de = np.linalg.norm(run[2][1] - ref[2][1])
        dt = np.linalg.norm(run[4][1] - ref[4][1])
        print(f"  {name}")
        print(f"    elbow ({f4(run[2][1][0])}, {f4(run[2][1][1])}) heading {wrap180(np.degrees(run[2][0])):.1f} deg"
              f"   {f4(de)} m from the correct elbow")
        print(f"    tip   ({f4(run[4][1][0])}, {f4(run[4][1][1])}) heading {wrap180(np.degrees(run[4][0])):.1f} deg"
              f"   {f4(dt)} m from the correct tip   |p| = {f4(np.linalg.norm(run[4][1]))} m")
    # The inverse pose the module prints beside the correct chain: ^E X^W = (^W X^E)^-1, so
    # ^E p^W = -R(q1+q2)^T ^W p^E, which is not the negative of ^W p^E.
    th, pw = ref[4]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pEW = -R.T @ pw
    print(f"    ^E p^W = ({f4(pEW[0])}, {f4(pEW[1])}) m,  not -(^W p^E) = ({f4(-pw[0])}, {f4(-pw[1])}) m")
    print(f"  l1 + l2 = {f4(L1 + L2)} m")
    headings = sorted({round(wrap180(np.degrees(chain(list(f))[4][0])), 6)
                       for f in itertools.permutations([R1, T1, R2, T2])})
    print(f"  headings over all 24 orderings of the four factors: {headings} deg")


if __name__ == "__main__":
    print(f"=== two-link-arm module, l1 = {L1} m, l2 = {L2} m, |ẋ| = {VMAG} m/s ===")

    print("\n--- part 1: the original view, unchanged (q = (30, 45) deg) ---")
    q = np.deg2rad([30.0, 45.0])
    J, p = jacobian(q), fk(q)
    S = np.linalg.svd(J, compute_uv=False)
    print(f"  ^W p^E = ({f4(p[0])}, {f4(p[1])}) m   heading 75 deg")
    print(f"  J = [[{f4(J[0,0])}, {f4(J[0,1])}], [{f4(J[1,0])}, {f4(J[1,1])}]]")
    print(f"  det J = {f4(np.linalg.det(J))}   singular values {f3(S[0])}, {f3(S[1])}")
    qd = np.linalg.solve(J, np.array([0.0, 0.1]))
    print(f"  v = (0, 0.1) m/s straight up: q̇ = ({f3(qd[0])}, {f3(qd[1])}) rad/s, |q̇| = {f3(np.linalg.norm(qd))} rad/s")

    print("\n--- part 2: a commanded tip velocity you can point ---")
    print("  default of the section: tangential at q = (30, 45) deg")
    velocity_block(45.0, "tangential")
    velocity_block(45.0, "radial")
    # 10, 1 and 0 degrees are the approach to the stretched singularity; 180 degrees is the
    # "Folded" preset button, the module's second singular pose, which has no URL of its own
    # but is one click away and prints a full readout.
    for q2d in (10.0, 1.0, 0.0, 180.0):
        velocity_block(q2d, "tangential")
        velocity_block(q2d, "radial")
    print("  the slider in free mode, ?v=90 (the fixed command of part 1's readout):")
    q = np.deg2rad([30.0, 45.0])
    qd = np.linalg.solve(jacobian(q), np.array([0.0, 0.1]))
    print(f"    q̇ = ({f3(qd[0])}, {f3(qd[1])}) rad/s, |q̇| = {f3(np.linalg.norm(qd))} rad/s"
          "   (the same number part 1 prints, from the same J)")

    print("\n--- part 3: multiply on the wrong side (?order=wrong, the default) ---")
    # The wrong-order drawing follows the joint sliders, so it is checked at every configuration
    # the URL presets above can put the page in, not only at the worked example.
    for q2d in (45.0, 10.0, 1.0, 0.0, 180.0):
        print(f"  q = (30, {q2d:.0f}) deg")
        order_block(30.0, q2d)
