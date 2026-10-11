"""Default rotations-3d numbers and key nondefault teaching cases, from NumPy.

Run:  .venv/bin/python modules/rotations-3d/check_numbers.py
Compare with the module's readouts (or with the DOM dump the README describes). The module's
JavaScript implements the same formulas without three.js, so any disagreement is a bug in one
of the two, not a rendering artefact. Formulas: session 2.2 notes §3.3 (elementary rotations),
§3.4 (quaternion), §4.2 (columns, RᵀR, det), §4.5 (order), §4.6 (Rodrigues), §4.7 (gimbal lock).
"""
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):      # a Windows code page must not stop the printout
    sys.stdout.reconfigure(errors="replace")

np.set_printoptions(precision=4, suppress=True)
d = np.deg2rad


def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skew(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


def rodrigues(u, t):
    K = skew(u)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def quat_from_axis_angle(u, t):          # r = (w, x, y, z)
    return np.array([np.cos(t / 2), *(np.sin(t / 2) * np.asarray(u))])


def mat_from_quat(r):
    w, x, y, z = r
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle_from_quat(r):
    w, v = r[0], r[1:]
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else v), 2 * np.arctan2(n, w)


print("=== tab 1: R_z(30°), the tray ===")
R = Rz(d(30))
print(R)
print("RᵀR − I max |entry| =", np.abs(R.T @ R - np.eye(3)).max(), "  det R =", np.linalg.det(R))
p_B = np.array([0.30, 0.40, 0.06])
print("active : ^B p =", p_B, " ^W p = R ^B p =", R @ p_B)
print("passive: ^W p =", p_B, " ^F p = Rᵀ ^W p =", R.T @ p_B)
Rm = R.copy(); Rm[:, 0] *= -1
print("column x negated: det =", np.linalg.det(Rm), " RᵀR − I max =", np.abs(Rm.T @ Rm - np.eye(3)).max())

print("\n=== tab 2, order: a = R_x(90°), b = R_z(90°) ===")
RL = Rz(d(90)) @ Rx(d(90)); RR = Rx(d(90)) @ Rz(d(90))
print("left  (a then b) = R_z R_x:\n", RL, "\n  first column", RL[:, 0])
print("right (b then a) = R_x R_z:\n", RR, "\n  first column", RR[:, 0])
print("max |R_left − R_right| =", np.abs(RL - RR).max())

print("\n=== tab 2, order: current-stage matrices and commuting exceptions ===")
stages = [(np.eye(3), np.eye(3)), (Rx(d(90)), Rz(d(90))), (RL, RR)]
expected_columns = [([1, 0, 0], [1, 0, 0]), ([1, 0, 0], [0, 1, 0]), ([0, 1, 0], [0, 0, 1])]
for stage, ((left, right), (left_x, right_x)) in enumerate(zip(stages, expected_columns)):
    np.testing.assert_allclose(left[:, 0], left_x, atol=1e-12)
    np.testing.assert_allclose(right[:, 0], right_x, atol=1e-12)
    print(f"stage {stage}: first columns {left[:, 0]}, {right[:, 0]}; max difference {np.abs(left-right).max():.4f}")
np.testing.assert_allclose(Rx(d(30)) @ Rx(d(70)), Rx(d(70)) @ Rx(d(30)), atol=1e-12)
np.testing.assert_allclose(Rx(d(180)) @ Ry(d(180)), Ry(d(180)) @ Rx(d(180)), atol=1e-12)
assert np.abs(RL - RR).max() > 0.9
print("same-axis turns commute; perpendicular half turns commute; default quarter turns do not")

print("\n=== tab 2, gimbal: ψ = 10°, ϑ_p = 90°, ϕ = 0° ===")
def euler(psi, pit, phi): return Rz(d(psi)) @ Ry(d(pit)) @ Rx(d(phi))
E = euler(10, 90, 0)
print(E)
roll_axis = (Rz(d(10)) @ Ry(d(90)))[:, 0]
print("roll axis (body x after yaw, pitch) =", roll_axis, " angle between its line and the world z line =", np.degrees(np.arccos(abs(roll_axis[2]))), "°  (anti-parallel: locked)")
print("max |R(10,90,0) − R(20,90,10)| =", np.abs(E - euler(20, 90, 10)).max())
print("unlocked check, pitch 30°: max |R(10,30,0) − R(20,30,10)| =", np.abs(euler(10, 30, 0) - euler(20, 30, 10)).max())
for pitch, roll_delta, axis_z in [(90, 10, -1), (-90, -10, 1)]:
    axis = (Rz(d(10)) @ Ry(d(pitch)))[:, 0]
    np.testing.assert_allclose(axis, [0, 0, axis_z], atol=1e-12)
    unchanged = np.abs(euler(10, pitch, 0) - euler(20, pitch, roll_delta)).max()
    changed = np.abs(euler(10, pitch, 0) - euler(20, pitch, -roll_delta)).max()
    assert unchanged < 1e-12 and changed > 0.3
    print(f"pitch {pitch:+d}°: roll axis z = {axis_z:+d}; yaw +10°, roll {roll_delta:+d}° cancel ({unchanged:.2e}); other sign changes R ({changed:.5f})")
unlocked_axis = (Rz(d(10)) @ Ry(d(30)))[:, 0]
assert abs(unlocked_axis[2]) < 0.999

print("\n=== tab 3: 30° about z ===")
u, t = np.array([0.0, 0.0, 1.0]), d(30)
print("[û]_× =\n", skew(u))
Rr = rodrigues(u, t)
print("Rodrigues R =\n", Rr, "\n  max |R − R_z(30°)| =", np.abs(Rr - Rz(t)).max())
r = quat_from_axis_angle(u, t)
print("quaternion r = (w, x, y, z) =", r, " |r| =", np.linalg.norm(r))
print("max |mat(r) − R| =", np.abs(mat_from_quat(r) - Rr).max())
print('RAI:  X: "t(0.40 0.20 0.75) d(30 0 0 1)"   X: [0.40 0.20 0.75 %.4f %.4f %.4f %.4f]' % tuple(r))
rm = np.array([r[3], r[0], r[1], r[2]])           # read as (x, y, z, w): w' = z, (x', y', z') = (w, x, y)
um, tm = axis_angle_from_quat(rm)
print("misread as (x,y,z,w): w' = %.4f, axis' = %s, angle' = %.1f°" % (rm[0], um, np.degrees(tm)))
print("max |R_ghost − R| =", np.abs(mat_from_quat(rm) - Rr).max())
