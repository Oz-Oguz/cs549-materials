import numpy as np

np.set_printoptions(precision=6, suppress=True)

def position(q):
    a, b = q
    return np.array([0.4*np.cos(a) + 0.3*np.cos(a+b),
                     0.4*np.sin(a) + 0.3*np.sin(a+b)])

def jacobian(q):
    a, b = q
    return np.array([[-0.4*np.sin(a)-0.3*np.sin(a+b), -0.3*np.sin(a+b)],
                     [ 0.4*np.cos(a)+0.3*np.cos(a+b),  0.3*np.cos(a+b)]])

# 1. The inverse round-trip of section 3: frame B at (0.40, 0.20) m
#    turned by 30 deg, and the point P at (0.15, 0.05) m in B's axes.
th = np.deg2rad(30.0)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
p_AB = np.array([0.40, 0.20])
p_BP = np.array([0.15, 0.05])
p_AP = p_AB + R @ p_BP          # equation (3): B-coordinates to A
print("P in A [m]:", p_AP)
# equation (5): subtract the offset, then rotate back
print("P back in B [m]:", R.T @ (p_AP - p_AB))
row = np.array([[0.0, 0.0, 1.0]])
X = np.block([[R, p_AB[:, None]], [row]])
X_inv = np.block([[R.T, -(R.T @ p_AB)[:, None]], [row]])
print("X times its inverse is the identity:",
      np.allclose(X @ X_inv, np.eye(3)))
print("plain transpose of X applied to (P in A, 1):",
      X.T @ np.append(p_AP, 1.0))

# 2. The Jacobian at (30 deg, 45 deg), checked entry by entry
#    against central finite differences.
q = np.deg2rad([30.0, 45.0])
J = jacobian(q)
h = 1e-6
J_fd = np.column_stack([(position(q + h*e) - position(q - h*e)) / (2*h)
                        for e in np.eye(2)])
print("tip position [m]:", position(q))
print("position Jacobian [m/rad]:\n", J)
print("largest |analytic - finite difference| [m/rad]:",
      np.abs(J - J_fd).max())

# 3. Differential IK for an upward request, then one integrated
#    command: a rate times a time.
requested = np.array([0.0, 0.1])     # world frame, meters per second
rates = np.linalg.solve(J, requested)   # the square case (11)
print("joint rates [rad/s]:", rates,
      " achieved velocity [m/s]:", J @ rates)
dt = 0.01                               # seconds
step = dt * rates
actual = position(q + step) - position(q)
print("joint increment [rad]:", step, " = [deg]:", np.rad2deg(step))
print("actual tip increment [m]:", actual,
      " model error [m]:", np.linalg.norm(actual - dt*requested))

# 4. One joint-position update toward a static target: a displacement
#    in meters, no time step. Every error is target minus tip.
y_star = np.array([0.5, 0.2])
e = y_star - position(q)
dq = np.linalg.solve(J, e)              # the local equation J dq = e
q_new = q + dq
e_new = y_star - position(q_new)
print("remaining displacement e [m]:", e, " |e| [m]:", np.linalg.norm(e))
print("update dq [rad]:", dq, " = [deg]:", np.rad2deg(dq))
print("tip after the update [m]:", position(q_new))
print("e_new [m]:", e_new, " |e_new| [m]:", np.linalg.norm(e_new))

# 5. At full extension the pseudo-inverse keeps a reachable request
#    and drops an unreachable one.
J_straight = jacobian(np.zeros(2))
for command in ([0.0, 0.1], [0.1, 0.0]):
    rates_straight = np.linalg.pinv(J_straight) @ command
    print("straight-arm request [m/s]:", command,
          " achieved [m/s]:", J_straight @ rates_straight)
