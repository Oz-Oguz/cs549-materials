"""Chapter 02, section 1.1: predict one joint update, then check it with FK.

Run with Python 3 and NumPy. Angles are radians and positions are meters.
There is no time step, dynamics, joint limit or collision model in this example.
"""

import numpy as np

L1, L2 = 0.4, 0.3


def phi(q):
    """Return the tip position in the fixed planar world frame."""
    a, b = q
    return np.array([L1*np.cos(a) + L2*np.cos(a+b),
                     L1*np.sin(a) + L2*np.sin(a+b)])


def jac(q):
    """Return the position Jacobian in meters per radian."""
    a, b = q
    return np.array([[-L1*np.sin(a)-L2*np.sin(a+b), -L2*np.sin(a+b)],
                     [L1*np.cos(a)+L2*np.cos(a+b), L2*np.cos(a+b)]])


def example():
    """Compute each intermediate quantity from unrounded inputs."""
    q_initial = np.deg2rad([30.0, 45.0])
    target = np.array([0.5, 0.2])
    p_initial = phi(q_initial)
    J = jac(q_initial)
    e = target - p_initial
    delta = np.linalg.solve(J, e)
    p_predicted = p_initial + J @ delta
    q_checked = q_initial + delta
    p_checked = phi(q_checked)
    r_initial = p_initial - target
    r_checked = p_checked - target
    return dict(q_initial=q_initial, target=target, p_initial=p_initial,
                J=J, e=e, delta=delta, p_predicted=p_predicted,
                q_checked=q_checked, p_checked=p_checked,
                r_initial=r_initial, r_checked=r_checked,
                distance_initial=np.linalg.norm(r_initial),
                distance_checked=np.linalg.norm(r_checked),
                cost_initial=0.5 * float(r_initial @ r_initial),
                cost_checked=0.5 * float(r_checked @ r_checked))


if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    result = example()
    for name, value in result.items():
        print(f"{name}: {value}")
    print("delta [degrees]:", np.rad2deg(result["delta"]))
    print("q_checked [degrees]:", np.rad2deg(result["q_checked"]))
