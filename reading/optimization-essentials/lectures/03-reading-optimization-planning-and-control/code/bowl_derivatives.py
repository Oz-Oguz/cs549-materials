"""Chapter 02, section 1.2: read a cost surface, its slopes and its curvature.

Run with Python 3 and NumPy from any working directory. The bowl's cost,
gradient and Hessian come from Deck 04's existing descent.py. No figures,
simulation, optimization iterations or files are produced.
"""

import importlib.util
from pathlib import Path

import numpy as np


def shared_bowl():
    """Load the lecture's functions without retaining its print settings."""
    source = (Path(__file__).resolve().parents[2]
              / "03-D1-optimization-1" / "code" / "descent.py")
    spec = importlib.util.spec_from_file_location("chapter02_shared_bowl", source)
    module = importlib.util.module_from_spec(spec)
    with np.printoptions():
        spec.loader.exec_module(module)
    return module


def central_gradient(cost, z, epsilon=1e-5):
    """Estimate each slope using only cost values on either side of z."""
    basis = np.eye(len(z))
    return np.array([(cost(z + epsilon * e) - cost(z - epsilon * e))
                     / (2 * epsilon) for e in basis])


def central_hessian(cost, z, epsilon=1e-3):
    """Estimate all second derivatives using only cost evaluations."""
    basis = np.eye(len(z))
    H = np.zeros((len(z), len(z)))
    for i, e in enumerate(basis):
        H[i, i] = (cost(z + epsilon * e) - 2 * cost(z)
                   + cost(z - epsilon * e)) / epsilon**2
        for j in range(i):
            a, b = epsilon * e, epsilon * basis[j]
            H[i, j] = H[j, i] = (
                cost(z + a + b) - cost(z + a - b)
                - cost(z - a + b) + cost(z - a - b)
            ) / (4 * epsilon**2)
    return H


def saddle_cost(z):
    """Return the separate counterexample cost used to explain stationarity."""
    return float(z[0]**2 - z[1]**2)


def saddle_hessian(z):
    """Differentiate the slopes 2*z[0] and -2*z[1] at any point z."""
    return np.diag([2.0, -2.0])


def example():
    """Compute the subsection's displayed values from the shared bowl."""
    bowl = shared_bowl()
    z = bowl.Z0.copy()
    origin = np.zeros(2)
    probes = []
    for epsilon in (0.1, 0.01):
        for coordinate, e in enumerate(np.eye(2), start=1):
            changed = z + epsilon * e
            difference = bowl.f(changed) - bowl.f(z)
            probes.append(dict(epsilon=epsilon, coordinate=coordinate,
                               point=changed, cost=bowl.f(changed),
                               cost_change=difference,
                               average_slope=difference / epsilon,
                               gradient=bowl.grad(changed)))
    contour_points = np.array([[np.sqrt(10.0), 0.0],
                               [-np.sqrt(10.0), 0.0],
                               [0.0, 1.0], [0.0, -1.0]])
    delta = np.array([0.1, 0.1])
    linear = float(bowl.grad(z) @ delta)
    quadratic = 0.5 * float(delta @ bowl.hess(z) @ delta)
    saddle_points = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    return dict(
        z=z, cost=bowl.f(z), origin_cost=bowl.f(origin),
        gradient=bowl.grad(z), hessian=bowl.hess(z),
        origin_gradient=bowl.grad(origin),
        slice_constants=np.array([bowl.f(np.array([0.0, z[1]])),
                                  bowl.f(np.array([z[0], 0.0]))]),
        contour_points=contour_points,
        contour_costs=np.array([bowl.f(p) for p in contour_points]),
        probes=probes, delta=delta, linear=linear, quadratic=quadratic,
        linear_prediction=bowl.f(z) + linear,
        quadratic_prediction=bowl.f(z) + linear + quadratic,
        checked_point=z + delta, checked_cost=bowl.f(z + delta),
        saddle_points=saddle_points,
        saddle_costs=np.array([saddle_cost(p) for p in saddle_points]),
        saddle_origin_gradient=central_gradient(saddle_cost, origin),
        saddle_hessian=saddle_hessian(origin),
    )


def check_derivatives():
    """Independently check analytic derivatives and the quadratic identity."""
    bowl = shared_bowl()
    gradient_error = hessian_error = 0.0
    for z in (bowl.Z0, np.zeros(2), np.array([-1.25, 0.75])):
        numerical_gradient = central_gradient(bowl.f, z)
        numerical_hessian = central_hessian(bowl.f, z)
        np.testing.assert_allclose(numerical_gradient, bowl.grad(z),
                                   rtol=0, atol=1e-8)
        np.testing.assert_allclose(numerical_hessian, bowl.hess(z),
                                   rtol=0, atol=1e-7)
        gradient_error = max(gradient_error,
                             float(np.max(abs(numerical_gradient - bowl.grad(z)))))
        hessian_error = max(hessian_error,
                            float(np.max(abs(numerical_hessian - bowl.hess(z)))))
        for delta in (np.array([0.1, 0.1]), np.array([-0.2, 0.3])):
            model = (bowl.f(z) + bowl.grad(z) @ delta
                     + 0.5 * delta @ bowl.hess(z) @ delta)
            np.testing.assert_allclose(model, bowl.f(z + delta),
                                       rtol=0, atol=1e-12)
    np.testing.assert_allclose(central_gradient(saddle_cost, np.zeros(2)),
                               np.zeros(2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(central_hessian(saddle_cost, np.zeros(2)),
                               saddle_hessian(np.zeros(2)), rtol=0, atol=1e-7)
    assert saddle_cost(np.array([0.1, 0.0])) > saddle_cost(np.zeros(2))
    assert saddle_cost(np.array([0.0, 0.1])) < saddle_cost(np.zeros(2))

    # The bowl has zero mixed derivatives, so it cannot check the mixed stencil's
    # scale. For this coupled cost the slopes are 2*z[0] + 3*z[1] and
    # 3*z[0] + 20*z[1]; differentiating gives the exact Hessian below.
    def coupled_cost(z):
        return float(z[0]**2 + 3*z[0]*z[1] + 10*z[1]**2)

    expected_hessian = np.array([[2.0, 3.0], [3.0, 20.0]])
    coupled_hessian_error = 0.0
    for z in (np.zeros(2), np.array([0.7, -1.3]), np.array([-1.25, 0.75])):
        numerical_hessian = central_hessian(coupled_cost, z)
        np.testing.assert_allclose(numerical_hessian, expected_hessian,
                                   rtol=0, atol=1e-7)
        coupled_hessian_error = max(
            coupled_hessian_error,
            float(np.max(abs(numerical_hessian - expected_hessian))))
    return dict(max_gradient_error=gradient_error,
                max_hessian_error=hessian_error,
                max_coupled_hessian_error=coupled_hessian_error)


def main():
    data = example()
    with np.printoptions(precision=10, suppress=True):
        for name in ("z", "cost", "origin_cost", "gradient", "hessian",
                     "origin_gradient", "slice_constants", "contour_points",
                     "contour_costs"):
            value = data[name]
            print(f"{name}: {value:.10g}" if np.isscalar(value)
                  else f"{name}: {value}")
        print("\nCoordinate probes (only the named coordinate changes):")
        for probe in data["probes"]:
            print(f"epsilon={probe['epsilon']:.2f}, coordinate={probe['coordinate']}, "
                  f"cost={probe['cost']:.4f}, change={probe['cost_change']:.4f}, "
                  f"average slope={probe['average_slope']:.4f}, "
                  f"gradient={probe['gradient']}")
        print("\nOne chosen increment, without an optimization algorithm:")
        for name in ("delta", "linear", "quadratic", "linear_prediction",
                     "quadratic_prediction", "checked_point", "checked_cost",
                     "saddle_points", "saddle_costs", "saddle_origin_gradient",
                     "saddle_hessian"):
            value = data[name]
            print(f"{name}: {value:.10g}" if np.isscalar(value)
                  else f"{name}: {value}")
        checks = check_derivatives()
        print("\nPASS: cost-only derivative checks, exact quadratic model, "
              "stationary saddle, and coupled mixed-partial check.")
        for name, value in checks.items():
            print(f"{name}: {value:.3e}")


if __name__ == "__main__":
    main()
