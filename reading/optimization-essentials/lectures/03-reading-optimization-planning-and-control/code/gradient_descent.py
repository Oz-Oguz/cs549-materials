"""Chapter 02, section 1.3: direction, fixed step size and conditioning.

Run with Python 3 and NumPy from any working directory. The cost, gradient,
Hessian and numerical iterations come from Deck 04's descent.py. Independent
coordinate powers check the paths; exact fractions check the 45-step count.
This script prints results and writes no files or figures.
"""

from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np


STEP_SIZES = (0.02, 0.05, 0.08, 0.10, 0.11)
FIGURE_STEPS = 12


def shared_bowl():
    """Load the deck's functions without retaining print settings or writing pyc."""
    source = (Path(__file__).resolve().parents[2]
              / "03-D1-optimization-1" / "code" / "descent.py")
    spec = importlib.util.spec_from_file_location("chapter02_shared_descent", source)
    module = importlib.util.module_from_spec(spec)
    with np.printoptions():
        # Compile the source directly so loading it cannot create a bytecode file.
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def diagonal_balance(hessian):
    """Balance the two coordinate factors of a positive diagonal Hessian."""
    hessian = np.asarray(hessian, dtype=float)
    if (hessian.shape != (2, 2) or not np.all(np.isfinite(hessian))
            or not np.array_equal(hessian, np.diag(np.diag(hessian)))
            or np.any(np.diag(hessian) <= 0)):
        raise ValueError("This calculation requires a positive diagonal 2-by-2 Hessian.")
    entries = tuple(Fraction(str(value)) for value in np.diag(hessian))
    alpha = 2 / sum(entries)
    factors = tuple(1 - alpha * entry for entry in entries)
    worst = max(abs(factor) for factor in factors)
    return dict(alpha_exact=alpha, factors_exact=factors, factor_exact=worst,
                cost_factor_exact=worst**2,
                condition_number=float(max(entries) / min(entries)))


def example():
    """Compute the chapter's values using the deck's functions and iterates."""
    bowl = shared_bowl()
    z = bowl.Z0.copy()
    gradient = bowl.grad(z)
    direction = -gradient
    hessian = bowl.hess(z)
    curvatures = np.diag(hessian)  # Preserve coordinate order.
    balanced = diagonal_balance(hessian)

    first_steps = []
    for alpha in (0.05, 0.5):
        point = bowl.gd_fixed(z, alpha, 1)[-1]
        first_steps.append(dict(alpha=alpha, factors=1 - alpha * curvatures,
                                point=point, cost=bowl.f(point)))

    fixed_runs = []
    for alpha in STEP_SIZES:
        path = bowl.gd_fixed(z, alpha, FIGURE_STEPS)
        fixed_runs.append(dict(alpha=alpha, factors=1 - alpha * curvatures,
                               path=path, costs=np.array([bowl.f(p) for p in path])))

    best_path = bowl.gd_fixed(z, float(balanced["alpha_exact"]), 45)
    scale = 0.01
    increment = scale * direction
    return dict(
        start=z, cost=bowl.f(z), gradient=gradient, hessian=hessian,
        direction=direction, slope=float(gradient @ direction),
        scale=scale, increment=increment, scaled_point=z + increment,
        scaled_cost=bowl.f(z + increment),
        direction_length=float(np.linalg.norm(direction)),
        increment_length=float(np.linalg.norm(increment)),
        first_steps=first_steps, fixed_runs=fixed_runs,
        curvature_range=np.array([min(curvatures), max(curvatures)]),
        condition_number=balanced["condition_number"],
        conditioning_example=diagonal_balance(np.diag([2.0, 200.0])),
        best=dict(**balanced, path=best_path,
                  first_cost=bowl.f(best_path[1]),
                  cost44=bowl.f(best_path[44]), cost45=bowl.f(best_path[45])),
    )


def closed_form_path(z0, alpha, iters):
    """Independently apply the exact powers of 1-2*alpha and 1-20*alpha.

    This uses the known polynomial coefficients, not a gradient or numerical
    update. A zero initial coordinate stays zero even if its factor's magnitude
    exceeds one.
    """
    alpha = Fraction(str(alpha)) if not isinstance(alpha, Fraction) else alpha
    start = [Fraction(str(value)) for value in z0]
    factors = (1 - 2 * alpha, 1 - 20 * alpha)
    return np.array([[float(value * factor**i)
                      for value, factor in zip(start, factors)]
                     for i in range(iters + 1)])


def check_updates():
    """Check repeated updates, their qualitative behavior and the exact count."""
    before = np.get_printoptions()
    bowl = shared_bowl()
    data = example()
    assert np.get_printoptions() == before
    np.testing.assert_array_equal(data["start"], [3.0, 2.0])
    np.testing.assert_array_equal(data["gradient"], [6.0, 40.0])
    np.testing.assert_array_equal(data["hessian"], np.diag([2.0, 20.0]))
    np.testing.assert_allclose(data["increment"], [-0.06, -0.4], rtol=0, atol=1e-15)
    assert data["slope"] == -1636.0
    np.testing.assert_allclose(data["scaled_cost"], 34.2436, rtol=0, atol=1e-12)

    max_path_error = 0.0
    starts = (bowl.Z0, np.zeros(2), np.array([0.0, 2.0]),
              np.array([3.0, 0.0]), np.array([-1.25, 0.75]))
    for start in starts:
        for alpha in (*STEP_SIZES, Fraction(1, 11)):
            numerical = bowl.gd_fixed(start, float(alpha), FIGURE_STEPS)
            exact = closed_form_path(start, alpha, FIGURE_STEPS)
            np.testing.assert_allclose(numerical, exact, rtol=1e-12, atol=1e-13)
            max_path_error = max(max_path_error,
                                 float(np.max(np.abs(numerical - exact))))
            # The independent cost calculation uses the polynomial itself.
            expected_costs = exact[:, 0]**2 + 10 * exact[:, 1]**2
            np.testing.assert_allclose([bowl.f(p) for p in numerical],
                                       expected_costs, rtol=1e-12, atol=1e-13)

    for first, expected_point, expected_cost in zip(
            data["first_steps"], ([2.7, 0.0], [0.0, -18.0]), (7.29, 3240.0)):
        np.testing.assert_allclose(first["point"], expected_point, rtol=0, atol=1e-14)
        np.testing.assert_allclose(first["cost"], expected_cost, rtol=0, atol=1e-12)

    runs = {run["alpha"]: run["path"] for run in data["fixed_runs"]}
    assert np.all(np.diff(np.abs(runs[0.02]), axis=0) < 0)
    assert np.all(runs[0.05][1:, 1] == 0)
    assert np.all(np.diff(np.abs(runs[0.05][:, 0])) < 0)
    assert np.all(np.diff(np.abs(runs[0.08]), axis=0) < 0)
    for alpha in (0.08, 0.10, 0.11):
        assert np.all(runs[alpha][:-1, 1] * runs[alpha][1:, 1] < 0)
    np.testing.assert_array_equal(np.abs(runs[0.10][:, 1]), 2.0)
    assert np.all(np.diff(np.abs(runs[0.11][:, 1])) > 0)

    # For unit directions, the quadratic form is 2 + 18*d_2**2.
    # Its minimum and maximum occur at the two coordinate axes.
    for unit in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                 np.array([1.0, 1.0]) / np.sqrt(2)):
        curvature = float(unit @ bowl.hess(bowl.Z0) @ unit)
        np.testing.assert_allclose(curvature, 2 + 18 * unit[1]**2,
                                   rtol=0, atol=1e-13)
        assert 2 <= curvature <= 20
    assert data["condition_number"] == 10

    best = data["best"]
    assert best["alpha_exact"] == Fraction(1, 11)
    assert best["factors_exact"] == (Fraction(9, 11), Fraction(-9, 11))
    assert best["factor_exact"] == Fraction(9, 11)
    assert best["cost_factor_exact"] == Fraction(81, 121)
    # Reversing coordinate order reverses the factors, unlike sorted eigenvalues.
    reversed_balance = diagonal_balance(np.diag([20.0, 2.0]))
    assert reversed_balance["factors_exact"] == (Fraction(-9, 11), Fraction(9, 11))
    more_unequal = data["conditioning_example"]
    assert more_unequal["condition_number"] == 100
    assert more_unequal["alpha_exact"] == Fraction(1, 101)
    assert more_unequal["factors_exact"] == (Fraction(99, 101), Fraction(-99, 101))
    for hessian in (np.diag([2.0, 20.0]), np.diag([20.0, 2.0]),
                    np.diag([2.0, 200.0])):
        balance = diagonal_balance(hessian)
        updated = bowl.Z0 - float(balance["alpha_exact"]) * (hessian @ bowl.Z0)
        factors = np.array([float(value) for value in balance["factors_exact"]])
        np.testing.assert_allclose(updated, factors * bowl.Z0, rtol=0, atol=1e-14)
    threshold = Fraction(1, 1_000_000)  # A worked cost target, not a solver stop.
    exact_costs = [Fraction(49) * Fraction(81, 121)**i for i in range(46)]
    first_below = next(i for i, cost in enumerate(exact_costs) if cost < threshold)
    assert first_below == 45
    assert exact_costs[44] >= threshold > exact_costs[45]
    np.testing.assert_allclose([bowl.f(p) for p in best["path"]],
                               [float(cost) for cost in exact_costs],
                               rtol=1e-12, atol=1e-15)
    assert data["first_steps"][0]["cost"] < best["first_cost"]
    assert np.get_printoptions() == before
    return dict(max_path_error=max_path_error,
                checked_paths=len(starts) * (len(STEP_SIZES) + 1),
                first_below=first_below, exact_cost44=exact_costs[44],
                exact_cost45=exact_costs[45])


def vector_text(vector):
    """Format a short vector without changing NumPy's global print options."""
    return "[" + ", ".join(f"{value:.12g}" for value in vector) + "]"


def main():
    before = np.get_printoptions()
    data = example()
    print("Chapter 02 section 1.3: shared Deck 04 bowl and gradient descent")
    print(f"start = {vector_text(data['start'])}; cost = {data['cost']:.12g}")
    print(f"gradient = {vector_text(data['gradient'])}")
    print(f"direction = -gradient = {vector_text(data['direction'])}")
    print(f"gradient dot direction = {data['slope']:.12g}")
    print(f"scale = {data['scale']:.12g}; increment = {vector_text(data['increment'])}")
    print(f"direction length = {data['direction_length']:.12g}; "
          f"increment length = {data['increment_length']:.12g}")
    print(f"scaled next point = {vector_text(data['scaled_point'])}; "
          f"cost = {data['scaled_cost']:.12g}")
    print("\nFirst-step comparison from the same start:")
    for first in data["first_steps"]:
        print(f"alpha = {first['alpha']:.2f}; factors = {vector_text(first['factors'])}; "
              f"next = {vector_text(first['point'])}; cost = {first['cost']:.12g}")
    print("\nFixed-step paths (iteration zero is the start; the figure uses 12 updates):")
    for run in data["fixed_runs"]:
        print(f"alpha = {run['alpha']:.2f}; factors = {vector_text(run['factors'])}")
        for i, (point, cost) in enumerate(zip(run["path"], run["costs"])):
            print(f"  i = {i:2d}: z = {vector_text(point)}; cost = {cost:.12g}")
    print("\nUnit-direction curvature = 2 + 18*d_2^2, with 0 <= d_2^2 <= 1")
    print(f"curvature range = {vector_text(data['curvature_range'])}; "
          f"condition number = {data['condition_number']:.12g}")
    best = data["best"]
    print("balanced alpha (minimizes the larger coordinate-factor magnitude) = "
          f"{best['alpha_exact']} = {float(best['alpha_exact']):.16g}")
    print("coordinate factors = " + ", ".join(str(value) for value in best["factors_exact"]))
    print(f"worst factor magnitude = {best['factor_exact']}; "
          f"cost factor = {best['cost_factor_exact']}")
    print(f"balanced-alpha first point = {vector_text(best['path'][1])}; "
          f"cost = {best['first_cost']:.16g}")
    print(f"balanced-alpha cost after 44 updates = {best['cost44']:.16g}")
    print(f"balanced-alpha cost after 45 updates = {best['cost45']:.16g}")
    more_unequal = data["conditioning_example"]
    print("\nDiagonal example f(z) = z_1^2 + 100*z_2^2; Hessian entries = 2, 200")
    print(f"condition number = {more_unequal['condition_number']:.12g}; "
          f"balanced alpha = {more_unequal['alpha_exact']}; factors = "
          + ", ".join(str(value) for value in more_unequal["factors_exact"]))
    checks = check_updates()
    print("\nPASS: shared iterations match independent coordinate powers on "
          f"{checks['checked_paths']} paths; shrinking, oscillation and growth agree.")
    print(f"maximum path error = {checks['max_path_error']:.3e}")
    print("PASS: exact Fraction arithmetic puts the first cost below 1e-6 at "
          f"{checks['first_below']} updates.")
    print(f"exact-arithmetic cost44 as decimal = {float(checks['exact_cost44']):.16g}")
    print(f"exact-arithmetic cost45 as decimal = {float(checks['exact_cost45']):.16g}")
    assert np.get_printoptions() == before
    print("PASS: unit-curvature bounds, coordinate-order and condition-100 checks, "
          "and NumPy print-option preservation.")


if __name__ == "__main__":
    main()
