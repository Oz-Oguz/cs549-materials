"""Chapter 02, section 1.4: trial steps and Armijo backtracking.

Run with Python 3 and NumPy from any working directory. The worked bowl reuses
Deck 04's cost and gradient. Fraction arithmetic independently checks the trial
tables, six accepted updates and the floor counterexample. This script prints
results and writes no files or figures.
"""

from fractions import Fraction
import importlib.util
from pathlib import Path

import numpy as np


def shared_bowl():
    """Load the deck's functions without retaining print settings or writing pyc."""
    source = (Path(__file__).resolve().parents[2]
              / "03-D1-optimization-1" / "code" / "descent.py")
    spec = importlib.util.spec_from_file_location("chapter02_shared_descent", source)
    module = importlib.util.module_from_spec(spec)
    with np.printoptions():
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def backtrack(cost, gradient, z, direction=None, *, alpha0=1.0, c=0.01,
              shrink=0.5, floor=1e-6, max_trials=60):
    """Return an accepted step or a named failure, with every tested trial.

    A rejected trial never moves the iterate. The first tested scale at or below
    floor is still tested; failure there returns stalled/floor. A finite trial
    must satisfy Armijo and change the floating-point point. Domain errors and
    nonfinite trial values are rejected and retried at a smaller scale.

    Status is accepted, stationary (exactly zero gradient), not_descent,
    nonfinite (invalid current data), stalled, or limit. Invalid parameter values
    raise ValueError. No gradient tolerance or convergence claim is used here.
    """
    z = np.asarray(z, dtype=float).copy()
    if z.ndim != 1 or z.size == 0:
        raise ValueError("z must be a nonempty vector.")
    if (not np.isfinite(alpha0) or alpha0 <= 0
            or not 0 < c < 1 or not 0 < shrink < 1
            or not np.isfinite(floor) or floor < 0
            or not isinstance(max_trials, int) or max_trials < 1):
        raise ValueError("Require positive alpha0, 0<c,shrink<1, floor>=0 and a trial limit.")
    result = dict(point=z.copy(), alpha=None, increment=None, f0=None,
                  gradient=None, direction=None, slope=None, trials=[])

    def finish(status, reason):
        result.update(status=status, reason=reason)
        return result

    if not np.all(np.isfinite(z)):
        return finish("nonfinite", "current point")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            f0 = float(cost(z))
            g = np.asarray(gradient(z), dtype=float)
    except (ValueError, ArithmeticError):
        return finish("nonfinite", "current evaluation failed")
    if g.shape != z.shape:
        raise ValueError("gradient must have the same shape as z.")
    result.update(f0=f0, gradient=g.copy())
    if not np.isfinite(f0) or not np.all(np.isfinite(g)):
        return finish("nonfinite", "current cost or gradient")
    if np.all(g == 0):
        return finish("stationary", "exactly zero gradient")
    d = -g if direction is None else np.asarray(direction, dtype=float)
    if d.shape != z.shape:
        raise ValueError("direction must have the same shape as z.")
    result["direction"] = d.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        slope = float(g @ d)
    result["slope"] = slope
    if not np.all(np.isfinite(d)) or not np.isfinite(slope):
        return finish("nonfinite", "direction or directional slope")
    if slope >= 0:
        return finish("not_descent", "directional slope is not negative")

    alpha = float(alpha0)
    for _ in range(max_trials):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            increment = alpha * d
            trial_point = z + increment
            threshold = f0 + c * alpha * slope
            trial_cost = np.nan
            if np.all(np.isfinite(trial_point)):
                try:
                    trial_cost = float(cost(trial_point))
                except (ValueError, ArithmeticError):
                    pass
        changed = not np.array_equal(trial_point, z)
        finite = bool(np.isfinite(trial_cost) and np.isfinite(threshold))
        accepted = bool(finite and changed and trial_cost <= threshold)
        result["trials"].append(dict(alpha=alpha, point=trial_point.copy(),
                                     cost=trial_cost, threshold=threshold,
                                     finite=finite, accepted=accepted))
        if accepted:
            result.update(point=trial_point, alpha=alpha, increment=increment)
            return finish("accepted", "Armijo condition")
        if not changed:
            return finish("stalled", "point unchanged by floating-point rounding")
        if alpha <= floor:
            return finish("stalled", "floor")
        next_alpha = shrink * alpha
        if next_alpha == 0 or next_alpha == alpha:
            return finish("stalled", "scale cannot shrink")
        alpha = next_alpha
    return finish("limit", "trial limit")


def run_armijo(z0, updates=6, *, alpha0=1.0, grow=1.2, c=0.01,
               shrink=0.5, floor=1e-6, max_trials=60):
    """Run the shared bowl, growing the next initial scale after acceptance."""
    if not isinstance(updates, int) or updates < 1 or not np.isfinite(grow) or grow <= 0:
        raise ValueError("Require a positive update count and positive finite growth.")
    bowl = shared_bowl()
    z = np.asarray(z0, dtype=float).copy()
    path, searches, alphas = [z.copy()], [], []
    alpha = alpha0
    status, reason = "limit", "requested update count reached"
    for _ in range(updates):
        search = backtrack(bowl.f, bowl.grad, z, alpha0=alpha, c=c,
                           shrink=shrink, floor=floor, max_trials=max_trials)
        searches.append(search)
        if search["status"] != "accepted":
            status, reason = search["status"], search["reason"]
            break
        z = search["point"].copy()
        path.append(z)
        alphas.append(search["alpha"])
        alpha = grow * search["alpha"]
        if not np.isfinite(alpha):
            status, reason = "nonfinite", "grown initial scale"
            break
    return dict(path=np.array(path), alphas=np.array(alphas), searches=searches,
                status=status, reason=reason)


def fraction_run(start=(3, 2), c=Fraction(1, 100), updates=6):
    """Independent exact reference from the expanded one-variable polynomial.

    At (x,y), negative-gradient trials have cost
    x^2+10*y^2 - alpha*(4*x^2+400*y^2)
                  + alpha^2*(4*x^2+4000*y^2).
    This calculation calls neither the shared functions nor backtrack().
    """
    x, y = (Fraction(str(value)) for value in start)
    alpha = Fraction(1)
    path, searches = [(x, y)], []
    for _ in range(updates):
        f0 = x*x + 10*y*y
        slope = -4*x*x - 400*y*y
        quadratic = 4*x*x + 4000*y*y
        trials = []
        while True:
            value = f0 + alpha*slope + alpha*alpha*quadratic
            threshold = f0 + c*alpha*slope
            accepted = value <= threshold
            trials.append(dict(alpha=alpha, cost=value, threshold=threshold,
                               accepted=accepted))
            if accepted:
                break
            alpha /= 2
        x, y = (1 - 2*alpha)*x, (1 - 20*alpha)*y
        path.append((x, y))
        searches.append(trials)
        alpha *= Fraction(6, 5)
    return dict(path=path, searches=searches)


def floor_example():
    """A descent direction whose first scale below the floor still fails."""
    bowl = shared_bowl()
    direction = -(2**17) * bowl.grad(bowl.Z0)
    stopped = backtrack(bowl.f, bowl.grad, bowl.Z0, direction, floor=1e-6)
    continued = backtrack(bowl.f, bowl.grad, bowl.Z0, direction, floor=0)
    return dict(direction=direction, stopped=stopped, continued=continued)


def example():
    """Keep the deck's numerical start and the app's different start separate."""
    return dict(deck=run_armijo([3.0, 2.0]),
                app=run_armijo([-2.5, 0.8], c=1e-4), floor=floor_example())


def check_examples():
    """Check exact references, the shared implementation and failure outcomes."""
    before = np.get_printoptions()
    errors = np.geterr()
    bowl = shared_bowl()
    data = example()
    max_error = 0.0
    for name, start, c in (("deck", (3, 2), Fraction(1, 100)),
                           ("app", (-2.5, 0.8), Fraction(1, 10000))):
        run = data[name]
        exact = fraction_run(start, c)
        expected = np.array([[float(v) for v in p] for p in exact["path"]])
        np.testing.assert_allclose(run["path"], expected, rtol=0, atol=2e-15)
        assert run["status"] == "limit" and len(run["alphas"]) == 6
        assert [len(s["trials"]) for s in run["searches"]] == [5, 1, 1, 1, 1, 2]
        for search, trials in zip(run["searches"], exact["searches"]):
            assert search["status"] == "accepted"
            assert len(search["trials"]) == len(trials)
            for actual, reference in zip(search["trials"], trials):
                assert actual["accepted"] == reference["accepted"]
                for key in ("alpha", "cost", "threshold"):
                    target = float(reference[key])
                    np.testing.assert_allclose(actual[key], target, rtol=2e-14, atol=2e-14)
                    max_error = max(max_error, abs(actual[key] - target))
    path, alphas = bowl.gd_armijo(bowl.Z0, 6, verbose_first=False)
    np.testing.assert_allclose(data["deck"]["path"], path, rtol=0, atol=0)
    np.testing.assert_allclose(data["deck"]["alphas"], alphas, rtol=0, atol=0)
    assert bowl.armijo_backtracking(bowl.Z0, -bowl.grad(bowl.Z0), verbose=False) == 1/16

    floor = data["floor"]
    assert floor["stopped"]["status"] == "stalled"
    assert floor["stopped"]["reason"] == "floor"
    assert floor["stopped"]["alpha"] is None
    assert floor["stopped"]["slope"] < 0
    np.testing.assert_array_equal(floor["stopped"]["point"], bowl.Z0)
    assert len(floor["stopped"]["trials"]) == 21
    assert len(floor["continued"]["trials"]) == 22
    # Effective ordinary-gradient scales are exactly 1/8 and 1/16.
    for search, alpha in ((floor["stopped"], Fraction(1, 2**20)),
                           (floor["continued"], Fraction(1, 2**21))):
        effective = alpha * 2**17
        value = 49 - 1636*effective + 16036*effective**2
        threshold = 49 - Fraction(1636, 100)*effective
        trial = search["trials"][-1]
        assert trial["alpha"] == float(alpha)
        assert trial["cost"] == float(value)
        assert trial["threshold"] == float(threshold)
        assert trial["accepted"] == (value <= threshold)
    assert floor["continued"]["status"] == "accepted"

    stationary = backtrack(bowl.f, bowl.grad, [0.0, 0.0])
    assert stationary["status"] == "stationary" and not stationary["trials"]
    for direction in (bowl.grad(bowl.Z0), np.zeros(2)):
        result = backtrack(bowl.f, bowl.grad, bowl.Z0, direction)
        assert result["status"] == "not_descent" and not result["trials"]
    assert backtrack(bowl.f, bowl.grad, [np.inf, 2])["status"] == "nonfinite"
    assert backtrack(lambda z: np.nan, bowl.grad, bowl.Z0)["status"] == "nonfinite"
    assert backtrack(bowl.f, lambda z: np.array([np.nan, 0]), bowl.Z0)["status"] == "nonfinite"
    assert backtrack(bowl.f, bowl.grad, bowl.Z0, [np.inf, 0])["status"] == "nonfinite"
    assert backtrack(bowl.f, bowl.grad, bowl.Z0, max_trials=1)["status"] == "limit"
    rounding = backtrack(bowl.f, bowl.grad, bowl.Z0, alpha0=1e-300, floor=0)
    assert (rounding["status"], rounding["reason"]) == (
        "stalled", "point unchanged by floating-point rounding")
    # A smooth quadratic and a very large direction make a changed trial even
    # at the smallest positive scale. Its rejection exposes scale underflow.
    underflow = backtrack(lambda z: 1e18*z[0]**2 + z[0],
                          lambda z: np.array([2e18*z[0] + 1]), [0.0],
                          [-1e308], alpha0=np.nextafter(0.0, 1.0),
                          floor=0, max_trials=1)
    assert (underflow["status"], underflow["reason"]) == (
        "stalled", "scale cannot shrink")
    assert underflow["trials"][0]["point"][0] != 0
    assert not underflow["trials"][0]["accepted"]
    np.testing.assert_array_equal(underflow["point"], [0.0])

    def domain_cost(z):
        if np.linalg.norm(z - bowl.Z0) > 1:
            raise ValueError("Trial outside this test function's domain.")
        return bowl.f(z)

    domain = backtrack(domain_cost, bowl.grad, bowl.Z0)
    assert domain["status"] == "accepted"
    assert not domain["trials"][0]["finite"] and domain["alpha"] == 1/64
    invalid_trials = backtrack(lambda z: bowl.f(z) if np.array_equal(z, bowl.Z0) else np.inf,
                               bowl.grad, bowl.Z0, max_trials=3)
    assert invalid_trials["status"] == "limit"
    assert all(not trial["finite"] for trial in invalid_trials["trials"])
    np.testing.assert_array_equal(invalid_trials["point"], bowl.Z0)
    np.testing.assert_array_equal(bowl.Z0, [3.0, 2.0])
    assert np.get_printoptions() == before and np.geterr() == errors
    return dict(max_trial_error=max_error, exact_traces=2,
                stationary=stationary["status"], domain=domain["status"])


def vector_text(vector):
    return "[" + ", ".join(f"{value:.12g}" for value in vector) + "]"


def main():
    before = np.get_printoptions()
    errors = np.geterr()
    data = example()
    print("Chapter 02 section 1.4: Armijo trials on the shared Deck 04 bowl")
    for name, label in (("deck", "Deck: start (3, 2), c=0.01"),
                         ("app", "App settings: start (-2.5, 0.8), c=0.0001")):
        run = data[name]
        first = run["searches"][0]
        print(f"\n{label}; alpha0=1, shrink=0.5, next initial scale=1.2*accepted scale")
        print(f"initial cost = {first['f0']:.12g}; direction = "
              f"{vector_text(first['direction'])}; slope = {first['slope']:.12g}")
        print("First search: alpha | trial cost | Armijo threshold | decision")
        for trial in first["trials"]:
            print(f"  {trial['alpha']:.12g} | {trial['cost']:.12g} | "
                  f"{trial['threshold']:.12g} | "
                  f"{'accept' if trial['accepted'] else 'reject'}")
        print("Accepted updates: index | alpha | point | cost | trial count")
        for i, search in enumerate(run["searches"], 1):
            print(f"  {i} | {search['alpha']:.12g} | {vector_text(search['point'])} | "
                  f"{search['trials'][-1]['cost']:.15g} | {len(search['trials'])}")
        print(f"Run outcome: {run['status']} ({run['reason']}); "
              f"{sum(len(s['trials']) for s in run['searches'])} trial evaluations")
    print("\nThe app values reproduce its settings numerically; they are not a browser check.")
    print("\nFloor example: d = -2^17*gradient at (3, 2), floor=1e-6")
    for key in ("stopped", "continued"):
        search = data["floor"][key]
        trial = search["trials"][-1]
        print(f"  {key}: {search['status']} ({search['reason']}); "
              f"{len(search['trials'])} trials; last alpha={trial['alpha']:.16g}; "
              f"cost={trial['cost']:.12g}; threshold={trial['threshold']:.12g}; "
              f"accepted={trial['accepted']}")
    checks = check_examples()
    print("\nPASS: two exact Fraction traces, shared deck iterates and scales, "
          "and exact floor rejection followed by acceptance.")
    print(f"maximum floating-point trial error = {checks['max_trial_error']:.3e}")
    print("PASS: stationary, non-descent, nonfinite current data, rejected nonfinite "
          "trials, domain failure and recovery, trial limit, and rounding stall.")
    assert np.get_printoptions() == before and np.geterr() == errors
    print("PASS: NumPy print options and error settings preserved from entry to main.")


if __name__ == "__main__":
    main()
