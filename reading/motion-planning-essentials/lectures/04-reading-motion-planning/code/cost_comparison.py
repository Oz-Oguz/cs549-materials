"""Chapter 03: equality-constrained motion costs with the Section 1 endpoint.

Run from the repository root with .venv/bin/python <this file>.
Reports independent KKT and route checks. Writes the chapter figure
fig/cost-comparison.png only when --write is given.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'modules/motion-planning'))
import staged_two_link as S


def solve(order):
    T, dt = S.T_K, S.komo_dt()
    # D acts on q0,...,qT. At t=1 the second difference repeats q0.
    D = np.zeros((T, T + 1))
    for t in range(1, T + 1):
        for offset, coef in ([(0, 1), (-1, -1)] if order == 1 else
                             [(0, 1), (-1, -2), (-2, 1)]):
            D[t - 1, max(0, t + offset)] += coef
    factor = np.sqrt(dt) / dt**order
    A = factor * np.kron(D[:, 1:], np.eye(2))
    b = factor * (D[:, :1] * np.asarray(S.Q_START)).ravel()
    E = np.zeros((4, 2 * T))
    E[:2, -2:] = np.eye(2)
    E[2:, -4:-2], E[2:, -2:] = -np.eye(2) / dt, np.eye(2) / dt
    target = np.r_[S.Q_GOAL, 0., 0.]
    K = np.block([[2 * A.T @ A, E.T], [E, np.zeros((4, 4))]])
    x = np.linalg.solve(K, np.r_[-2 * A.T @ b, target])
    z, multiplier = x[:2*T], x[2*T:]
    Q = np.vstack([S.Q_START, z.reshape(T, 2)])
    residual = A @ z + b
    eq_error = np.max(np.abs(E @ z - target))
    stationarity = np.max(np.abs(2*A.T @ residual + E.T @ multiplier))
    direction = np.asarray(S.Q_GOAL) - S.Q_START
    progress = (Q - S.Q_START) @ direction / (direction @ direction)
    route_error = np.max(np.abs(Q - (S.Q_START + progress[:, None]*direction)))
    steps = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    assert eq_error < 1e-11 and stationarity < 1e-8 and route_error < 1e-10
    assert np.min(np.diff(progress)) > -1e-12
    assert np.min(np.linalg.eigvalsh(2*A.T @ A)) > 0  # unique quadratic minimizer
    if order == 1:
        np.testing.assert_allclose(steps[:-1], np.linalg.norm(direction)/(T-1), atol=1e-12)
    else:
        assert np.argmax(steps) == 9
        assert np.all(np.diff(steps[:10]) > 0) and np.all(np.diff(steps[9:]) < 0)
    assert steps[-1] < 1e-12
    return Q, dict(cost=float(residual@residual), equality_error=float(eq_error),
                   stationarity_error=float(stationarity), route_error=float(route_error),
                   steps_deg=np.degrees(steps).tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='write fig/cost-comparison.png (default: print only)')
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('review_figures', ROOT / 'lectures/04-D2-motion-planning/code/make_review_figures.py')
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    answers, report = {}, {}
    for order in (1, 2):
        Q, report[str(order)] = solve(order)
        answers[order, False] = (Q, None, report[str(order)]['cost'])
    # Without --write the figure is drawn into a discarded temporary folder, so the
    # printed report and the figure's own checks stay the same in both modes.
    with tempfile.TemporaryDirectory() as scratch:
        output = (Path(__file__).resolve().parents[1] if args.write else Path(scratch)) / 'fig/cost-comparison.png'
        report['figure'] = figures.cost_comparison(
            SimpleNamespace(Q0=np.asarray(S.Q_START)), answers, goal=S.Q_GOAL,
            output_path=output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
