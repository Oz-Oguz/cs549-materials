"""Chapter 02 §1.5: one Jacobian-transpose update and a stationarity check.

Run with Python 3 and NumPy. Reuses the chapter's arm and safeguarded search.
No physical time, joint limits or collision constraints are modeled.
"""
import numpy as np

from local_step import phi, jac
from armijo import backtrack


def example():
    q = np.deg2rad([30., 45.])
    target = np.array([.5, .2])

    def cost(x):
        r = phi(x) - target
        return .5 * float(r @ r)

    def gradient(x):
        return jac(x).T @ (phi(x) - target)

    g = gradient(q)
    h = 1e-6
    basis = np.eye(2)
    numerical = np.array([(cost(q + h*b) - cost(q - h*b))/(2*h)
                          for b in basis])
    search = backtrack(cost, gradient, q)
    bridge = np.linalg.solve(jac(q), target - phi(q))
    cosine = (-g @ bridge)/(np.linalg.norm(g)*np.linalg.norm(bridge))
    result = dict(q=q, target=target, J=jac(q), residual=phi(q)-target,
                  gradient=g, finite_difference=numerical,
                  fd_error=np.max(np.abs(g-numerical)),
                  search=search, q1_degrees=np.rad2deg(search['point']),
                  tip1=phi(search['point']),
                  distance0=np.linalg.norm(phi(q)-target),
                  distance1=np.linalg.norm(phi(search['point'])-target),
                  direction_angle_degrees=np.rad2deg(np.arccos(np.clip(cosine,-1,1))))
    # A reachable target can have nonzero error and zero cost gradient.
    straight = np.zeros(2)
    sideways_target = np.array([.5, 0.])
    r = phi(straight) - sideways_target
    result['stationary_residual'] = r
    result['stationary_gradient'] = jac(straight).T @ r
    assert search['status'] == 'accepted' and search['alpha'] == 1.
    assert result['fd_error'] < 1e-9
    np.testing.assert_allclose(result['q1_degrees'], [20.8282234569,42.4499414684])
    np.testing.assert_allclose(result['stationary_gradient'], 0., atol=1e-15)
    assert np.linalg.norm(r) > .19
    return result


if __name__ == '__main__':
    np.set_printoptions(precision=12, suppress=True)
    for name, value in example().items():
        print(f'{name}: {value}')
