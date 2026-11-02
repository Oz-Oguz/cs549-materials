"""Recompute Chapter 03 Section 2, including derivatives and strict edge failures.

Run from the repository root. JSON output includes the solver records of both starts.
Does not change demo code, figures, or solver settings.

The KOMO solves use RAI's loop (rai_nlp.py through staged_two_link.solve_komo) on the converged reference of the
whole-link model: W-auglag-tight-line-s10 and W-auglag-tight-shortcut-s10, RAI's penalty schedule with every
constraint row times 10 and the tight stops, each from its exact start. Their results are compared with the records
of the same case IDs in the real-RAI fixture modules/motion-planning/fixtures/stage4_rai.json (sos to 1e-6
relative); a missing fixture or case fails. The multipliers are physical (10 times RAI's dual). The margin
sensitivity solves W-auglag-tight-line-s10-m5.1-warm (the margin 5.1 mm, warm-started at the reference answer) and
compares the measured cost change with 1e-4 m times the sum of the physical multipliers, to 0.003 relative.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'modules/motion-planning'))
import staged_two_link as S
import rrt_two_link as R
from cost_comparison import solve


def fd(fun, x, eps=1e-6):
    return np.column_stack([(fun(x+eps*e)-fun(x-eps*e))/(2*eps) for e in np.eye(len(x))])


def main():
    report = {'quadratics': {str(k): solve(k)[1] for k in (1, 2)}}
    dt = S.komo_dt()
    np.testing.assert_allclose(np.diff([0., .05, .10])/dt, [1., 1.])
    assert abs((.15-2*.05)/dt**2-20) < 1e-12
    assert abs((np.sqrt(dt)/dt**2)**2-8000) < 1e-10
    z0 = S.linear_init()
    jac_errors = {}
    for name, fun, rows in [('acceleration',S.acc_residuals,40),('equalities',S.eq_constraints,4),('clearance',S.ineq_constraints,40)]:
        y, J = fun(z0)
        assert y.shape == (rows,) and J.shape == (rows,40)
        err = np.max(np.abs(J-fd(lambda z: fun(z)[0],z0)))
        assert err < 1e-7
        jac_errors[name] = float(err)
    report['full_row_jacobian_errors'] = jac_errors
    theta = np.arctan2(S.OBS_C[1],S.OBS_C[0])
    cases = [('start, link 1',np.array(S.Q_START),1),('start, link 2',np.array(S.Q_START),2),
             ('base endpoint',np.array([theta+2.,.5]),1),
             ('interior-to-tip transition',np.array([theta+np.arccos(S.L1/np.linalg.norm(S.OBS_C)),.5]),1),
             ('interior-to-base transition',np.array([theta+np.pi/2,.5]),1)]
    deriv = []
    for name,q,link in cases:
        p,d,J,u = S.segment_point_and_jac(q,link)
        analytic = (p-S.OBS_C)/d @ J
        numeric = fd(lambda x: np.atleast_1d(S.segment_point_and_jac(x,link)[1]),q)[0]
        err=float(np.max(np.abs(analytic-numeric)))
        assert err < 2e-7
        deriv.append(dict(case=name,fraction=float(u),error=err))
    report['projection_cases'] = deriv
    with contextlib.redirect_stdout(io.StringIO()):
        tree=R.rrt(7,3000,'links')
        path,_=S.shortcut([tree['nodes'][i] for i in tree['path']],7,100)
    fixture = ROOT / 'modules/motion-planning/fixtures/stage4_rai.json'
    assert fixture.exists(), f'the real-RAI fixture {fixture} is missing'
    records = json.loads(fixture.read_text())['cases']
    starts = {'straight': 'W-auglag-tight-line-s10', 'supplied': 'W-auglag-tight-shortcut-s10'}
    assert np.array_equal(S.shortcut_init(7), S.path_init(path))
    solutions = {}
    report['solutions'] = {}
    for name, cid in starts.items():
        assert cid in records, f'{cid} is not recorded in {fixture.name}'
        rec = records[cid]
        answer, lam, kap, info = S.solve_komo(cid)
        sos = info['errors_buffered']['sos']
        assert abs(sos - rec['ret']['sos']) <= 1e-6 * rec['ret']['sos'], f"{cid}: sos {sos} against RAI's {rec['ret']['sos']}"
        assert rec['stop'].endswith(info['stop'])
        Q = S.knots(answer)
        assert np.max(np.abs(Q)) <= np.pi
        h, _ = S.eq_constraints(answer); g, _ = S.ineq_constraints(answer)
        active = np.flatnonzero(lam > 1e-6)
        assert active.tolist() == [23, 25]
        report['solutions'][name] = dict(case=cid, start='exact start', initial_cost=S.cost(np.array(info['x_start'])), cost=S.cost(answer),
            sos_rai=rec['ret']['sos'], evals=info['evals'], evals_rai=rec['ret']['evals'], stop=info['stop'],
            endpoint_max=float(np.abs(h[:2]).max()), rest_max=float(np.abs(h[2:]).max()),
            max_g=float(g.max()), active_multipliers=lam[active].tolist(), multiplier_units='per metre (10 x RAI dual)',
            validation=S.validate_path([tuple(q) for q in Q]),
            solver={k: info[k] for k in ('case', 'evals', 'outer_iters', 'stop', 'feasible', 'errors_buffered', 'errors_at_x', 'options_passed', 'mu')})
        solutions[name] = answer
    difference = float(np.max(np.abs(solutions['straight'] - solutions['supplied'])))
    rai_difference = float(np.max(np.abs(np.array(records[starts['straight']]['x']) - np.array(records[starts['supplied']]['x']))))
    assert difference <= rai_difference + 1e-8
    report['solution_difference_rad'] = dict(port=difference, rai=rai_difference, cases=list(starts.values()))
    # Multipliers predict the local cost change when every clearance row
    # tightens by the same amount. The warm case keeps the reference's answer
    # as its exact start; solve_komo restores the 5 mm model afterwards.
    warm = 'W-auglag-tight-line-s10-m5.1-warm'
    assert warm in records, f'{warm} is not recorded in {fixture.name}'
    margin_change = 0.0001
    base = solutions['straight']
    predicted = margin_change * sum(report['solutions']['straight']['active_multipliers'])
    tighter, multipliers, _, info = S.solve_komo(warm, z0=base)
    assert abs(info['errors_buffered']['sos'] - records[warm]['ret']['sos']) <= 1e-6 * records[warm]['ret']['sos']
    assert np.flatnonzero(multipliers > 1e-6).tolist() == [23, 25]
    with S.case_model(warm):
        assert abs(S.MARGIN - 0.0051) < 1e-15
        assert S.ineq_constraints(tighter)[0].max() < 1e-9
    actual = S.cost(tighter) - S.cost(base)
    assert abs(actual - 0.023473) < 5e-7
    assert abs(actual - predicted) / actual < 0.003
    report['margin_sensitivity'] = dict(
        case=warm, parent=starts['straight'], parent_in_fixture=records[warm]['parent'],
        start="warm start at the port's exact-start reference answer (the fixture's warm case starts at its seeded parent's ret.x, within 1e-9 rad of it)",
        margin_change_m=margin_change, predicted_cost_change=predicted,
        solved_cost_change=actual, solved_cost_change_rai=records[warm]['ret']['sos'] - records[records[warm]['parent']]['ret']['sos'],
        new_cost=S.cost(tighter), evals=info['evals'], evals_rai=records[warm]['ret']['evals'], stop=info['stop'])
    Q=S.knots(solutions['straight'])
    first=report['solutions']['straight']['validation']['fail']
    assert first['edge']==11 and first['j']==4
    n=max(1,int(np.ceil(np.max(np.abs(Q[13]-Q[12]))/R.SUB)))
    assert n==4
    deficits=[-R.clearance(tuple((1-j/n)*Q[12]+j/n*Q[13]),'links')[0]*1000 for j in range(n+1)]
    np.testing.assert_allclose(deficits[1:4],[1.37950838596,1.843636534,1.37401490596],atol=1e-6)
    report['interval_12_13_deficit_mm']=dict(case=starts['straight'],start='exact start',deficits_mm=deficits)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
