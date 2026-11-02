"""Reproduce Chapter 03 §§3–5 without changing any demo, figure or fixture.

Print JSON for the roadmap, steering, bidirectional route, RRT* insertion,
shortcutting, refinement, clearance certificates and time rescaling.
Run from the repository root with the course Python environment.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'modules/motion-planning'))
import rrt_two_link as R
import staged_two_link as S
import rai_bidir_rrt as B
import rrtstar_reference as X
import refine_figures as F


def scan(path, spacing_deg):
    """Include both endpoints of every subinterval; report surface clearance."""
    minimum = math.inf
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        n = max(1, math.ceil(np.max(np.abs(b-a)) / (spacing_deg * R.DEG)))
        for j in range(n+1):
            minimum = min(minimum, R.clearance(tuple(a+(b-a)*j/n))[0] + R.MARGIN)
    return minimum * 1000


def close(value, expected, tol=5e-3):
    assert abs(value - expected) <= tol, (value, expected)


def main():
    report = {}
    p = S.prm(seed=7, n_samples=400, k=5)
    route = [p['nodes'][i] for i in p['path']]
    assert p['path'] == [0, 378, 154, 238, 74, 1]
    # Independent Bellman relaxation of all edges checks the reported Dijkstra cost.
    d = np.full(len(p['nodes']), np.inf); d[0] = 0
    for _ in range(len(d)-1):
        old = d.copy()
        for u, edges in p['edges'].items():
            for v, weight in edges:
                d[v] = min(d[v], old[u]+weight)
        if np.array_equal(old, d):
            break
    length = S.path_length(route)
    assert abs(length-d[1]) < 1e-12
    partial = S.path_length(route[:-1]) / R.DEG
    last = R.dist(*route[-2:]) / R.DEG
    close(partial, 58.2709); close(last, 11.6150)
    report['prm'] = dict(samples=400, valid=p['n_valid'], vertices=len(p['nodes']),
                         route=p['path'], length_deg=length/R.DEG,
                         last_relaxation_deg=[partial, last, d[1]/R.DEG],
                         clearance_1deg_mm=scan(route, 1))
    trace = []; r = R.rrt(7, 3000, trace=trace)
    original = [r['nodes'][i] for i in r['path']]
    ev = trace[10]; near = np.array(r['nodes'][ev['near']]); target = np.array(ev['q_rand'])
    delta = target-near; scale = min(1., R.STEP/np.linalg.norm(delta))
    new = near+scale*delta
    np.testing.assert_allclose(new, ev['q_new'], atol=1e-15)
    assert ev['near']==6 and ev['result']=='accepted' and ev['n']==10
    assert R.rrt(7,20)['outcome']=='budget' and r['iterations']==42
    report['rrt'] = dict(length_deg=S.path_length(original)/R.DEG, edges=len(original)-1,
                         budget20='no route', iterations=r['iterations'],
                         clearance_fine_mm=scan(original,.05),
                         extension=dict(parent=6, near_deg=(near/R.DEG).tolist(),
                                        target_deg=(target/R.DEG).tolist(), distance_deg=ev['d']/R.DEG,
                                        scale=scale,new_deg=(new/R.DEG).tolist(), checks=ev['n']))
    b = B.plan(seed=7); bv=B.check_path(b['path'])
    assert b['joined'] and bv['pass'] and B.join_case(b)[0]=='on-segment'
    bad=B.plan(seed=1327); bad_check=B.check_path(bad['path'])
    assert bad['joined'] and not bad_check['pass'] and B.join_case(bad)[0]=='off-line'
    report['bidir'] = dict(vertices=len(b['path']),iterations=b['iterations'],queries=b['queries'],
                           check=bv,join_case=B.join_case(b)[0], failing_example=bad_check)
    x=X.run(seed=17,sample=141,radius_deg=10)
    assert not x['invariants']
    done=x['events'][-1]; goal=done['goal']
    assert abs(done['goal_after']-S.path_length([x['final']['nodes'][i] for i in done['route']]))<1e-12
    close(done['goal_before']/R.DEG,78.60409);close(done['goal_after']/R.DEG,76.97545)
    rewire=next(e for e in x['events'] if e['kind']=='rewire' and e['result']=='rewired')
    saving=rewire['cost']-rewire['after']
    assert len(rewire['descendants'])==4
    assert all(abs(e['before']-e['after']-saving)<1e-12 for e in rewire['descendants'])
    report['rrtstar']=dict(parent_events=[e for e in x['events'] if e['kind']=='parent'],
                          goal_before_deg=done['goal_before']/R.DEG,goal_after_deg=done['goal_after']/R.DEG,
                          saving_deg=saving/R.DEG,rewired_node=rewire['k'],descendants=4)
    shortened, log=S.shortcut(original,7,100)
    assert S.validate_path(shortened)['valid']
    assert all(e['length_after_deg']<=e['length_before_deg']+1e-12 for e in log if e['result']=='accepted')
    close(S.path_length(shortened)/R.DEG,58.34)
    fine=scan(shortened,.05);close(fine,4.912,tol=.001)
    report['shortcut']=dict(before_deg=S.path_length(original)/R.DEG,after_deg=S.path_length(shortened)/R.DEG,
                            accepted=sum(e['result']=='accepted' for e in log), edges=len(shortened)-1,
                            clearance_1deg_mm=scan(shortened,1),clearance_fine_mm=fine)
    data=F.compute(); assert not F.check_against_native(data)
    report['refinement']={k:data[k] for k in ['cost_init','cost_res','deficit_init_mm','deficit_res_mm','knot_deficit_res_mm','same_deg']}
    assert not data['v_init']['valid'] and not data['v_res']['valid']
    assert data['v_init']['deepest']['edge']==14
    z,lam,_,info=S.solve_komo('W-auglag-tight-line-s10-T40')
    with S.case_model('W-auglag-tight-line-s10-T40'):
        rep40=S.physical_report('W-auglag-tight-line-s10-T40',z,lam)
    recorded=json.loads((ROOT/'modules/motion-planning/fixtures/stage4_rai.json').read_text())['cases']['W-auglag-tight-line-s10-T40-seed1']
    close(rep40['cost'],recorded['ret']['sos'],tol=1e-6)
    close(rep40['between_knot_deficit_mm'],recorded['residuals_at_x']['between_knot_deficit_mm'],tol=1e-5)
    report['forty_intervals']={k:v for k,v in rep40.items() if k!='pts'}
    # Uniform coordinate spacing gives a conservative half-interval bound.
    half1=(R.L1+2*R.L2)*R.DEG/2*1000
    half_fine=half1*.05
    report['certificates_mm']=dict(half_bound_1deg=half1,half_bound_fine=half_fine,
        prm_lower=report['prm']['clearance_1deg_mm']-half1,
        rrt_lower=report['rrt']['clearance_fine_mm']-half_fine)
    assert report['certificates_mm']['prm_lower']>5 and report['certificates_mm']['rrt_lower']>5
    # Check the chain/product rule against central differences on a smooth probe.
    path=lambda s: np.array([s*s, s*s*s])
    schedule=lambda t: t*t
    t=.4; s=schedule(t); ds=2*t; dds=2.
    v=np.array([2*s,3*s*s])*ds
    a=np.array([2.,6*s])*ds**2+np.array([2*s,3*s*s])*dds
    q=lambda t:path(schedule(t))
    h=1e-4
    np.testing.assert_allclose((q(t+h)-q(t-h))/(2*h),v,atol=1e-7)
    np.testing.assert_allclose((q(t+h)-2*q(t)+q(t-h))/h**2,a,atol=1e-7)
    qslow=lambda t:q(t/2)
    np.testing.assert_allclose((qslow(2*t+h)-qslow(2*t-h))/(2*h),v/2,atol=1e-7)
    np.testing.assert_allclose((qslow(2*t+h)-2*qslow(2*t)+qslow(2*t-h))/h**2,a/4,atol=1e-7)
    report['timing']='chain rule and uniform duration doubling verified by finite differences'
    print(json.dumps(report,indent=2,default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o)))


if __name__=='__main__':
    main()
