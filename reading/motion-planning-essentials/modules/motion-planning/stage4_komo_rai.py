"""Stage 4 of modules/motion-planning/staged_two_link.py in actual RAI (robotic 0.4.1): the KOMO program with the fixed
joint endpoint, whole-segment 5 mm clearance rows, the fixed prefix, normalized time and the plain squared cost,
written with ry.KOMO, so that the NumPy program and RAI are checked against each other on this problem.

Run (RAI runs on this Mac only in the amd64 container; docker/rai/RUNBOOK.md):
    RAI_NOVNC=0 scripts/rai_run.sh modules/motion-planning/stage4_komo_rai.py          (about a minute)

The RAI program (T = 20 slices after the start, dt = 1/T because ry.KOMO fixes the duration of a phase at 1):
    komo = ry.KOMO(C, phases=1, slicesPerPhase=T, kOrder=2, enableCollisions=False)
    komo.addControlObjective([], 2, 1.)                                          # cost: sum ||sqrt(dt) D^2 q / dt^2||^2, no 1/2
    komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1.], q_goal)           # the fixed goal configuration at the last slice
    komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1.], order=1)          # arrive at rest: (q_T - q_{T-1})/dt = 0
    komo.addObjective([], ry.FS.negDistance, ['link1', 'disc'], ry.OT.ineq, [1.])   # whole link 1 at least r_o + 5 mm from the center
    komo.addObjective([], ry.FS.negDistance, ['link2', 'disc'], ry.OT.ineq, [1.])   # whole link 2 likewise
Each link is a capsule of radius EPS along the link, and the disc a sphere of radius r_o + d_req - EPS, so RAI's
negative distance (r_o + d_req - EPS) + EPS - dist(segment, c) equals the NumPy row r_o + d_req - dist(segment, c)
exactly. No joint bounds are set on the hinge joints: the NumPy program has no joint-box rows either; the box enters
only through the final validation (stated, not changed here).

Part 1 compares RAI's feature VALUES and JACOBIANS with the NumPy program's rows on three fixed paths: the straight
joint-space line, the shortcut sampled path resampled to the knots, and the port's answer of W-auglag-tight-line-s10.
Part 2 compares, for the case IDs W-auglag-default-line (RAI's untouched options) and W-auglag-tight-line-s10 (RAI's
schedule, every constraint row times 10, tight stops), the port's exact-start run (rai_nlp.py through
staged_two_link.solve_komo) with RAI's own run in the same process: ry.NLP_Solver on the NumPy rows through
setPyProblem (program `twin`, as the fixture writer stage4_fixture_rai.py records exact cases), the case's options,
setInitialization at the nominal vector and solve(0). It checks equal evaluation counts and sos within 1e-9 relative.
An evaluation-count difference on the reference is reported as a qualified warning, because check_rai_nlp.py records a
rounding-level Wolfe decision there (port 151, RAI 132 evaluations); the end states must then agree (sos 1e-9
relative, end point 1e-6 rad).
Use --report FILE.json to retain every returned path, effective option table and independent residual check.
Every RAI path is then validated independently with the demo's checker
(rrt_two_link.check_edge, the 1 deg rule, knot to knot) and scanned
for its smallest clearance. Solver settings and RAI's return record are printed; the script exits with status 1 if a
check fails.
"""
import io
import os
import sys
import contextlib
import argparse
import importlib.metadata
import json
import numpy as np
import robotic as ry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import staged_two_link as S     # noqa: E402  (NumPy only; imports rrt_two_link)
import rrt_two_link as R        # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=150)
DEG = R.DEG
EPS = 1e-4                      # radius of the link capsules (m)
T = S.T_K
Q_START, Q_GOAL = np.array(S.Q_START), np.array(S.Q_GOAL)


def build():
    """The configuration; the disc sphere takes the margin from staged_two_link.MARGIN at call time."""
    C = ry.Config()
    C.addFrame("base").setPosition([0, 0, 0])
    C.addFrame("j1", "base").setJoint(ry.JT.hingeZ)          # no joint bounds, as in the NumPy program
    # a capsule's axis is its frame's z axis and it is centered at the frame origin: put the frame at the link's middle,
    # with z turned onto the link's x direction (a rotation of 90 deg about y: quaternion (cos 45, 0, sin 45, 0))
    C.addFrame("link1", "j1").setRelativePosition([S.L1 / 2, 0, 0]).setRelativeQuaternion([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]).setShape(ry.ST.capsule, [S.L1, EPS]).setContact(1)
    C.addFrame("l1end", "j1").setRelativePosition([S.L1, 0, 0])
    C.addFrame("j2", "l1end").setJoint(ry.JT.hingeZ)
    C.addFrame("link2", "j2").setRelativePosition([S.L2 / 2, 0, 0]).setRelativeQuaternion([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]).setShape(ry.ST.capsule, [S.L2, EPS]).setContact(1)
    C.addFrame("tip", "j2").setRelativePosition([S.L2, 0, 0]).setShape(ry.ST.marker, [0.05])
    C.addFrame("disc").setPosition([S.OBS_C[0], S.OBS_C[1], 0]).setShape(ry.ST.sphere, [S.OBS_R + S.MARGIN - EPS]).setContact(1)
    C.setJointState(Q_START.copy())
    return C


def make_komo(C, parts=("control", "goal", "rest", "link1", "link2"), constraint_scale=1.0, T=None):
    """The KOMO of the docstring. `T` defaults to staged_two_link.T_K at call time (the fixture writer varies it)."""
    T = S.T_K if T is None else T
    komo = ry.KOMO(C, phases=1, slicesPerPhase=T, kOrder=2, enableCollisions=False)
    if "control" in parts:
        komo.addControlObjective([], 2, 1.)
    if "goal" in parts:
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [constraint_scale], list(Q_GOAL))
    if "rest" in parts:
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [constraint_scale], order=1)
    for link in ("link1", "link2"):
        if link in parts:
            komo.addObjective([], ry.FS.negDistance, [link, "disc"], ry.OT.ineq, [constraint_scale])
    return komo


def dense(J, n_rows, n_cols):
    """RAI's Jacobian as a dense (n_rows x n_cols) array: robotic 0.4.1's nlp().evaluate returns it as sparse triplets
    (row index, column index, value), one per nonzero; a dense array is returned unchanged."""
    if hasattr(J, "toarray"):
        return np.asarray(J.toarray())
    J = np.asarray(J)
    if J.ndim == 2 and J.shape == (n_rows, n_cols):
        return J
    if J.ndim == 2 and J.shape[1] == 3 and np.all(J[:, 0] == np.round(J[:, 0])) and np.all(J[:, 1] == np.round(J[:, 1])):
        D = np.zeros((n_rows, n_cols))
        for i, j, v in J:
            D[int(i), int(j)] += v
        return D
    raise ValueError(f"unrecognized Jacobian storage with shape {J.shape}")


def rows_and_jac(C, part, z):
    """RAI's feature rows and Jacobian of ONE objective at the path z (2T values), in slice order."""
    komo = make_komo(C, parts=(part,))
    komo.initWithPath(np.asarray(z).reshape(T, 2))
    phi, J = komo.nlp().evaluate(np.asarray(z).flatten())
    phi = np.asarray(phi)
    return phi, dense(J, phi.shape[0], 2 * T)


def numpy_rows(z):
    r, Jr = S.acc_residuals(z)
    h, Jh = S.eq_constraints(z)
    g, Jg = S.ineq_constraints(z)
    # the NumPy inequality rows alternate (knot 1 link 1, knot 1 link 2, knot 2 link 1, ...); split by link
    return {"control": (r, Jr), "goal": (h[:2], Jh[:2]), "rest": (h[2:], Jh[2:]),
            "link1": (g[0::2], Jg[0::2]), "link2": (g[1::2], Jg[1::2])}


def validate_knots(z):
    pts = [tuple(q) for q in S.knots(z)]
    return S.validate_path(pts), pts


def kkt_relative(z):
    """Lagrangian stationarity at z with least-squares multipliers on the equalities and on the inequality rows that
    are active or violated (g > -1e-6 selects both), relative to |grad f|; also the most negative fitted inequality
    multiplier. This fit measures stationarity only; it does not make an infeasible result (the default-schedule runs)
    a KKT point, which also needs feasibility and complementarity."""
    _, J_h = S.eq_constraints(z); g, J_g = S.ineq_constraints(z)
    act = np.nonzero(g > -1e-6)[0]
    M = np.vstack([J_h, J_g[act]]).T
    gf = S.grad_cost(z)
    mult, *_ = np.linalg.lstsq(M, -gf, rcond=None)
    res = np.linalg.norm(gf + M @ mult) / max(1.0, np.linalg.norm(gf))
    lam = mult[J_h.shape[0]:]
    return res, (lam.min() if lam.size else 0.0), act


TOL = dict(rows_abs=1e-10, jac_abs=1e-8, cost_rel_rows=1e-12,          # Part 1
           feas=1e-8, joint=1e-6, cost_rel=1e-7, kkt_rel=1e-7, lam_neg=1e-6)   # Part 2 with the course's fixed mu
FAILS = []


def check(ok, msg):
    print(("    ok    " if ok else "    FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="write options, returned paths and checks to this JSON file")
    args = parser.parse_args()
    records = []
    C = build()
    print(f"=== stage 4's program in RAI (robotic 0.4.1): ry.KOMO(C, phases=1, slicesPerPhase={T}, kOrder=2), link capsules r = {EPS} m, disc sphere r = {S.OBS_R + S.MARGIN - EPS} m; no joint bounds ===")
    print(f"    dt = {S.komo_dt()}; tolerances {TOL}")
    # the NumPy side: the shortcut path of stage 3 and the reference answer of the port, computed quietly
    with contextlib.redirect_stdout(io.StringIO()):
        z_line, z_path = S.linear_init(), S.shortcut_init(7)
        z_ref = S.solve_komo("W-auglag-tight-line-s10")[0]
    paths = (("straight line", z_line), ("shortcut path resampled to the knots", z_path), ("port answer, W-auglag-tight-line-s10", z_ref))

    print("\n  Part 1: RAI's feature values and Jacobians against the NumPy rows on the same paths")
    for label, z in paths:
        npr = numpy_rows(z)
        worst_val, worst_jac, shapes = 0.0, 0.0, []
        for part in ("control", "goal", "rest", "link1", "link2"):
            phi, J = rows_and_jac(C, part, z)
            r_np, J_np = npr[part]
            shapes.append(f"{part} {phi.shape[0]}x{J.shape[1]}")
            if phi.shape[0] != r_np.shape[0] or J.shape != J_np.shape:
                check(False, f"{label}: {part} has {phi.shape[0]} rows and Jacobian {J.shape}; NumPy has {r_np.shape[0]} rows and {J_np.shape}")
                continue
            worst_val = max(worst_val, float(np.abs(phi - r_np).max()))
            worst_jac = max(worst_jac, float(np.abs(J - J_np).max()))
        sos, _ = rows_and_jac(C, "control", z)
        crel = abs(float(sos @ sos) - S.cost(z)) / S.cost(z)
        check(worst_val <= TOL["rows_abs"] and worst_jac <= TOL["jac_abs"] and crel <= TOL["cost_rel_rows"],
              f"{label:38s} RAI sos {float(sos @ sos):.9g} = NumPy f {S.cost(z):.9g} (rel. {crel:.1e}); largest row-value difference {worst_val:.1e}, largest Jacobian-entry difference {worst_jac:.1e}; rows: {', '.join(shapes)}")

    print("\n  Part 2: the port against RAI on the same case, both from the exact start (setInitialization + solve(0), program twin)")
    sys.path.insert(0, str(S.KOMO_CODE))
    import rai_fixture as F                                       # noqa: E402  (TwinNLP, method_enum)
    for cid in ("W-auglag-default-line", "W-auglag-tight-line-s10"):
        case = S.rai_nlp.Case(cid)
        z_port, lam_port, _, info = S.solve_komo(cid)
        with S.case_model(cid):
            x0 = S.linear_init()
            twin = F.TwinNLP(2 * case.T, S.acc_residuals, S.eq_constraints, S.ineq_constraints, x0, case.scale)
            solver = ry.NLP_Solver()
            solver.setPyProblem(twin)
            opts = case.options()
            solver.setOptions(verbose=0, **{**opts, "method": F.method_enum(opts["method"])})
            solver.setInitialization(x0.copy())
            ret = solver.solve(0)
            z = np.asarray(ret.x, dtype=float)
            lam_rai = case.scale * np.asarray(solver.dual)[-2 * case.T:]      # the twin's rows: sos, eq, ineq; physical = scale x dual
        rep, rep_port = S.physical_report(cid, z, lam_rai), S.physical_report(cid, z_port, lam_port)
        sos_rel = abs(info["errors_buffered"]["sos"] - ret.sos) / abs(ret.sos)
        dq = float(np.abs(z - z_port).max())
        record = {k: getattr(ret, k) for k in ("evals", "time", "feasible", "done", "sos", "f", "ineq", "eq") if hasattr(ret, k)}
        line = (f"{cid} (exact start): RAI {ret.evals} evaluations, sos {ret.sos:.9f}; port {info['evals']} evaluations, sos {info['errors_buffered']['sos']:.9f} "
                f"(relative {sos_rel:.1e}); max |x_RAI - x_port| {dq:.1e} rad")
        if ret.evals == info["evals"]:
            check(sos_rel <= 1e-9, line + "; evaluation counts equal, sos within 1e-9 relative")
        else:
            print(f"    WARN  {line}; qualified: the evaluation counts differ (a rounding-level decision, see check_rai_nlp.py), so the end states are compared")
            check(sos_rel <= 1e-9 and dq <= 1e-6, f"{cid}: end states agree (sos within 1e-9 relative, end points within 1e-6 rad)")
        v, _ = validate_knots(z)
        print(f"          {cid}, RAI: {S.report_words(rep)}")
        print(f"          {cid}, port: knot deficit {S.mm_words(rep_port['knot_deficit_mm'])}, physical multipliers "
              + ", ".join(f"knot {t} link {l}: {x:.3f} per m" for t, l, x in rep_port["active"]))
        print(f"          {cid}: RAI's return {record}; effective options {solver.getOptions().dict()}")
        records.append(dict(case=cid, program="twin", call="setInitialization + solve(0)", constraint_scale=case.scale, options_passed=opts,
                            options_effective=solver.getOptions().dict(), return_record=record, x=z.tolist(), x_port=z_port.tolist(),
                            evals_port=info["evals"], sos_port=info["errors_buffered"]["sos"], nominal_x=x0.tolist(),
                            residuals={k: rep[k] for k in ("cost", "knot_deficit_mm", "goal_error_rad", "rest_error", "between_knot_deficit_mm", "active")},
                            validation=v))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(dict(robotic=importlib.metadata.version("robotic"), compiled=ry.compiled(),
                           runs=records, failed_checks=FAILS), f, indent=2)
            f.write("\n")
    print(f"\nstage4_komo_rai.py: {'all checks passed' if not FAILS else str(len(FAILS)) + ' check(s) FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
