"""Session 4.1 Drake demo: constrained inverse kinematics on the Panda with the course's augmented-Lagrangian loop
(lectures/04-D1-constraints-and-komo/code/constrained_ik.py, notes section 4.9) and one visible joint limit.

Problem:   min_q  f(q) = 1/2 |phi(q) - y*|^2     s.t.   g(q) = q_4 - q_4max <= 0,    q_4max = -2.2 rad
The stop is a software limit chosen for this demonstration, inside the hardware range [-3.07, -0.07] rad of joint 4:
the elbow may not straighten past -2.2 rad (126 degrees of bend). In Deck 05's example (b) the limit kept the two-link
elbow from bending further; here it keeps the Panda's elbow from opening. phi is Drake's forward kinematics of the
flange (panda_link8) and J its 3x7 translational Jacobian, both from MultibodyPlant, as in gn_ik_panda.py; grad g = e_4.

Method (the course's loop: mu fixed, lambda learned):
    A(q) = f(q) + lambda g(q) + mu [g(q) > 0 or lambda > 0] g(q)^2
    inner solve, lambda fixed:  damped Gauss-Newton on A with H = J^T J + 2 mu [row included] e_4 e_4^T + rho I, rho = 1e-3
                                (Deck 04's damping, because J^T J has rank 3 of 7), Armijo backtracking restarted from
                                alpha = 1 (c = 0.01, halving, floor 1e-12; a trial at the floor that fails rejects the
                                direction and stalls the run); stop at |grad A| < 1e-7 or after 60 accepted steps
    outer update, q fixed:      lambda <- max(0, lambda + 2 mu g(q)); stop when max(g, 0) < 1e-6 rad,
                                |grad f + lambda grad g| < 1e-6 and |lambda g| < 1e-8 (the KKT triplet with the updated
                                multiplier, as in constrained_ik.py), or after 8 outer iterations.
    Outer iterations are numbered from 0 as in Deck 05 (outer iteration 0 is the squared penalty at mu = 1); the inner
    steps within each are numbered from 1. The two-link code stops the inner solve at 1e-9 with undamped Newton; on the
    Panda the Armijo test on A is decided by rounding below an inner gradient of about 2e-9, so the inner tolerance is
    1e-7, and the complementarity tolerance 1e-8 is approximately the multiplier magnitude times the feasibility
    tolerance (with 1e-9 the loop would run one more outer iteration on rounding noise).
Two runs from the ready pose. Run A, target (0.65, 0, 0.35) m: the unconstrained damped Gauss-Newton answer reaches
the target with the elbow 0.41 rad past the stop; the loop keeps the stop, ends 4.3 cm short with lambda* = 0.0117
m^2/rad after four multiplier updates, and the violation after each inner solve falls by the factor a/(a + 2 mu)
of notes section 4.9, about 0.0375 (a 27-fold reduction), with a the curvature of the best attainable f along the
stop. That answer is a LOCAL constrained result reached from the ready pose, not a proof that the target is out of
reach: the loop's problem has only the upper bound and joint angles are periodic, so the unconstrained answer with
one full turn subtracted from q_4 (-8.08 rad) satisfies the bound and reaches the target within 23 nm; it lies
outside the hardware range. Run B, the plan's scene target (0.45, 0.10, 0.45) m: the unconstrained answer keeps the
elbow 0.03 rad inside the stop, so the row is never included, lambda stays 0 and the loop is plain damped
Gauss-Newton (Deck 05's example (e)). SNOPT and IPOPT solve the same objective and stop PLUS the URDF joint box,
which excludes the wrapped configuration (and without which IPOPT can wander to configurations many turns away):
their agreement with the loop's answer from the same start supports the local result; it is not an equivalence of
the two programs and not a global certificate. Reported: their exit status, task error and q_4, the stop's dual with
Drake's sign rule (lambda = -dual for an upper bound), the largest box dual, and the KKT triplet of every answer
recomputed here; relaxing the stop by 1e-3 rad changes f* by about -lambda * 1e-3 (the multiplier as the price of the
constraint). IPOPT documents a bound relaxation (bound_relax_factor), hence its small positive violation. No claim
that the configurations coincide. The hardware limits are not constraints of the loop's problem; every displayed
configuration is checked against them and the exceptions are listed (one accepted iterate of run A has joint 6 two
degrees past its limit).

Run:  python constrained_ik_panda.py                  prints both runs (one line per outer iteration), the final answers and the SNOPT/IPOPT comparison
      python constrained_ik_panda.py --replay --app   paused replay as a web page around Meshcat: the robot at the event's q, a faded ghost
                                                      at the previous iterate for an inner step, the target and the flange joined by the
                                                      task-error line, a protractor fixed to the upper arm at the elbow (the allowed arc, the
                                                      stop, the current angle), readouts, a per-outer table, plots, and the controls Step,
                                                      Back, Reset, a slider over the events, multiplier-update jumps, run buttons, toggles
                                                      for the rejected trials, the unconstrained answer and an elbow close-up; nothing
                                                      moves by itself
      python constrained_ik_panda.py --replay         the same replay with Meshcat's own controls only (no web page)
      python constrained_ik_panda.py --record FILE    writes both runs' events as JSON (cs549-trace/1)
      python constrained_ik_panda.py --figure FILE    task error, violation, multiplier and inner step per event (matplotlib)
      python constrained_ik_panda.py --selftest [BASELINE.json]  recomputes every event from FK and the Jacobian, checks the counts,
                                                      the solver comparison, the contraction, the source hashes, and compares the iterates
                                                      with the saved regression baseline and the independent reference

Added 2026-10-14. Tested with drake 1.46.0 on macOS.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02-gauss-newton-ik"))
import gn_ik_panda as gn                                        # the plant, FK, the Jacobian, the camera fit, the web wrapper's folder
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, IpoptSolver, MathematicalProgram, MeshcatVisualizer,
                         Parser, Rgba, RigidTransform, SnoptSolver, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
JOINT, JOINT_NAME, Q4_MAX = 3, "panda_joint4", -2.2            # g(q) = q[JOINT] - Q4_MAX <= 0
MU, RHO = 1.0, 1e-3
INNER_TOL, INNER_MAX = 1e-7, 60
FEAS_TOL, STAT_TOL, COMP_TOL, OUTER_MAX = 1e-6, 1e-6, 1e-8, 8    # FEAS_TOL is also the activity tolerance (on the stop within it)
ARMIJO_C, ARMIJO_FLOOR = 0.01, 1e-12
RUNS = {"runA": {"label": "Run A: far target (0.65, 0, 0.35) m, the stop binds", "target": [0.65, 0.0, 0.35],
                 "note": "the unconstrained answer opens the elbow past the stop; the loop keeps the stop and ends short of the target"},
        "runB": {"label": "Run B: the plan target (0.45, 0.10, 0.45) m, the stop never binds", "target": [0.45, 0.10, 0.45],   # no apostrophe: Meshcat builds control code from button labels
                 "note": "the unconstrained answer keeps the elbow inside the stop, so the row is never included and lambda stays 0 (Deck 05's example (e))"}}
PROTRACTOR_R = 0.16                                             # radius of the elbow protractor [m], outside the elbow housing
SOURCE_FILES = ("constrained_ik_panda.py", "panel.html", "../02-gauss-newton-ik/replay_panel.py")


def state_of(g):
    """Plan section 7's three states, with the feasibility tolerance as the activity tolerance."""
    return "violated" if g > FEAS_TOL else ("on the stop" if g > -FEAS_TOL else "inside")


# ----------------------------------------------------------------------------------------------- the loop
def al_ik(phi, J, q0, y, j=JOINT, lim=Q4_MAX, mu=MU, rho=RHO, on_event=None):
    """The course's augmented-Lagrangian loop for one inequality g = q_j - lim <= 0 (constrained_ik.py, function
    augmented_lagrangian, with Deck 04's damped Gauss-Newton step as the inner solver). Returns
    (q, lambda, status, rows) with one row per outer iteration. on_event (optional; None in print mode) observes
    the run without changing it: {"type": "accepted_update"} per accepted inner step (with the previous iterate,
    the accepted step after Armijo shortening, the full direction, alpha, the number of rejected trials, whether
    the row was included in the inner objective at the previous iterate, the values of f, g and A after the step
    and the Armijo quantities), {"type": "trial_evaluation"} per rejected Armijo trial, and {"type":
    "outer_update"} per multiplier update (q fixed; lambda before and after; the KKT residuals with the updated
    multiplier)."""
    e = np.zeros(7); e[j] = 1.0                                   # grad g
    gfun = lambda qq: float(qq[j] - lim)
    f = lambda qq: 0.5 * float(np.sum((phi(qq) - y) ** 2))
    q, lam = np.array(q0, float), 0.0
    rows, status = [], "limit"
    for k in range(OUTER_MAX):
        incl = lambda qq: (gfun(qq) > 0) or (lam > 0)             # the row is included while it is violated or priced
        A = lambda qq: f(qq) + lam * gfun(qq) + (mu * gfun(qq) ** 2 if incl(qq) else 0.0)
        gradA = lambda qq: J(qq).T @ (phi(qq) - y) + e * (lam + (2 * mu * gfun(qq) if incl(qq) else 0.0))
        n_in, st = 0, "limit"
        gr = gradA(q)
        for _ in range(INNER_MAX):
            gr = gradA(q)
            if np.linalg.norm(gr) < INNER_TOL:
                st = "stationary"; break
            Jq = J(q); included = incl(q)
            H = Jq.T @ Jq + (2 * mu * np.outer(e, e) if included else 0.0) + rho * np.eye(7)
            d = -np.linalg.solve(H, gr)
            a, A0, slope = 1.0, A(q), float(gr @ d)
            ok = lambda aa: np.isfinite(A(q + aa * d)) and A(q + aa * d) <= A0 + ARMIJO_C * aa * slope
            rejected = 0
            while not ok(a) and a > ARMIJO_FLOOR:
                if on_event:
                    on_event({"type": "trial_evaluation", "outer": k, "inner": n_in + 1, "alpha": a, "q": (q + a * d).copy(),
                              "A": A(q + a * d), "included": bool(incl(q + a * d))})
                a *= 0.5; rejected += 1
            if not ok(a):
                st = "stalled"; break
            q_prev = q.copy(); q = q + a * d; n_in += 1
            if on_event:
                gq = gfun(q)
                on_event({"type": "accepted_update", "outer": k, "inner": n_in, "q_prev": q_prev, "q": q.copy(), "step": (a * d).copy(),
                          "direction": d.copy(), "alpha": a, "rejected": rejected, "included": bool(included), "lam": lam, "mu": mu,
                          "f": f(q), "A": A(q), "A_prev": A0, "slope": slope, "g": gq, "residual": float(np.linalg.norm(phi(q) - y)),
                          "step_norm": float(np.linalg.norm(a * d)), "grad_norm": float(np.linalg.norm(gradA(q))),
                          "lam_estimate": lam + (2 * mu * gq if incl(q) else 0.0)})
        gq = gfun(q); lam_new = max(lam + 2 * mu * gq, 0.0)
        feas = max(gq, 0.0)
        stat = float(np.linalg.norm(J(q).T @ (phi(q) - y) + e * lam_new))
        comp = abs(lam_new * gq)
        rows.append(dict(outer=k, lam_before=lam, lam_after=lam_new, q=q.copy(), g=gq, n_inner=n_in, inner_status=st,
                         grad_norm=float(np.linalg.norm(gr)), feas=feas, stat=stat, comp=comp,
                         passes={"feas": bool(feas < FEAS_TOL), "stat": bool(stat < STAT_TOL), "comp": bool(comp < COMP_TOL)},
                         residual=float(np.linalg.norm(phi(q) - y)), f=f(q)))
        if on_event:
            on_event({"type": "outer_update", "outer": k, "inner": n_in, "q": q.copy(), "g": gq, "lam_before": lam, "lam_after": lam_new, "mu": mu,
                      "feas": feas, "stat": stat, "comp": comp, "residual": float(np.linalg.norm(phi(q) - y)), "f": f(q),
                      "inner_steps": n_in, "inner_status": st, "grad_norm": float(np.linalg.norm(gr)),
                      "included": bool(gq > 0 or lam_new > 0), "lam_estimate": lam_new})
        lam = lam_new
        if st == "stalled":
            status = "stalled"; break
        if feas < FEAS_TOL and stat < STAT_TOL and comp < COMP_TOL:
            status = "converged"; break
    return q, lam, status, rows


def free_answer(phi, J, q0, y):
    """One unconstrained answer: damped Gauss-Newton (60 full steps, rho = 1e-3) from the same start; what the stop prevents."""
    rows = gn.gauss_newton_ik(phi, J, q0, y, RHO, iters=60)
    q = rows[-1][1]
    return q, float(np.linalg.norm(phi(q) - y))


# ----------------------------------------------------------------------------------------------- the solver comparison
def solver_check(plant, phi, J, y, q0, j=JOINT, lim=Q4_MAX, rows=None):
    """SNOPT and IPOPT on the same objective and stop PLUS the URDF joint box: a custom cost 1/2|phi(q) - y*|^2 on an
    AutoDiff plant, the bounding-box row q_j <= lim and the box q_lo <= q <= q_hi. The box is not part of the loop's
    problem: it excludes the wrapped configuration (q_4 - 2 pi) that reaches the target, and without it IPOPT can
    wander to configurations many turns away and still report success. The solvers' agreement with the loop's answer
    from the same start therefore supports the loop's LOCAL result; it is not an equivalence of the two programs and
    not a global certificate. Returns per solver the exit status, the answer, the stop's dual with Drake's sign rule
    for an upper bound (lambda = -dual, an estimate), the largest box dual (inactive when small), the KKT triplet
    recomputed here and, for SNOPT, the sensitivity of f* to relaxing the stop by 1e-3 rad and the curvature a of f*
    along the stop (two re-solves at +-1e-2 rad), which predicts the contraction factor a/(a + 2 mu) of the violation
    per multiplier update (notes section 4.9). IPOPT documents a bound relaxation (bound_relax_factor, 1e-8 by
    default), hence its small positive violation of the stop."""
    plant_ad = plant.ToAutoDiffXd(); ctx_ad = plant_ad.CreateDefaultContext()
    ee_ad = plant_ad.GetFrameByName("panda_link8")
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()

    def cost_fn(qv):
        if qv.dtype == object:                                      # AutoDiffXd: the gradient through Drake's FK
            plant_ad.SetPositions(ctx_ad, qv)
            p = plant_ad.CalcRelativeTransform(ctx_ad, plant_ad.world_frame(), ee_ad).translation()
            dd = p - y
            return 0.5 * (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2])
        return 0.5 * float(np.sum((phi(qv) - y) ** 2))

    def solve(solver, limit, guess):
        prog = MathematicalProgram(); q = prog.NewContinuousVariables(7, "q")
        prog.AddCost(cost_fn, q)
        box = prog.AddBoundingBoxConstraint(lo, hi, q)
        stop = prog.AddBoundingBoxConstraint(-np.inf, limit, q[j])
        res = solver.Solve(prog, guess, None)
        return res, res.GetSolution(q), float(res.GetDualSolution(stop)[0]), float(np.abs(res.GetDualSolution(box)).max())

    e = np.zeros(7); e[j] = 1.0
    out = {}
    for name, solver in (("SNOPT", SnoptSolver()), ("IPOPT", IpoptSolver())):
        res, qs, dual, box_dual = solve(solver, lim, q0)
        lam = -dual                                                 # Drake: a negative dual on a bounding box marks the upper bound
        r = phi(qs) - y
        det = res.get_solver_details()
        exit_info = f"info {det.info}" if name == "SNOPT" else f"status {det.status} ({det.ConvertStatusToString()})"
        entry = dict(success=bool(res.is_success()), solution_result=str(res.get_solution_result()), exit=exit_info, options="solver defaults",
                     start="the ready pose", cost=float(res.get_optimal_cost()), q=qs.tolist(), q_j=float(qs[j]), residual=float(np.linalg.norm(r)),
                     dual=dual, lam=lam, lam_rule="lambda = -dual for an upper bound (Drake's bounding-box sign; an estimate)",
                     box_dual_max=box_dual, g=float(qs[j] - lim), feas=float(max(qs[j] - lim, 0.0)),
                     stat=float(np.linalg.norm(J(qs).T @ r + e * lam)), comp=float(abs(lam * (qs[j] - lim))),
                     within_hardware_limits=bool(np.all(qs >= lo - 1e-9) and np.all(qs <= hi + 1e-9)),
                     program="the loop's objective and stop plus the URDF joint box (not in the loop's problem); agreement supports the loop's local result from the same start, not program equivalence")
        if name == "IPOPT":
            entry["note"] = "IPOPT relaxes bounds by bound_relax_factor (1e-8 by default; https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_bound_relax_factor), hence a small positive violation"
        if name == "SNOPT":
            delta, d2 = 1e-3, 1e-2
            res2, _, _, _ = solve(solver, lim + delta, qs)
            entry["sensitivity"] = dict(delta=delta, f_star=float(res.get_optimal_cost()), f_relaxed=float(res2.get_optimal_cost()),
                                        change=float(res2.get_optimal_cost() - res.get_optimal_cost()), minus_lam_delta=float(-lam * delta))
            rp, _, _, _ = solve(solver, lim + d2, qs); rm, _, _, _ = solve(solver, lim - d2, qs)
            a = float((rp.get_optimal_cost() - 2 * res.get_optimal_cost() + rm.get_optimal_cost()) / d2 ** 2)
            entry["contraction"] = dict(delta=d2, a=a, predicted_ratio=float(a / (a + 2 * MU)),
                                        observed_ratios=[float(rows[i + 1]["g"] / rows[i]["g"]) for i in range(len(rows) - 1)] if rows and len(rows) > 1 and all(r_["g"] > 0 for r_ in rows) else [])
        out[name] = entry
    return out


# ----------------------------------------------------------------------------------------------- geometry helpers
def make_protractor(plant, pc, Panda):
    """World pose of the elbow joint's parent frame F, fixed to panda_link3 (the protractor's disc is F's x-y plane,
    perpendicular to the joint axis, F's z; F's x axis rotated by q_4 about that axis is link 4's x axis), the
    pointer tip, the hardware range and the constant pose of F in link 3."""
    joint = plant.GetJointByName(JOINT_NAME, Panda)
    F = joint.frame_on_parent()
    lo, hi = float(joint.position_lower_limits()[0]), float(joint.position_upper_limits()[0])
    X_L3F = F.GetFixedPoseInBodyFrame().GetAsMatrix4().tolist()

    def pose(q):
        plant.SetPositions(pc, Panda, q)
        return plant.CalcRelativeTransform(pc, plant.world_frame(), F)

    def tip(q, r=PROTRACTOR_R):
        X = pose(q)
        return (X @ np.array([r * np.cos(q[JOINT]), r * np.sin(q[JOINT]), 0.0])).tolist()

    return pose, tip, (lo, hi), X_L3F


def source_info():
    here = Path(__file__).resolve().parent
    files = {}
    for rel in SOURCE_FILES:
        p = (here / rel).resolve()
        files[str(p.relative_to(here.parent.parent))] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--"] + [str((here / r).resolve()) for r in SOURCE_FILES],
                                    capture_output=True, text=True, cwd=here).stdout.strip())
    except Exception:
        git, dirty = "unknown", True
    return {"files": files, "git": git, "dirty": dirty}


# ----------------------------------------------------------------------------------------------- record
def record_runs(plant, pc, Panda, phi, J, with_checks=True):
    origins = gn.make_origins(plant, pc, Panda)
    pose, tip, hw, X_L3F = make_protractor(plant, pc, Panda)
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    outside = lambda q: [jj + 1 for jj in range(7) if q[jj] < lo[jj] - 1e-9 or q[jj] > hi[jj] + 1e-9]
    from importlib.metadata import version
    rec = {"schema": "cs549-trace/1",
           "problem": "constrained IK: min 1/2|phi(q) - y*|^2 s.t. g(q) = q_4 - q_4max <= 0 on the Panda (one elbow stop, a software limit chosen for this demonstration)",
           "stage": "B1 (IK with one hard constraint, the squared task-error objective kept)",
           "platform": f"drake {version('drake')}",
           "solver": f"course augmented Lagrangian (mu = {MU:g} fixed, lambda <- max(0, lambda + 2 mu g)), damped Gauss-Newton inner loop "
                     f"(rho = {RHO:g}, Armijo c = {ARMIJO_C}), NumPy on Drake FK and Jacobian; drake/04-constrained-ik-al/constrained_ik_panda.py",
           "event_source": "own-loop hook",
           "estimator": "intermediate multipliers are the loop's own estimates: lambda + 2 mu g at the end of each inner solve, which the multiplier update stores (notes section 4.9, eq. 23)",
           "timeline": "the slider walks accepted joint updates and multiplier updates only (every Step changes q or lambda, never both); rejected Armijo trials attach to their accepted update behind a toggle",
           "limitations": ["intermediate multipliers are the loop's own estimates (lambda + 2 mu g after each inner solve)",
                           "the inner solve stops at |grad A| < 1e-7 (60 steps at most) and the outer loop on the KKT triplet (1e-6 rad, 1e-6 m^2/rad, 1e-8 m^2), looser than the two-link code: below an inner gradient of about 2e-9 the Armijo test on A is decided by rounding",
                           "SNOPT and IPOPT are a comparison of the final answers only, on the same objective and stop plus the URDF joint box (inactive at every answer); their stop duals are reported as lambda with Drake's sign rule and are estimates; no claim that the configurations coincide",
                           "the hardware joint limits are not constraints of the loop's problem; every displayed configuration is checked against them and the exceptions are listed per run (hardware_exceptions)",
                           "most inner steps after the first overshoot move the flange by less than a millimetre: the per-outer table and the plots carry what the scene cannot show",
                           "the shortened steps and rejected trials of run A come mostly from the task term's curvature at a 4.3 cm residual (as in Deck 04's run 3b), not from the stop's term",
                           "run A's answer is a local constrained result reached from the ready pose, not a proof that the target is out of reach: the loop's problem has only the upper bound and joint angles are periodic, so the unconstrained answer with one full turn subtracted from q_4 satisfies the bound and reaches the target (outside the hardware range); the comparison program's box excludes that configuration"],
           "units": {"q": "rad", "p": "m", "residual": "m", "g": "rad", "lam": "m^2/rad (units of f per rad)", "mu": "m^2/rad^2", "f": "m^2", "A": "m^2",
                     "stat": "m^2/rad", "comp": "m^2", "feas": "rad", "step_norm": "rad", "grad_norm": "m^2/rad"},
           "tolerances": {"inner_grad": INNER_TOL, "feasibility_and_activity": FEAS_TOL, "stationarity": STAT_TOL, "complementarity": COMP_TOL,
                          "complementarity_rule": "approximately the multiplier magnitude times the feasibility tolerance"},
           "hardware_ranges": [[float(a), float(b)] for a, b in zip(lo, hi)],
           "model": "package://drake_models/franka_description/urdf/panda_arm.urdf, panda_link8 as the flange",
           "joint": {"index": JOINT, "name": JOINT_NAME, "display": "q_4", "limit": Q4_MAX, "side": "upper (q_4 <= limit): the elbow may not straighten past the stop",
                     "hardware_range": list(hw), "X_L3F": X_L3F, "note": "the stop is a software limit chosen for this demonstration, inside the hardware range; in Deck 05's example (b) the limit kept the two-link elbow from bending further"},
           "method": {"mu": MU, "rho": RHO, "inner_tol": INNER_TOL, "inner_max": INNER_MAX, "feas_tol": FEAS_TOL, "stat_tol": STAT_TOL,
                      "comp_tol": COMP_TOL, "outer_max": OUTER_MAX, "armijo_c": ARMIJO_C, "armijo_floor": ARMIJO_FLOOR},
           "q0": gn.Q_READY.tolist(), "protractor_radius": PROTRACTOR_R, "source": source_info(), "runs": {}}
    for key, spec in RUNS.items():
        y = np.array(spec["target"])
        events, trials = [], []

        def snap(q):
            return {"q": np.array(q).tolist(), "p": phi(q).tolist(), "origins": origins(q), "elbow": pose(q).translation().tolist(),
                    "axis": pose(q).rotation().matrix()[:, 2].tolist(), "pointer": tip(q), "outside_hardware": outside(np.array(q))}

        g0 = float(gn.Q_READY[JOINT] - Q4_MAX)
        events.append({"e": 0, "type": "start", "outer": 0, "inner": 0, "lam": 0.0, "mu": MU, "residual": float(np.linalg.norm(phi(gn.Q_READY) - y)),
                       "g": g0, "state": state_of(g0), "included": False, "lam_estimate": 0.0, **snap(gn.Q_READY)})

        def on_event(ev):
            if ev["type"] == "trial_evaluation":
                trials.append({"outer": ev["outer"], "inner": ev["inner"], "alpha": ev["alpha"], "q": ev["q"].tolist(), "p": phi(ev["q"]).tolist(),
                               "A": ev["A"], "included": ev["included"], "outside_hardware": outside(ev["q"])})
                return
            out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ev.items()}
            out["e"] = len(events); out["state"] = state_of(ev["g"])
            out.update(snap(ev["q"]))
            events.append(out)

        q, lam, status, rows = al_ik(phi, J, gn.Q_READY, y, on_event=on_event)
        q_free, r_free = free_answer(phi, J, gn.Q_READY, y)
        run = {"label": spec["label"], "note": spec["note"], "target": y.tolist(), "outcome": status, "n_outer": len(rows), "n_events": len(events) - 1,
               "n_inner": sum(r["n_inner"] for r in rows), "n_trials": len(trials),
               "outer_events": [ev["e"] for ev in events if ev["type"] == "outer_update"],
               "final": {"q": q.tolist(), "lam": lam, "residual": float(np.linalg.norm(phi(q) - y)), "g": float(q[JOINT] - Q4_MAX), "state": state_of(float(q[JOINT] - Q4_MAX)),
                         "outside_hardware": outside(q)},
               "free": {"label": "one unconstrained answer: damped Gauss-Newton from the same start, 60 full steps", "q": q_free.tolist(), "residual": r_free,
                        "g": float(q_free[JOINT] - Q4_MAX), "p": phi(q_free).tolist(), "origins": origins(q_free), "outside_hardware": outside(q_free)},
               "wrapped": {"label": "the unconstrained answer with one full turn subtracted from q_4: the same pose, inside the stop, outside the hardware range",
                           "q": (q_free - 2 * np.pi * np.eye(7)[JOINT]).tolist(), "q_j": float(q_free[JOINT] - 2 * np.pi), "satisfies_stop": bool(q_free[JOINT] - 2 * np.pi <= Q4_MAX),
                           "residual": float(np.linalg.norm(phi(q_free - 2 * np.pi * np.eye(7)[JOINT]) - y)), "outside_hardware": outside(q_free - 2 * np.pi * np.eye(7)[JOINT])},
               "rows": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in rows],
               "events": events, "trial_evaluations": trials}
        run["hardware_exceptions"] = ([f"event {ev['e']} (outer {ev['outer']}, inner {ev['inner']}): joints {ev['outside_hardware']}" for ev in events if ev["outside_hardware"]]
                                      + [f"rejected trial (outer {t['outer']}, inner {t['inner']}, alpha {t['alpha']:g}): joints {t['outside_hardware']}" for t in trials if t["outside_hardware"]]
                                      + ([f"the unconstrained answer: joints {run['free']['outside_hardware']}"] if run["free"]["outside_hardware"] else []))
        if with_checks:
            run["check"] = solver_check(plant, phi, J, y, gn.Q_READY, rows=rows)
            for name in ("SNOPT", "IPOPT"):
                if not run["check"][name]["within_hardware_limits"]:
                    run["hardware_exceptions"].append(f"the {name} answer: outside the hardware range")
        rec["runs"][key] = run
    return rec


# ----------------------------------------------------------------------------------------------- print mode
def print_mode():
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    print("=== Constrained IK on the Panda: the course's augmented Lagrangian with one elbow stop ===")
    print(f"problem: min 1/2|phi(q) - y*|^2  s.t.  g(q) = q_4 - ({Q4_MAX}) <= 0 (the elbow may not straighten past the stop, a software limit for this demonstration);"
          f"  hardware range of joint 4 [{lo[JOINT]:.4f}, {hi[JOINT]:.4f}] rad;  start = ready pose (q_4 = {gn.Q_READY[JOINT]:.3f} rad, inside)")
    print(f"method: mu = {MU:g} m^2/rad^2 fixed, rho = {RHO:g}; inner stop |grad A| < {INNER_TOL:g} ({INNER_MAX} steps at most); outer stop max(g,0) < {FEAS_TOL:g} rad,"
          f" |grad f + lambda grad g| < {STAT_TOL:g} m^2/rad, |lambda g| < {COMP_TOL:g} m^2 ({OUTER_MAX} at most); outer iterations numbered from 0")
    rec = record_runs(plant, pc, Panda, phi, J, with_checks=True)
    for key, run in rec["runs"].items():
        print(f"\n--- {run['label']} ---")
        fr = run["free"]
        print(f"one unconstrained answer (damped Gauss-Newton, 60 full steps, no stop): task error {fr['residual']:.1e} m, q_4 = {fr['q'][JOINT]:+.4f} rad"
              f" ({fr['g']:+.4f} rad relative to the stop){'' if not fr['outside_hardware'] else '; OUTSIDE the hardware range: joints ' + str(fr['outside_hardware'])}")
        w = run["wrapped"]
        if w["satisfies_stop"] and run["final"]["residual"] > 1e-3 and w["residual"] < 1e-6:     # only where the loop's answer misses the target
            print(f"the same pose with one full turn subtracted from q_4 (q_4 = {w['q_j']:+.4f} rad) satisfies the stop and has task error {w['residual']:.1e} m, outside the hardware range"
                  f" (joints {w['outside_hardware']}): the loop's answer below is a LOCAL constrained result reached from the ready pose, not a proof that the target is out of reach")
        print("outer  lambda before   inner steps (status)   q_4 (rad)    g (rad)      lambda after   task error (m)   feas (<1e-6)  stat (<1e-6)  comp (<1e-8)")
        for r in run["rows"]:
            mark = lambda name: "pass" if r["passes"][name] else "fail"
            print(f"  {r['outer']:<4} {r['lam_before']:<15.7f} {r['n_inner']:>3} ({r['inner_status']:<10})     {r['q'][JOINT]:+.5f}    {r['g']:+.2e}    {r['lam_after']:<14.7f} {r['residual']:.4e}       "
                  f"{r['feas']:.1e} {mark('feas')}  {r['stat']:.1e} {mark('stat')}  {r['comp']:.1e} {mark('comp')}")
        fin = run["final"]
        print(f"{run['outcome']} after {run['n_outer']} outer iterations ({run['n_inner']} accepted inner steps, {run['n_trials']} rejected trials); {run['n_events']} events on the timeline")
        print(f"q* = {np.array(fin['q'])} rad;  task error {fin['residual']:.6f} m;  q_4 = {fin['q'][JOINT]:+.7f} ({fin['state']}, g = {fin['g']:+.1e} rad);"
              f"  lambda* = {fin['lam']:.7f} m^2/rad")
        print("displayed configurations outside the hardware range: " + ("; ".join(run["hardware_exceptions"]) if run["hardware_exceptions"] else "none"))
        for name in ("SNOPT", "IPOPT"):
            c = run["check"][name]
            print(f"compared with {name} (same objective and stop plus the URDF box; {c['exit']}, {c['solution_result']}): task error {c['residual']:.6f} m, q_4 = {c['q_j']:+.7f},"
                  f" stop dual {c['dual']:+.7f} -> lambda = {c['lam']:.7f} (estimate); largest box dual {c['box_dual_max']:.1e};"
                  f" feas {c['feas']:.1e}, stationarity {c['stat']:.1e}, complementarity {c['comp']:.1e}; within the hardware range: {'yes' if c['within_hardware_limits'] else 'NO'}")
        s = run["check"]["SNOPT"]["sensitivity"]; ct = run["check"]["SNOPT"]["contraction"]
        print(f"sensitivity (SNOPT): relaxing the stop by {s['delta']:g} rad changes f* from {s['f_star']:.6e} to {s['f_relaxed']:.6e} ({s['change']:+.3e} m^2);"
              f"  -lambda * delta = {s['minus_lam_delta']:+.3e} m^2")
        if ct["observed_ratios"]:
            print(f"contraction: curvature of f* along the stop a = {ct['a']:.4f} m^2/rad^2 (re-solves at +-{ct['delta']:g} rad); predicted factor of successive violations a/(a + 2 mu) = {ct['predicted_ratio']:.4f}"
                  f" (a {1 / ct['predicted_ratio']:.0f}-fold reduction); observed {', '.join(f'{x:.4f}' for x in ct['observed_ratios'])}")
    print("\nEvents per run (start + accepted inner steps + multiplier updates):",
          ", ".join(f"{k}: {run['n_events']} ({run['n_inner']} inner, {run['n_outer']} outer, {run['n_trials']} rejected trials)" for k, run in rec["runs"].items()))


def record_mode(out):
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    rec = record_runs(plant, pc, Panda, phi, J)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{k}: {run['n_events']} events" for k, run in rec["runs"].items()))


# ----------------------------------------------------------------------------------------------- self-test
EXPECTED = {"runA": {"n_outer": 4, "n_inner": 27, "n_trials": 26, "outcome": "converged"}, "runB": {"n_outer": 1, "n_inner": 5, "n_trials": 0, "outcome": "converged"}}


def selftest_mode(baseline=None):
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    pose, tip, hw, X_L3F = make_protractor(plant, pc, Panda)
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    fails = 0

    def check(msg, ok, info=""):
        nonlocal fails
        print(("  ok    " if ok else "  FAIL  ") + msg + ("" if ok or not info else f"  ::  {info}"))
        fails += 0 if ok else 1

    rec = record_runs(plant, pc, Panda, phi, J, with_checks=True)
    src = source_info()
    check("the record's source hashes match the files on disk (" + ("dirty" if src["dirty"] else "clean") + f" at {src['git']})", rec["source"]["files"] == src["files"])
    e4 = np.zeros(7); e4[JOINT] = 1.0
    for key, run in rec["runs"].items():
        y = np.array(run["target"]); events = run["events"]; exp = EXPECTED[key]
        q_plain, lam_plain, st_plain, rows_plain = al_ik(phi, J, gn.Q_READY, y)     # no hook
        check(f"{key}: the hook does not change the run (q bit-identical, lambda {lam_plain:.7f}, status {st_plain})",
              np.array_equal(np.array(run["final"]["q"]), q_plain) and run["final"]["lam"] == lam_plain and run["outcome"] == st_plain)
        check(f"{key}: counts as documented: {run['n_outer']} outer, {run['n_inner']} inner, {run['n_trials']} rejected trials, {run['outcome']}",
              all(run[k_] == v for k_, v in exp.items()), str({k_: run[k_] for k_ in exp}))
        check(f"{key}: event indices are contiguous and the start is event 0", [ev["e"] for ev in events] == list(range(len(events))) and events[0]["type"] == "start")
        f = lambda qq: 0.5 * float(np.sum((phi(qq) - y) ** 2))
        g = lambda qq: float(qq[JOINT] - Q4_MAX)
        worst = {"q": 0.0, "A": 0.0, "grad": 0.0, "dir": 0.0, "fk": 0.0, "pointer": 0.0, "est": 0.0}
        armijo_ok, order_ok, outer_ok, state_ok, hw_ok = True, True, True, True, True
        prev_q = np.array(events[0]["q"]); lam_cur = 0.0; k_cur = 0
        for ev in events:
            q = np.array(ev["q"])
            state_ok &= ev["state"] == state_of(g(q)) and abs(ev["g"] - g(q)) < 1e-15
            hw_ok &= ev["outside_hardware"] == [jj + 1 for jj in range(7) if q[jj] < lo[jj] - 1e-9 or q[jj] > hi[jj] + 1e-9]
        for ev in events[1:]:
            q = np.array(ev["q"])
            worst["fk"] = max(worst["fk"], abs(np.linalg.norm(phi(q) - y) - ev["residual"]), np.linalg.norm(np.array(ev["p"]) - phi(q)))
            worst["pointer"] = max(worst["pointer"], np.linalg.norm(np.array(ev["pointer"]) - np.array(tip(q))))
            if ev["type"] == "accepted_update":
                order_ok &= ev["outer"] == k_cur and ev["lam"] == lam_cur
                qp = np.array(ev["q_prev"]); step = np.array(ev["step"]); dirn = np.array(ev["direction"])
                order_ok &= np.array_equal(qp, prev_q)
                worst["q"] = max(worst["q"], np.linalg.norm(qp + step - q), np.linalg.norm(ev["alpha"] * dirn - step))
                lam, mu = ev["lam"], ev["mu"]
                incl = lambda qq: (g(qq) > 0) or (lam > 0)
                A = lambda qq: f(qq) + lam * g(qq) + (mu * g(qq) ** 2 if incl(qq) else 0.0)
                gradA = lambda qq: J(qq).T @ (phi(qq) - y) + e4 * (lam + (2 * mu * g(qq) if incl(qq) else 0.0))
                worst["A"] = max(worst["A"], abs(A(q) - ev["A"]), abs(f(q) - ev["f"]), abs(A(qp) - ev["A_prev"]))
                worst["grad"] = max(worst["grad"], abs(np.linalg.norm(gradA(q)) - ev["grad_norm"]), abs(float(gradA(qp) @ dirn) - ev["slope"]))
                worst["est"] = max(worst["est"], abs(ev["lam_estimate"] - (lam + (2 * mu * g(q) if incl(q) else 0.0))))
                state_ok &= ev["included"] == incl(qp)
                Jp = J(qp); H = Jp.T @ Jp + (2 * mu * np.outer(e4, e4) if incl(qp) else 0.0) + RHO * np.eye(7)
                worst["dir"] = max(worst["dir"], np.linalg.norm(H @ dirn + gradA(qp)))
                armijo_ok &= A(q) <= A(qp) + ARMIJO_C * ev["alpha"] * float(gradA(qp) @ dirn) + 1e-15
                prev_q = q
            elif ev["type"] == "outer_update":
                outer_ok &= np.array_equal(q, prev_q) and ev["outer"] == k_cur and ev["lam_before"] == lam_cur
                lam_new = max(ev["lam_before"] + 2 * ev["mu"] * g(q), 0.0)
                outer_ok &= abs(lam_new - ev["lam_after"]) < 1e-15 and ev["lam_estimate"] == ev["lam_after"]
                outer_ok &= abs(ev["feas"] - max(g(q), 0.0)) < 1e-15 and abs(ev["stat"] - np.linalg.norm(J(q).T @ (phi(q) - y) + e4 * lam_new)) < 1e-12 and abs(ev["comp"] - abs(lam_new * g(q))) < 1e-15
                outer_ok &= ev["included"] == ((g(q) > 0) or (lam_new > 0))
                lam_cur = ev["lam_after"]; k_cur += 1
        check(f"{key}: inner events: q = q_prev + step, step = alpha * direction (worst {worst['q']:.1e})", worst["q"] < 1e-12)
        check(f"{key}: inner events: f, A, A_prev, |grad A|, the slope and the multiplier estimate recomputed from FK and the Jacobian (worst {max(worst['A'], worst['grad'], worst['est']):.1e})",
              max(worst["A"], worst["grad"], worst["est"]) < 1e-12)
        check(f"{key}: inner events: the direction solves the damped system H d = -grad A at q_prev (worst residual {worst['dir']:.1e})", worst["dir"] < 1e-10)
        check(f"{key}: inner events: every accepted step passes the Armijo test with c = {ARMIJO_C}", armijo_ok)
        check(f"{key}: events are in algorithm order (outer counter, lambda in force, q_prev chains)", order_ok)
        check(f"{key}: outer events: q unchanged, lambda <- max(0, lambda + 2 mu g), the KKT triplet and the inclusion flag recomputed", outer_ok)
        check(f"{key}: the state (violated / on the stop / inside, tolerance {FEAS_TOL:g} rad), the inclusion flag and the hardware flags recomputed for every event", state_ok and hw_ok)
        check(f"{key}: FK, flange and protractor pointer recomputed (worst {max(worst['fk'], worst['pointer']):.1e} m)", max(worst["fk"], worst["pointer"]) < 1e-12)
        tr_ok = True
        for t in run["trial_evaluations"]:
            inner = [ev for ev in events if ev["type"] == "accepted_update" and ev["outer"] == t["outer"] and ev["inner"] == t["inner"]]
            tr_ok &= len(inner) == 1 and ARMIJO_FLOOR < t["alpha"] <= 1.0
            if inner:
                ev = inner[0]; lam, mu = ev["lam"], ev["mu"]; qp = np.array(ev["q_prev"]); dirn = np.array(ev["direction"])
                incl = lambda qq: (g(qq) > 0) or (lam > 0)
                A = lambda qq: f(qq) + lam * g(qq) + (mu * g(qq) ** 2 if incl(qq) else 0.0)
                gradA = lambda qq: J(qq).T @ (phi(qq) - y) + e4 * (lam + (2 * mu * g(qq) if incl(qq) else 0.0))
                qt = np.array(t["q"])
                tr_ok &= np.linalg.norm(qt - (qp + t["alpha"] * dirn)) < 1e-12 and abs(A(qt) - t["A"]) < 1e-12 and t["included"] == incl(qt)
                tr_ok &= not (A(qt) <= A(qp) + ARMIJO_C * t["alpha"] * float(gradA(qp) @ dirn))        # it was rejected
        check(f"{key}: {run['n_trials']} rejected trials are tied to an inner step, lie on its direction and fail the Armijo test", tr_ok)
        fin = run["final"]
        check(f"{key}: final answer {fin['state']} (g = {fin['g']:+.1e} rad), lambda* = {fin['lam']:.7f}; hardware exceptions listed: {len(run['hardware_exceptions'])}",
              fin["state"] in ("on the stop", "inside") and not fin["outside_hardware"] and (len(run["hardware_exceptions"]) == (1 if key == "runA" else 0)))
        w = run["wrapped"]; qw = np.array(w["q"])
        check(f"{key}: the wrapped unconstrained answer (q_4 - 2 pi = {w['q_j']:+.4f}) recomputed: same pose (task error {np.linalg.norm(phi(qw) - y):.1e} m), stop {'satisfied' if w['satisfies_stop'] else 'violated'}, outside the hardware range {w['outside_hardware']}",
              abs(np.linalg.norm(phi(qw) - y) - w["residual"]) < 1e-15 and w["satisfies_stop"] == (qw[JOINT] <= Q4_MAX) and (key != "runA" or (w["satisfies_stop"] and w["residual"] < 1e-6 and 4 in w["outside_hardware"])))
        for name, tol_lam, tol_g in (("SNOPT", 1e-6, 1e-9), ("IPOPT", 1e-7, 1e-7)):
            c = run["check"][name]
            r_s = phi(np.array(c["q"])) - y
            stat_s = np.linalg.norm(J(np.array(c["q"])).T @ r_s + e4 * c["lam"])
            check(f"{key}: {name} ({c['exit']}) solved the same program within the box (task error {c['residual']:.6f} vs {fin['residual']:.6f} m; lambda {c['lam']:.7f} vs {fin['lam']:.7f}, tolerance {tol_lam:g}; g {c['g']:+.1e} <= {tol_g:g}; box dual {c['box_dual_max']:.1e}; stationarity {stat_s:.1e})",
                  c["success"] and abs(c["residual"] - fin["residual"]) < 1e-5 and abs(c["lam"] - fin["lam"]) < tol_lam and c["g"] <= tol_g and stat_s < 1e-5 and c["within_hardware_limits"] and c["box_dual_max"] < 1e-5)
        s = run["check"]["SNOPT"]["sensitivity"]; ct = run["check"]["SNOPT"]["contraction"]
        if fin["lam"] > 1e-6:
            check(f"{key}: sensitivity: f* changes by {s['change']:+.3e} for a {s['delta']:g} rad relaxation; -lambda delta = {s['minus_lam_delta']:+.3e} (within 5%)",
                  abs(s["change"] - s["minus_lam_delta"]) < 0.05 * abs(s["minus_lam_delta"]))
            check(f"{key}: contraction: predicted a/(a + 2 mu) = {ct['predicted_ratio']:.4f} with a = {ct['a']:.4f}; observed violation ratios {', '.join(f'{x:.4f}' for x in ct['observed_ratios'])} (within 10%)",
                  len(ct["observed_ratios"]) >= 2 and all(abs(x - ct["predicted_ratio"]) < 0.1 * ct["predicted_ratio"] for x in ct["observed_ratios"]))
        else:
            check(f"{key}: lambda* = 0 and the row was never included: the loop is plain damped Gauss-Newton here (task error {fin['residual']:.1e} m, f* change {s['change']:+.1e})",
                  fin["lam"] == 0.0 and not any(ev.get("included") for ev in events) and fin["residual"] < 1e-6 and abs(s["change"]) < 1e-12)
    ref = Path(__file__).resolve().parents[2] / "planning/optimization-step-visuals-2026-10-14/increment2/reference-iterates.json"
    if ref.exists():
        R = json.loads(ref.read_text())
        for key, run in rec["runs"].items():
            qs = np.array([ev["q"] for ev in run["events"]]); lams = np.array([ev.get("lam_after", ev.get("lam", 0.0)) for ev in run["events"]])
            rq, rl = np.array(R[key]["q"]), np.array(R[key]["lam"])
            same = qs.shape == rq.shape and np.abs(qs - rq).max() < 1e-12 and np.abs(lams - rl).max() < 1e-12
            check(f"{key}: the {len(qs)} events agree with the independent reference implementation (increment2/reference_al.py) to 1e-12 (max |diff| {np.abs(qs - rq).max() if qs.shape == rq.shape else float('nan'):.1e})", same)
    else:
        print(f"  note  no reference file ({ref}); the reference comparison was skipped")
    if baseline and Path(baseline).exists():
        base = json.loads(Path(baseline).read_text())
        for key, run in rec["runs"].items():
            qs = np.array([ev["q"] for ev in run["events"]]); lams = np.array([ev.get("lam_after", ev.get("lam", 0.0)) for ev in run["events"]])
            bq = np.array(base[key]["q"]); bl = np.array(base[key]["lam"])
            same = qs.shape == bq.shape and np.array_equal(qs, bq) and np.array_equal(lams, bl)
            check(f"{key}: the {len(qs)} events equal the saved regression baseline at full precision (max |diff| {np.abs(qs - bq).max() if qs.shape == bq.shape else float('nan'):.1e})", same)
    else:
        print(f"  note  no baseline file ({baseline}); the baseline comparison was skipped")
    print(f"{fails} failures")
    return fails


# ----------------------------------------------------------------------------------------------- figure
def figure_mode(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    rec = record_runs(plant, pc, Panda, phi, J, with_checks=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    colors = {"runA": "#7a1f3d", "runB": "#1f6f8b"}
    for key, run in rec["runs"].items():
        ev = run["events"]; e = np.array([x["e"] for x in ev]); c = colors[key]
        lab = key.replace("run", "run ")
        axes[0, 0].semilogy(e, [x["residual"] for x in ev], "-o", ms=3, color=c, label=lab)
        axes[0, 1].semilogy(e, [max(x["g"], 1e-12) for x in ev], "-o", ms=3, color=c, label=lab)
        lam = [x.get("lam_after", x.get("lam", 0.0)) for x in ev]
        axes[1, 0].step(e, lam, where="post", color=c, label=lab)
        inner = [x for x in ev if x["type"] == "accepted_update"]
        axes[1, 1].semilogy([x["e"] for x in inner], [x["step_norm"] for x in inner], "-o", ms=3, color=c, label=lab)
        for ax in axes.flat:
            for oe in run["outer_events"]:
                ax.axvline(oe, color=c, alpha=0.25, lw=1)
    axes[0, 0].set(title="task error |phi(q^(e)) - y*| per event", xlabel="event e", ylabel="m")
    axes[0, 1].set(title="violation max(g, 0), g = q_4 - q_4max\nper event (1e-12 means 0 or below)", xlabel="event e", ylabel="rad")
    axes[1, 0].set(title="multiplier lambda in force after each event [m^2/rad]", xlabel="event e", ylabel="lambda")
    axes[1, 1].set(title="accepted inner step |q^(e) - q^(e-1)| (inner events only)", xlabel="event e", ylabel="rad")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Constrained IK on the Panda, the course's augmented Lagrangian with one elbow stop (q_4 <= -2.2 rad); vertical lines mark the multiplier updates", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------- replay
def replay_mode(app=False, app_port=0):
    """Paused replay of both runs, one event at a time: the robot at the event's q; for an inner step the faded
    ghost at the previous iterate; for a multiplier update nothing moves and the readouts change. Nothing advances
    by itself. With app=True a web page (panel.html, served by the Gauss-Newton demo's replay_panel.py) wraps the
    Meshcat view with the explanation, readouts, the per-outer table, plots and the same controls."""
    d0, plant0, pc0, ctx0, Panda0, ee0 = gn.build()
    phi, J = gn.make_fk(plant0, pc0, Panda0, ee0)
    rec = record_runs(plant0, pc0, Panda0, phi, J, with_checks=True)
    pose, tip, hw, X_L3F = make_protractor(plant0, pc0, Panda0)
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    (Panda,) = parser.AddModelsFromUrl(gn.PANDA_URL)
    parser.SetAutoRenaming(True)
    (Ghost,) = parser.AddModelsFromUrl(gn.PANDA_URL)
    for m in (Panda, Ghost):
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", m))
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext(); pc = plant.GetMyContextFromRoot(ctx)
    ghost_path = f"/drake/visualizer/{plant.GetModelInstanceName(Ghost)}"
    gold, maroon, grey, red, green = Rgba(0.8, 0.6, 0.0, 0.55), Rgba(0.55, 0.1, 0.25, 1.0), Rgba(0.45, 0.45, 0.45, 0.9), Rgba(0.85, 0.15, 0.1, 0.9), Rgba(0.1, 0.6, 0.2, 0.9)
    black = Rgba(0.1, 0.1, 0.1, 1.0)
    meshcat.SetObject("/replay/target", Sphere(0.012), gold)
    meshcat.SetObject("/replay/flange", Sphere(0.007), maroon)
    # the protractor lives in the elbow joint's parent frame F (fixed to the upper arm, panda_link3): beads along the
    # allowed arc (larger, green) and the forbidden arc (smaller, red) over the hardware range, the stop, the pointer
    for n, th in enumerate(np.arange(hw[0], Q4_MAX + 1e-9, 0.04)):
        meshcat.SetObject(f"/replay/protractor/allowed/{n}", Sphere(0.006), green)
        meshcat.SetTransform(f"/replay/protractor/allowed/{n}", RigidTransform([PROTRACTOR_R * np.cos(th), PROTRACTOR_R * np.sin(th), 0.0]))
    for n, th in enumerate(np.arange(Q4_MAX + 0.04, hw[1] + 1e-9, 0.04)):
        meshcat.SetObject(f"/replay/protractor/forbidden/{n}", Sphere(0.0045), red)
        meshcat.SetTransform(f"/replay/protractor/forbidden/{n}", RigidTransform([PROTRACTOR_R * np.cos(th), PROTRACTOR_R * np.sin(th), 0.0]))
    stop_dir = np.array([np.cos(Q4_MAX), np.sin(Q4_MAX), 0.0])
    meshcat.SetLine("/replay/protractor/stop", np.column_stack([0.5 * PROTRACTOR_R * stop_dir, 1.25 * PROTRACTOR_R * stop_dir]), 4.0, Rgba(0.55, 0.1, 0.25, 1.0))
    meshcat.SetObject("/replay/protractor/stop_bead", Sphere(0.009), maroon)
    meshcat.SetTransform("/replay/protractor/stop_bead", RigidTransform(1.25 * PROTRACTOR_R * stop_dir))
    meshcat.SetObject("/replay/protractor/pointer_bead", Sphere(0.008), black)
    keys = list(rec["runs"])
    state = {"run": "runA", "e": 0, "trials": False, "free": False, "closeup": False, "camera": None, "applied_action_id": 0}
    wide_pts, wide = {}, {}
    for k in keys:
        run = rec["runs"][k]; pts = [[0.0, 0.0, 0.0], run["target"], run["free"]["p"]] + list(run["free"]["origins"].values())
        for ev in run["events"]:
            pts += list(ev["origins"].values()) + [ev["p"], ev["pointer"]]
            el = np.array(ev["elbow"]); pts += [(el + np.array([0.2, 0, 0])).tolist(), (el - np.array([0.2, 0, 0])).tolist(), (el + np.array([0, 0, 0.2])).tolist(), (el - np.array([0, 0, 0.2])).tolist()]
        wide_pts[k] = pts; wide[k] = gn.fit_camera(pts)
    panel = None
    if app:
        from replay_panel import ReplayPanel                         # drake/02-gauss-newton-ik/replay_panel.py (on sys.path through gn)
        panel = ReplayPanel(meshcat.web_url(), rec, port=app_port, page=Path(__file__).with_name("panel.html"))

    def set_camera(eye, target):
        meshcat.SetCameraPose(eye, target); state["camera"] = [np.asarray(eye).tolist(), np.asarray(target).tolist()]

    def set_toggle(name, on):
        state[name] = on
        if not on and not state["closeup"] and name == "trials":
            set_camera(*wide[state["run"]])

    SLIDER, SLIDER_MAX = "event e", max(run["n_events"] for run in rec["runs"].values())
    buttons = ["Step", "Back", "Reset", "Next outer", "Previous outer"] + [rec["runs"][k]["label"] for k in keys] + ["Show / hide rejected trials", "Show / hide the unconstrained answer", "Stop"]
    meshcat.AddSlider(SLIDER, 0, SLIDER_MAX, 1, 0)
    for b in buttons:
        meshcat.AddButton(b)
    clicks = {b: 0 for b in buttons}
    readouts = {"labels": []}

    def refresh_readouts(labels):
        if labels != readouts["labels"]:
            for old in readouts["labels"]:
                meshcat.DeleteButton(old)
            for lab in labels:
                meshcat.AddButton(lab)
            readouts["labels"] = labels

    def load_run(key):
        run = rec["runs"][key]
        meshcat.SetTransform("/replay/target", RigidTransform(np.array(run["target"])))
        meshcat.Delete("/replay/free"); meshcat.Delete("/replay/trials")
        state["run"], state["e"] = key, 0
        if not state["closeup"]:
            set_camera(*wide[key])

    def draw_free(run):
        meshcat.Delete("/replay/free")
        if state["free"]:
            pts = list(run["free"]["origins"].values()) + [run["free"]["p"]]
            for n, p in enumerate(pts):
                meshcat.SetObject(f"/replay/free/beads/{n}", Sphere(0.012), red)
                meshcat.SetTransform(f"/replay/free/beads/{n}", RigidTransform(np.array(p)))
            meshcat.SetLine("/replay/free/line", np.array(pts).T, 3.0, red)

    def draw(e):
        run = rec["runs"][state["run"]]; events = run["events"]; n = run["n_events"]
        e = int(np.clip(e, 0, n)); state["e"] = e
        ev = events[e]; q = np.array(ev["q"])
        plant.SetPositions(pc, Panda, q)
        inner = ev["type"] == "accepted_update"
        prev = np.array(ev["q_prev"]) if inner else q
        plant.SetPositions(pc, Ghost, prev)
        meshcat.SetProperty(ghost_path, "visible", inner)
        meshcat.SetTransform("/replay/flange", RigidTransform(np.array(ev["p"])))
        meshcat.SetLine("/replay/error_line", np.column_stack([np.array(ev["p"]), np.array(run["target"])]), 2.0, black)
        X_WF = pose(q)
        meshcat.SetTransform("/replay/protractor", X_WF)
        ptr = PROTRACTOR_R * np.array([np.cos(q[JOINT]), np.sin(q[JOINT]), 0.0])
        violating = ev["state"] == "violated"                           # past the stop by more than the tolerance: red and larger
        meshcat.SetLine("/replay/protractor/pointer", np.column_stack([np.zeros(3), ptr]), 4.0, Rgba(0.85, 0.15, 0.1, 1.0) if violating else black)
        meshcat.SetObject("/replay/protractor/pointer_bead", Sphere(0.012 if violating else 0.008), Rgba(0.85, 0.15, 0.1, 1.0) if violating else black)
        meshcat.SetTransform("/replay/protractor/pointer_bead", RigidTransform(ptr))
        meshcat.Delete("/replay/trials")
        trials = [t for t in run["trial_evaluations"] if inner and t["outer"] == ev["outer"] and t["inner"] == ev["inner"]]
        if state["trials"] and trials:
            for j, t in enumerate(trials):
                meshcat.SetObject(f"/replay/trials/{j}", Sphere(0.012), grey)
                meshcat.SetTransform(f"/replay/trials/{j}", RigidTransform(np.array(t["p"])))
            if not state["closeup"]:
                set_camera(*gn.fit_camera(wide_pts[state["run"]] + [t["p"] for t in trials]))
        draw_free(run)
        lam_now = ev.get("lam_after", ev.get("lam", 0.0))
        if ev["type"] == "start":
            what = "start (no update yet)"
        elif inner:
            what = f"accepted joint update: outer iteration {ev['outer']}, inner step {ev['inner']} of {run['rows'][ev['outer']]['n_inner']}" + (f"; alpha {ev['alpha']:g}; {ev['rejected']} rejected" if ev["rejected"] else "")
        else:
            what = f"multiplier update after outer iteration {ev['outer']}: lambda {ev['lam_before']:.7f} -> {ev['lam_after']:.7f}"
        labels = [f"e = {e} of {n}  ({state['run']})", f"event: {what}", f"task error = {ev['residual']:.3e} m",
                  f"violation max(g,0) = {max(ev['g'], 0.0):.3e} rad, {ev['state']}  (q_4 = {q[JOINT]:+.5f}, stop {Q4_MAX})", f"lambda = {lam_now:.7f} m^2/rad, mu = {MU:g}",
                  f"outcome at the end: {run['outcome']} after {run['n_outer']} outer iterations; task error {run['final']['residual']:.4f} m, lambda {run['final']['lam']:.7f}"]
        if ev["outside_hardware"]:
            labels.append(f"joints outside the hardware range at this event: {ev['outside_hardware']} (not part of the problem)")
        if state["closeup"]:                                         # look along the elbow axis at the protractor, from the wide view's side
            axis = X_WF.rotation().matrix()[:, 2]; s = 1.0 if float(axis @ gn.CAMERA_DIR) >= 0 else -1.0
            set_camera(X_WF.translation() + 0.5 * s * axis, X_WF.translation())
        if panel is None:
            refresh_readouts(labels)
        diagram.ForcedPublish(ctx)
        if panel is not None:
            panel.publish({"run": state["run"], "e": e, "n": n, "type": ev["type"], "outer": ev["outer"], "inner": ev["inner"], "trials": state["trials"],
                           "free": state["free"], "closeup": state["closeup"], "camera": state["camera"], "meshcat_url": meshcat.web_url(),
                           "labels": labels, "p": ev["p"], "origins": ev["origins"], "pointer": ev["pointer"], "elbow": ev["elbow"],
                           "state": ev["state"], "applied_action_id": state["applied_action_id"]})
        print(f"{state['run']} e = {e}: " + " | ".join(labels[1:5]))

    load_run("runA"); draw(0)
    meshcat.SetProperty(ghost_path, "modulated_opacity", 0.3)
    faded_at = time.time()
    if panel is not None:
        print("Demo app at", panel.start(), flush=True)              # the launcher waits for this line in --app mode
    print("Meshcat at", meshcat.web_url())
    print("  Paused replay of the augmented-Lagrangian loop. Nothing moves until you press Step, Back, Reset, Next outer or Previous outer, or")
    print("  move the 'event e' slider. An accepted joint update moves the arm (the ghost is the previous iterate); a multiplier update moves")
    print("  nothing and changes lambda. The protractor at the elbow shows the allowed arc (green), the stop (maroon) and the current angle (the")
    print("  pointer, red while the stop is violated by more than the tolerance). 'Stop' quits.")

    def jump_outer(forward):
        run = rec["runs"][state["run"]]; oe = run["outer_events"]
        if forward:
            later = [x for x in oe if x > state["e"]]; return later[0] if later else run["n_events"]
        earlier = [x for x in oe if x < state["e"]]; return earlier[-1] if earlier else 0

    while meshcat.GetButtonClicks("Stop") == 0:
        changed = None
        for b in buttons:
            c = meshcat.GetButtonClicks(b)
            if c != clicks[b]:
                clicks[b] = c
                n = rec["runs"][state["run"]]["n_events"]
                if b == "Step": changed = min(state["e"] + 1, n)
                elif b == "Back": changed = max(state["e"] - 1, 0)
                elif b == "Reset": changed = 0
                elif b == "Next outer": changed = jump_outer(True)
                elif b == "Previous outer": changed = jump_outer(False)
                elif b == "Show / hide rejected trials": set_toggle("trials", not state["trials"]); changed = state["e"]
                elif b == "Show / hide the unconstrained answer": set_toggle("free", not state["free"]); changed = state["e"]
                elif b in [rec["runs"][k]["label"] for k in keys]:
                    load_run(next(k for k in keys if rec["runs"][k]["label"] == b)); changed = 0
                if changed is not None:
                    state["e"] = changed
                    meshcat.SetSliderValue(SLIDER, float(changed))
        for action_id, do in (panel.pop_actions() if panel is not None else []):    # the web page's requests, applied in order
            n = rec["runs"][state["run"]]["n_events"]
            changed = state["e"]
            if do == "step": changed = min(state["e"] + 1, n)
            elif do == "back": changed = max(state["e"] - 1, 0)
            elif do == "reset": changed = 0
            elif do == "next-outer": changed = jump_outer(True)
            elif do == "prev-outer": changed = jump_outer(False)
            elif do.startswith("set:"): changed = int(np.clip(int(float(do[4:])), 0, n))
            elif do.startswith("run:") and do[4:] in rec["runs"]: load_run(do[4:]); changed = 0
            elif do.startswith("trials:"): set_toggle("trials", do.endswith("1")); changed = state["e"]
            elif do.startswith("free:"): set_toggle("free", do.endswith("1")); changed = state["e"]
            elif do.startswith("closeup:"):
                state["closeup"] = do.endswith("1"); changed = state["e"]
                if not state["closeup"]:
                    set_camera(*wide[state["run"]])
            state["e"] = changed                                      # the next queued action starts from this result
            state["applied_action_id"] = action_id                     # bounds and other no-ops are acknowledged too
            meshcat.SetSliderValue(SLIDER, float(changed))
        sv = int(round(meshcat.GetSliderValue(SLIDER)))
        n = rec["runs"][state["run"]]["n_events"]
        if sv > n:
            meshcat.SetSliderValue(SLIDER, float(n)); sv = n
        if sv != state["e"] or changed is not None:
            draw(sv)
        if time.time() - faded_at > 1.0:
            meshcat.SetProperty(ghost_path, "modulated_opacity", 0.3); faded_at = time.time()
        time.sleep(0.05)
    meshcat.DeleteAddedControls()
    if panel is not None:
        panel.close()


if __name__ == "__main__":
    if "--replay" in sys.argv:
        port = int(sys.argv[sys.argv.index("--app-port") + 1]) if "--app-port" in sys.argv else 0
        replay_mode(app="--app" in sys.argv, app_port=port)
    elif "--record" in sys.argv:
        record_mode(sys.argv[sys.argv.index("--record") + 1])
    elif "--figure" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--figure") + 1])
    elif "--selftest" in sys.argv:
        k = sys.argv.index("--selftest")
        default_baseline = str(Path(__file__).with_name("iterates-baseline.json"))
        sys.exit(1 if selftest_mode(sys.argv[k + 1] if len(sys.argv) > k + 1 and not sys.argv[k + 1].startswith("--") else default_baseline) else 0)
    else:
        print_mode()
