"""Session 4.2 Drake demo: Deck 05's IK program solved by the course's augmented-Lagrangian loop, one event at a time.

    min_q  f(q) = 1/2 |q - q_ref|^2                          the posture deviation is the objective [rad^2]; q_ref = the ready pose
    s.t.   h(q) = phi(q) - y* = 0                            the target is a rail: three equality rows [m]
           g(q) = (q_lo - q, q - q_hi) <= 0                  the URDF joint box as 14 fence rows [rad]

This is the program of drake/03-ik-as-nlp/ik_nlp_panda.py (Deck 05 slides 18-19), which SNOPT solves there; here the
course's loop (constrained_ik.py, komo_two_link.py: mu fixed, multipliers learned) solves it, so that students can
watch the inner solves and the multiplier updates. What changed against the earlier demos: there the squared target
error was the objective and the distance a displayed diagnostic; here the distance measures the violation of the
rail. The rail and the joint limits must hold at a successful answer within tolerance, while intermediate iterates
may violate them; neither the posture cost nor a violation need change monotonically through the solve.

    A(q) = f + kappa^T h + mu_h |h|^2 + sum_i [g_i > 0 or lambda_i > 0] (lambda_i g_i + mu_g g_i^2)
    inner (kappa, lambda frozen): damped Gauss-Newton on A, H = I + 2 mu_h J^T J + 2 mu_g sum_incl e_i e_i^T + rho I,
                                  Armijo from alpha = 1 (c = 0.01); stop at |grad A| < 3e-7, or when the predicted decrease
                                  -grad A . d falls below 1e-13 max(1, |A|) (the precision floor of the Armijo test), or after 100 steps
    outer (q fixed):              kappa <- kappa + 2 mu_h h(q),   lambda_i <- max(0, lambda_i + 2 mu_g g_i(q));
                                  stop when max|h| < 1e-6 m, max g+ < 1e-6 rad, |grad f + J^T kappa + sum lambda_i grad g_i| < 1e-6 rad
                                  and max|lambda_i g_i| < 1e-8 rad^2 (a fixed threshold; in run B stationarity and complementarity both
                                  fail at outer iteration 3 and both pass at iteration 4), or after 30 outer iterations
    The rail rows are in metres and the fence rows in radians, so the two penalty weights are separate: mu_h = 1000
    rad^2/m^2 and mu_g = 100 (dimensionless), chosen from --sweep (increment3/sweep.txt: mu_h = 1 does not converge in
    30 outer iterations; mu_h = 100 needs 5 and 11 to 18 outer iterations, and run B does not converge with mu_g = 1;
    1000/100 needs 3 and 5). The inner tolerance 3e-7 was raised from 1e-7 because at 1e-7 the last inner steps of
    run B were decided by rounding and the counts changed with a 1e-13 change of the start; --robustness perturbs the
    starts and reports the counts. The coefficient acting in an inner solve is the estimate kappa_hat = kappa + 2 mu_h h
    (and max(0, lambda + 2 mu_g g) for an included fence row), which the update stores at the last inner iterate; the
    stored multipliers are frozen during the solve.
Runs: run A = Deck 05's run 1 (the scene target (0.45, 0.10, 0.45) m, no fence active) from the ready pose; run B =
run 2 (the target behind the robot (-0.20, 0, 0.60) m, joint 2 at its lower limit) from the ready pose with joint 1
turned by +0.01 rad: the target lies on the robot's symmetry plane; from the symmetric ready pose SNOPT returns the
symmetric KKT point (cost 1.729, lambda = 4.196), which the second-order check (the reduced Hessian of the Lagrangian
on the tangent space of the active rows, a negative eigenvalue -1.40) shows to be a constrained saddle, and from the
symmetric start the loop's sideways direction would be decided by rounding. The offset breaks the symmetry and makes
the run repeatable, but the whole run decides the side (the first steps cross the plane); offsets of 0.02 or 0.05 rad
reach the mirror answer, so a changed start or changed weights must re-check the sign of kappa_y against slide 19.
The initial guess and the posture reference are separate (q_init and q_ref). The run-B answer (cost 1.499, lambda =
1.053) and its mirror image (joint 1 turned the other way, the same cost) are local minima with strict
complementarity; solver answers are compared on cost, feasibility, stationarity and complementarity (with the
solvers' tolerances tightened to meet the loop's stops), multipliers and prices on the same branch, and nothing
here is a global claim.

Run:  python ik_nlp_al_panda.py                  prints both runs (one line per outer iteration), the KKT quartet with pass marks, the comparison with
                                                 SNOPT (Drake's InverseKinematics, the identical program) and IPOPT, the price checks and the second-order checks
      python ik_nlp_al_panda.py --sweep          measures outer and inner counts over the penalty weights (the choice above is documented from it)
      python ik_nlp_al_panda.py --robustness     perturbs the starts by 1e-12, 1e-9 and 1e-6 rad and reports whether the counts and the answers hold
      python ik_nlp_al_panda.py --write-baseline FILE   writes the regression baseline (q, kappa and lambda per event) from the demo's own record
      python ik_nlp_al_panda.py --replay --app   paused replay as a web page around Meshcat: the robot at the event's q, a faded ghost at the
                                                 previous iterate for an inner step, the target and the flange joined by the rail-violation line,
                                                 an optional arrow -kappa_hat at the flange, a protractor at joint 2 in run B, readouts, a per-outer
                                                 table, plots, and the controls Step, Back, Reset, a slider over the events, multiplier-update jumps,
                                                 run buttons, toggles; nothing moves by itself
      python ik_nlp_al_panda.py --replay         the same replay with Meshcat's own controls only
      python ik_nlp_al_panda.py --record FILE    writes both runs' events as JSON (cs549-trace/1)
      python ik_nlp_al_panda.py --figure FILE    rail residual, posture cost, |kappa| and lambda_2 per event (matplotlib)
      python ik_nlp_al_panda.py --selftest [BASELINE.json]  recomputes every event from FK and the Jacobian, checks the counts, the solver
                                                 comparison, the second-order checks, the source hashes, the independent reference and the baseline

Added 2026-10-15. Tested with drake 1.46.0 on macOS.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02-gauss-newton-ik"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03-ik-as-nlp"))
import gn_ik_panda as gn
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, IpoptSolver, MeshcatVisualizer, Parser, Rgba, RigidTransform,
                         SnoptSolver, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
Q_REF = gn.Q_READY                                               # the posture reference of the objective (fixed)
Q_INIT_B = gn.Q_READY + np.array([0.01, 0, 0, 0, 0, 0, 0])        # run B starts 0.01 rad off the symmetry plane (joint 1), documented
RUNS = {"runA": {"label": "Run A: the scene target (0.45, 0.10, 0.45) m, no fence active", "target": [0.45, 0.10, 0.45], "q_init": gn.Q_READY,
                 "init_note": "the ready pose, which is also the posture reference", "gauge_joint": None,
                 "note": "the rail is reached as kappa is learned; no joint limit is active at the answer"},
        "runB": {"label": "Run B: the target behind the robot (-0.20, 0, 0.60) m, joint 2 at its lower limit", "target": [-0.20, 0.0, 0.60], "q_init": Q_INIT_B,
                 "init_note": "the ready pose with joint 1 turned by +0.01 rad (0.57 degrees): the target lies on the robot's symmetry plane, where SNOPT from the exactly symmetric pose stops at a KKT point that is not a minimum and where the loop's sideways direction would be decided by rounding; the offset breaks the symmetry and makes the run repeatable, but the whole run decides the side (the first steps cross the plane: q_1 = +0.006, +0.077, -0.238, +0.001 at events 1 to 4, settling on the positive side from event 10), and offsets of 0.02 or 0.05 rad reach the mirror answer, so any change of q_init or of the weights must re-check the sign of kappa_y against slide 19; the reference posture stays the ready pose, so the initial posture cost is 5e-5 rad^2",
                 "gauge_joint": 1, "note": "joint 2 reaches its lower limit: a fence row joins the rail, with its own lambda"}}
MU_H, MU_G, RHO = 1000.0, 100.0, 1e-3                            # chosen from --sweep (2026-10-15); see the checkpoint
INNER_TOL, INNER_MAX, OUTER_MAX = 3e-7, 100, 30                 # 3e-7: at 1e-7 the last inner steps of run B sat below the rounding resolution of the Armijo test on A, and the event counts changed with a 1e-13 change of the start (reviewer, 2026-10-15)
FLOOR_REL = 1e-13                                                 # the inner solve also ends when the predicted decrease -grad A . d is below FLOOR_REL * max(1, |A|): the precision floor of the Armijo test
FEAS_H_TOL, FEAS_G_TOL, STAT_TOL, COMP_TOL = 1e-6, 1e-6, 1e-6, 1e-8   # m, rad, rad, rad^2; the complementarity threshold is a fixed 1e-8 rad^2
ACT_TOL = FEAS_G_TOL                                              # one activity threshold for every bound (the primal gap), recorded once
LAM_FLOOR = 1e-6                                                  # strict complementarity: a primal-active bound needs lambda above this for the "local minimum" label
SOLVER_TOL = {"SNOPT": {"feas_h": 1e-6, "feas_g": 1e-6, "stat": 1e-6, "comp": 1e-8}, "IPOPT": {"feas_h": 1e-6, "feas_g": 1e-6, "stat": 1e-6, "comp": 1e-8}}   # with the solver options below; measured values in the record
REGRESSION_TOL = {"cost": 1e-6, "q": 1e-4, "kappa": 1e-5, "lam": 1e-4}   # how far the solvers' answers may sit from the loop's on the same branch: regression tolerances, not guarantees of the stopping rules
ARMIJO_C, ARMIJO_FLOOR = 0.01, 1e-12
ARROW_SCALE, ARROW_CLAMP = 0.02, 0.30                            # metres of arrow per rad^2/m of -kappa_hat, clamped (kappa_hat is large before the multipliers settle)


def arrow_tip(p, kappa_hat):
    """The arrow's tip: p - kappa_hat scaled, its length clamped at ARROW_CLAMP metres."""
    n = float(np.linalg.norm(kappa_hat))
    return (np.array(p) - np.array(kappa_hat) / n * min(ARROW_SCALE * n, ARROW_CLAMP)).tolist() if n > 0 else list(p)
PROTRACTOR_R = 0.16
SOURCE_FILES = ("ik_nlp_al_panda.py", "panel.html", "../02-gauss-newton-ik/replay_panel.py", "../02-gauss-newton-ik/gn_ik_panda.py", "../03-ik-as-nlp/ik_nlp_panda.py",
                "../../planning/optimization-step-visuals-2026-10-14/increment3/reference_al_eq.py")
E_ROWS = np.vstack([-np.eye(7), np.eye(7)])                      # grad g rows: g = (lo - q, q - hi)


def al_ik_nlp(phi, J, lo, hi, q_init, y, q_ref=None, mu_h=MU_H, mu_g=MU_G, rho=RHO, inner_tol=INNER_TOL, inner_max=INNER_MAX, outer_max=OUTER_MAX, on_event=None, tols=None):
    """The course's augmented-Lagrangian loop on Deck 05's program (rail + joint box), started at q_init with the
    posture reference q_ref (the ready pose by default; the two are separate). Returns (q, kappa, lam, status, rows).
    on_event (optional; None in print mode) observes the run without changing it."""
    q0 = Q_REF if q_ref is None else np.array(q_ref, float)        # the objective's reference, not the start
    feas_h_tol, feas_g_tol, stat_tol, comp_tol = tols or (FEAS_H_TOL, FEAS_G_TOL, STAT_TOL, COMP_TOL)
    E = E_ROWS
    gfun = lambda qq: np.concatenate([lo - qq, qq - hi])
    f = lambda qq: 0.5 * float(np.sum((qq - q0) ** 2))
    q, kap, lam = np.array(q_init, float), np.zeros(3), np.zeros(14)
    rows, status = [], "limit"
    for it in range(outer_max):                                     # outer iterations are numbered from 0 as in Deck 05; k is the Markov order in this course
        incl = lambda qq: (gfun(qq) > 0) | (lam > 0)
        def A(qq):
            h = phi(qq) - y; g = gfun(qq); a = incl(qq)
            return f(qq) + float(kap @ h) + mu_h * float(h @ h) + float(np.sum(a * (lam * g + mu_g * g ** 2)))
        def gradA(qq):
            h = phi(qq) - y; g = gfun(qq); a = incl(qq)
            return (qq - q0) + J(qq).T @ (kap + 2 * mu_h * h) + E.T @ (a * (lam + 2 * mu_g * g))
        n_in, st, gr = 0, "limit", gradA(q)
        for _ in range(inner_max):
            gr = gradA(q)
            if np.linalg.norm(gr) < inner_tol:
                st = "stationary"; break
            Jq = J(q); a = incl(q)
            H = np.eye(7) + 2 * mu_h * Jq.T @ Jq + 2 * mu_g * E.T @ np.diag(a.astype(float)) @ E + rho * np.eye(7)
            d = -np.linalg.solve(H, gr)
            al, A0, slope = 1.0, A(q), float(gr @ d)
            if -slope < FLOOR_REL * max(1.0, abs(A0)):
                st = "stopped at precision floor"; break
            ok = lambda aa: np.isfinite(A(q + aa * d)) and A(q + aa * d) <= A0 + ARMIJO_C * aa * slope
            rejected = 0
            while not ok(al) and al > ARMIJO_FLOOR:
                if on_event: on_event({"type": "trial_evaluation", "outer": it, "inner": n_in + 1, "alpha": al, "q": (q + al * d).copy(), "A": A(q + al * d), "included": [int(i) for i in np.nonzero(incl(q + al * d))[0]]})
                al *= 0.5; rejected += 1
            if not ok(al):
                st = "stalled"; break
            q_prev = q.copy(); q = q + al * d; n_in += 1
            if on_event:
                h = phi(q) - y; g = gfun(q); a_now = incl(q)
                on_event({"type": "accepted_update", "outer": it, "inner": n_in, "q_prev": q_prev, "q": q.copy(), "step": (al * d).copy(), "direction": d.copy(),
                          "alpha": al, "rejected": rejected, "included": [int(i) for i in np.nonzero(a)[0]], "kappa": kap.copy(), "lam": lam.copy(), "mu_h": mu_h, "mu_g": mu_g,
                          "f": f(q), "A": A(q), "A_prev": A0, "slope": slope, "h": h.copy(), "g": g.copy(), "rail_residual": float(np.linalg.norm(h)),
                          "kappa_hat": (kap + 2 * mu_h * h).copy(), "lam_estimate": np.where(a_now, np.maximum(lam + 2 * mu_g * g, 0.0), lam).copy(),
                          "step_norm": float(np.linalg.norm(al * d)), "grad_norm": float(np.linalg.norm(gradA(q)))})
        h = phi(q) - y; g = gfun(q)
        kap_new = kap + 2 * mu_h * h
        lam_new = np.maximum(lam + 2 * mu_g * g, 0.0)
        feas_h = float(np.abs(h).max()); feas_g = float(max(g.max(), 0.0))
        stat = float(np.linalg.norm((q - q0) + J(q).T @ kap_new + E.T @ lam_new))
        comp = float(np.abs(lam_new * g).max())
        passes = {"feas_h": bool(feas_h < feas_h_tol), "feas_g": bool(feas_g < feas_g_tol), "stat": bool(stat < stat_tol), "comp": bool(comp < comp_tol)}
        rows.append(dict(outer=it, kappa_before=kap.copy(), kappa_after=kap_new.copy(), lam_before=lam.copy(), lam_after=lam_new.copy(), q=q.copy(),
                         h=h.copy(), g=g.copy(), n_inner=n_in, inner_status=st, grad_norm=float(np.linalg.norm(gr)), feas_h=feas_h, feas_g=feas_g,
                         stat=stat, comp=comp, passes=passes, f=f(q), rail_residual=float(np.linalg.norm(h)), included=[int(i) for i in np.nonzero((g > 0) | (lam_new > 0))[0]]))
        if on_event:
            on_event({"type": "outer_update", "outer": it, "inner": n_in, "q": q.copy(), "h": h.copy(), "g": g.copy(), "kappa_before": kap.copy(), "kappa_after": kap_new.copy(),
                      "lam_before": lam.copy(), "lam_after": lam_new.copy(), "mu_h": mu_h, "mu_g": mu_g, "feas_h": feas_h, "feas_g": feas_g, "stat": stat, "comp": comp,
                      "passes": passes, "f": f(q), "rail_residual": float(np.linalg.norm(h)), "inner_steps": n_in, "inner_status": st, "grad_norm": float(np.linalg.norm(gr)),
                      "included": [int(i) for i in np.nonzero((g > 0) | (lam_new > 0))[0]],
                      "kappa_hat": kap_new.copy(), "lam_estimate": lam_new.copy()})      # the update stores the estimates
        kap, lam = kap_new, lam_new
        if st == "stalled":
            status = "stalled"; break
        if all(passes.values()):
            status = "converged"; break
    return q, kap, lam, status, rows


def reduced_hessian(phi, J, lo, hi, q, kap, lam, eps=1e-5):
    """Eigenvalues of Z^T H_L Z, with H_L the Hessian of the Lagrangian L = f + kappa^T h + lambda^T g at q (central
    finite differences of grad L, step eps) and Z an orthonormal basis of the tangent space of the active constraints
    (the three rail rows and the bounds with lambda > 0). All positive: a local minimum (second-order sufficient
    condition, within the finite-difference accuracy); a negative one: a constrained saddle, although every KKT
    condition holds."""
    E = E_ROWS
    gradL = lambda qq: (qq - Q_REF) + J(qq).T @ kap + E.T @ lam
    H = np.zeros((7, 7))
    for i in range(7):
        d = np.zeros(7); d[i] = eps
        H[:, i] = (gradL(q + d) - gradL(q - d)) / (2 * eps)
    H = 0.5 * (H + H.T)
    g = np.concatenate([lo - q, q - hi])
    active = [i for i in range(14) if abs(g[i]) <= ACT_TOL]           # activity by the primal gap (a solver leaves small duals on inactive bounds)
    C = np.vstack([J(q)] + [E[i:i + 1] for i in active])
    Z = np.linalg.svd(C)[2][C.shape[0]:].T
    return np.sort(np.linalg.eigvalsh(Z.T @ H @ Z)), len(active)


def second_order_verdict(ev, lam, active):
    """'local minimum' needs all reduced-Hessian eigenvalues positive and strict complementarity (every primal-active bound
    priced above LAM_FLOOR); a negative eigenvalue is a 'constrained saddle'; anything else is 'inconclusive'."""
    if ev.size and ev.min() < 0:
        return "constrained saddle"
    if ev.size and ev.min() > 0 and all(lam[i] > LAM_FLOOR for i in active):
        return "local minimum"
    return "inconclusive (a zero eigenvalue or a primal-active bound with a zero multiplier)"


def solver_reference(plant, pc, panda, phi, J, lo, hi, y, q_init, loop_q=None):
    """SNOPT through Drake's InverseKinematics (the slide-19 program; its default joint bounds make the program identical)
    and IPOPT on the same MathematicalProgram, both started at the run's q_init; the duals with Drake's sign rules
    (kappa = -dual of the position equality; lambda = +dual at a lower bound, -dual at an upper bound); the price
    checks by re-solving from the answer (a 1 mm target shift in x; the active bound tightened by 1e-3 rad); the KKT
    quartet recomputed here; the second-order check; and SNOPT from the symmetric ready pose as the saddle diagnostic
    when the target lies on the symmetry plane."""
    import ik_nlp_panda as nlp
    from pydrake.all import InverseKinematics, SolverOptions
    assert np.array_equal(nlp.Q_READY, Q_REF), "the comparison program's posture reference must be the loop's"
    opts = SolverOptions()
    opts.SetOption(SnoptSolver.id(), "Major optimality tolerance", 1e-9); opts.SetOption(SnoptSolver.id(), "Major feasibility tolerance", 1e-9)
    opts.SetOption(IpoptSolver.id(), "tol", 1e-10); opts.SetOption(IpoptSolver.id(), "constr_viol_tol", 1e-9); opts.SetOption(IpoptSolver.id(), "compl_inf_tol", 1e-10)
    OPTIONS_TEXT = "SNOPT: Major optimality tolerance 1e-9, Major feasibility tolerance 1e-9; IPOPT: tol 1e-10, constr_viol_tol 1e-9, compl_inf_tol 1e-10 (tightened so that the answers meet the loop's own stops)"
    out = {}
    def build(target, extra=None):
        ik = InverseKinematics(plant, pc); q = ik.q(); prog = ik.prog()
        prog.AddQuadraticErrorCost(0.5 * np.eye(7), Q_REF, q)
        Ef = plant.GetFrameByName("panda_link8", panda)
        pos = ik.AddPositionConstraint(Ef, np.zeros(3), plant.world_frame(), target, target)
        box = prog.bounding_box_constraints()[0]
        if extra is not None: prog.AddBoundingBoxConstraint(extra[0], extra[1], q)
        return prog, q, pos, box
    def entry_of(res, q, pos, box, qs):
        kap = -res.GetDualSolution(pos); dual_box = res.GetDualSolution(box)
        lam = np.concatenate([np.maximum(dual_box, 0), np.maximum(-dual_box, 0)])
        g = np.concatenate([lo - qs, qs - hi]); h = phi(qs) - y
        raw_inactive = float(np.abs(np.where(np.abs(g) > ACT_TOL, lam, 0.0)).max())
        lam = np.where(np.abs(g) <= ACT_TOL, lam, 0.0)                   # activity by the primal gap: SNOPT leaves duals of 1e-5 to 1e-4 on inactive bounds
        active = [int(i) for i in np.nonzero(np.abs(g) <= ACT_TOL)[0]]
        ev, n_act = reduced_hessian(phi, J, lo, hi, qs, kap, lam)
        det = res.get_solver_details()
        return kap, lam, dict(cost=float(res.get_optimal_cost()), q=qs.tolist(), kappa=kap.tolist(), lam=lam.tolist(), feas_h=float(np.abs(h).max()),
                              feas_g=float(max(g.max(), 0.0)), stat=float(np.linalg.norm((qs - Q_REF) + J(qs).T @ kap + E_ROWS.T @ lam)),
                              comp=float(np.abs(lam * g).max()), largest_dual_on_inactive_bounds=raw_inactive, active=active,
                              success=bool(res.is_success()), exit=(f"info {det.info}" if hasattr(det, "info") else f"status {det.status} ({det.ConvertStatusToString()})"),
                              reduced_hessian_eigenvalues=ev.tolist(), second_order=second_order_verdict(ev, lam, active))
    def resolve(S, target, guess, extra=None):
        prog, q, pos, box = build(target, extra); res = S.Solve(prog, guess, opts); qs = res.GetSolution(q)
        lo_f, hi_f = (np.maximum(lo, extra[0]), np.minimum(hi, extra[1])) if extra is not None else (lo, hi)      # the bounds in force
        g = np.concatenate([lo_f - qs, qs - hi_f])
        return dict(success=bool(res.is_success()), cost=float(res.get_optimal_cost()), q=qs.tolist(), active=[int(i) for i in np.nonzero(np.abs(g) <= ACT_TOL)[0]])
    for name, S in (("SNOPT", SnoptSolver()), ("IPOPT", IpoptSolver())):
        prog, q, pos, box = build(y)
        res = S.Solve(prog, q_init, opts); qs = res.GetSolution(q)
        kap, lam, entry = entry_of(res, q, pos, box, qs)
        if abs(y[1]) < 1e-12 and loop_q is not None and qs[0] * loop_q[0] < 0:            # the solver landed on the mirror branch: map it onto the loop's
            entry["branch"] = "mirror branch, mapped onto the loop's (q_1, q_3, q_5 and kappa_y negated)"
            qs = qs.copy(); qs[[0, 2, 4]] *= -1; kap = kap.copy(); kap[1] *= -1
            entry["q"], entry["kappa"] = qs.tolist(), kap.tolist()
        else:
            entry["branch"] = "the loop's branch" if (abs(y[1]) >= 1e-12 or loop_q is None or qs[0] * loop_q[0] > 0) else "undetermined"
        entry.update(start="the run's q_init", options=OPTIONS_TEXT, tolerances=SOLVER_TOL[name],
                     passes={k_: bool(entry[k_] < SOLVER_TOL[name][k_]) for k_ in ("feas_h", "feas_g", "stat", "comp")},
                     program="Drake's InverseKinematics with its default joint bounds plus the position equality and the posture cost: the loop's program")
        if name == "SNOPT":
            # the price of the rail: re-solves from this answer with the target shifted by +-delta in x (forward and central differences; the
            # forward difference carries an O(delta) curvature term); each re-solve's success and active set are recorded
            d = 1e-3; yp = y.copy(); yp[0] += d; ym = y.copy(); ym[0] -= d
            rp, rm = resolve(S, yp, qs), resolve(S, ym, qs)
            entry["price_target_x"] = dict(delta=d, change=rp["cost"] - entry["cost"], change_central=(rp["cost"] - rm["cost"]) / 2, predicted=float(-kap[0] * d),
                                           resolves_succeeded=rp["success"] and rm["success"], same_active_set=rp["active"] == entry["active"] == rm["active"])
            if entry["active"]:
                i = entry["active"][0]; j, side = (i, "lower") if i < 7 else (i - 7, "upper")
                lo_t, hi_t = lo.copy(), hi.copy()
                if side == "lower": lo_t[j] += d
                else: hi_t[j] -= d
                rt = resolve(S, y, qs, (lo_t, hi_t))                                  # only tightening: the URDF bound cannot be relaxed inside this program
                entry["price_bound"] = dict(joint=j + 1, side=side, delta=d, change=rt["cost"] - entry["cost"], change_central=None, predicted=float(lam[i] * d),
                                            resolves_succeeded=rt["success"], same_active_set=rt["active"] == entry["active"],
                                            note="a forward difference only (the URDF bound cannot be relaxed within the program); it carries an O(delta) curvature term")
            if abs(y[1]) < 1e-12 and np.linalg.norm(q_init - Q_REF) > 0:
                progs, qsym, poss, boxs = build(y); ress = S.Solve(progs, Q_REF, opts)
                _, _, sym = entry_of(ress, qsym, poss, boxs, ress.GetSolution(qsym))
                sym.update(start="the symmetric ready pose", tolerances=SOLVER_TOL[name], passes={k_: bool(sym[k_] < SOLVER_TOL[name][k_]) for k_ in ("feas_h", "feas_g", "stat", "comp")},
                           kkt_point=bool(sym["success"] and all(sym[k_] < SOLVER_TOL[name][k_] for k_ in ("feas_h", "feas_g", "stat", "comp"))))
                entry["symmetric_start"] = sym
        out[name] = entry
    return out


def make_protractor(plant, pc, Panda, joint_name):
    joint = plant.GetJointByName(joint_name, Panda)
    F = joint.frame_on_parent()
    lo_j, hi_j = float(joint.position_lower_limits()[0]), float(joint.position_upper_limits()[0])
    X_PF = F.GetFixedPoseInBodyFrame().GetAsMatrix4().tolist()
    jidx = joint.position_start()
    def pose(q):
        plant.SetPositions(pc, Panda, q)
        return plant.CalcRelativeTransform(pc, plant.world_frame(), F)
    def tip(q, r=PROTRACTOR_R):
        return (pose(q) @ np.array([r * np.cos(q[jidx]), r * np.sin(q[jidx]), 0.0])).tolist()
    return pose, tip, (lo_j, hi_j), {"name": joint_name, "index": jidx, "parent_link": F.body().name(), "child_link": joint.frame_on_child().body().name(), "X_PF": X_PF, "hardware_range": [lo_j, hi_j]}


def source_info():
    here = Path(__file__).resolve().parent
    # A copy outside the course repository (the students' demos folder) lacks the planning reference; record it as absent.
    files = {str((here / rel).resolve().relative_to(here.parent.parent)): hashlib.sha256((here / rel).resolve().read_bytes()).hexdigest()
             for rel in SOURCE_FILES if (here / rel).resolve().exists()}
    missing = [rel for rel in SOURCE_FILES if not (here / rel).resolve().exists()]
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--"] + [str((here / r).resolve()) for r in SOURCE_FILES if (here / r).resolve().exists()], capture_output=True, text=True, cwd=here).stdout.strip())
    except Exception:
        git, dirty = "unknown", True
    import platform
    return {"files": files, "missing": missing, "git": git, "dirty": dirty, "python": platform.python_version(), "numpy": np.__version__, "platform": f"{platform.system()} {platform.release()} {platform.machine()}"}


def setup():
    d, plant, pc, ctx, Panda, ee = gn.build()
    phi, J = gn.make_fk(plant, pc, Panda, ee)
    return plant, pc, Panda, phi, J, plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()


# ----------------------------------------------------------------------------------------------- record
def record_runs(plant, pc, Panda, phi, J, lo, hi, with_checks=True):
    origins = gn.make_origins(plant, pc, Panda)
    from importlib.metadata import version
    tl = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
    rec = {"schema": "cs549-trace/1",
           "problem": "Deck 05's IK program: min 1/2|q - q_ref|^2 s.t. phi(q) - y* = 0 (a rail, three rows in m) and the URDF joint box (14 fence rows in rad) on the Panda",
           "stage": "B2 (IK as Deck 05's nonlinear program: the target as an equality, the posture deviation as the objective)",
           "platform": f"drake {version('drake')}",
           "solver": f"course augmented Lagrangian (mu_h = {MU_H:g} rad^2/m^2 and mu_g = {MU_G:g} fixed; kappa <- kappa + 2 mu_h h, lambda <- max(0, lambda + 2 mu_g g)), damped Gauss-Newton inner loop "
                     f"(rho = {RHO:g}, Armijo c = {ARMIJO_C}), NumPy on Drake FK and Jacobian; drake/05-ik-nlp-al/ik_nlp_al_panda.py",
           "event_source": "own-loop hook",
           "estimator": "intermediate multipliers are the loop's own estimates: kappa_hat = kappa + 2 mu_h h and max(0, lambda + 2 mu_g g) at the iterate (notes section 4.9, eq. 23); the multiplier update stores the values at the last inner iterate; the stored multipliers are frozen during an inner solve",
           "timeline": "the slider walks accepted joint updates and multiplier updates only (every Step changes q or the multipliers, never both); rejected Armijo trials attach to their accepted update behind a toggle",
           "what_changed": "in the earlier demos the squared target error was the objective and the distance a displayed diagnostic; here the distance measures the violation of the rail, which must hold at a successful answer within tolerance, while intermediate iterates may violate it; neither the posture cost nor a violation need change monotonically",
           "limitations": ["intermediate multipliers are the loop's own estimates",
                           "the loop finds a KKT point from its start: run B's answer is a local minimum (second-order check with strict complementarity) with a mirror image of equal cost; SNOPT's answer from the symmetric ready pose is a constrained saddle; no global claim",
                           "run B's side (the sign of q_1 and kappa_y) is decided by the whole run, not by the sign of the 0.01 rad offset: offsets of 0.02 or 0.05 rad reach the mirror answer; a changed start or changed weights must re-check the sign against slide 19",
                           "solver answers are compared on cost, feasibility, stationarity and complementarity with the solver tolerances tightened (solver_options) so that they meet the loop's stops; the agreement on q, kappa, lambda and cost on the same branch (the solvers start at the run's q_init) is checked against regression tolerances, which are tighter than the stopping rules guarantee",
                           "the reference implementation is a re-transcription of the course's rules in the same operation order (it shares FK and the Jacobian): it checks the event hook and the bookkeeping, and the regression baseline is written from the demo's own record",
                           "the inner solve stops at |grad A| < 3e-7 or at its precision floor (predicted decrease below 1e-13 max(1, |A|)): at 1e-7 without the floor the last inner steps of run B were decided by rounding and the event counts changed with a 1e-13 change of the start; now the counts are the same over random perturbations of the start up to 1e-6 rad (checked by --robustness); an inner solve that ends at the floor is labelled 'stopped at precision floor', not 'stationary'",
                           "the fence rows are the hardware limits themselves, so a violated fence row (beyond the fence tolerance) is a configuration outside the hardware range (flagged per event with the same tolerance)",
                           "the arrow (off by default) is -kappa_hat at the flange in the world frame, 0.02 m per rad^2/m and clamped at 0.30 m (kappa_hat is large before the multipliers settle); its units are rad^2/m with this posture objective, so it is not a force; only at convergence does moving the target along it raise the optimal cost fastest",
                           f"tolerances: max|h| < {FEAS_H_TOL:g} m, max g+ < {FEAS_G_TOL:g} rad, stationarity < {STAT_TOL:g} rad, complementarity < {COMP_TOL:g} rad^2 (a fixed threshold; in run B stationarity and complementarity both fail at outer iteration 3 and both pass at iteration 4); inner |grad A| < {INNER_TOL:g} or the precision floor"],
           "units": {"q": "rad", "q_prev": "rad", "step": "rad", "direction": "rad", "alpha": "1", "p": "m", "origins": "m", "h": "m", "rail_residual": "m", "g": "rad", "kappa": "rad^2/m", "kappa_hat": "rad^2/m",
                     "kappa_before": "rad^2/m", "kappa_after": "rad^2/m", "lam": "rad", "lam_estimate": "rad", "lam_before": "rad", "lam_after": "rad", "mu_h": "rad^2/m^2", "mu_g": "1",
                     "f": "rad^2", "A": "rad^2", "A_prev": "rad^2", "slope": "rad^2", "cost": "rad^2", "stat": "rad", "comp": "rad^2", "feas_h": "m", "feas_g": "rad", "step_norm": "rad",
                     "grad_norm": "rad", "reduced_hessian_eigenvalues": "1 (curvature of the Lagrangian per rad^2 of tangent motion)", "price change": "rad^2", "gauge.pointer": "m", "arrow": "m"},
           "tolerances": {"inner_grad": INNER_TOL, "feas_h": FEAS_H_TOL, "feas_g": FEAS_G_TOL, "stat": STAT_TOL, "comp": COMP_TOL,
                          "complementarity_rule": "a fixed 1e-8 rad^2; in run B stationarity and complementarity both fail at outer iteration 3 and both pass at iteration 4", "inner_floor_rel": FLOOR_REL,
                          "activity": ACT_TOL, "strict_complementarity_floor": LAM_FLOOR, "solver": SOLVER_TOL, "regression": REGRESSION_TOL, "price_check_relative": 0.02, "price_check_central_relative": 0.001,
                          "arrow_clamp_m": ARROW_CLAMP, "hardware_flag": FEAS_G_TOL},
           "hardware_ranges": [[float(a), float(b)] for a, b in zip(lo, hi)],
           "model": "package://drake_models/franka_description/urdf/panda_arm.urdf, panda_link8 as the flange",
           "method": {"mu_h": MU_H, "mu_g": MU_G, "rho": RHO, "inner_tol": INNER_TOL, "inner_max": INNER_MAX, "outer_max": OUTER_MAX, "armijo_c": ARMIJO_C, "armijo_floor": ARMIJO_FLOOR},
           "q_ref": Q_REF.tolist(), "arrow": {"definition": "-kappa_hat = -(kappa + 2 mu_h h) at the flange, world frame", "scale_m_per_unit": ARROW_SCALE, "clamp_m": ARROW_CLAMP, "units": "rad^2/m", "default": "off"},
           "protractor_radius": PROTRACTOR_R, "source": source_info(), "runs": {}}
    for key, spec in RUNS.items():
        y = np.array(spec["target"]); q_init = np.array(spec["q_init"])
        gauge = None
        if spec["gauge_joint"] is not None:
            pose, tip, hw_j, ginfo = make_protractor(plant, pc, Panda, f"panda_joint{spec['gauge_joint'] + 1}")
            gauge = ginfo
        events, trials = [], []

        def snap(q):
            g = np.concatenate([lo - q, q - hi]); h = phi(q) - y
            d = {"q": np.array(q).tolist(), "p": phi(q).tolist(), "origins": origins(q), "h": h.tolist(), "rail_residual": float(np.linalg.norm(h)),
                 "rail_state": "on the rail" if np.abs(h).max() < FEAS_H_TOL else "off the rail", "g": g.tolist(),
                 "violated_bounds": [int(i) for i in np.nonzero(g > FEAS_G_TOL)[0]], "active_bounds": [int(i) for i in np.nonzero(np.abs(g) <= FEAS_G_TOL)[0]],
                 "outside_hardware": sorted({int(i % 7) + 1 for i in np.nonzero(g > FEAS_G_TOL)[0]})}      # the fence rows are the hardware limits: the same tolerance
            if gauge is not None:
                d["gauge"] = {"elbow": pose(q).translation().tolist(), "axis": pose(q).rotation().matrix()[:, 2].tolist(), "pointer": tip(q)}
            return d

        events.append({"e": 0, "type": "start", "outer": 0, "inner": 0, "kappa": [0.0] * 3, "lam": [0.0] * 14, "kappa_hat": (2 * MU_H * (phi(q_init) - y)).tolist(),
                       "lam_estimate": np.maximum(2 * MU_G * np.concatenate([lo - q_init, q_init - hi]), 0.0).tolist(), "included": [],
                       "f": 0.5 * float(np.sum((q_init - Q_REF) ** 2)), "mu_h": MU_H, "mu_g": MU_G, **snap(q_init)})

        def on_event(ev):
            if ev["type"] == "trial_evaluation":
                trials.append({"outer": ev["outer"], "inner": ev["inner"], "alpha": ev["alpha"], "q": ev["q"].tolist(), "p": phi(ev["q"]).tolist(), "A": ev["A"], "included": ev["included"],
                               "outside_hardware": sorted({int(i % 7) + 1 for i in np.nonzero(np.concatenate([lo - ev["q"], ev["q"] - hi]) > FEAS_G_TOL)[0]})})
                return
            out = {k: tl(v) for k, v in ev.items()}
            out["e"] = len(events); out.update(snap(ev["q"]))
            events.append(out)

        q, kap, lam, status, rows = al_ik_nlp(phi, J, lo, hi, q_init, y, on_event=on_event)
        ev_loop, n_act = reduced_hessian(phi, J, lo, hi, q, kap, lam)
        active_final = [int(i) for i in np.nonzero(np.abs(np.concatenate([lo - q, q - hi])) <= ACT_TOL)[0]]
        run = {"label": spec["label"], "note": spec["note"], "target": y.tolist(), "q_init": q_init.tolist(), "init_note": spec["init_note"], "gauge_joint": spec["gauge_joint"], "gauge": gauge,
               "outcome": status, "n_outer": len(rows), "n_events": len(events) - 1, "n_inner": sum(r["n_inner"] for r in rows), "n_trials": len(trials),
               "outer_events": [ev["e"] for ev in events if ev["type"] == "outer_update"],
               "final": {"q": q.tolist(), "kappa": kap.tolist(), "lam": lam.tolist(), "f": 0.5 * float(np.sum((q - Q_REF) ** 2)), "rail_residual": float(np.linalg.norm(phi(q) - y)),
                         "rail_state": events[-1]["rail_state"], "active_bounds": events[-1]["active_bounds"], "included": rows[-1]["included"],
                         "reduced_hessian_eigenvalues": ev_loop.tolist(), "second_order": second_order_verdict(ev_loop, lam, active_final)},
               "rows": [{k: tl(v) for k, v in r.items()} for r in rows], "events": events, "trial_evaluations": trials}
        run["hardware_exceptions"] = ([f"event {ev['e']} (outer {ev['outer']}, inner {ev['inner']}): joints {ev['outside_hardware']}" for ev in events if ev["outside_hardware"]]
                                      + [f"rejected trial (outer {t['outer']}, inner {t['inner']}, alpha {t['alpha']:g}): joints {t['outside_hardware']}" for t in trials if t["outside_hardware"]])
        if abs(y[1]) < 1e-12:
            qm = q_init.copy(); qm[0] = Q_REF[0] - (q_init[0] - Q_REF[0])
            qq, kk, ll, stt, rr = al_ik_nlp(phi, J, lo, hi, qm, y)
            evm, _ = reduced_hessian(phi, J, lo, hi, qq, kk, ll)
            run["mirror"] = {"q_init": qm.tolist(), "q": qq.tolist(), "kappa": kk.tolist(), "lam": ll.tolist(), "f": 0.5 * float(np.sum((qq - Q_REF) ** 2)), "outcome": stt, "n_outer": len(rr),
                             "n_inner": sum(r_["n_inner"] for r_ in rr), "feas_h": rr[-1]["feas_h"], "feas_g": rr[-1]["feas_g"], "stat": rr[-1]["stat"], "comp": rr[-1]["comp"],
                             "reduced_hessian_eigenvalues": evm.tolist()}
        if with_checks:
            run["check"] = solver_reference(plant, pc, Panda, phi, J, lo, hi, y, q_init, loop_q=q)
        rec["runs"][key] = run
    return rec


# ----------------------------------------------------------------------------------------------- print mode and sweep
def print_mode():
    plant, pc, Panda, phi, J, lo, hi = setup()
    print("=== Deck 05's IK program solved by the course's augmented-Lagrangian loop ===")
    print("program: min 1/2|q - q_ref|^2  s.t.  phi(q) - y* = 0 (three rail rows, m), q_lo <= q <= q_hi (14 fence rows, rad); q_ref = the ready pose")
    print(f"weights: mu_h = {MU_H:g} rad^2/m^2 (rail), mu_g = {MU_G:g} (fences), rho = {RHO:g}; inner |grad A| < {INNER_TOL:g} ({INNER_MAX} at most);"
          f" outer stop: max|h| < {FEAS_H_TOL:g} m, max g+ < {FEAS_G_TOL:g} rad, |grad f + J^T kappa + sum lambda grad g| < {STAT_TOL:g} rad, max|lambda g| < {COMP_TOL:g} rad^2 ({OUTER_MAX} at most)")
    rec = record_runs(plant, pc, Panda, phi, J, lo, hi, with_checks=True)
    for key, run in rec["runs"].items():
        print(f"\n--- {run['label']} ---")
        print(f"start q_init = {np.array(run['q_init'])} ({run['init_note']}); reference q_ref = {np.array(rec['q_ref'])}; initial posture cost {run['events'][0]['f']:.1e} rad^2,"
              f" initial rail residual {run['events'][0]['rail_residual']:.4f} m")
        print("outer  inner (status)   |h| (m)     max g+ (rad)  kappa (rad^2/m)                 lambda_2   f (rad^2)   feas_h   feas_g   stat     comp")
        for r in run["rows"]:
            m = lambda n: "pass" if r["passes"][n] else "fail"
            print(f"  {r['outer']:<4} {r['n_inner']:>3} ({r['inner_status']:<10}) {r['rail_residual']:.2e}    {r['feas_g']:.1e}       {np.array2string(np.array(r['kappa_after']), precision=4):<30} {r['lam_after'][1]:.4f}     {r['f']:.5f}     "
                  f"{r['feas_h']:.0e} {m('feas_h')} {r['feas_g']:.0e} {m('feas_g')} {r['stat']:.0e} {m('stat')} {r['comp']:.0e} {m('comp')}")
        fin = run["final"]
        print(f"{run['outcome']} after {run['n_outer']} outer iterations ({run['n_inner']} inner steps, {run['n_trials']} rejected trials); {run['n_events']} events on the timeline")
        print(f"q* = {np.array(fin['q'])}; f* = {fin['f']:.8f}; kappa* = {np.array(fin['kappa'])}; included fence rows at the end: {fin['included']} (0-6 lower bounds of joints 1-7, 7-13 upper);"
              f" lambda of included rows: {[round(float(fin['lam'][i]), 5) for i in fin['included']]}; {fin['rail_state']}")
        print(f"second-order check at the answer: reduced-Hessian eigenvalues {np.array(fin['reduced_hessian_eigenvalues'])}: {fin['second_order']}")
        print("configurations outside the hardware range: " + ("; ".join(run["hardware_exceptions"]) if run["hardware_exceptions"] else "none"))
        if "mirror" in run:
            print(f"mirror branch (joint 1 turned the other way at the start): {run['mirror']['outcome']} after {run['mirror']['n_outer']} outer; q_1 = {run['mirror']['q'][0]:+.4f}, kappa_y = {run['mirror']['kappa'][1]:+.4f}, f* = {run['mirror']['f']:.8f} (the same cost)")
        for name in ("SNOPT", "IPOPT"):
            c = run["check"][name]; m = lambda n_: "pass" if c["passes"][n_] else "FAIL"
            act = ", ".join(("lower" if i < 7 else "upper") + f" bound of joint {(i % 7) + 1}" for i in c["active"]) or "none"
            print(f"compared with {name} from q_init (the identical program; {c['exit']}; options tightened): f* {c['cost']:.5f}, kappa {np.array(c['kappa'])}, active {act}, lambda {[round(float(c['lam'][i]), 5) for i in c['active']]};"
                  f" max|h| {c['feas_h']:.1e} {m('feas_h')}, max g+ {c['feas_g']:.1e} {m('feas_g')}, stationarity {c['stat']:.1e} {m('stat')}, complementarity {c['comp']:.1e} {m('comp')} (largest dual left on an inactive bound {c['largest_dual_on_inactive_bounds']:.1e});"
                  f" second order: {c['second_order']}")
        c = run["check"]["SNOPT"]
        p = c["price_target_x"]; print(f"price check (SNOPT, same branch; re-solves {'succeeded' if p['resolves_succeeded'] else 'FAILED'}, active set {'unchanged' if p['same_active_set'] else 'CHANGED'}): target x +- {p['delta']:g} m: forward difference {p['change']:+.5f}, central difference {p['change_central']:+.5f}; -kappa_x * delta = {p['predicted']:+.5f}; the loop's -kappa_x * delta = {-fin['kappa'][0]*p['delta']:+.5f}")
        if "price_bound" in c:
            p = c["price_bound"]; print(f"price check (SNOPT, same branch; the re-solve {'succeeded' if p['resolves_succeeded'] else 'FAILED'}, active set {'unchanged' if p['same_active_set'] else 'CHANGED'}): joint {p['joint']}'s {p['side']} bound tightened by {p['delta']:g} rad: forward difference {p['change']:+.5f} (tightening only: the URDF bound cannot be relaxed inside this program); lambda * delta = {p['predicted']:+.5f}; the loop's lambda * delta = {fin['lam'][p['joint']-1 if p['side']=='lower' else p['joint']+6]*p['delta']:+.5f}")
        if "symmetric_start" in c:
            sy = c["symmetric_start"]; m = lambda n_: "pass" if sy["passes"][n_] else "FAIL"
            print(f"SNOPT from the symmetric ready pose ({sy['exit']}): f* {sy['cost']:.5f}, q_1 = {sy['q'][0]:+.4f}, lambda_2 = {sy['lam'][1]:.5f}, kappa {np.array(sy['kappa'])};"
                  f" max|h| {sy['feas_h']:.1e} {m('feas_h')}, max g+ {sy['feas_g']:.1e} {m('feas_g')}, stationarity {sy['stat']:.1e} {m('stat')}, complementarity {sy['comp']:.1e} {m('comp')}: {'a KKT point' if sy['kkt_point'] else 'NOT a KKT point within the tolerances'};"
                  f" reduced-Hessian eigenvalues {np.array(sy['reduced_hessian_eigenvalues'])}: {sy['second_order']} (its multipliers describe that stationary branch)")


def sweep():
    plant, pc, Panda, phi, J, lo, hi = setup()
    print("=== sweep: outer iterations (inner steps, rejected trials) to the KKT quartet (1e-6 m, 1e-6 rad, 1e-6 rad, 1e-8 rad^2), 30 outer at most ===")
    for key, spec in RUNS.items():
        y = np.array(spec["target"]); print(f"\n--- {spec['label']} (start {spec['init_note'].split(':')[0]}) ---")
        print("  mu_h    mu_g   status      outer  inner  rejected  feas_h     feas_g     stat       comp       f         kappa                      lambda_2 (lower)  branch (sign of q_1)")
        for mu_h in (1.0, 10.0, 100.0, 1000.0):
            for mu_g in (1.0, 10.0, 100.0):
                ev = []; q, kap, lam, st, rows = al_ik_nlp(phi, J, lo, hi, spec["q_init"], y, mu_h=mu_h, mu_g=mu_g, on_event=ev.append)
                r = rows[-1]
                print(f"  {mu_h:<7g} {mu_g:<6g} {st:<11} {len(rows):<6} {sum(x['n_inner'] for x in rows):<6} {sum(1 for e in ev if e['type']=='trial_evaluation'):<9} "
                      f"{r['feas_h']:.1e}    {r['feas_g']:.1e}    {r['stat']:.1e}    {r['comp']:.1e}    {r['f']:.5f}   {np.array2string(kap, precision=4):<26} {lam[1]:.4f}            {'+' if q[0] > 1e-3 else '-' if q[0] < -1e-3 else '0'}")


def robustness_mode(n=30, seed=0):
    """Perturb each run's start by random offsets of 1e-12, 1e-9 and 1e-6 rad and report whether the event counts and the
    final answer (to 1e-8 rad) stay the same: the inner tolerance is chosen so that they do."""
    plant, pc, Panda, phi, J, lo, hi = setup()
    rng = np.random.default_rng(seed); worst_total = 0
    for key, spec in RUNS.items():
        y = np.array(spec["target"]); q0 = np.array(spec["q_init"])
        ev0 = []; q_ref_ans, kap_ref, lam_ref, st0, rows0 = al_ik_nlp(phi, J, lo, hi, q0, y, on_event=ev0.append)
        base = (len(rows0), sum(r["n_inner"] for r in rows0), sum(1 for e in ev0 if e["type"] == "trial_evaluation"))
        for size in (1e-12, 1e-9, 1e-6):
            same, worst = 0, 0.0
            for _ in range(n):
                ev = []; q, kap, lam, st, rows = al_ik_nlp(phi, J, lo, hi, q0 + size * rng.standard_normal(7), y, on_event=ev.append)
                counts = (len(rows), sum(r["n_inner"] for r in rows), sum(1 for e in ev if e["type"] == "trial_evaluation"))
                same += counts == base; worst = max(worst, float(np.linalg.norm(q - q_ref_ans)))
            worst_total = max(worst_total, worst)
            print(f"{key}: starts perturbed by {size:g} rad: {same}/{n} runs keep the counts {base}; largest final |q - q*| {worst:.1e} rad")
    print(f"largest final difference overall {worst_total:.1e} rad")


def record_mode(out):
    plant, pc, Panda, phi, J, lo, hi = setup()
    rec = record_runs(plant, pc, Panda, phi, J, lo, hi)
    Path(out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {out}: " + ", ".join(f"{k}: {run['n_events']} events" for k, run in rec["runs"].items()))


# ----------------------------------------------------------------------------------------------- self-test
EXPECTED = {"runA": {"n_outer": 3, "n_inner": 16, "n_trials": 0, "outcome": "converged"}, "runB": {"n_outer": 5, "n_inner": 54, "n_trials": 3, "outcome": "converged"}}   # measured 2026-10-15 (--robustness: stable over 90 perturbed starts)


def selftest_mode(baseline=None):
    plant, pc, Panda, phi, J, lo, hi = setup()
    fails = 0

    def check(msg, ok, info=""):
        nonlocal fails
        print(("  ok    " if ok else "  FAIL  ") + msg + ("" if ok or not info else f"  ::  {info}"))
        fails += 0 if ok else 1

    rec = record_runs(plant, pc, Panda, phi, J, lo, hi, with_checks=True)
    src = source_info()
    check("the record's source hashes match the files on disk (" + ("dirty" if src["dirty"] else "clean") + f" at {src['git']})", rec["source"]["files"] == src["files"])
    E = E_ROWS
    for key, run in rec["runs"].items():
        y = np.array(run["target"]); q_init = np.array(run["q_init"]); events = run["events"]; exp = EXPECTED[key]
        q_plain, kap_plain, lam_plain, st_plain, _ = al_ik_nlp(phi, J, lo, hi, q_init, y)
        q_again, kap_again, lam_again, _, _ = al_ik_nlp(phi, J, lo, hi, q_init, y)
        check(f"{key}: the hook does not change the run and a second run is bit-identical (f* {run['final']['f']:.8f}, status {st_plain})",
              np.array_equal(np.array(run["final"]["q"]), q_plain) and np.array_equal(np.array(run["final"]["kappa"]), kap_plain) and np.array_equal(np.array(run["final"]["lam"]), lam_plain)
              and np.array_equal(q_plain, q_again) and np.array_equal(kap_plain, kap_again) and run["outcome"] == st_plain)
        check(f"{key}: counts as documented: {run['n_outer']} outer, {run['n_inner']} inner, {run['n_trials']} rejected trials, {run['outcome']}", all(run[k_] == v for k_, v in exp.items()), str({k_: run[k_] for k_ in exp}))
        check(f"{key}: the start is q_init ({run['init_note'][:40]}...), the reference is the ready pose, separately recorded",
              np.array_equal(np.array(events[0]["q"]), q_init) and np.array_equal(np.array(rec["q_ref"]), gn.Q_READY) and abs(events[0]["f"] - 0.5 * float(np.sum((q_init - gn.Q_READY) ** 2))) < 1e-15)
        f = lambda qq: 0.5 * float(np.sum((qq - gn.Q_READY) ** 2))
        g = lambda qq: np.concatenate([lo - qq, qq - hi])
        worst = {"q": 0.0, "A": 0.0, "grad": 0.0, "dir": 0.0, "fk": 0.0, "est": 0.0, "ptr": 0.0}
        armijo_ok, order_ok, outer_ok, state_ok = True, True, True, True
        prev_q = np.array(events[0]["q"]); kap_cur = np.zeros(3); lam_cur = np.zeros(14); k_cur = 0
        for ev in events:
            q = np.array(ev["q"]); h = phi(q) - y; gq = g(q)
            worst["fk"] = max(worst["fk"], np.linalg.norm(np.array(ev["h"]) - h), abs(ev["rail_residual"] - np.linalg.norm(h)), np.linalg.norm(np.array(ev["p"]) - phi(q)), np.abs(np.array(ev["g"]) - gq).max())
            state_ok &= ev["rail_state"] == ("on the rail" if np.abs(h).max() < FEAS_H_TOL else "off the rail")
            state_ok &= ev["violated_bounds"] == [int(i) for i in np.nonzero(gq > FEAS_G_TOL)[0]] and ev["active_bounds"] == [int(i) for i in np.nonzero(np.abs(gq) <= FEAS_G_TOL)[0]]
            state_ok &= ev["outside_hardware"] == sorted({int(i % 7) + 1 for i in np.nonzero(gq > FEAS_G_TOL)[0]})
            if "gauge" in ev and run["gauge"] is not None:
                pose, tip, _, _ = make_protractor(plant, pc, Panda, run["gauge"]["name"])
                worst["ptr"] = max(worst["ptr"], np.linalg.norm(np.array(ev["gauge"]["pointer"]) - np.array(tip(q))))
        for ev in events[1:]:
            q = np.array(ev["q"])
            if ev["type"] == "accepted_update":
                order_ok &= ev["outer"] == k_cur and np.array_equal(np.array(ev["kappa"]), kap_cur) and np.array_equal(np.array(ev["lam"]), lam_cur)
                qp = np.array(ev["q_prev"]); step = np.array(ev["step"]); dirn = np.array(ev["direction"])
                order_ok &= np.array_equal(qp, prev_q)
                worst["q"] = max(worst["q"], np.linalg.norm(qp + step - q), np.linalg.norm(ev["alpha"] * dirn - step))
                kap, lam, mu_h, mu_g = np.array(ev["kappa"]), np.array(ev["lam"]), ev["mu_h"], ev["mu_g"]
                incl = lambda qq: (g(qq) > 0) | (lam > 0)
                A = lambda qq: f(qq) + float(kap @ (phi(qq) - y)) + mu_h * float((phi(qq) - y) @ (phi(qq) - y)) + float(np.sum(incl(qq) * (lam * g(qq) + mu_g * g(qq) ** 2)))
                gradA = lambda qq: (qq - gn.Q_READY) + J(qq).T @ (kap + 2 * mu_h * (phi(qq) - y)) + E.T @ (incl(qq) * (lam + 2 * mu_g * g(qq)))
                worst["A"] = max(worst["A"], abs(A(q) - ev["A"]), abs(f(q) - ev["f"]), abs(A(qp) - ev["A_prev"]))
                worst["grad"] = max(worst["grad"], abs(np.linalg.norm(gradA(q)) - ev["grad_norm"]), abs(float(gradA(qp) @ dirn) - ev["slope"]))
                worst["est"] = max(worst["est"], np.linalg.norm(np.array(ev["kappa_hat"]) - (kap + 2 * mu_h * (phi(q) - y))),
                                   np.linalg.norm(np.array(ev["lam_estimate"]) - np.where(incl(q), np.maximum(lam + 2 * mu_g * g(q), 0.0), lam)))
                state_ok &= ev["included"] == [int(i) for i in np.nonzero(incl(qp))[0]]
                Jp = J(qp); a = incl(qp).astype(float)
                H = np.eye(7) + 2 * mu_h * Jp.T @ Jp + 2 * mu_g * E.T @ np.diag(a) @ E + RHO * np.eye(7)
                worst["dir"] = max(worst["dir"], np.linalg.norm(H @ dirn + gradA(qp)))
                armijo_ok &= A(q) <= A(qp) + ARMIJO_C * ev["alpha"] * float(gradA(qp) @ dirn) + 1e-15
                prev_q = q
            elif ev["type"] == "outer_update":
                outer_ok &= np.array_equal(q, prev_q) and ev["outer"] == k_cur and np.array_equal(np.array(ev["kappa_before"]), kap_cur) and np.array_equal(np.array(ev["lam_before"]), lam_cur)
                h = phi(q) - y; gq = g(q)
                kap_new = kap_cur + 2 * ev["mu_h"] * h; lam_new = np.maximum(lam_cur + 2 * ev["mu_g"] * gq, 0.0)
                outer_ok &= np.abs(np.array(ev["kappa_after"]) - kap_new).max() < 1e-12 and np.abs(np.array(ev["lam_after"]) - lam_new).max() < 1e-12
                outer_ok &= abs(ev["feas_h"] - np.abs(h).max()) < 1e-15 and abs(ev["feas_g"] - max(gq.max(), 0.0)) < 1e-15
                outer_ok &= abs(ev["stat"] - np.linalg.norm((q - gn.Q_READY) + J(q).T @ kap_new + E.T @ lam_new)) < 1e-10 and abs(ev["comp"] - np.abs(lam_new * gq).max()) < 1e-13
                outer_ok &= ev["included"] == [int(i) for i in np.nonzero((gq > 0) | (lam_new > 0))[0]]
                outer_ok &= np.array_equal(np.array(ev["kappa_hat"]), np.array(ev["kappa_after"])) and np.array_equal(np.array(ev["lam_estimate"]), np.array(ev["lam_after"]))
                kap_cur, lam_cur = np.array(ev["kappa_after"]), np.array(ev["lam_after"]); k_cur += 1
        check(f"{key}: inner events: q = q_prev + step, step = alpha * direction (worst {worst['q']:.1e})", worst["q"] < 1e-12)
        check(f"{key}: inner events: f, A, A_prev, |grad A|, the slope and the multiplier estimates recomputed from FK and the Jacobian (worst {max(worst['A'], worst['grad'], worst['est']):.1e})", max(worst["A"], worst["grad"], worst["est"]) < 1e-9)
        check(f"{key}: inner events: the direction solves the damped system H d = -grad A at q_prev (worst residual {worst['dir']:.1e})", worst["dir"] < 1e-8)
        check(f"{key}: inner events: every accepted step passes the Armijo test with c = {ARMIJO_C}", armijo_ok)
        check(f"{key}: events are in algorithm order (outer counter, frozen multipliers, q_prev chains)", order_ok)
        check(f"{key}: outer events: q unchanged, kappa and lambda updates, the KKT quartet and the inclusion rows recomputed", outer_ok)
        check(f"{key}: the rail state, the violated and active bounds, the hardware flags and the inclusion rows recomputed for every event", state_ok)
        check(f"{key}: FK, the rail residual, the fence rows and the gauge pointer recomputed (worst {max(worst['fk'], worst['ptr']):.1e})", max(worst["fk"], worst["ptr"]) < 1e-12)
        tr_ok = True
        for t in run["trial_evaluations"]:
            inner = [ev for ev in events if ev["type"] == "accepted_update" and ev["outer"] == t["outer"] and ev["inner"] == t["inner"]]
            tr_ok &= len(inner) == 1 and ARMIJO_FLOOR < t["alpha"] <= 1.0
            if inner:
                ev = inner[0]; kap, lam, mu_h, mu_g = np.array(ev["kappa"]), np.array(ev["lam"]), ev["mu_h"], ev["mu_g"]; qp = np.array(ev["q_prev"]); dirn = np.array(ev["direction"])
                incl = lambda qq: (g(qq) > 0) | (lam > 0)
                A = lambda qq: f(qq) + float(kap @ (phi(qq) - y)) + mu_h * float((phi(qq) - y) @ (phi(qq) - y)) + float(np.sum(incl(qq) * (lam * g(qq) + mu_g * g(qq) ** 2)))
                gradA = lambda qq: (qq - gn.Q_READY) + J(qq).T @ (kap + 2 * mu_h * (phi(qq) - y)) + E.T @ (incl(qq) * (lam + 2 * mu_g * g(qq)))
                qt = np.array(t["q"])
                tr_ok &= np.linalg.norm(qt - (qp + t["alpha"] * dirn)) < 1e-12 and abs(A(qt) - t["A"]) < 1e-9 and not (A(qt) <= A(qp) + ARMIJO_C * t["alpha"] * float(gradA(qp) @ dirn))
        check(f"{key}: {run['n_trials']} rejected trials are tied to an inner step, lie on its direction and fail the Armijo test", tr_ok)
        fin = run["final"]; ev_loop, _ = reduced_hessian(phi, J, lo, hi, np.array(fin["q"]), np.array(fin["kappa"]), np.array(fin["lam"]))
        check(f"{key}: the answer is on the rail and a local minimum (reduced-Hessian eigenvalues {np.array(fin['reduced_hessian_eigenvalues'])}, recomputed to {np.abs(ev_loop - np.array(fin['reduced_hessian_eigenvalues'])).max():.1e})",
              fin["rail_state"] == "on the rail" and fin["second_order"] == "local minimum" and ev_loop.min() > 0)
        rng = np.random.default_rng(1); keep = True
        for _ in range(3):
            qq, kk, ll, stt, rr = al_ik_nlp(phi, J, lo, hi, q_init + 1e-6 * rng.standard_normal(7), y)
            keep &= stt == run["outcome"] and len(rr) == run["n_outer"] and sum(r_["n_inner"] for r_ in rr) == run["n_inner"] and np.sign(qq[0]) == np.sign(np.array(fin["q"])[0]) and abs(0.5 * np.sum((qq - gn.Q_READY) ** 2) - fin["f"]) < 1e-9
        check(f"{key}: three starts perturbed by 1e-6 rad keep the counts, the branch (sign of q_1) and f* to 1e-9", keep)
        for name in ("SNOPT", "IPOPT"):
            c = run["check"][name]; T = SOLVER_TOL[name]; R = REGRESSION_TOL
            dq = np.linalg.norm(np.array(c["q"]) - np.array(fin["q"])); dk = np.linalg.norm(np.array(c["kappa"]) - np.array(fin["kappa"])); dl = np.abs(np.array(c["lam"]) - np.array(fin["lam"])).max()
            check(f"{key}: {name} from q_init ({c['exit']}, tightened options) meets the loop's stops: max|h| {c['feas_h']:.1e} < {T['feas_h']:g}, max g+ {c['feas_g']:.1e} < {T['feas_g']:g}, stationarity {c['stat']:.1e} < {T['stat']:g}, complementarity {c['comp']:.1e} < {T['comp']:g}",
                  c["success"] and all(c["passes"].values()))
            check(f"{key}: {name} agrees with the loop on the same branch within the regression tolerances: |df| {abs(c['cost'] - fin['f']):.1e} < {R['cost']:g}, |dq| {dq:.1e} < {R['q']:g}, |dkappa| {dk:.1e} < {R['kappa']:g}, |dlambda| {dl:.1e} < {R['lam']:g}, active {c['active']} vs {fin['active_bounds']}, second order {c['second_order']}",
                  abs(c["cost"] - fin["f"]) < R["cost"] and dq < R["q"] and dk < R["kappa"] and dl < R["lam"] and c["active"] == fin["active_bounds"] and c["second_order"] == "local minimum")
        c = run["check"]["SNOPT"]; p = c["price_target_x"]
        check(f"{key}: price check on the branch (re-solves {'succeeded' if p['resolves_succeeded'] else 'FAILED'}, active set {'unchanged' if p['same_active_set'] else 'CHANGED'}): target x +- 1 mm: forward {p['change']:+.5f} (within 2%), central {p['change_central']:+.5f} (within 0.1%) of -kappa_x delta = {p['predicted']:+.5f}; the loop's {-fin['kappa'][0]*1e-3:+.5f}",
              p["resolves_succeeded"] and p["same_active_set"] and abs(p["change"] - p["predicted"]) < 0.02 * abs(p["predicted"]) and abs(p["change_central"] - p["predicted"]) < 0.001 * abs(p["predicted"]) and abs(-fin["kappa"][0] * 1e-3 - p["predicted"]) < 0.001 * abs(p["predicted"]))
        if "price_bound" in c:
            p = c["price_bound"]; i = p["joint"] - 1 if p["side"] == "lower" else p["joint"] + 6
            check(f"{key}: price check on the branch (the re-solve {'succeeded' if p['resolves_succeeded'] else 'FAILED'}, active set {'unchanged' if p['same_active_set'] else 'CHANGED'}): joint {p['joint']}'s {p['side']} bound tightened by 1e-3 rad: forward {p['change']:+.5f} (within 2%) of lambda delta = {p['predicted']:+.5f}; the loop's {fin['lam'][i]*1e-3:+.5f} (within 0.1%)",
                  p["resolves_succeeded"] and p["same_active_set"] and abs(p["change"] - p["predicted"]) < 0.02 * abs(p["predicted"]) and abs(fin["lam"][i] * 1e-3 - p["predicted"]) < 0.001 * abs(p["predicted"]))
        if "symmetric_start" in c:
            sy = c["symmetric_start"]
            check(f"{key}: SNOPT from the symmetric ready pose ({sy['exit']}) is a KKT point within the solver tolerances (max|h| {sy['feas_h']:.1e}, max g+ {sy['feas_g']:.1e}, stat {sy['stat']:.1e}, comp {sy['comp']:.1e}) and a constrained saddle (eigenvalues {np.array(sy['reduced_hessian_eigenvalues'])}) of higher cost ({sy['cost']:.5f} > {fin['f']:.5f})",
                  sy["kkt_point"] and sy["second_order"] == "constrained saddle" and sy["cost"] > fin["f"])
        if "mirror" in run:
            check(f"{key}: the mirror branch has the same cost ({run['mirror']['f']:.8f} vs {fin['f']:.8f}) with joint 1 and kappa_y mirrored", abs(run["mirror"]["f"] - fin["f"]) < 1e-7 and run["mirror"]["q"][0] * fin["q"][0] < 0 and run["mirror"]["kappa"][1] * fin["kappa"][1] < 0)
    ref = Path(__file__).resolve().parents[2] / "planning/optimization-step-visuals-2026-10-14/increment3/reference-iterates.json"
    if ref.exists():
        R = json.loads(ref.read_text())
        for key, run in rec["runs"].items():
            qs = np.array([ev["q"] for ev in run["events"]]); rq = np.array(R[key]["q"])
            ks = np.array([ev.get("kappa_after", ev.get("kappa")) for ev in run["events"]]); rk = np.array(R[key]["kappa"])
            ls = np.array([ev.get("lam_after", ev.get("lam")) for ev in run["events"]]); rl = np.array(R[key]["lam"])
            same = qs.shape == rq.shape and np.abs(qs - rq).max() < 1e-10 and ks.shape == rk.shape and np.abs(ks - rk).max() < 1e-8 and ls.shape == rl.shape and np.abs(ls - rl).max() < 1e-8
            check(f"{key}: the {len(qs)} events (q, kappa, lambda) agree with the reference re-transcription of the course's rules (increment3/reference_al_eq.py; the same operation order, so this checks the hook and the bookkeeping) (max |q diff| {np.abs(qs - rq).max() if qs.shape == rq.shape else float('nan'):.1e})", same)
    else:
        print(f"  note  no reference file ({ref}); the reference comparison was skipped")
    if baseline and Path(baseline).exists():
        base = json.loads(Path(baseline).read_text())
        for key, run in rec["runs"].items():
            qs = np.array([ev["q"] for ev in run["events"]]); bq = np.array(base[key]["q"])
            ks = np.array([ev.get("kappa_after", ev.get("kappa")) for ev in run["events"]]); bk = np.array(base[key]["kappa"])
            ls = np.array([ev.get("lam_after", ev.get("lam")) for ev in run["events"]]); bl = np.array(base[key].get("lam", ls.tolist()))
            same = qs.shape == bq.shape and np.array_equal(qs, bq) and np.array_equal(ks, bk) and np.array_equal(ls, bl)
            check(f"{key}: the {len(qs)} events (q, kappa, lambda) equal the saved regression baseline at full precision (written from the demo's own record by --write-baseline)", same)
    else:
        print(f"  note  no baseline file ({baseline}); the baseline comparison was skipped")
    print(f"{fails} failures")
    return fails


# ----------------------------------------------------------------------------------------------- figure
def figure_mode(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plant, pc, Panda, phi, J, lo, hi = setup()
    rec = record_runs(plant, pc, Panda, phi, J, lo, hi, with_checks=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    colors = {"runA": "#7a1f3d", "runB": "#1f6f8b"}
    for key, run in rec["runs"].items():
        ev = run["events"]; e = np.array([x["e"] for x in ev]); c = colors[key]; lab = key.replace("run", "run ")
        axes[0, 0].semilogy(e, [max(x["rail_residual"], 1e-12) for x in ev], "-o", ms=3, color=c, label=lab)
        axes[0, 1].plot(e, [x["f"] for x in ev], "-o", ms=3, color=c, label=lab)
        axes[1, 0].step(e, [np.linalg.norm(x.get("kappa_after", x.get("kappa"))) for x in ev], where="post", color=c, label=lab)
        axes[1, 1].step(e, [x.get("lam_after", x.get("lam"))[1] for x in ev], where="post", color=c, label=lab)
        for ax in axes.flat:
            for oe in run["outer_events"]:
                ax.axvline(oe, color=c, alpha=0.25, lw=1)
    axes[0, 0].set(title="Target equality error", xlabel="update event", ylabel="m")
    axes[0, 1].set(title="Posture cost", xlabel="update event", ylabel=r"rad$^2$")
    axes[1, 0].set(title="Equality-multiplier norm", xlabel="update event", ylabel=r"rad$^2$/m")
    axes[1, 1].set(title="Joint 2 lower-bound multiplier", xlabel="update event", ylabel="rad")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3); ax.legend(fontsize=16)
        ax.tick_params(labelsize=16)
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
    fig.suptitle("Hard-target inverse kinematics with a fixed posture objective\nvertical lines mark multiplier updates", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------- replay
def replay_mode(app=False, app_port=0):
    plant0, pc0, Panda0, phi, J, lo, hi = setup()
    rec = record_runs(plant0, pc0, Panda0, phi, J, lo, hi, with_checks=True)
    gauges = {k: make_protractor(plant0, pc0, Panda0, f"panda_joint{run['gauge_joint'] + 1}") for k, run in rec["runs"].items() if run["gauge_joint"] is not None}
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
    gold, maroon, grey, red, green, black, teal = (Rgba(0.8, 0.6, 0.0, 0.55), Rgba(0.55, 0.1, 0.25, 1.0), Rgba(0.45, 0.45, 0.45, 0.9), Rgba(0.85, 0.15, 0.1, 0.9),
                                                   Rgba(0.1, 0.6, 0.2, 0.9), Rgba(0.1, 0.1, 0.1, 1.0), Rgba(0.1, 0.55, 0.55, 1.0))
    meshcat.SetObject("/replay/target", Sphere(0.012), gold)
    meshcat.SetObject("/replay/flange", Sphere(0.007), maroon)
    keys = list(rec["runs"])
    state = {"run": "runA", "e": 0, "trials": False, "arrow": False, "closeup": False, "camera": None, "applied_action_id": 0}
    wide_pts, wide = {}, {}
    for k in keys:
        run = rec["runs"][k]; pts = [[0.0, 0.0, 0.0], run["target"]]
        for ev in run["events"]:
            pts += list(ev["origins"].values()) + [ev["p"]]
            if "gauge" in ev:
                el = np.array(ev["gauge"]["elbow"]); pts += [(el + np.array([0.2, 0, 0])).tolist(), (el - np.array([0.2, 0, 0])).tolist(), (el + np.array([0, 0, 0.2])).tolist(), (el - np.array([0, 0, 0.2])).tolist()]
        wide_pts[k] = pts; wide[k] = gn.fit_camera(pts)
    panel = None
    if app:
        from replay_panel import ReplayPanel
        panel = ReplayPanel(meshcat.web_url(), rec, port=app_port, page=Path(__file__).with_name("panel.html"))

    def set_camera(eye, target):
        meshcat.SetCameraPose(eye, target); state["camera"] = [np.asarray(eye).tolist(), np.asarray(target).tolist()]

    def set_toggle(name, on):
        state[name] = on
        if not on and not state["closeup"] and name == "trials":
            set_camera(*wide[state["run"]])

    SLIDER, SLIDER_MAX = "event e", max(run["n_events"] for run in rec["runs"].values())
    buttons = ["Step", "Back", "Reset", "Next outer", "Previous outer"] + [rec["runs"][k]["label"] for k in keys] + ["Show / hide rejected trials", "Show / hide the multiplier arrow", "Stop"]
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

    def draw_protractor(key):
        meshcat.Delete("/replay/protractor")
        if key not in gauges:
            return
        pose, tip, (lo_j, hi_j), info = gauges[key]
        for n, th in enumerate(np.arange(lo_j, hi_j + 1e-9, 0.06)):
            meshcat.SetObject(f"/replay/protractor/allowed/{n}", Sphere(0.006), green)
            meshcat.SetTransform(f"/replay/protractor/allowed/{n}", RigidTransform([PROTRACTOR_R * np.cos(th), PROTRACTOR_R * np.sin(th), 0.0]))
        for n, th in enumerate(np.arange(lo_j - 0.6, lo_j, 0.06)):
            meshcat.SetObject(f"/replay/protractor/forbidden/{n}", Sphere(0.0045), red)
            meshcat.SetTransform(f"/replay/protractor/forbidden/{n}", RigidTransform([PROTRACTOR_R * np.cos(th), PROTRACTOR_R * np.sin(th), 0.0]))
        sd = np.array([np.cos(lo_j), np.sin(lo_j), 0.0])
        meshcat.SetLine("/replay/protractor/stop", np.column_stack([0.5 * PROTRACTOR_R * sd, 1.25 * PROTRACTOR_R * sd]), 4.0, Rgba(0.55, 0.1, 0.25, 1.0))
        meshcat.SetObject("/replay/protractor/stop_bead", Sphere(0.009), maroon)
        meshcat.SetTransform("/replay/protractor/stop_bead", RigidTransform(1.25 * PROTRACTOR_R * sd))

    def load_run(key):
        run = rec["runs"][key]
        meshcat.SetTransform("/replay/target", RigidTransform(np.array(run["target"])))
        meshcat.Delete("/replay/trials"); meshcat.Delete("/replay/arrow")
        draw_protractor(key)
        state["run"], state["e"] = key, 0
        if not state["closeup"]:
            set_camera(*wide[key])

    def draw(e):
        run = rec["runs"][state["run"]]; events = run["events"]; n = run["n_events"]
        e = int(np.clip(e, 0, n)); state["e"] = e
        ev = events[e]; q = np.array(ev["q"])
        plant.SetPositions(pc, Panda, q)
        inner = ev["type"] == "accepted_update"
        plant.SetPositions(pc, Ghost, np.array(ev["q_prev"]) if inner else q)
        meshcat.SetProperty(ghost_path, "visible", inner)
        p_now = np.array(ev["p"])
        meshcat.SetTransform("/replay/flange", RigidTransform(p_now))
        meshcat.SetLine("/replay/rail_line", np.column_stack([p_now, np.array(run["target"])]), 2.0, black)
        meshcat.Delete("/replay/arrow")
        kap_hat = np.array(ev["kappa_hat"])
        if state["arrow"]:
            tip_a = np.array(arrow_tip(p_now, kap_hat))
            meshcat.SetLine("/replay/arrow/line", np.column_stack([p_now, tip_a]), 4.0, teal)
            meshcat.SetObject("/replay/arrow/tip", Sphere(0.009), teal)
            meshcat.SetTransform("/replay/arrow/tip", RigidTransform(tip_a))
        if state["run"] in gauges:
            pose, tip, (lo_j, hi_j), info = gauges[state["run"]]
            X_WF = pose(q); j = info["index"]
            meshcat.SetTransform("/replay/protractor", X_WF)
            ptr = PROTRACTOR_R * np.array([np.cos(q[j]), np.sin(q[j]), 0.0])
            violating = j in [i for i in ev["violated_bounds"]] or (j + 7) in ev["violated_bounds"]
            meshcat.SetLine("/replay/protractor/pointer", np.column_stack([np.zeros(3), ptr]), 4.0, Rgba(0.85, 0.15, 0.1, 1.0) if violating else black)
            meshcat.SetObject("/replay/protractor/pointer_bead", Sphere(0.012 if violating else 0.008), Rgba(0.85, 0.15, 0.1, 1.0) if violating else black)
            meshcat.SetTransform("/replay/protractor/pointer_bead", RigidTransform(ptr))
        meshcat.Delete("/replay/trials")
        trials = [t for t in run["trial_evaluations"] if inner and t["outer"] == ev["outer"] and t["inner"] == ev["inner"]]
        if state["trials"] and trials:
            for jj, t in enumerate(trials):
                meshcat.SetObject(f"/replay/trials/{jj}", Sphere(0.012), grey)
                meshcat.SetTransform(f"/replay/trials/{jj}", RigidTransform(np.array(t["p"])))
            if not state["closeup"]:
                set_camera(*gn.fit_camera(wide_pts[state["run"]] + [t["p"] for t in trials]))
        kap_now = np.array(ev.get("kappa_after", ev.get("kappa"))); lam_now = np.array(ev.get("lam_after", ev.get("lam")))
        if ev["type"] == "start":
            what = "start (no update yet)"
        elif inner:
            what = f"accepted joint update: outer iteration {ev['outer']}, inner step {ev['inner']} of {run['rows'][ev['outer']]['n_inner']}" + (f"; alpha {ev['alpha']:g}; {ev['rejected']} rejected" if ev["rejected"] else "")
        else:
            what = f"multiplier update after outer iteration {ev['outer']}: kappa {np.array(ev['kappa_before'])} -> {np.array(ev['kappa_after'])}"
        labels = [f"e = {e} of {n}  ({state['run']})", f"event: {what}", f"rail violation |h| = {ev['rail_residual']:.3e} m ({ev['rail_state']})", f"posture cost f = {ev['f']:.5f} rad^2",
                  f"kappa = {kap_now}, lambda of included rows {[round(float(lam_now[i]), 4) for i in ev['included']]}",
                  f"outcome at the end: {run['outcome']} after {run['n_outer']} outer iterations; f* {run['final']['f']:.5f}, |h| {run['final']['rail_residual']:.1e} m"]
        if ev["outside_hardware"]:
            labels.append(f"joints outside the hardware range (violated fence rows) at this event: {ev['outside_hardware']}")
        if state["closeup"]:                                         # the flange from the side, following it
            set_camera(p_now + np.array([0.30, -0.36, -0.06]), p_now)
        if panel is None:
            refresh_readouts(labels)
        diagram.ForcedPublish(ctx)
        if panel is not None:
            panel.publish({"run": state["run"], "e": e, "n": n, "type": ev["type"], "outer": ev["outer"], "inner": ev["inner"], "trials": state["trials"], "arrow": state["arrow"],
                           "closeup": state["closeup"], "camera": state["camera"], "meshcat_url": meshcat.web_url(), "labels": labels, "p": ev["p"], "origins": ev["origins"],
                           "arrow_tip": arrow_tip(p_now, kap_hat), "gauge": ev.get("gauge"), "rail_state": ev["rail_state"], "applied_action_id": state["applied_action_id"]})
        print(f"{state['run']} e = {e}: " + " | ".join(labels[1:5]))

    load_run("runA"); draw(0)
    meshcat.SetProperty(ghost_path, "modulated_opacity", 0.3)
    faded_at = time.time()
    if panel is not None:
        print("Demo app at", panel.start(), flush=True)
    print("Meshcat at", meshcat.web_url())
    print("  Paused replay of the augmented-Lagrangian loop on Deck 05's IK program. Nothing moves until you press Step, Back, Reset, Next outer or")
    print("  Previous outer, or move the 'event e' slider. The line from the flange to the target is the rail violation; the optional arrow is")
    print("  -kappa_hat at the flange (0.02 m per rad^2/m, clamped at 0.30 m; an estimate, not a force). 'Stop' quits.")

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
                elif b == "Show / hide the multiplier arrow": set_toggle("arrow", not state["arrow"]); changed = state["e"]
                elif b in [rec["runs"][k]["label"] for k in keys]:
                    load_run(next(k for k in keys if rec["runs"][k]["label"] == b)); changed = 0
                if changed is not None:
                    state["e"] = changed
                    meshcat.SetSliderValue(SLIDER, float(changed))
        for action_id, do in (panel.pop_actions() if panel is not None else []):
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
            elif do.startswith("arrow:"): set_toggle("arrow", do.endswith("1")); changed = state["e"]
            elif do.startswith("closeup:"):
                state["closeup"] = do.endswith("1"); changed = state["e"]
                if not state["closeup"]:
                    set_camera(*wide[state["run"]])
            state["e"] = changed
            state["applied_action_id"] = action_id
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
    if "--sweep" in sys.argv:
        sweep()
    elif "--robustness" in sys.argv:
        robustness_mode()
    elif "--write-baseline" in sys.argv:
        plant, pc, Panda, phi, J, lo, hi = setup()
        rec = record_runs(plant, pc, Panda, phi, J, lo, hi, with_checks=False)
        base = {k: {"q": [ev["q"] for ev in run["events"]], "kappa": [ev.get("kappa_after", ev.get("kappa")) for ev in run["events"]],
                    "lam": [ev.get("lam_after", ev.get("lam")) for ev in run["events"]]} for k, run in rec["runs"].items()}
        out = sys.argv[sys.argv.index("--write-baseline") + 1]
        Path(out).write_text(json.dumps(base)); print(f"wrote the regression baseline {out} from the demo's own record: " + ", ".join(f"{k}: {len(v['q'])} events" for k, v in base.items()))
    elif "--replay" in sys.argv:
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
