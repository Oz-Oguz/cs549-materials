"""Session 4.2 Drake demo: inverse kinematics as a nonlinear program on the Franka Panda (Spring 2026 deck 03, motion planning, p.53 and p.60).

    min_q   1/2 || q - q_ready ||^2
    s.t.    ^W p^E(q) = y*                                   equality: the flange reaches the target
            q_lo <= q <= q_hi                                the Panda's joint limits from the URDF
            || p_s(q) - c || >= r_o + r_s  for every collision sphere s of links 1 to 8 (44 rows): the whole arm clear of
                                     an obstacle sphere (c, r_o) (only in run 3 and in the live modes)

Solved by SNOPT through Drake's InverseKinematics class. For each run the script prints the solution, which
constraints are active, and the multipliers, which Drake calls dual solutions. The sign of a Drake dual depends
on the kind of constraint: for an inequality it is +lambda at a LOWER bound and -lambda at an UPPER bound
(both checked here by tightening the bound by 1e-3 rad and comparing the cost change with lambda * delta), and for
an equality it is -kappa (checked by shifting the target and comparing with -kappa * delta). Earlier versions
of this file said "Drake reports -lambda" and took abs(); corrected 2026-09-20. The printout applies the rule.

Runs (print mode):
  1  a target in free space: every inequality inactive, so the seven joint-limit duals are zero, but the position
     equality still pulls: kappa = (-2.0705, -0.6748, 0.7056), and a 1 mm target shift in x confirms -kappa_x * 0.001
  2  a target behind the robot, on the robot's symmetry plane: joint 2 runs into its lower limit, its dual is
     nonzero, and a sensitivity check confirms that tightening the limit by 0.001 rad raises the optimal cost by
     about lambda * 0.001. Solved from the ready pose with joint 1 turned by +0.01 rad (2026-10-15): from the
     symmetric ready pose SNOPT returns the symmetric KKT point (q_1 = q_3 = 0, cost 1.729, lambda = 4.196), which
     the second-order check below shows to be a constrained saddle (a negative eigenvalue of the reduced Hessian of
     the Lagrangian on the tangent space of the active rows); the local minimum reached from the offset start has
     cost 1.499, lambda = 1.053, and its mirror image (joint 1 turned the other way) the same cost. Multipliers and
     prices are compared on the same branch; nothing here is a global claim.
  3  the run-1 target with an obstacle sphere above the elbow, c = (0.175, 0.15, 0.75) m, r_o = 0.08 m: two collision
     spheres of link 5 (the forearm) touch it, the cost rises from 0.209 to 0.364, and growing r_o by 1 mm confirms
     the sum of their multipliers. The Panda's collision model in drake_models is spheres (collision_spheres()), so
     the whole-arm clearance is one smooth point-to-point distance row per sphere. Run 1's answer, which ignores the
     sphere, would put the forearm 26.9 mm inside it.
     Then the live demonstration's moves are printed with their costs and active rows (Deck 05 slide 18's cues).

     The multiplier is also recomputed by hand: least squares on the stationarity condition
     grad f + J^T kappa - lambda e_2 = 0, with J the 3x7 translational Jacobian of the flange at the solution.

--app      the live demonstration as a web page around Meshcat (the launcher's /run/ik-nlp, Deck 05 slide 18): target
           sliders and fields (x 0.1..0.9, y -0.6..0.6, z 0.05..0.9 m, steps of 0.01), the cues' target presets, the
           program with its parameters, the cost beside the same target's cost without the sphere, the active rows with
           their multipliers, kappa, each link's smallest clearance, the start that won and a history of the solves.
           Every target change is solved from the last shown configuration and from q_ready, and the lower-cost
           solution is shown. The active set is decided by the primal gap (a joint within 1e-5 rad of a bound, a sphere
           within 1e-6 m of the obstacle), not by the size of a dual. When SNOPT fails from both starts, the arm stays at
           its last solution and the target turns grey; the page says whether the failure is also a proof (a target
           inside the sphere: the flange point lies inside link 7's collision sphere), and whether the kept pose overlaps
           a sphere that has moved since it was solved. The ready pose q_ready is drawn as a faded second arm (a toggle, on
           at the start). The obstacle's centre and radius are editable (centre x -0.8..0.9, y -0.8..0.8, z 0..1.2 m,
           radius 0.02..0.30 m; a "default sphere" button); a change re-solves the current target, and the preset
           "inside the sphere" follows the current sphere. A second toggle shows the arm's collision spheres. Nothing
           solves until the target or the sphere changes.
--meshcat  the same solves with the sliders in Meshcat and one console line per solve, the default sphere, and a button
           that shows or hides the ready pose ("Stop" ends it).

Run:  python ik_nlp_panda.py            prints the three runs and the live moves (notes §5 (k))
      python ik_nlp_panda.py --app      the web page (scripts/demo_server.py id ik-nlp); check: drake/03-ik-as-nlp/check_ik_nlp.mjs
      python ik_nlp_panda.py --meshcat  Meshcat sliders and console lines only
      python ik_nlp_panda.py --png FILE renders run 3 offline from the front left (the arm with the obstacle in orange and the
                                        target in red; the forearm on the sphere's surface): Deck 05 slide 18

Tested 2026-09-28 with drake 1.46.0 on macOS (print mode). On 2026-10-25 the whole-arm model, the new scene and the --app page
were checked headless by drake/03-ik-as-nlp/check_ik_nlp.mjs through the launcher.
"""
import sys
import time
from pathlib import Path
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, InverseKinematics, JacobianWrtVariable,
                         MeshcatVisualizer, Parser, Role, Rgba, RigidTransform, SnoptSolver, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm.urdf"
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


_SPHERES = None


def collision_spheres():
    """The Panda's collision model in drake_models: spheres on links 0 to 8. Returns, for the moving links 1 to 8, a list
    of (link name, centre in the link frame, radius). Clearance of the whole arm from a spherical obstacle (c, r_o) is then
    one smooth row per collision sphere s: || p_s(q) - c || >= r_o + r_s, with p_s(q) the sphere's centre in the world."""
    global _SPHERES
    if _SPHERES is None:
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
        plant.Finalize()
        inspector = scene_graph.model_inspector()
        _SPHERES = []
        for k in range(1, 9):
            body = plant.GetBodyByName(f"panda_link{k}", panda)
            for g in plant.GetCollisionGeometriesForBody(body):
                shape = inspector.GetShape(g)
                _SPHERES.append((body.name(), inspector.GetPoseInFrame(g).translation(), shape.radius()))
    return _SPHERES


# The obstacle sphere (c, r_o) of run 3 and the live demo: above the elbow, chosen by
# planning/ik-nlp-whole-arm-2026-10-25/scene_search.py so that lowering the target changes the active link.
OBSTACLE = (np.array([0.175, 0.15, 0.75]), 0.08)
ACTIVE_JOINT_TOL, ACTIVE_GAP_TOL = 1e-5, 1e-6          # rad, m: the active set is decided by the primal gaps


def sphere_gaps(plant, pc, panda, q, obstacle):
    """Clearance of every collision sphere of links 1 to 8 from the obstacle at q, || p_s(q) - c || - r_o - r_s (m),
    computed from forward kinematics."""
    c, r_o = obstacle
    plant.SetPositions(pc, panda, q)
    out = []
    for name, p_BS, r_s in collision_spheres():
        p = plant.CalcPointsPositions(pc, plant.GetFrameByName(name, panda), p_BS.reshape(3, 1), plant.world_frame()).ravel()
        out.append((name, p, r_s, float(np.linalg.norm(p - c) - r_o - r_s)))
    return out


def obstacle_rows(result, b, obstacle):
    """One entry per obstacle row: link, sphere radius, clearance (m) and the multiplier lambda >= 0 of the row
    g = r_o + r_s - || p_s(q) - c || <= 0. Drake evaluates the squared distance and reports its dual for a lower
    bound on it; lambda for g is 2 d times that dual (checked by the radius price check in print mode)."""
    c, r_o = obstacle
    rows = []
    for name, p_BS, r_s, bd in b["dist"]:
        d = float(np.sqrt(result.EvalBinding(bd)[0]))
        rows.append(dict(link=name, r_s=float(r_s), gap=d - r_o - r_s, lam=2 * d * float(result.GetDualSolution(bd)[0])))
    for r in rows:
        r["active"] = bool(r["gap"] < ACTIVE_GAP_TOL)
    return rows


def link_summary(rows):
    """Per link: the smallest clearance of its rows, the number of active rows and the sum of their multipliers."""
    out = {}
    for r in rows:
        e = out.setdefault(r["link"], dict(gap=9.0, n_active=0, lam=0.0, n_rows=0))
        e["gap"] = min(e["gap"], r["gap"]); e["n_rows"] += 1
        if r["active"]:
            e["n_active"] += 1; e["lam"] += r["lam"]
    return out


def joint_rows(plant, result, q, b):
    """The joint limits at a bound (within ACTIVE_JOINT_TOL) with lambda = +dual at a lower bound, -dual at an upper bound."""
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    dual = result.GetDualSolution(b["limits"])
    out = []
    for j in range(7):
        if abs(q[j] - lo[j]) < ACTIVE_JOINT_TOL:
            out.append(dict(joint=j + 1, side="lower", lam=float(dual[j])))
        elif abs(q[j] - hi[j]) < ACTIVE_JOINT_TOL:
            out.append(dict(joint=j + 1, side="upper", lam=float(-dual[j])))
    return out


def build(meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda))
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    return diagram, ctx, plant, plant.GetMyContextFromRoot(ctx), panda


def solve_ik(plant, pc, panda, target, obstacle=None, q_guess=Q_READY, extra_limits=None):
    """Returns (result, q, bindings) for the NLP above. extra_limits = (lo, hi) adds tightened joint limits (sensitivity check).
    With an obstacle, every collision sphere of links 1 to 8 stays clear of it (collision_spheres())."""
    ik = InverseKinematics(plant, pc)               # creates q and adds q_lo <= q <= q_hi
    q = ik.q()
    prog = ik.prog()
    prog.AddQuadraticErrorCost(0.5*np.eye(7), Q_READY, q)     # 1/2 ||q - q_ready||^2
    E = plant.GetFrameByName("panda_link8", panda)
    pos = ik.AddPositionConstraint(E, np.zeros(3), plant.world_frame(), target, target)
    dist = []
    if obstacle is not None:
        c, r = obstacle
        for name, p_BS, r_s in collision_spheres():              # a binding's row: r_o + r_s - || p_s(q) - c || <= 0
            dist.append((name, p_BS, r_s, ik.AddPointToPointDistanceConstraint(plant.GetFrameByName(name, panda), p_BS,
                                                                               plant.world_frame(), c, r + r_s, 10.0)))
    limits = prog.bounding_box_constraints()[0]
    if extra_limits is not None:
        prog.AddBoundingBoxConstraint(extra_limits[0], extra_limits[1], q)
    result = SnoptSolver().Solve(prog, q_guess, None)
    return result, result.GetSolution(q), dict(pos=pos, dist=dist, limits=limits, obstacle=obstacle)


def reduced_hessian(plant, pc, panda, result, q_sol, b, eps=1e-5):
    """Eigenvalues of Z^T H_L Z: H_L the Hessian of the Lagrangian f + kappa^T h + sum lambda g at the answer (central
    finite differences of its gradient), Z an orthonormal basis of the tangent space of the active rows (the three
    position rows and the bounds with a nonzero dual). All positive: a local minimum (second-order sufficient
    condition, within the finite-difference accuracy); a negative one: a constrained saddle."""
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    kappa = -result.GetDualSolution(b["pos"]); dual = result.GetDualSolution(b["limits"])
    E = plant.GetFrameByName("panda_link8", panda)
    def jac(qq):
        plant.SetPositions(pc, panda, qq)
        return plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, E, np.zeros(3), plant.world_frame(), plant.world_frame())
    rows = [(j, "lower" if abs(q_sol[j] - lo[j]) < 1e-5 else "upper") for j in range(7) if abs(q_sol[j] - lo[j]) < 1e-5 or abs(q_sol[j] - hi[j]) < 1e-5]
    grad_g = np.zeros(7)
    for j, side in rows:
        lam = dual[j] if side == "lower" else -dual[j]
        grad_g[j] += -lam if side == "lower" else lam
    gradL = lambda qq: (qq - Q_READY) + jac(qq).T @ kappa + grad_g
    H = np.zeros((7, 7))
    for i in range(7):
        d = np.zeros(7); d[i] = eps
        H[:, i] = (gradL(q_sol + d) - gradL(q_sol - d)) / (2 * eps)
    H = 0.5 * (H + H.T)
    C = np.vstack([jac(q_sol)] + [np.eye(7)[j:j + 1] for j, _ in rows])
    Z = np.linalg.svd(C)[2][C.shape[0]:].T
    return np.sort(np.linalg.eigvalsh(Z.T @ H @ Z))


def report(plant, pc, panda, result, q_sol, b, label):
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    plant.SetPositions(pc, panda, q_sol)
    p = plant.CalcRelativeTransform(pc, plant.world_frame(), plant.GetFrameByName("panda_link8", panda)).translation()
    print(f"\n--- {label} ---")
    print(f"  success = {result.is_success()},  cost 1/2||q - q_ready||^2 = {result.get_optimal_cost():.5f}")
    print(f"  q* = {q_sol}  (rad)\n  ^W p^E(q*) = {p}")
    dual_lim = result.GetDualSolution(b["limits"])
    active = [(j, "lower" if abs(q_sol[j]-lo[j]) < 1e-5 else "upper") for j in range(7)
              if abs(q_sol[j]-lo[j]) < 1e-5 or abs(q_sol[j]-hi[j]) < 1e-5]
    if active:
        for j, side in active:
            lam = dual_lim[j] if side == "lower" else -dual_lim[j]          # +dual at a lower bound, -dual at an upper bound
            print(f"  joint {j+1} at its {side} limit ({q_sol[j]:+.4f} rad); Drake dual = {dual_lim[j]:+.5f}  ->  lambda = {lam:.5f}")
    else:
        print(f"  no joint at a limit; all seven joint-limit duals = {dual_lim}")
    if b["dist"]:
        rows = obstacle_rows(result, b, b["obstacle"])
        for k, r in enumerate(rows):
            if r["active"]:
                print(f"  obstacle row {k} ({r['link']}, collision sphere r_s = {r['r_s']:.3f} m): clearance {r['gap']:+.1e} m, ACTIVE, lambda = {r['lam']:.4f}")
        print("  per link: " + "; ".join(f"{n[6:]} {v['gap']*1000:+.1f} mm" + (f" ({v['n_active']} active, lambda sum {v['lam']:.4f})" if v["n_active"] else "")
                                          for n, v in link_summary(rows).items()))
    print(f"  equality duals (Drake sign) = {result.GetDualSolution(b['pos'])}  ->  kappa = -dual = {-result.GetDualSolution(b['pos'])}")
    return active


def print_mode():
    diagram, ctx, plant, pc, panda = build()
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    print("=== IK as a nonlinear program on the Franka Panda (SNOPT through Drake's InverseKinematics) ===")
    print(f"  joint limits (rad): lower {plant.GetPositionLowerLimits()}\n                      upper {hi}")

    y1 = np.array([0.45, 0.10, 0.45])
    res, q, b = solve_ik(plant, pc, panda, y1)
    report(plant, pc, panda, res, q, b, f"run 1: free-space target y* = {y1}")
    kappa = -res.GetDualSolution(b["pos"])                    # the sign rule, checked: shift the target and compare
    for i, delta in ((0, 0.001), (2, 0.001)):
        y_shift = y1.copy(); y_shift[i] += delta
        res_s, _, _ = solve_ik(plant, pc, panda, y_shift, q_guess=q)
        print(f"  sensitivity: shift the target's {'xyz'[i]} by +{delta} m -> cost {res.get_optimal_cost():.5f} -> {res_s.get_optimal_cost():.5f},"
              f" change {res_s.get_optimal_cost()-res.get_optimal_cost():+.5f};  -kappa_{'xyz'[i]} * delta = {-kappa[i]*delta:+.5f}")

    y2 = np.array([-0.20, 0.00, 0.60])          # behind the robot: the shoulder joint 2 runs into its lower limit
    q_init2 = Q_READY + np.array([0.01, 0, 0, 0, 0, 0, 0])      # 0.01 rad off the symmetry plane: the symmetric start leads SNOPT to a saddle (below)
    res, q, b = solve_ik(plant, pc, panda, y2, q_guess=q_init2)
    active = report(plant, pc, panda, res, q, b, f"run 2: target behind the robot y* = {y2}, started 0.01 rad off the symmetry plane (joint 1)")
    ev_main = reduced_hessian(plant, pc, panda, res, q, b)
    verdict = lambda ev: "all positive, a local minimum" if ev.min() > 0 else "a negative eigenvalue, a constrained saddle" if ev.min() < 0 else "a zero eigenvalue, inconclusive"
    print(f"  second-order check at this answer: reduced Hessian of the Lagrangian on the tangent space of the active rows, eigenvalues {ev_main}: {verdict(ev_main)}")
    res_m, q_m, b_m = solve_ik(plant, pc, panda, y2, q_guess=Q_READY - np.array([0.01, 0, 0, 0, 0, 0, 0]))
    mirror = abs(res_m.get_optimal_cost() - res.get_optimal_cost()) < 1e-6 and q_m[0] * q[0] < 0 and abs(abs(q_m[0]) - abs(q[0])) < 1e-4
    print(f"  the mirror branch (joint 1 turned by -0.01 rad at the start): cost {res_m.get_optimal_cost():.5f}, q_1 = {q_m[0]:+.4f}, q_3 = {q_m[2]:+.4f}, kappa = {-res_m.GetDualSolution(b_m['pos'])}:"
          f" {'the same cost, the mirror image' if mirror else 'NOT the mirror image'}")
    res_s, q_s, b_s = solve_ik(plant, pc, panda, y2)
    kap_s = -res_s.GetDualSolution(b_s["pos"]); lam_s = res_s.GetDualSolution(b_s["limits"])[1]
    plant.SetPositions(pc, panda, q_s); Es = plant.GetFrameByName("panda_link8", panda)
    p_s = plant.CalcRelativeTransform(pc, plant.world_frame(), Es).translation()
    J_s = plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, Es, np.zeros(3), plant.world_frame(), plant.world_frame())
    feas_h = float(np.abs(p_s - y2).max()); feas_g = float(max(np.concatenate([lo - q_s, q_s - hi]).max(), 0.0))
    stat_s = float(np.linalg.norm((q_s - Q_READY) + J_s.T @ kap_s - lam_s * np.eye(7)[1])); comp_s = float(abs(lam_s * (lo[1] - q_s[1])))
    kkt = res_s.is_success() and feas_h < 1e-6 and feas_g < 1e-6 and stat_s < 1e-5 and comp_s < 1e-8
    ev_s = reduced_hessian(plant, pc, panda, res_s, q_s, b_s)
    print(f"  from the symmetric ready pose SNOPT returns q = {q_s}, cost {res_s.get_optimal_cost():.5f}, joint 2 at its lower limit with lambda = {lam_s:.5f}, kappa = {kap_s};"
          f" KKT residuals max|h| {feas_h:.1e} m, max g+ {feas_g:.1e} rad, stationarity {stat_s:.1e}, complementarity {comp_s:.1e}: {'a KKT point' if kkt else 'NOT a KKT point within 1e-6 / 1e-6 / 1e-5 / 1e-8'};"
          f" reduced-Hessian eigenvalues {ev_s}: {verdict(ev_s)} (its multipliers describe that stationary branch; Deck 05 slide 19 quoted it as run 2 until 2026-10-15)")
    if active:
        j, side = active[0]
        dual = res.GetDualSolution(b["limits"])[j]
        lo_t, hi_t, delta = lo.copy(), hi.copy(), 0.001      # the verified sensitivity step; a larger step can change the active set (notes §5 (k))
        if side == "upper": hi_t[j] -= delta
        else:               lo_t[j] += delta
        res2, q2, _ = solve_ik(plant, pc, panda, y2, q_guess=q, extra_limits=(lo_t, hi_t))   # from the answer: the price check stays on the same branch
        print(f"  sensitivity: tighten joint {j+1}'s {side} limit by {delta} rad (solve success = {res2.is_success()}) -> cost {res.get_optimal_cost():.5f} -> {res2.get_optimal_cost():.5f},"
              f" change {res2.get_optimal_cost()-res.get_optimal_cost():+.5f};  lambda * delta = {(dual if side == 'lower' else -dual)*delta:+.5f}")
        # the multiplier by hand: least squares on the stationarity condition grad f + J^T kappa + lambda * grad g = 0,
        # with grad f = q - q_ready, J the 3x7 translational Jacobian of the flange, and grad g = -e_j for a lower bound (g = lo_j - q_j)
        plant.SetPositions(pc, panda, q)
        E = plant.GetFrameByName("panda_link8", panda)
        Jq = plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, E, np.zeros(3), plant.world_frame(), plant.world_frame())
        e = np.zeros(7); e[j] = -1.0 if side == "lower" else 1.0
        M = np.column_stack([Jq.T, e])
        mult, *_ = np.linalg.lstsq(M, -(q - Q_READY), rcond=None)
        print(f"  multiplier by hand (least squares on grad f + J^T kappa + lambda grad g = 0): lambda = {mult[3]:.5f}, kappa = {mult[:3]},"
              f" stationarity residual {np.linalg.norm((q - Q_READY) + M @ mult):.1e}  (Drake's lambda: {(dual if side == 'lower' else -dual):.5f})")

    obstacle = OBSTACLE
    res, q, b = solve_ik(plant, pc, panda, y1, obstacle=obstacle)
    report(plant, pc, panda, res, q, b, f"run 3: run-1 target with an obstacle sphere at {obstacle[0]}, r_o = {obstacle[1]}; every collision sphere of links 1 to 8 stays clear ({len(b['dist'])} rows)")
    rows = obstacle_rows(res, b, obstacle); lam_sum = sum(r["lam"] for r in rows if r["active"])
    delta = 0.001
    res_r, _, _ = solve_ik(plant, pc, panda, y1, obstacle=(obstacle[0], obstacle[1] + delta), q_guess=q)
    print(f"  sensitivity: grow r_o by {delta} m -> cost {res.get_optimal_cost():.5f} -> {res_r.get_optimal_cost():.5f}, change {res_r.get_optimal_cost()-res.get_optimal_cost():+.6f};"
          f"  (sum of the active lambdas) * delta = {lam_sum*delta:+.6f}")
    res_1, q_1, _ = solve_ik(plant, pc, panda, y1)
    worst = min(sphere_gaps(plant, pc, panda, q_1, obstacle), key=lambda t: t[3])
    print(f"  run 1's answer (cost {res_1.get_optimal_cost():.5f}), which ignores the sphere, would have the clearance {worst[3]*1000:+.1f} mm ({worst[0]}):"
          f" the rows move the arm around the sphere")

    print("\n--- the live demonstration's moves (each solved from the previous answer and from q_ready; the lower cost is kept) ---")
    q_prev = q
    for y in ([0.45, 0.10, 0.35], [0.45, 0.10, 0.30], [0.45, 0.10, 0.45], [0.18, 0.10, 0.75], [0.45, 0.10, 0.45], [0.10, 0.0, 0.25], [0.84, 0.10, 0.45], [0.85, 0.10, 0.45]):
        y = np.array(y)
        st = solve_live(plant, pc, panda, y, q_prev, obstacle)
        if st["success"]:
            q_prev = np.array(st["q"])
        print("  " + live_line(st))
    st0 = solve_live(plant, pc, panda, y1, q_prev, obstacle)
    st1 = solve_live(plant, pc, panda, y1, np.array(st0["q"]), (obstacle[0], obstacle[1] + 0.001))
    lam0 = sum(r["lam"] for r in st0["rows_active"])
    print(f"  the page's price check at the start: radius {obstacle[1]} -> {obstacle[1] + 0.001} m, cost {st0['cost']:.5f} -> {st1['cost']:.5f},"
          f" change {st1['cost'] - st0['cost']:+.5f}; (sum of the active lambdas) * 0.001 = {lam0 * 0.001:+.5f}")




def render_png(out, q, spheres, title, eye=(-0.02, 1.84, 1.13), look_at=(0.30, 0.05, 0.40)):
    """Offline picture with Drake's VTK render engine (no browser): the Panda at q with coloured spheres
    (target, obstacle) registered as world visuals, seen from eye toward look_at (figure_mode passes FIG_EYE and FIG_LOOK_AT)."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
                             MakeRenderEngineVtk, RenderCameraCore, RenderEngineVtkParams, RgbdSensor, RotationMatrix)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda))
    for i, (centre, radius, rgba) in enumerate(spheres):
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(centre), Sphere(radius), f"sphere{i}", rgba)
    plant.Finalize()
    eye, target = np.array(eye, dtype=float), np.array(look_at, dtype=float)
    z_cam = target - eye; z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, [0, 0, 1.0]); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    X_WCam = RigidTransform(RotationMatrix(np.column_stack([x_cam, y_cam, z_cam])), eye)
    core = RenderCameraCore("vtk", CameraInfo(1280, 960, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WCam, ColorRenderCamera(core, False),
                                          DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(ctx), panda, q)
    img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(ctx))
    Image.fromarray(np.array(img.data)[:, :, :3]).save(out)
    autocrop(out)
    print(f"rendered {out}: {title}")


def autocrop(path, margin=40):
    """Crop the rendered picture to the arm and the spheres plus a margin, so the slide does not carry empty sky."""
    from PIL import Image
    im = np.array(Image.open(path)); background = im[0, 0]
    ys, xs = np.where(np.any(im != background, axis=2))
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, im.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, im.shape[1])
    Image.fromarray(im[y0:y1, x0:x1]).save(path)


FIG_EYE, FIG_LOOK_AT = (1.05, -1.45, 1.0), (0.28, 0.08, 0.52)   # from the front left: the forearm under the sphere and the target


def figure_mode(out):
    diagram, ctx, plant, pc, panda = build()
    y1 = np.array([0.45, 0.10, 0.45]); obstacle = OBSTACLE
    res, q, b = solve_ik(plant, pc, panda, y1, obstacle=obstacle)
    # opaque orange for the obstacle: a grey semi-transparent sphere is invisible against the grey links
    render_png(out, q, [(y1, 0.02, [0.9, 0.1, 0.1, 1.0]), (obstacle[0], obstacle[1], [0.95, 0.55, 0.10, 1.0])],
               f"run 3: success = {result_word(res)}, cost {res.get_optimal_cost():.5f}, link 5 on the sphere",
               eye=FIG_EYE, look_at=FIG_LOOK_AT)


def result_word(res):
    return "True" if res.is_success() else "False"


def solve_live(plant, pc, panda, y, q_prev, obstacle):
    """One move of the live demonstration: solve from the last shown configuration and from q_ready, keep the lower
    cost, and describe the answer (JSON-ready). A warm start alone can fail, or land in a worse valley, for a target
    that q_ready solves (seen 2026-10-06 after an excursion toward the reach limit); two starts keep the shown answer
    reproducible. Also solves without the obstacle, so that the obstacle's effect on the cost can be read."""
    t0 = time.time()
    y = np.asarray(y, dtype=float)
    starts = [("the last pose", q_prev)] + ([("q_ready", Q_READY)] if np.any(q_prev != Q_READY) else [])
    solved = []
    for label, q_guess in starts:
        res, q, b = solve_ik(plant, pc, panda, y, obstacle=obstacle, q_guess=q_guess)
        if res.is_success():
            solved.append((res.get_optimal_cost(), label, q, b, res))
    res_free, _, _ = solve_ik(plant, pc, panda, y, q_guess=q_prev)
    c, r_o = obstacle
    st = dict(target=y.tolist(), obstacle=[c.tolist(), r_o], success=bool(solved),
              free_cost=float(res_free.get_optimal_cost()) if res_free.is_success() else None,
              target_inside=bool(np.linalg.norm(y - c) < r_o), target_to_centre=float(np.linalg.norm(y - c)))
    if solved:
        solved.sort(key=lambda t: t[0])
        cost, label, q, b, res = solved[0]
        rows = obstacle_rows(res, b, obstacle)
        gaps = sphere_gaps(plant, pc, panda, q, obstacle)
        worst = min(gaps, key=lambda t: t[3])
        st.update(cost=float(cost), q=q.tolist(), start=label, kappa=(-res.GetDualSolution(b["pos"])).tolist(),
                  joints=joint_rows(plant, res, q, b),
                  rows_active=[dict(index=k, **r) for k, r in enumerate(rows) if r["active"]],
                  links=link_summary(rows),
                  arm_gaps={n[6:]: min(g for m_, _, _, g in gaps if m_ == n) for n in dict.fromkeys(m_ for m_, _, _, _ in gaps)},
                  arm_min=[worst[0], worst[3]], spheres=[[p.tolist(), r_s, g] for _, p, r_s, g in gaps],
                  other_cost=float(solved[1][0]) if len(solved) == 2 and abs(solved[0][0] - solved[1][0]) > 1e-4 else None,
                  other_start=solved[1][1] if len(solved) == 2 else None, n_rows=len(rows))
        plant.SetPositions(pc, panda, q)
    st["ms"] = round(1000 * (time.time() - t0))
    return st


def live_line(st):
    y = np.array(st["target"])
    if not st["success"]:
        why = ("The target lies inside the sphere, so this failure is also a proof: the flange point sits inside link 7's collision sphere"
               if st["target_inside"] else "This is a solver failure, not a proof of infeasibility")
        return f"y* = {y}  no feasible point found by SNOPT from the last pose or from q_ready. {why}. The arm stays at its last solution; the target is drawn grey."
    act = [f"joint {j['joint']} ({j['side']}, lambda {j['lam']:.4f})" for j in st["joints"]]
    act += [f"{n[6:]} ({v['n_active']} sphere{'s' if v['n_active'] > 1 else ''}, lambda {v['lam']:.4f})" for n, v in st["links"].items() if v["n_active"]]
    valleys = (f"  [two local solutions: cost {st['cost']:.4f} from {st['start']}, cost {st['other_cost']:.4f} from {st['other_start']}; the lower one is shown]"
               if st["other_cost"] is not None else "")
    free = f", without the sphere {st['free_cost']:.4f}" if st["free_cost"] is not None else ""
    arm = f"; whole-arm clearance {st['arm_min'][1]*1000:+.1f} mm ({st['arm_min'][0][6:]})"
    kappa = "; kappa (" + ", ".join(f"{k + 0.0:.3f}" for k in np.round(st["kappa"], 3)) + ")"
    return f"y* = {y}  cost = {st['cost']:.4f}{free}  active: {', '.join(act) if act else 'none'}{arm}{kappa}{valleys}"


PRESETS = (("start", "the start (run 3)", [0.45, 0.10, 0.45]), ("lower10", "10 cm lower", [0.45, 0.10, 0.35]),
           ("lower15", "15 cm lower", [0.45, 0.10, 0.30]), ("base", "near the base", [0.10, 0.0, 0.25]), ("inside", "inside the sphere", [0.18, 0.10, 0.75]),
           ("far84", "far: x = 0.84 m", [0.84, 0.10, 0.45]), ("far85", "far: x = 0.85 m", [0.85, 0.10, 0.45]))
SLIDER_RANGES = (("x", 0.1, 0.9, 0.45), ("y", -0.6, 0.6, 0.10), ("z", 0.05, 0.9, 0.45))


OBSTACLE_RANGES = (("x", -0.8, 0.9), ("y", -0.8, 0.8), ("z", 0.0, 1.2), ("r", 0.02, 0.30))   # m: the page's sphere fields


def build_display(meshcat):
    """The scene drawn in Meshcat: the solved Panda and a second Panda kept at q_ready, drawn faded (the ghost; the
    pattern of drake/05-ik-nlp-al). The solves use build()'s plant, which has only one robot, so the ghost adds no
    decision variables."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    (panda,) = parser.AddModelsFromUrl(PANDA_URL)
    parser.SetAutoRenaming(True)
    (ghost,) = parser.AddModelsFromUrl(PANDA_URL)
    for m in (panda, ghost):
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", m))
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pc = plant.GetMyContextFromRoot(ctx)
    plant.SetPositions(pc, ghost, Q_READY)
    return diagram, ctx, plant, pc, panda, f"/drake/visualizer/{plant.GetModelInstanceName(ghost)}"


def meshcat_mode(app=False, app_port=0):
    """--meshcat: target sliders in Meshcat and one console line per solve. --app: the same solves behind a web page
    (Meshcat embedded) that shows the program, the cost, the active rows with their multipliers, the per-link
    clearances, a history of the solves, the target presets of the cues, editable obstacle-sphere parameters, and
    two toggles: the ready pose q_ready as a faded ghost (on at the start) and the robot's collision spheres.
    Nothing solves until the target or the sphere changes."""
    meshcat = StartMeshcat()
    _, _, plant, pc, panda = build()                       # the solves' plant: one robot
    diagram_d, ctx_d, plant_d, pc_d, panda_d, ghost_path = build_display(meshcat)
    red, grey = Rgba(0.9, 0.1, 0.1, 1.0), Rgba(0.45, 0.45, 0.45, 1.0)
    maroon, pale = Rgba(0.55, 0.1, 0.25, 0.55), Rgba(0.2, 0.45, 0.8, 0.18)
    meshcat.SetObject("/target", Sphere(0.015), red)
    meshcat.SetCameraPose([1.0, -0.95, 0.95], [0.22, 0.05, 0.48])        # from the front left: the forearm under the sphere and the target
    state = dict(spheres=False, ghost=True, obstacle=(OBSTACLE[0].copy(), OBSTACLE[1]), applied_action_id=0,
                 history=[], last=None, q_prev=Q_READY, st=None)
    panel = None
    if app:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02-gauss-newton-ik"))
        from replay_panel import ReplayPanel
        record = dict(obstacle=[OBSTACLE[0].tolist(), OBSTACLE[1]], q_ready=Q_READY.tolist(), presets=[dict(key=k, label=l, target=t) for k, l, t in PRESETS],
                      sliders=[dict(axis=a, lo=lo, hi=hi, value=v) for a, lo, hi, v in SLIDER_RANGES], n_spheres=len(collision_spheres()),
                      obstacle_ranges=[dict(key=k, lo=lo, hi=hi) for k, lo, hi in OBSTACLE_RANGES],
                      lower=plant.GetPositionLowerLimits().tolist(), upper=plant.GetPositionUpperLimits().tolist(),
                      tolerances=dict(joint=ACTIVE_JOINT_TOL, gap=ACTIVE_GAP_TOL))
        panel = ReplayPanel(meshcat.web_url(), record, port=app_port, page=Path(__file__).with_name("panel.html"))
    else:
        for a, lo, hi, v in SLIDER_RANGES:
            meshcat.AddSlider(f"target {a}", lo, hi, 0.01, v)
        meshcat.AddButton("Show / hide the ready pose")
        meshcat.AddButton("Stop")

    def draw_obstacle():
        c, r_o = state["obstacle"]
        meshcat.SetObject("/obstacle", Sphere(r_o), Rgba(0.95, 0.55, 0.10, 0.45))
        meshcat.SetTransform("/obstacle", RigidTransform(c))

    def set_ghost(on):
        state["ghost"] = on
        meshcat.SetProperty(ghost_path, "visible", on)
        meshcat.SetProperty(ghost_path, "modulated_opacity", 0.25)

    def draw_spheres(st):
        meshcat.Delete("/arm_spheres")
        if not st or not st["success"]:
            return
        for k, (p, r_s, g) in enumerate(st["spheres"]):
            touching = abs(g) <= ACTIVE_GAP_TOL
            if state["spheres"] or touching:
                meshcat.SetObject(f"/arm_spheres/{k}", Sphere(r_s), maroon if touching else pale)
                meshcat.SetTransform(f"/arm_spheres/{k}", RigidTransform(np.array(p)))

    def solve_and_show(y):
        state["last"] = y
        meshcat.SetTransform("/target", RigidTransform(y))
        st = solve_live(plant, pc, panda, y, state["q_prev"], state["obstacle"])
        if st["success"]:
            state["q_prev"] = np.array(st["q"])
            meshcat.SetObject("/target", Sphere(0.015), red)
        else:
            meshcat.SetObject("/target", Sphere(0.015), grey)
            worst = min(sphere_gaps(plant, pc, panda, state["q_prev"], state["obstacle"]), key=lambda t: t[3])
            st["kept_min"] = [worst[0], worst[3]]                 # the kept pose against the current sphere, which may have moved
        plant_d.SetPositions(pc_d, panda_d, state["q_prev"])      # a failed solve keeps the last answer
        diagram_d.ForcedPublish(ctx_d)
        state["st"] = st
        draw_spheres(st)
        default = np.allclose(state["obstacle"][0], OBSTACLE[0]) and abs(state["obstacle"][1] - OBSTACLE[1]) < 1e-12
        state["history"] = ([dict(target=st["target"], success=st["success"], cost=st.get("cost"),
                                  obstacle=None if default else [state["obstacle"][0].tolist(), state["obstacle"][1]],
                                  active=live_line(st).split("active: ")[1].split(";")[0] if st["success"] else "no feasible point found")] + state["history"])[:8]
        print("  " + live_line(st), flush=True)
        publish()

    def publish():
        if panel is not None:
            panel.publish(dict(st=state["st"], spheres=state["spheres"], ghost=state["ghost"],
                               obstacle=[state["obstacle"][0].tolist(), state["obstacle"][1]], history=state["history"],
                               applied_action_id=state["applied_action_id"], meshcat_url=meshcat.web_url()))

    print("Meshcat at", meshcat.web_url(), flush=True)
    print("  Every collision sphere of links 1 to 8 must stay clear of the orange sphere; the flange must sit on the target.")
    print("  The faded arm is the ready pose q_ready, the reference of the cost.")
    print("  Each move is solved from the last shown configuration and from q_ready, and the lower-cost solution is shown.")
    print("  The active set is decided by the primal gaps (1e-5 rad, 1e-6 m), not by the size of a dual.")
    draw_obstacle()
    solve_and_show(np.array([v for _, _, _, v in SLIDER_RANGES]))
    set_ghost(True)
    faded_at = time.time()
    if panel is not None:
        print("Demo app at", panel.start(), flush=True)
    ghost_clicks, stop = 0, False
    while not stop:
        if panel is None:
            if meshcat.GetButtonClicks("Stop") > 0:
                break
            if meshcat.GetButtonClicks("Show / hide the ready pose") != ghost_clicks:
                ghost_clicks = meshcat.GetButtonClicks("Show / hide the ready pose"); set_ghost(not state["ghost"])
            y = np.array([meshcat.GetSliderValue(f"target {a}") for a in "xyz"])
            if np.any(y != state["last"]):
                solve_and_show(y)
        else:
            for action_id, do in panel.pop_actions():
                state["applied_action_id"] = action_id
                if do.startswith("target:"):
                    vals = [float(v) for v in do[7:].split(",")]
                    y = np.array([np.clip(v, lo, hi) for v, (_, lo, hi, _) in zip(vals, SLIDER_RANGES)])
                    if np.any(y != state["last"]):
                        solve_and_show(y); continue
                elif do.startswith("obstacle:"):
                    if do == "obstacle:default":
                        new = (OBSTACLE[0].copy(), OBSTACLE[1])
                    else:
                        vals = [float(v) for v in do[9:].split(",")]
                        if len(vals) != 4 or not all(np.isfinite(vals)):
                            publish(); continue
                        vals = [float(np.clip(v, lo, hi)) for v, (_, lo, hi) in zip(vals, OBSTACLE_RANGES)]
                        new = (np.array(vals[:3]), vals[3])
                    if not (np.allclose(new[0], state["obstacle"][0]) and new[1] == state["obstacle"][1]):
                        state["obstacle"] = new; draw_obstacle(); solve_and_show(state["last"]); continue
                elif do.startswith("spheres:"):
                    state["spheres"] = do.endswith("1"); draw_spheres(state["st"])
                elif do.startswith("ghost:"):
                    set_ghost(do.endswith("1"))
                elif do == "stop":
                    stop = True
                publish()
        if state["ghost"] and time.time() - faded_at > 1.0:         # Meshcat resets the opacity when the meshes finish loading
            meshcat.SetProperty(ghost_path, "modulated_opacity", 0.25); faded_at = time.time()
        time.sleep(0.05)
    if panel is None:
        meshcat.DeleteAddedControls()
    else:
        panel.close()
    print("stopped", flush=True)


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    elif "--app" in sys.argv:
        meshcat_mode(app=True, app_port=int(sys.argv[sys.argv.index("--app-port") + 1]) if "--app-port" in sys.argv else 0)
    elif "--meshcat" in sys.argv:
        meshcat_mode()
    else:
        print_mode()
