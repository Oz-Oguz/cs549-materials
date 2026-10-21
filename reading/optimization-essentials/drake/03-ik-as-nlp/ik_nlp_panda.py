"""Session 4.2 Drake demo: inverse kinematics as a nonlinear program on the Franka Panda (Spring 2026 deck 03, motion planning, p.53 and p.60).

    min_q   1/2 || q - q_ready ||^2
    s.t.    ^W p^E(q) = y*                                   equality: the flange reaches the target
            q_lo <= q <= q_hi                                the Panda's joint limits from the URDF
            || p(q) - c || >= r_o   for each of three wrist points p  an obstacle sphere (only in run 3 and in --meshcat)

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
  3  the run-1 target with an obstacle sphere next to it: the distance constraint becomes active

     The multiplier is also recomputed by hand: least squares on the stationarity condition
     grad f + J^T kappa - lambda e_2 = 0, with J the 3x7 translational Jacobian of the flange at the solution.

--meshcat  sliders for the target position (x 0.1..0.9, y -0.6..0.6, z 0.05..0.9 m, steps of 0.01); the sliders
           start at run 3's target and the sphere is always drawn, so the first solve is run 3. Every move is
           solved from the last shown configuration and from q_ready, and the lower-cost solution is shown
           (the console says when the two starts end in different local solutions). The console prints the
           active set, decided by the primal gap (a joint within 1e-5 rad of a bound, a wrist point within
           1e-6 m of the sphere), not by the size of a dual, which SNOPT leaves at 1e-5 to 1e-4 on inactive
           bounds. When SNOPT fails from both starts, the arm stays at its last solution, the target turns
           grey, and the console says that this is a solver failure and not a proof of infeasibility.
           Target (red) and obstacle (grey) are drawn. "Stop" ends it. Run 2's target lies behind the
           robot, outside the x slider's range; it is a print-mode result only.

Run:  python ik_nlp_panda.py            prints the three runs (notes §5 (k))
      python ik_nlp_panda.py --meshcat  target sliders; the arm re-solves and the console prints the active set
      python ik_nlp_panda.py --png FILE renders run 3 offline (the arm solved with the obstacle sphere in orange and the
                                        target in red; the wrist frame sits on the sphere's surface): Deck 05 slide 18

Tested 2026-09-28 with drake 1.46.0 on macOS (print mode). On 2026-10-06 the Meshcat mode was checked headless by a script
that moves the sliders through Meshcat's own controls and reads this console.
"""
import sys
import time
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, InverseKinematics, JacobianWrtVariable,
                         MeshcatVisualizer, Parser, Rgba, RigidTransform, SnoptSolver, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm.urdf"
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
WRIST_POINTS = ["panda_link8", "panda_link7", "panda_link6"]        # frame origins kept out of the obstacle


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
    """Returns (result, q, bindings) for the NLP above. extra_limits = (lo, hi) adds tightened joint limits (sensitivity check)."""
    ik = InverseKinematics(plant, pc)               # creates q and adds q_lo <= q <= q_hi
    q = ik.q()
    prog = ik.prog()
    prog.AddQuadraticErrorCost(0.5*np.eye(7), Q_READY, q)     # 1/2 ||q - q_ready||^2
    E = plant.GetFrameByName("panda_link8", panda)
    pos = ik.AddPositionConstraint(E, np.zeros(3), plant.world_frame(), target, target)
    dist = []
    if obstacle is not None:
        c, r = obstacle
        for name in WRIST_POINTS:
            dist.append((name, ik.AddPointToPointDistanceConstraint(plant.GetFrameByName(name, panda), np.zeros(3),
                                                                    plant.world_frame(), c, r, 10.0)))
    limits = prog.bounding_box_constraints()[0]
    if extra_limits is not None:
        prog.AddBoundingBoxConstraint(extra_limits[0], extra_limits[1], q)
    result = SnoptSolver().Solve(prog, q_guess, None)
    return result, result.GetSolution(q), dict(pos=pos, dist=dist, limits=limits)


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
    for name, bd in b["dist"]:
        d = np.sqrt(result.EvalBinding(bd)[0])            # Drake evaluates the squared distance
        dual = result.GetDualSolution(bd)[0]
        print(f"  distance({name}, obstacle centre) = {d:.4f} m  (>= r_o); Drake dual (squared-distance form) = {dual:+.4f}"
              + (f"   ACTIVE  ->  lambda for g = r_o - d: {2*d*dual:.4f}" if abs(dual) > 1e-7 else ""))   # a lower bound on d^2: +dual
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

    obstacle = (np.array([0.45, 0.10, 0.53]), 0.07)
    res, q, b = solve_ik(plant, pc, panda, y1, obstacle=obstacle)
    report(plant, pc, panda, res, q, b, f"run 3: run-1 target with an obstacle sphere at {obstacle[0]}, r_o = {obstacle[1]}")




def render_png(out, q, spheres, title):
    """Offline picture with Drake's VTK render engine (no browser): the Panda at q with coloured spheres
    (target, obstacle) registered as world visuals. The camera looks from the +y side, because from the
    front the solved wrist of run 3 hides the obstacle behind link 7 (checked 2026-09-29)."""
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
    eye, target = np.array([-0.02, 1.84, 1.13]), np.array([0.30, 0.05, 0.40])   # from +y: the wrist wraps the obstacle in view
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


def figure_mode(out):
    diagram, ctx, plant, pc, panda = build()
    y1 = np.array([0.45, 0.10, 0.45]); obstacle = (np.array([0.45, 0.10, 0.53]), 0.07)
    res, q, b = solve_ik(plant, pc, panda, y1, obstacle=obstacle)
    # opaque orange for the obstacle: a grey semi-transparent sphere is invisible against the grey links
    render_png(out, q, [(y1, 0.02, [0.9, 0.1, 0.1, 1.0]), (obstacle[0], obstacle[1], [0.95, 0.55, 0.10, 1.0])],
               f"run 3: success = {result_word(res)}, cost {res.get_optimal_cost():.5f}, wrist point on the sphere")


def result_word(res):
    return "True" if res.is_success() else "False"


def meshcat_mode():
    meshcat = StartMeshcat()
    diagram, ctx, plant, pc, panda = build(meshcat)
    obstacle = (np.array([0.45, 0.10, 0.53]), 0.07)
    red, grey = Rgba(0.9, 0.1, 0.1, 1.0), Rgba(0.45, 0.45, 0.45, 1.0)
    meshcat.SetObject("/obstacle", Sphere(obstacle[1]), Rgba(0.5, 0.5, 0.5, 0.6))
    meshcat.SetTransform("/obstacle", RigidTransform(obstacle[0]))
    meshcat.SetObject("/target", Sphere(0.015), red)
    for name, lo, hi, v in (("target x", 0.1, 0.9, 0.45), ("target y", -0.6, 0.6, 0.10), ("target z", 0.05, 0.9, 0.45)):
        meshcat.AddSlider(name, lo, hi, 0.01, v)
    meshcat.AddButton("Stop")
    print("Meshcat at", meshcat.web_url(), "- move the target sliders; the arm re-solves; 'Stop' to quit.")
    print("  The sliders start at run 3's target, with the sphere drawn, so the first solve is run 3 of the printout.")
    print("  Each move is solved twice, from the last shown configuration and from q_ready, and the lower-cost")
    print("  solution is shown. The active set is decided by the primal gaps, not by the size of a dual.")
    lo, hi = plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    last, q_prev = None, Q_READY
    while meshcat.GetButtonClicks("Stop") == 0:
        y = np.array([meshcat.GetSliderValue(f"target {a}") for a in "xyz"])
        if last is None or np.any(y != last):
            last = y
            meshcat.SetTransform("/target", RigidTransform(y))
            # Two starts: the last shown configuration (the warm start) and q_ready. A warm start alone can
            # fail, or land in a worse valley, for a target that q_ready solves (seen 2026-10-06 after an
            # excursion toward the reach limit); solving from both keeps the shown answer reproducible.
            starts = [("the last pose", q_prev)] + ([("q_ready", Q_READY)] if np.any(q_prev != Q_READY) else [])
            solved = []
            for label, q_guess in starts:
                res, q, b = solve_ik(plant, pc, panda, y, obstacle=obstacle, q_guess=q_guess)
                if res.is_success():
                    solved.append((res.get_optimal_cost(), label, q, b))
            if solved:
                solved.sort(key=lambda s: s[0])
                cost, label, q, b = solved[0]
                q_prev = q
                plant.SetPositions(pc, panda, q)
                diagram.ForcedPublish(ctx)
                meshcat.SetObject("/target", Sphere(0.015), red)
                act = [f"joint {j+1} ({'lower' if abs(q[j]-lo[j]) < 1e-5 else 'upper'})" for j in range(7) if min(abs(q[j]-lo[j]), abs(q[j]-hi[j])) < 1e-5]
                for name, _ in b["dist"]:
                    p_k = plant.CalcRelativeTransform(pc, plant.world_frame(), plant.GetFrameByName(name, panda)).translation()
                    if abs(np.linalg.norm(p_k - obstacle[0]) - obstacle[1]) < 1e-6:
                        act.append(f"obstacle/{name}")
                valleys = ""
                if len(solved) == 2 and abs(solved[0][0] - solved[1][0]) > 1e-4:
                    valleys = (f"  [two local solutions: cost {solved[0][0]:.4f} from {solved[0][1]}, "
                               f"cost {solved[1][0]:.4f} from {solved[1][1]}; the lower one is shown]")
                print(f"  y* = {y}  cost = {cost:.4f}  active: {act if act else 'none'}{valleys}")
            else:
                meshcat.SetObject("/target", Sphere(0.015), grey)
                plant.SetPositions(pc, panda, q_prev)
                p_prev = plant.CalcRelativeTransform(pc, plant.world_frame(), plant.GetFrameByName("panda_link8", panda)).translation()
                print(f"  y* = {y}  no feasible point found by SNOPT from the last pose or from q_ready."
                      f" This is a solver failure, not a proof of infeasibility."
                      f" (A target inside the sphere is infeasible by construction: the flange is one of the three constrained points.)"
                      f" The arm stays at its last solution, flange at {p_prev}; the target is drawn grey.")
        time.sleep(0.05)
    meshcat.DeleteAddedControls()


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    elif "--meshcat" in sys.argv:
        meshcat_mode()
    else:
        print_mode()
