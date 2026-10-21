"""Session 3.2 Drake demo: the damped Gauss-Newton IK step of Spring 2026 deck 03 (motion planning) p.56 on a real 7-joint arm.

    q <- q + J#(q) (y* - phi(q)),    J# = (J^T J + rho I)^{-1} J^T          (K = I, D = rho I)

phi is Drake's forward kinematics of the Panda's end-effector position and J its 3x7 translational
Jacobian, both from MultibodyPlant, so the only thing written here is the update rule itself.
Three runs:
  (1) ready start, target 10 cm away, rho = 1e-3: converges in four steps (redundant arm: J^T J is
      7x7 of rank 3, so with D = 0 the normal equations have infinitely many solutions; D = rho I,
      or the minimum-norm pseudo-inverse J+, picks one);
  (2) start at a stretched pose (elbow at its straight limit, arm horizontal), target 5 cm further
      along the direction the arm can least move (the smallest left-singular direction of J): the
      solution is regular (sigma_min = 0.026), and the 18 steps at rho = 1e-3 are the damping's own
      linear rate rho/(sigma_min^2 + rho) = 0.6; with rho = 1e-8 or J+ the same run takes 5 steps;
  (3) the same start, target 30 cm along that direction: with full steps the iteration
      wanders with steps of about 1 to 4 rad (the two-link blow-up on a real arm); with Armijo
      backtracking on the same direction its residual is within 1e-6 m of a numerical stationary reference residual
      by step 40, the tip is 0.6 mm from that reference tip (no global certificate), and it is still moving (the two-link
      case is notes section 5 (k), where Armijo stops 1 cm from the closest point); fixed rho = 1e-2
      with full steps does not converge here, adaptive rho does. Print mode computes all of these.
  (The all-zero configuration of the Panda is outside its joint limits and, unlike the two-link
  arm's stretched pose, keeps all three translations available thanks to the 8.8 cm flange
  offset; its singularity is in rotation only. Hence the stretched horizontal pose is used here.)

Run:  python gn_ik_panda.py            prints the three runs (notes §5 (o))
      python gn_ik_panda.py --meshcat  animates runs (1) and (2) in the browser, one iterate per second
      python gn_ik_panda.py --replay --app   paused replay of runs 1, 2, 3a, 3b as a web page around Meshcat: explanation,
                                       equations, legend, readable readouts, three plots with a cursor at the selected
                                       iterate, and the controls (Step, Back, Reset, a discrete iterate slider, the run
                                       buttons, a trials toggle, a flange close-up); the robot at q^(i), a faded ghost at
                                       q^(i-1), the linearized tip prediction made from q^(i-1) beside the FK result, one
                                       bead per past iterate; nothing moves by itself
      python gn_ik_panda.py --replay   the same replay with Meshcat's own controls only (no web page)
      python gn_ik_panda.py --record FILE    writes the four runs' accepted updates and rejected trials as JSON (cs549-trace/1)
      python gn_ik_panda.py --figure FILE    draws task error, joint-step magnitude and prediction gap per iterate (matplotlib)
      python gn_ik_panda.py --selftest [BASELINE.json]  checks the record against FK and the Jacobian and against a saved baseline
      python gn_ik_panda.py --png FILE renders run 2's end (the stretched start solved to its near-boundary target, the
                                       target as a gold sphere) offline with Drake's VTK engine: the picture on Deck 04 slide 31

Tested 2026-09-28 with drake 1.46.0 on macOS (print mode; the Meshcat mode was started headless only). The replay, record,
figure and self-test modes were added 2026-10-14; the print mode's output
is unchanged by them (diffed to zero bytes), and the optional on_event hook below is None in every mode but those.
"""
import sys
import time
from pathlib import Path
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, JacobianWrtVariable,
                         MeshcatVisualizer, Parser, Rgba, RigidTransform, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm.urdf"
Q_HOME = np.zeros(7)
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])      # Franka's standard ready pose
Q_STRETCH = np.array([0.0, 1.571, 0.0, -0.07, 0.0, 1.641, 0.785])      # arm horizontal, elbow at its straight limit (q4 = -0.07 rad)


def build(meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (Panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", Panda))
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    return diagram, plant, plant.GetMyContextFromRoot(ctx), ctx, Panda, plant.GetFrameByName("panda_link8", Panda)


def make_fk(plant, pc, Panda, ee):
    def phi(q):
        plant.SetPositions(pc, Panda, q)
        return plant.CalcRelativeTransform(pc, plant.world_frame(), ee).translation()
    def J(q):
        plant.SetPositions(pc, Panda, q)
        return plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, ee, np.zeros(3),
                                                       plant.world_frame(), plant.world_frame())
    return phi, J


def make_origins(plant, pc, Panda):
    """World positions of the link origins panda_link1..8 at q (for the record and the scene checks)."""
    def origins(q):
        plant.SetPositions(pc, Panda, q)
        return {f"panda_link{k}": plant.CalcRelativeTransform(pc, plant.world_frame(), plant.GetFrameByName(f"panda_link{k}", Panda)).translation().tolist()
                for k in range(1, 9)}
    return origins


LAST = {"status": None}     # the outcome word of the last run, printed by report(): Deck 04 slide 27's four words


def outcome(phi, J, q, y, res, stalled, tol=1e-6, gtol=1e-8):
    """solved (residual below tol) / stalled (the line search rejected the direction at its floor) /
    stationary (the relative gradient test |J^T r| <= gtol |J|_F |r| of Deck 04's gauss_newton_ik.py, K = I) / limit."""
    if res < tol:
        return "solved"
    if stalled:
        return "stalled"
    r = phi(q) - y
    if np.linalg.norm(J(q).T @ r) <= gtol * np.linalg.norm(J(q)) * np.linalg.norm(r):
        return "stationary"
    return "limit"


def gauss_newton_ik(phi, J, q0, y, rho, iters=20, tol=1e-6, on_step=None, linesearch=False, on_event=None):
    """q <- q + J#(y* - phi), J# = (J^T J + rho I)^-1 J^T (K = I, D = rho I). With linesearch=True the
    direction is kept and the step length is halved from alpha = 1 (restarted at every iteration) until
    the Armijo test on the cost 1/2|phi - y*|^2 passes (c = 0.01): session 3.1's safety check applied
    to session 3.2's step. A direction whose trial at the floor 1e-6 fails is rejected and the run stops
    (it never happens in the three runs below). The outcome word is stored in LAST and printed by report().
    on_event (optional, None in print mode) observes the run without changing it: it receives one dict per
    rejected Armijo trial ({"type": "trial_evaluation", ...}, an evaluated candidate the arm never moved
    through) and one per accepted update ({"type": "accepted_update", ...} with the accepted step after
    any Armijo shortening, the linearized tip prediction p_prev + J_prev @ step and the FK result)."""
    q = q0.copy()
    rows = [(0, q.copy(), np.linalg.norm(phi(q) - y), 0.0)]
    cost = lambda qq: 0.5 * float(np.sum((phi(qq) - y) ** 2))
    stalled = False
    for k in range(1, iters + 1):
        Jq = J(q)
        step = np.linalg.solve(Jq.T @ Jq + rho * np.eye(7), Jq.T @ (y - phi(q)))   # J# (y* - phi)
        q_prev, direction, a = q.copy(), step.copy(), 1.0
        if linesearch:
            a, c0, g = 1.0, cost(q), Jq.T @ (phi(q) - y)
            while cost(q + a * step) > c0 + 0.01 * a * float(g @ step) and a > 1e-6:
                if on_event:
                    on_event({"type": "trial_evaluation", "i": k, "alpha": a, "q": (q + a * step).copy(), "cost": cost(q + a * step)})
                a *= 0.5
            if cost(q + a * step) > c0 + 0.01 * a * float(g @ step):
                print(f"    step {k}: the trial at the floor 1e-6 failed the Armijo test; the direction is rejected and the run stops here")
                stalled = True
                break
            step = a * step
        q = q + step
        rows.append((k, q.copy(), np.linalg.norm(phi(q) - y), np.linalg.norm(step)))
        if on_step: on_step(q)
        if on_event:
            p_prev = phi(q_prev)
            on_event({"type": "accepted_update", "i": k, "q_prev": q_prev, "q": q.copy(), "step": step.copy(),
                      "direction": direction, "alpha": a, "p_prev": p_prev, "prediction": p_prev + Jq @ step,
                      "p": phi(q), "residual": rows[-1][2], "step_norm": rows[-1][3]})
        if rows[-1][2] < tol:
            break
    LAST["status"] = outcome(phi, J, rows[-1][1], y, rows[-1][2], stalled, tol)
    return rows


def report(title, rows):
    print(f"\n--- {title} ---")
    print("     i   |phi(q) - y*| [m]   |step| [rad]")
    for k, q, res, st in rows:
        print(f"    {k:2d}   {res:16.2e}   {st:10.4f}")
    print("    final q [rad] =", rows[-1][1])
    print(f"    outcome: {LAST['status']}   (solved / stationary / stalled / limit, Deck 04 slide 27)")


def print_mode():
    d, plant, pc, ctx, Panda, ee = build()
    phi, J = make_fk(plant, pc, Panda, ee)
    print("=== damped Gauss-Newton IK on the Panda: q <- q + (J^T J + rho I)^{-1} J^T (y* - phi(q)) ===")
    y = phi(Q_READY) + np.array([0.0, 0.10, -0.05])
    print("\n(1) ready start, target = current tip + (0, 0.10, -0.05) m =", y, ", rho = 1e-3")
    Jb = J(Q_READY)
    print("    rank of J^T J (7x7) =", np.linalg.matrix_rank(Jb.T @ Jb, tol=1e-9), "of 7: with D = 0 the normal equations have infinitely many solutions; D = rho I (or J+) picks one")
    rows = gauss_newton_ik(phi, J, Q_READY, y, rho=1e-3)
    report("run 1", rows)
    sv1 = np.linalg.svd(J(rows[-1][1]), compute_uv=False)[-1]
    print(f"    late residual ratio {rows[-1][2] / rows[-2][2]:.3f}, and rho/(sigma_min^2 + rho) = {1e-3 / (sv1**2 + 1e-3):.4f} with sigma_min = {sv1:.3f}: fast at first, then the damping's own linear rate")
    Js = J(Q_STRETCH); U, S, Vt = np.linalg.svd(Js)
    u_min = U[:, -1]                                            # the tip direction the arm can least move
    y2 = phi(Q_STRETCH) + 0.05 * u_min
    print("\n(2) stretched start (elbow at its straight limit, arm horizontal), rho = 1e-3")
    print("    singular values of J(stretched) =", S, "  least-mobile tip direction u_min =", u_min)
    print("    target = tip + 0.05 m along u_min =", y2)
    print("    J^T (y* - phi) =", Js.T @ (y2 - phi(Q_STRETCH)), "  <- small: the cost is flat in the direction the arm can least move")
    rows = gauss_newton_ik(phi, J, Q_STRETCH, y2, rho=1e-3, iters=30)
    report("run 2 (rho = 1e-3: the damping's linear rate)", rows)
    sv = np.linalg.svd(J(rows[-1][1]), compute_uv=False)
    print(f"    singular values of J at the solution = {sv}: a regular solution; late residual ratio {rows[-1][2] / rows[-2][2]:.3f} = rho/(sigma_min^2 + rho) = {1e-3 / (sv[-1]**2 + 1e-3):.3f}")
    for label, rho in (("rho = 1e-8", 1e-8), ("J+ (minimum-norm pseudo-inverse)", None)):
        if rho is None:
            q = Q_STRETCH.copy(); k = 0
            while np.linalg.norm(phi(q) - y2) >= 1e-6 and k < 30:
                q = q + np.linalg.pinv(J(q)) @ (y2 - phi(q)); k += 1
            print(f"    same run with {label}: {k} steps to 1e-6")
        else:
            print(f"    same run with {label}: {len(gauss_newton_ik(phi, J, Q_STRETCH, y2, rho=rho, iters=30)) - 1} steps to 1e-6")
    print("    -> the 18 steps are the price of rho = 1e-3 along the direction with sigma = 0.026, not a property of the target")
    y3 = phi(Q_STRETCH) + 0.30 * u_min
    print("\n(3) same start, target = tip + 0.30 m along u_min =", y3, "(unreachable by the chain-length bound below)")
    origins = np.array(list(make_origins(plant, pc, Panda)(Q_STRETCH).values()))
    chain_bound = np.linalg.norm(np.diff(origins, axis=0), axis=1).sum()
    target_distance = np.linalg.norm(y3 - origins[0])
    print(f"    shoulder-to-target distance {target_distance:.3f} m exceeds the invariant shoulder-to-flange segment sum {chain_bound:.3f} m;"
          f" task-error lower bound is about {target_distance - chain_bound:.3f} m by the triangle inequality")
    print("    this geometric bound proves this target unreachable; it does not certify the globally smallest residual of the numerical reference")
    rows = gauss_newton_ik(phi, J, Q_STRETCH, y3, rho=1e-3, iters=40)
    report("run 3a: full steps, rho = 1e-3 (the two-link blow-up, on a real arm)", rows[:6] + rows[-2:])
    steps = [r[3] for r in rows[1:]]
    print(f"    steps of {min(steps):.1f} to {max(steps):.1f} rad and a residual that never settles: rho = 1e-3 is far too little damping\n"
          f"    for a target this far outside the reach; the last iterate has joint 2 at {rows[-1][1][1]:.2f} rad, past its upper limit of {plant.GetPositionUpperLimits()[1]:.2f} rad (the URDF's value), which no term of the cost knows")
    # a numerical reference for the far target: adaptive damping to a stationary point, without a global certificate
    q_ref, rho_ad = Q_STRETCH.copy(), 1e-3
    cost3 = lambda qq: 0.5 * float(np.sum((phi(qq) - y3) ** 2))
    for _ in range(300):
        while True:
            Jq = J(q_ref); s = np.linalg.solve(Jq.T @ Jq + rho_ad * np.eye(7), Jq.T @ (y3 - phi(q_ref)))
            if cost3(q_ref + s) < cost3(q_ref):
                q_ref = q_ref + s; rho_ad = max(rho_ad / 10, 1e-12); break
            rho_ad *= 10
            if rho_ad > 1e12: break
        if rho_ad > 1e12: break
    r_min = np.linalg.norm(phi(q_ref) - y3)
    print(f"    stationary reference by adaptive damping (rho x10 on a rise, /10 on a fall): residual {r_min:.5f} m, |grad f| = {np.linalg.norm(J(q_ref).T @ (phi(q_ref) - y3)):.1e}, tip {phi(q_ref)}")
    rows = gauss_newton_ik(phi, J, Q_STRETCH, y3, rho=1e-3, iters=40, linesearch=True)
    report("run 3b: same, with Armijo on the step", rows[:6] + rows[-2:])
    q3b = rows[-1][1]
    print(f"    |phi(final) - y*| = {rows[-1][2]:.5f} m, {rows[-1][2] - r_min:.1e} m above the stationary reference residual; the tip is {np.linalg.norm(phi(q3b) - phi(q_ref))*1000:.1f} mm from the stationary reference tip;"
          f" |grad f| = {np.linalg.norm(J(q3b).T @ (phi(q3b) - y3)):.1e}, last step {rows[-1][3]:.4f} rad: converging slowly, not stalled; trials were rejected, but no entire direction was rejected at the floor, so the run ends after 40 accepted updates at its iteration limit")
    q_fix = Q_STRETCH.copy()
    for _ in range(2000):
        Jq = J(q_fix); q_fix = q_fix + np.linalg.solve(Jq.T @ Jq + 1e-2 * np.eye(7), Jq.T @ (y3 - phi(q_fix)))
    print(f"    fixed rho = 1e-2 with full steps, 2000 iterations: residual {np.linalg.norm(phi(q_fix) - y3):.3f} m, |grad f| = {np.linalg.norm(J(q_fix).T @ (phi(q_fix) - y3)):.1e}: does not converge on this target; adaptive damping does (above)")




def render_png(out, q, spheres, title):
    """Offline picture with Drake's VTK render engine (no browser): the Panda at q with coloured spheres
    (target, obstacle) registered as world visuals, the camera of drake/01-fk-jacobian's --png mode."""
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
    eye, target = np.array([1.45, -1.25, 0.95]), np.array([0.30, 0.0, 0.45])
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
    d, plant, pc, ctx, Panda, ee = build()
    phi, J = make_fk(plant, pc, Panda, ee)
    Js = J(Q_STRETCH); u_min = np.linalg.svd(Js)[0][:, -1]
    y2 = phi(Q_STRETCH) + 0.05 * u_min
    rows = gauss_newton_ik(phi, J, Q_STRETCH, y2, rho=1e-3, iters=30)
    q_end = rows[-1][1]
    render_png(out, q_end, [(y2, 0.02, [0.8, 0.6, 0.0, 1.0])],
               f"run 2 solved from the stretched start in {rows[-1][0]} steps at rho = 1e-3, residual {rows[-1][2]:.1e} m, outcome {LAST['status']}")


def meshcat_mode():
    meshcat = StartMeshcat()
    d, plant, pc, ctx, Panda, ee = build(meshcat)
    phi, J = make_fk(plant, pc, Panda, ee)
    def show(q):
        plant.SetPositions(pc, Panda, q); d.ForcedPublish(ctx); time.sleep(1.0)
    print("Meshcat at", meshcat.web_url())
    Js = J(Q_STRETCH); u_min = np.linalg.svd(Js)[0][:, -1]
    for name, q0, y, rho in (("run 1: ready start", Q_READY, phi(Q_READY) + np.array([0.0, 0.10, -0.05]), 1e-3),
                             ("run 2: stretched start, target near the boundary", Q_STRETCH, phi(Q_STRETCH) + 0.05 * u_min, 1e-3)):
        print(name); show(q0)
        meshcat.SetObject("target", Sphere(0.02), Rgba(0.8, 0.6, 0.0, 1.0))
        meshcat.SetTransform("target", RigidTransform(y))
        rows = gauss_newton_ik(phi, J, q0, y, rho, iters=30, on_step=show)
        print(f"  {rows[-1][0]} iterations, final residual {rows[-1][2]:.2e} m")
        time.sleep(2.0)
    input("Press Enter to quit. ")


# ------------------------------------------------------------------ the recorded runs (replay, record, figure, self-test)
# The four runs of the print mode, with the same starts, targets, damping and iteration budgets. The damping
# rho penalizes the proposed joint increment inside each local step; the objective stays 1/2|phi(q) - y*|^2.
RUN_LABELS = {"run1": "Run 1: ready start, target 10 cm away", "run2": "Run 2: stretched start, target 5 cm along u_min",
              "run3a": "Run 3a: unreachable target, full steps", "run3b": "Run 3b: unreachable target, Armijo"}


def define_runs(phi, J):
    y1 = phi(Q_READY) + np.array([0.0, 0.10, -0.05])
    u_min = np.linalg.svd(J(Q_STRETCH))[0][:, -1]
    y2, y3 = phi(Q_STRETCH) + 0.05 * u_min, phi(Q_STRETCH) + 0.30 * u_min
    return [("run1", Q_READY, y1, 1e-3, 20, False), ("run2", Q_STRETCH, y2, 1e-3, 30, False),
            ("run3a", Q_STRETCH, y3, 1e-3, 40, False), ("run3b", Q_STRETCH, y3, 1e-3, 40, True)]


def record_runs(phi, J, origins=None):
    """One complete solve per run, observed through on_event. Returns the record (cs549-trace/1): the header
    names the solver and the event source, and each run holds its accepted updates (with the accepted step,
    the prediction made from the previous iterate and the FK result) and its rejected Armijo trials. origins(q),
    if given, returns the world positions of the link origins for the scene checks."""
    import subprocess, importlib.metadata
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    try:
        drake_version = importlib.metadata.version("drake")
    except Exception:
        drake_version = "unknown"
    rec = {"schema": "cs549-trace/1", "problem": "damped Gauss-Newton IK on the Panda (Deck 04 slide 31, runs 1, 2, 3a, 3b)",
           "stage": "A: squared-error IK; rho damps the step, the objective is 1/2|phi(q) - y*|^2",
           "platform": f"drake {drake_version}, numpy {np.__version__}",
           "solver": f"course damped Gauss-Newton loop (gn_ik_panda.py@{sha}) on Drake's FK and translational Jacobian",
           "event_source": "own-loop hook (on_event): accepted updates are the loop's own; trial evaluations are the rejected Armijo candidates",
           "limitations": "accepted updates and rejected trials are observed directly; no solver history is inferred",
           "units": {"q": "rad", "p": "m", "residual": "m", "step_norm": "rad"}, "model": PANDA_URL, "runs": {}}
    for key, q0, y, rho, iters, ls in define_runs(phi, J):
        events = []
        rows = gauss_newton_ik(phi, J, q0, y, rho, iters=iters, linesearch=ls, on_event=events.append)
        run = {"label": RUN_LABELS[key], "q0": q0.tolist(), "target": y.tolist(), "rho": rho, "iteration_budget": iters,
               "linesearch": ls, "outcome": LAST["status"], "n_updates": len(rows) - 1,
               "iterates": [{"i": r[0], "q": r[1].tolist(), "p": phi(r[1]).tolist(), "residual": float(r[2]), "step_norm": float(r[3]),
                             **({"origins": origins(r[1])} if origins else {})} for r in rows],
               "accepted_updates": [], "trial_evaluations": []}
        for e in events:
            if e["type"] == "accepted_update":
                run["accepted_updates"].append({"i": e["i"], "q_prev": e["q_prev"].tolist(), "q": e["q"].tolist(),
                                                "step": e["step"].tolist(), "direction": e["direction"].tolist(), "alpha": e["alpha"],
                                                "p_prev": e["p_prev"].tolist(), "prediction": e["prediction"].tolist(), "p": e["p"].tolist(),
                                                "gap": float(np.linalg.norm(e["prediction"] - e["p"])),
                                                "residual": float(e["residual"]), "step_norm": float(e["step_norm"])})
            else:
                run["trial_evaluations"].append({"i": e["i"], "alpha": e["alpha"], "q": e["q"].tolist(), "p": phi(e["q"]).tolist(), "cost": e["cost"]})
        rec["runs"][key] = run
    return rec


def record_mode(out):
    import json
    d, plant, pc, ctx, Panda, ee = build()
    rec = record_runs(*make_fk(plant, pc, Panda, ee), origins=make_origins(plant, pc, Panda))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    for key, run in rec["runs"].items():
        print(f"{key}: {run['n_updates']} accepted updates, {len(run['trial_evaluations'])} rejected trials, outcome {run['outcome']}, "
              f"final residual {run['iterates'][-1]['residual']:.3e} m")
    print(f"wrote {out}")


def selftest_mode(baseline=None):
    """Checks the record against Drake's FK and Jacobian, against the loop without hooks, and against a saved
    full-precision baseline. Prediction gaps are reported as measured; no bound is claimed."""
    import json
    d, plant, pc, ctx, Panda, ee = build()
    phi, J = make_fk(plant, pc, Panda, ee)
    rec = record_runs(phi, J)
    failures = []
    def check(ok, msg):
        print(("  ok    " if ok else "  FAIL  ") + msg)
        if not ok:
            failures.append(msg)
    base = json.load(open(baseline)) if baseline else None
    for key, q0, y, rho, iters, ls in define_runs(phi, J):
        run = rec["runs"][key]
        rows = gauss_newton_ik(phi, J, q0, y, rho, iters=iters, linesearch=ls)          # no hook: must be the same numbers
        same = len(rows) == len(run["iterates"]) and all(np.array_equal(r[1], np.array(it["q"])) for r, it in zip(rows, run["iterates"]))
        check(same, f"{key}: the hook does not change the iterates ({len(rows) - 1} updates, bit-identical to the loop without hooks)")
        ups = run["accepted_updates"]
        check(len(ups) == run["n_updates"], f"{key}: one accepted_update event per update ({len(ups)})")
        worst_pred = worst_fk = 0.0
        for u in ups:
            qp, q = np.array(u["q_prev"]), np.array(u["q"])
            pred = phi(qp) + J(qp) @ (q - qp)                      # independent recomputation from Drake's FK and Jacobian
            worst_pred = max(worst_pred, float(np.linalg.norm(pred - np.array(u["prediction"]))))
            worst_fk = max(worst_fk, float(np.linalg.norm(phi(q) - np.array(u["p"]))))
            check(np.allclose(q, qp + np.array(u["step"])), f"{key} update {u['i']}: q = q_prev + accepted step") if u["i"] == 1 else None
        check(worst_pred < 1e-12, f"{key}: prediction = p_prev + J_prev @ (q - q_prev) for every update (worst difference {worst_pred:.1e} m)")
        check(worst_fk < 1e-12, f"{key}: actual = FK(q_i) for every update (worst difference {worst_fk:.1e} m)")
        gaps = [u["gap"] for u in ups]
        decreasing = all(b < a for a, b in zip(gaps, gaps[1:]))
        print(f"        {key}: measured prediction gaps [m] = " + " ".join(f"{g:.2e}" for g in gaps[:8]) + (" ..." if len(gaps) > 8 else "")
              + f"; decreasing at every update: {decreasing} (an observation, not a guarantee)")
        if ls:
            trials = run["trial_evaluations"]
            check(trials and all(1e-6 < t["alpha"] <= 1.0 and 1 <= t["i"] <= run["n_updates"] for t in trials)
                  and all(not np.array_equal(np.array(t["q"]), np.array(it["q"])) for t in trials for it in run["iterates"]),
                  f"{key}: {len(trials)} rejected trials recorded as evaluations (alpha in (1e-6, 1], tied to an update, none equal to an iterate)")
        if base is not None:
            bq = np.array(base[key]["q"]); rq = np.array([it["q"] for it in run["iterates"]])
            diff = float(np.max(np.abs(bq - rq))) if bq.shape == rq.shape else float("inf")
            check(diff == 0.0, f"{key}: iterate array equals the saved full-precision baseline (shape {rq.shape}, max |diff| {diff:.1e})")
            check(base[key]["outcome"] == run["outcome"], f"{key}: outcome {run['outcome']} as in the baseline")
    counts = {k: rec["runs"][k]["n_updates"] for k in rec["runs"]}
    check(counts == {"run1": 4, "run2": 18, "run3a": 40, "run3b": 40}, f"update counts {counts} (4, 18, limit 40, limit 40)")
    check(abs(rec["runs"]["run3b"]["iterates"][-1]["residual"] - 0.24621) < 1e-5, "run 3b ends 0.24621 m from the unreachable target, as printed")
    print(f"{len(failures)} failures")
    return 1 if failures else 0


def figure_replay_mode(out):
    """Task error, joint-step magnitude and prediction gap per iterate for the four runs (scalar norms over
    all seven joints and the three tip coordinates; no joint-space projection is drawn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, plant, pc, ctx, Panda, ee = build()
    rec = record_runs(*make_fk(plant, pc, Panda, ee))
    keys = list(rec["runs"])
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 7.6), sharex="col")
    for c, key in enumerate(keys):
        run = rec["runs"][key]
        it = run["iterates"]; ups = run["accepted_updates"]
        i_all = [x["i"] for x in it]
        axes[0, c].semilogy(i_all, [x["residual"] for x in it], "o-", color="#7a1f3d", ms=3)
        axes[0, c].set_title(run["label"].replace(": ", ":\n", 1), fontsize=9.5)
        axes[1, c].plot([u["i"] for u in ups], [u["step_norm"] for u in ups], "o-", color="#1f6f8b", ms=3)
        axes[2, c].semilogy([u["i"] for u in ups], [max(u["gap"], 1e-16) for u in ups], "o-", color="#b8860b", ms=3)
        if run["trial_evaluations"]:
            axes[1, c].text(0.03, 0.92, f"{len(run['trial_evaluations'])} rejected Armijo trials\n(evaluations, not shown as steps)", transform=axes[1, c].transAxes, fontsize=8, va="top")
        axes[2, c].set_xlabel("iterate i (accepted updates)")
        for r in range(3):
            axes[r, c].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("task error [m]")
    axes[1, 0].set_ylabel("accepted joint step [rad]")
    axes[2, 0].set_ylabel("prediction gap [m]")
    fig.suptitle("Damped Gauss-Newton IK on the Panda, per accepted update i (rho = 1e-3 damps the step; the objective is 1/2|phi(q) - y*|^2).\n"
                 "task error = |phi(q^(i)) - y*|;  accepted joint step = |q^(i) - q^(i-1)| (a norm over all seven joints);  "
                 "prediction gap = |p^(i-1) + J^(i-1)(q^(i) - q^(i-1)) - phi(q^(i))|, a measured value of these runs, not a bound.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


CAMERA_DIR = np.array([0.70, -0.95, 0.40]) / np.linalg.norm([0.70, -0.95, 0.40])  # from the scene toward the eye (the --png viewpoint)


def fit_camera(points, direction=CAMERA_DIR, fov_deg=75.0, margin=0.78, pad=0.09):
    """The eye and look-at point that keep every point in view along a fixed direction. Meshcat's camera has a
    75 degree vertical field of view; the fit assumes a square view, so a wider window only shows more. Each
    point is padded by `pad` metres so that the link meshes around the joint origins stay in view too."""
    P = np.asarray(points, dtype=float); centre = (P.min(axis=0) + P.max(axis=0)) / 2
    f = direction / np.linalg.norm(direction); up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, f); right /= np.linalg.norm(right); up = np.cross(f, right)
    limit = np.tan(np.radians(fov_deg) / 2) * margin
    radius = np.linalg.norm(P - centre, axis=1).max()
    for d in np.arange(0.5 * radius, 8.0 * radius, 0.01):
        rel = P - (centre + d * f); depth = -rel @ f                  # positive in front of the eye
        if depth.min() > 0.05 and np.all(np.abs(rel @ right) + pad <= limit * depth) and np.all(np.abs(rel @ up) + pad <= limit * depth):
            return centre + d * f, centre
    return centre + 8.0 * radius * f, centre


def replay_mode(app=False, app_port=0):
    """Paused replay in Meshcat. Displayed iterate i shows the robot at q^(i), the ghost at q^(i-1), the
    prediction made from q^(i-1) for the accepted update, and the FK result at q^(i) (the current flange).
    Iterate 0 shows the start with its flange (the FK marker) and, ahead of the first Step, the prediction for update 1. Nothing advances by itself; Step, Back, Reset and the slider move it.
    With app=True a web page (panel/index.html, served by replay_panel.py) wraps the Meshcat view with the
    explanation, readouts, plots and the same controls; Meshcat then keeps only its buttons and slider."""
    import json
    d0, plant0, pc0, ctx0, Panda0, ee0 = build()                        # the solver's plant: one Panda, as in print mode
    rec = record_runs(*make_fk(plant0, pc0, Panda0, ee0), origins=make_origins(plant0, pc0, Panda0))
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    (Panda,) = parser.AddModelsFromUrl(PANDA_URL)
    parser.SetAutoRenaming(True)
    (Ghost,) = parser.AddModelsFromUrl(PANDA_URL)                        # the previous iterate, drawn faded
    for m in (Panda, Ghost):
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", m))
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext(); pc = plant.GetMyContextFromRoot(ctx)
    ghost_path = f"/drake/visualizer/{plant.GetModelInstanceName(Ghost)}"
    arm_path = f"/drake/visualizer/{plant.GetModelInstanceName(Panda)}"
    gold, teal, maroon, grey = Rgba(0.8, 0.6, 0.0, 0.55), Rgba(0.1, 0.6, 0.6, 0.6), Rgba(0.55, 0.1, 0.25, 1.0), Rgba(0.45, 0.45, 0.45, 0.9)
    # Marker radii stay below run 2's first prediction gap (9.9 mm) so that the close-up shows the gap as two
    # spheres; from the wide view the readouts carry the numbers. The target and the prediction are translucent,
    # so a converged tip stays visible inside them.
    meshcat.SetObject("/replay/target", Sphere(0.012), gold)
    meshcat.SetObject("/replay/predicted", Sphere(0.010), teal)                         # translucent, larger: the promise
    meshcat.SetObject("/replay/actual", Sphere(0.007), maroon)                          # solid, smaller: where FK put it
    keys = list(rec["runs"])
    state = {"run": "run2", "i": 0, "trials": False, "closeup": False, "camera": None, "applied_action_id": 0,
             "arm_opacity": 1.0,                                         # the solid arm's opacity; the ghost stays at 0.3
             "show": {"target": True, "predicted": True, "actual": True, "beads": True, "ghost": True}}
    MARKERS = tuple(state["show"])
    wide_pts = {k: [[0.0, 0.0, 0.0]] + [o for it in rec["runs"][k]["iterates"] for o in list(it["origins"].values()) + [it["p"]]]
                   + [u["prediction"] for u in rec["runs"][k]["accepted_updates"]] + [rec["runs"][k]["target"]] for k in keys}
    wide = {k: fit_camera(wide_pts[k]) for k in keys}                   # one wide view per run: the base and the run's accepted iterates

    def set_camera(eye, target):
        meshcat.SetCameraPose(eye, target); state["camera"] = [np.asarray(eye).tolist(), np.asarray(target).tolist()]

    def set_trials(on):
        state["trials"] = on
        if not on and not state["closeup"]:
            set_camera(*wide[state["run"]])                             # hiding the candidates restores the run's wide view
    panel = None
    if app:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from replay_panel import ReplayPanel
        panel = ReplayPanel(meshcat.web_url(), rec, port=app_port)
    SLIDER, SLIDER_MAX, OPACITY = "iterate i", 40, "arm opacity"
    buttons = ["Step", "Back", "Reset"] + [rec["runs"][k]["label"] for k in keys] + ["Show / hide rejected trials", "Show / hide markers", "Stop"]
    meshcat.AddSlider(SLIDER, 0, SLIDER_MAX, 1, 0)
    meshcat.AddSlider(OPACITY, 0, 1, 0.05, 1)

    def set_arm_opacity(value):
        # modulated_opacity multiplies every material's opacity in the browser, as for the ghost; it is resent
        # once a second below because the arm's meshes are parsed asynchronously after they are sent
        state["arm_opacity"] = float(np.clip(value, 0.0, 1.0))
        meshcat.SetProperty(arm_path, "modulated_opacity", state["arm_opacity"])
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
        meshcat.Delete("/replay/beads"); meshcat.Delete("/replay/trials")
        for j, it in enumerate(run["iterates"]):
            meshcat.SetObject(f"/replay/beads/{j}", Sphere(0.005), Rgba(0.8, 0.6, 0.0, 0.85))
            meshcat.SetTransform(f"/replay/beads/{j}", RigidTransform(np.array(it["p"])))
        meshcat.SetTransform("/replay/target", RigidTransform(np.array(run["target"])))
        state["run"], state["i"] = key, 0
        if not state["closeup"]:
            set_camera(*wide[key])                                      # the wide view is set per run; Step keeps a hand-moved view

    def draw(i):
        run = rec["runs"][state["run"]]; its = run["iterates"]; n = len(its) - 1
        i = int(np.clip(i, 0, n)); state["i"] = i
        plant.SetPositions(pc, Panda, np.array(its[i]["q"]))
        plant.SetPositions(pc, Ghost, np.array(its[max(i - 1, 0)]["q"]))
        show = state["show"]
        meshcat.SetProperty(ghost_path, "visible", i >= 1 and show["ghost"])
        meshcat.SetProperty("/replay/target", "visible", show["target"])
        for j in range(len(its)):
            meshcat.SetProperty(f"/replay/beads/{j}", "visible", j <= i and show["beads"])
        meshcat.Delete("/replay/trials"); meshcat.Delete("/replay/gap_line")
        labels = [f"i = {i} of {n}  ({state['run']})"]
        p_now = np.array(its[i]["p"])
        if i >= 1:
            u = run["accepted_updates"][i - 1]
            meshcat.SetTransform("/replay/predicted", RigidTransform(np.array(u["prediction"])))
            meshcat.SetTransform("/replay/actual", RigidTransform(np.array(u["p"])))
            meshcat.SetProperty("/replay/predicted", "visible", show["predicted"]); meshcat.SetProperty("/replay/actual", "visible", show["actual"])
            if show["predicted"] and show["actual"]:                    # the line joins two drawn spheres only
                meshcat.SetLine("/replay/gap_line", np.column_stack([np.array(u["prediction"]), np.array(u["p"])]), 2.0, Rgba(0.1, 0.1, 0.1, 1.0))
            trials = [t for t in run["trial_evaluations"] if t["i"] == i]
            if state["trials"]:
                for j, t in enumerate(trials):
                    meshcat.SetObject(f"/replay/trials/{j}", Sphere(0.012), grey)     # larger than the tip markers: no gap to show
                    meshcat.SetTransform(f"/replay/trials/{j}", RigidTransform(np.array(t["p"])))
                if trials and not state["closeup"]:                      # widen the view to include the candidates
                    set_camera(*fit_camera(wide_pts[state["run"]] + [t["p"] for t in trials]))
            labels += [f"event: accepted update {i}" + (f"; alpha {u['alpha']:g}; {len(trials)} rejected" if run["linesearch"] else ""),
                       f"task error = {its[i]['residual']:.3e} m", f"accepted joint step = {u['step_norm']:.4f} rad",
                       f"prediction gap = {u['gap']:.3e} m"]
        else:
            # iterate 0: the FK tip is the start's flange, and the teal sphere is already the model's promise for update 1
            # (p^(0) + J^(0) dq^(0), the point the same sphere marks at iterate 1); no gap line, because no update has happened
            first = run["accepted_updates"][0] if run["accepted_updates"] else None
            meshcat.SetTransform("/replay/actual", RigidTransform(p_now))
            if first is not None:
                meshcat.SetTransform("/replay/predicted", RigidTransform(np.array(first["prediction"])))
            meshcat.SetProperty("/replay/predicted", "visible", first is not None and show["predicted"])
            meshcat.SetProperty("/replay/actual", "visible", show["actual"])
            labels += ["event: start (no update yet)", f"task error = {its[0]['residual']:.3e} m", "accepted joint step = none yet", "prediction gap = none yet"]
        labels += ["teal = predicted tip, maroon = FK tip, ghost = q^(i-1)", f"rho = {run['rho']:g} damps the step", "objective = 1/2|phi(q) - y*|^2",
                   f"outcome at the end: {run['outcome']} after {n} updates",
                   "error history: " + " ".join(f"{x['residual']:.1e}" for x in its[max(0, i - 4):i + 1])]
        if state["closeup"]:                                             # the close-up follows the flange: 47 cm away, from the side and a little below
            set_camera(p_now + np.array([0.30, -0.36, -0.06]), p_now)
        if panel is None:
            refresh_readouts(labels)                                    # Meshcat-only mode: readouts as relabelled buttons
        diagram.ForcedPublish(ctx)
        if panel is not None:
            panel.publish({"run": state["run"], "i": i, "n": n, "trials": state["trials"], "closeup": state["closeup"],
                           "arm_opacity": state["arm_opacity"], "show": dict(state["show"]),
                           "camera": state["camera"], "meshcat_url": meshcat.web_url(), "labels": labels, "p": its[i]["p"],
                           "origins": its[i].get("origins"), "applied_action_id": state["applied_action_id"]})
        print(f"{state['run']} i = {i}: " + " | ".join(labels[1:5]))

    load_run("run2"); draw(0)
    # The ghost's fading is a client-side property (meshcat.js multiplies each material's opacity when the
    # property arrives), and the arm's meshes are parsed asynchronously after they are sent, so a property sent
    # once can land before any material exists. It is resent once a second, which also covers a browser tab that
    # connects later; a resend is one small message.
    meshcat.SetProperty(ghost_path, "modulated_opacity", 0.3)
    set_arm_opacity(state["arm_opacity"])
    faded_at = time.time()
    if panel is not None:
        print("Demo app at", panel.start(), flush=True)              # the launcher waits for this line in --app mode
    print("Meshcat at", meshcat.web_url())
    print("  Paused replay. Nothing moves until you press Step, Back or Reset or move the 'iterate i' slider (values past the")
    print("  run's last iterate snap back). The run buttons restart at iterate 0. The ghost is the previous iterate; the teal")
    print("  sphere is where the linear model predicted the flange, the maroon sphere where FK put it. 'Show / hide rejected")
    print("  trials' draws the Armijo candidates of the selected update only (run 3b). 'arm opacity' fades the arm so the spheres")
    print("  inside it stay visible; 'Show / hide markers' toggles the spheres and the ghost. 'Stop' quits. The grey lines are readouts.")
    while meshcat.GetButtonClicks("Stop") == 0:
        changed = None
        for b in buttons:
            c = meshcat.GetButtonClicks(b)
            if c != clicks[b]:
                clicks[b] = c
                n = len(rec["runs"][state["run"]]["iterates"]) - 1
                if b == "Step": changed = min(state["i"] + 1, n)
                elif b == "Back": changed = max(state["i"] - 1, 0)
                elif b == "Reset": changed = 0
                elif b == "Show / hide rejected trials": set_trials(not state["trials"]); changed = state["i"]
                elif b == "Show / hide markers":
                    on = not any(state["show"].values()); state["show"] = {m: on for m in MARKERS}; changed = state["i"]
                elif b in [rec["runs"][k]["label"] for k in keys]:
                    load_run(next(k for k in keys if rec["runs"][k]["label"] == b)); changed = 0
                if changed is not None:
                    state["i"] = changed
                    meshcat.SetSliderValue(SLIDER, float(changed))
        for action_id, do in (panel.pop_actions() if panel is not None else []):    # the web page's requests, applied in order
            n = len(rec["runs"][state["run"]]["iterates"]) - 1
            changed = state["i"]
            if do == "step": changed = min(state["i"] + 1, n)
            elif do == "back": changed = max(state["i"] - 1, 0)
            elif do == "reset": changed = 0
            elif do.startswith("set:"): changed = int(np.clip(int(float(do[4:])), 0, n))
            elif do.startswith("run:") and do[4:] in rec["runs"]: load_run(do[4:]); changed = 0
            elif do.startswith("trials:"): set_trials(do.endswith("1")); changed = state["i"]
            elif do.startswith("opacity:"):
                try:
                    set_arm_opacity(float(do[8:])); meshcat.SetSliderValue(OPACITY, state["arm_opacity"])
                except ValueError:
                    pass                                                # a malformed request is acknowledged and ignored
            elif do.startswith("show:"):
                parts = do.split(":")
                if len(parts) == 3 and parts[2] in ("0", "1") and (parts[1] in MARKERS or parts[1] == "all"):
                    for m in (MARKERS if parts[1] == "all" else (parts[1],)):
                        state["show"][m] = parts[2] == "1"
            elif do.startswith("closeup:"):
                state["closeup"] = do.endswith("1"); changed = state["i"]
                if not state["closeup"]:
                    set_camera(*wide[state["run"]])                     # back to the run's wide view
            state["i"] = changed                                      # the next queued action starts from this result
            state["applied_action_id"] = action_id                     # bounds and other no-ops are acknowledged too
            meshcat.SetSliderValue(SLIDER, float(changed))
        ov = meshcat.GetSliderValue(OPACITY)
        if abs(ov - state["arm_opacity"]) > 1e-9:                          # Meshcat's own opacity slider
            set_arm_opacity(ov); changed = state["i"]
        sv = int(round(meshcat.GetSliderValue(SLIDER)))
        n = len(rec["runs"][state["run"]]["iterates"]) - 1
        if sv > n:
            meshcat.SetSliderValue(SLIDER, float(n)); sv = n
        if sv != state["i"] or changed is not None:
            draw(sv)
        if time.time() - faded_at > 1.0:
            meshcat.SetProperty(ghost_path, "modulated_opacity", 0.3); set_arm_opacity(state["arm_opacity"]); faded_at = time.time()
        time.sleep(0.05)
    meshcat.DeleteAddedControls()
    if panel is not None:
        panel.close()


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    elif "--meshcat" in sys.argv:
        meshcat_mode()
    elif "--replay" in sys.argv:
        port = int(sys.argv[sys.argv.index("--app-port") + 1]) if "--app-port" in sys.argv else 0
        replay_mode(app="--app" in sys.argv, app_port=port)
    elif "--record" in sys.argv:
        record_mode(sys.argv[sys.argv.index("--record") + 1])
    elif "--figure" in sys.argv:
        figure_replay_mode(sys.argv[sys.argv.index("--figure") + 1])
    elif "--selftest" in sys.argv:
        k = sys.argv.index("--selftest")
        default_baseline = str(Path(__file__).with_name("iterates-baseline.json"))
        sys.exit(selftest_mode(sys.argv[k + 1] if len(sys.argv) > k + 1 and not sys.argv[k + 1].startswith("--") else default_baseline))
    else:
        print_mode()
