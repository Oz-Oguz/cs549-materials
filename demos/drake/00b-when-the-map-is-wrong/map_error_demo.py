"""Session 2.1 demo: when the map is wrong.

The classical stack plans on the *estimated* pose of the mug and executes against the *true*
one. Nothing in the stack knows the difference: inverse kinematics solves exactly the problem
it was given and the controller tracks the plan exactly. What happens then is decided by
geometry the stack never looked at. This demo plays it out with three kinematic rules (no
physics engine; each rule is what the real robot would do, or the simplest honest stand-in):

  1. APPROACH. The hand comes straight down onto the planned grasp. If a finger or the palm
     would enter the mug on the way, the arm STOPS there — a Franka does exactly that, its
     collision reflex halts the motion (a robot without one would push the mug over).
  2. CLOSE. If the approach ends with the mug between the open fingers, the fingers close on
     it and the parallel gripper centres the mug between the pads (Franka's fingers are
     mechanically coupled): a grasp, with the mug nudged by whatever the error was.
  3. If the mug is out of the fingers' reach altogether, the fingers close fully on air.

Numbers this produces (slide 9): the hand opens to 80 mm and the mug is 70 mm across, so the
mug is between the open fingers only if the estimate is right to 5 mm along the closing axis;
one millimetre more and a finger comes down on the mug's rim. Along the finger pads the
tolerance is about 4 cm, and a mug that is *higher* than believed meets the palm.

Run:  python map_error_demo.py             prints the plan/execute table for eight estimation errors
      python map_error_demo.py --meshcat   sliders for the estimation error (cm); "Plan on the
                                           estimate, execute on the truth" animates the approach
                                           and the closing under the rules above. The orange ghost
                                           is where the stack believes the mug is; the teal mug is
                                           where it is; the ghost turns green (grasp) or red (stopped
                                           on contact / closed on air). Moving a slider only moves
                                           the ghost; press the button again to re-plan and execute.
                                           "Back to the ready pose" resets the arm. "Stop" ends the loop.
      python map_error_demo.py --meshcat --auto   the same, runs the default error once on start,
                                           then waits for "Stop" or 90 s; add --html FILE to write
                                           Meshcat's self-contained page of the final scene and exit.
      python map_error_demo.py --png FILE  renders the default case (2 cm off along the closing axis:
                                           a finger stopped on the mug's rim) with Drake's VTK
                                           renderer, no browser: the slide-9 figure.

Same scene as 00-contact-changes-the-map (Franka Panda at its ready pose, the mug's centre 10.5 cm
below the flange so that its top clears the palm), with Drake's panda_arm_hand.urdf so that the
fingers exist. The hand's geometry used by the rules is a box model of the palm and the two
fingers, dimensions stated below as approximations of the Franka hand. Written 2026-09-19,
contact rules added 2026-09-19 so that the fingers cannot close through the mug.
"""
import pathlib
import sys
import time
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, Cylinder, DiagramBuilder, InverseKinematics,
                         MeshcatVisualizer, Parser, Rgba, RigidTransform, RotationMatrix, Solve, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)

PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
Q_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
FINGER_OPEN = 0.04                  # each finger's travel: the hand opens to 2 × 0.04 = 0.08 m (Panda hand)
MUG_R, MUG_H = 0.035, 0.08          # the mug of the contact demo: 70 mm across, 80 mm tall
P_HQ = np.array([0.0, 0.0, 0.105])  # the grasp point Q, 10.5 cm along the hand's z axis: the mug's centre when its top clears the
                                    # palm and the finger pads (9.5-11.2 cm below the flange) hold its middle — as in contact_demo.py
APPROACH = 0.12                     # the hand descends this far, straight down, onto the planned grasp
# the hand as boxes in the hand frame (x along the pads, y the closing direction, z down the approach); approximate Franka hand
PALM = dict(x=(-0.035, 0.035), y=(-0.10, 0.10), z=(0.0, 0.058))
FINGER_Z, FINGER_W, FINGER_T = (0.0585, 0.112), 0.02, 0.008  # finger span below the flange (finger base to fingertip, measured on Drake's meshes;
                                                             # the flat pads occupy 0.095-0.112); pad width along x; finger thickness behind the pad


def finger_box(side, f):
    """The finger on `side` (+1 or −1) with opening f: its pad's inner face at y = side·f, body outward from it."""
    lo, hi = sorted([side * f, side * (f + FINGER_T)])
    return dict(x=(-FINGER_W / 2, FINGER_W / 2), y=(lo, hi), z=FINGER_Z)


def box_hits_mug(box, c_H):
    """Does an axis-aligned box in the hand frame intersect the mug (a vertical cylinder of radius MUG_R, height MUG_H,
    centre c_H in the hand frame)? The mug's axis is the hand's z axis because the hand stays vertical."""
    dx = max(box["x"][0] - c_H[0], 0.0, c_H[0] - box["x"][1])
    dy = max(box["y"][0] - c_H[1], 0.0, c_H[1] - box["y"][1])
    in_plane = dx * dx + dy * dy < MUG_R ** 2
    z_lo, z_hi = c_H[2] - MUG_H / 2, c_H[2] + MUG_H / 2
    return in_plane and box["z"][0] < z_hi and z_lo < box["z"][1]


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
    return diagram, plant, plant.GetMyContextFromRoot(ctx), ctx, panda, plant.GetFrameByName("panda_hand", panda)


def full_q(q7, finger=FINGER_OPEN):
    return np.concatenate([q7, [finger, finger]])


def hand_pose(plant, pc, hand, q):
    plant.SetPositions(pc, q)
    X_WH = plant.CalcRelativeTransform(pc, plant.world_frame(), hand)
    return X_WH, X_WH.multiply(P_HQ)          # ^W X^H and the grasp point ^W p^Q


def plan(plant, pc, hand, p_WQ_target, R_WH_target, q0):
    """Inverse kinematics on the *estimated* mug: put the grasp point Q on the target, keep the hand's
    orientation, stay close to q0. Exactly the goal box of the stack, with the map it was given."""
    ik = InverseKinematics(plant, pc)
    ik.AddPositionConstraint(hand, P_HQ, plant.world_frame(), p_WQ_target - 1e-4, p_WQ_target + 1e-4)
    ik.AddOrientationConstraint(plant.world_frame(), R_WH_target, hand, RotationMatrix(), 1e-3)
    prog, q = ik.get_mutable_prog(), ik.q()
    prog.AddQuadraticErrorCost(np.eye(len(q0)), q0, q)
    prog.AddBoundingBoxConstraint(FINGER_OPEN, FINGER_OPEN, q[7:9])
    prog.SetInitialGuess(q, q0)
    result = Solve(prog)
    return result.GetSolution(q), result.is_success()


def execute(plant, pc, hand, p_WM_true, R_WH_ready, delta, q0, n_steps=60):
    """Plan on the estimate, then play the approach and the closing under the three rules.
    Returns a dict with the outcome, the joint path of the approach (for animation), the final finger
    opening, and where the mug ends up."""
    p_WM_est = p_WM_true + delta                                   # what the stack believes
    q_plan, ok = plan(plant, pc, hand, p_WM_est, R_WH_ready, q0)
    X_WH_plan, _ = hand_pose(plant, pc, hand, q_plan)
    R = X_WH_plan.rotation().matrix()
    # rule 1: approach straight down from APPROACH above the plan; stop at the first contact
    path, q_prev, hit, height = [], q0, None, 0.0
    for k in range(n_steps + 1):
        s = k / n_steps
        p_H = X_WH_plan.translation() + np.array([0, 0, APPROACH * (1 - s)])
        c_H = R.T @ (p_WM_true - p_H)                               # the true mug's centre in the hand frame
        parts = [("palm", PALM), ("finger +y", finger_box(+1, FINGER_OPEN)), ("finger −y", finger_box(-1, FINGER_OPEN))]
        hit = next((name for name, box in parts if box_hits_mug(box, c_H)), None)
        if hit:
            height = APPROACH * (1 - s)
            break
        q_k, _ = plan(plant, pc, hand, p_H + R @ P_HQ, X_WH_plan.rotation(), q_prev)
        path.append(q_k); q_prev = q_k
    q_end = path[-1]
    X_WH_end, p_WQ = hand_pose(plant, pc, hand, q_end)
    c_H = X_WH_end.rotation().matrix().T @ (p_WM_true - X_WH_end.translation())
    miss_H = -c_H + P_HQ                                            # grasp point − mug centre, hand axes
    r = dict(q_plan=q_plan, ik_ok=ok, path=path, q_end=q_end, miss_H=miss_H, delta=delta, mug_final=p_WM_true.copy())
    if hit:
        r.update(outcome="STOPPED", finger=FINGER_OPEN, success=False,
                 message=f"collision on approach: the {hit} meets the mug {100*height:.1f} cm above the planned grasp; "
                         f"the arm stops (a Franka's reflex; a robot without one pushes the mug over)")
        return r
    # rule 2 / 3: close; the fingers reach the mug if it lies between them (in y) and under the pads (in x)
    between = abs(c_H[1]) + MUG_R <= FINGER_OPEN + 1e-9
    under_pads = abs(c_H[0]) < MUG_R + FINGER_W / 2
    pads_on_mug = FINGER_Z[0] < c_H[2] + MUG_H / 2 and c_H[2] - MUG_H / 2 < FINGER_Z[1]
    if between and under_pads and pads_on_mug:
        shift_H = np.array([0.0, -c_H[1], 0.0])                     # the coupled fingers centre the mug between the pads
        r.update(outcome="GRASP", finger=MUG_R, success=True, mug_final=p_WM_true + X_WH_end.rotation().matrix() @ shift_H,
                 message=f"grasp: the mug was {1000*abs(c_H[1]):.1f} mm off the hand's centre line along the closing axis and "
                         f"{1000*abs(c_H[0]):.1f} mm along the pads; the coupled fingers centred it")
    else:
        why = ("the mug is beside the fingers, out of their reach" if not between else
               "the mug is beyond the pads along their width" if not under_pads else "the pads pass above the mug")
        r.update(outcome="AIR", finger=0.0, success=False, message=f"the fingers close on air: {why}")
    return r


def print_mode():
    d, plant, pc, ctx, panda, hand = build()
    q0 = full_q(Q_READY)
    X_WH0, p_WQ0 = hand_pose(plant, pc, hand, q0)
    p_WM_true = p_WQ0.copy()                            # the true mug: exactly where the ready pose's grasp point is
    R = X_WH0.rotation().matrix()
    print("Hand at the ready pose: ^W p^H =", X_WH0.translation(), " grasp point ^W p^Q =", p_WQ0)
    print("Hand axes in world coordinates: x_H (pads) =", R[:, 0], " y_H (closing) =", R[:, 1], " z_H (approach) =", R[:, 2])
    print(f"True mug centre ^W p^M = {p_WM_true}; radius {MUG_R} m, height {MUG_H} m; hand opens to {2*FINGER_OPEN} m")
    print(f"Between the open fingers only if the error along the closing axis is below {1000*(FINGER_OPEN-MUG_R):.0f} mm\n")
    cases = [("no error", [0, 0, 0]), ("4 mm in world y (across the fingers)", [0, 0.004, 0]), ("6 mm in world y", [0, 0.006, 0]),
             ("2 cm in world y", [0, 0.02, 0]), ("2 cm in world x (along the pads)", [0.02, 0, 0]),
             ("5 cm in world x", [0.05, 0, 0]), ("2 cm in world z (mug lower than believed)", [0, 0, 0.02]),
             ("−2 cm in world z (mug higher than believed)", [0, 0, -0.02])]
    print(f"{'estimation error':44s} {'outcome':8s} what happened")
    for label, delta in cases:
        r = execute(plant, pc, hand, p_WM_true, X_WH0.rotation(), np.array(delta, float), q0)   # 2 mm approach steps, as in the animation
        print(f"{label:44s} {r['outcome']:8s} {r['message']}")
    print("\nEvery plan was executed exactly. What happened next was decided by geometry the stack never saw.")


def draw_mugs(meshcat, p_WM_true, p_WM_est, colour):
    meshcat.SetObject("/mugs/true", Cylinder(MUG_R, MUG_H), Rgba(0.2, 0.5, 0.6, 1.0))
    meshcat.SetTransform("/mugs/true", RigidTransform(p_WM_true))
    meshcat.SetObject("/mugs/estimate", Cylinder(MUG_R * 1.01, MUG_H * 1.01), colour)
    meshcat.SetTransform("/mugs/estimate", RigidTransform(p_WM_est))


def meshcat_mode(auto=False):
    meshcat = StartMeshcat()
    d, plant, pc, ctx, panda, hand = build(meshcat)
    q0 = full_q(Q_READY)
    X_WH0, p_WQ0 = hand_pose(plant, pc, hand, q0)
    p_WM_true = p_WQ0.copy()
    meshcat.SetObject("/table", Box(0.8, 0.6, 0.04), Rgba(0.75, 0.72, 0.68, 1.0))
    meshcat.SetTransform("/table", RigidTransform(p_WM_true + np.array([0, 0, -MUG_H / 2 - 0.02])))
    GHOST, GOOD, BAD = Rgba(1.0, 0.6, 0.1, 0.45), Rgba(0.2, 0.7, 0.3, 0.45), Rgba(0.85, 0.15, 0.15, 0.45)
    for name in ("Δx (cm)", "Δy (cm)", "Δz (cm)"):
        meshcat.AddSlider(name, -5.0, 5.0, 0.1, 2.0 if name.startswith("Δy") else 0.0)
    meshcat.AddButton("Plan on the estimate, execute on the truth")
    meshcat.AddButton("Back to the ready pose")
    meshcat.AddButton("Stop")
    meshcat.SetCameraPose(np.array([1.25, -1.05, 0.95]), p_WM_true)
    q_now = q0.copy()
    plant.SetPositions(pc, q_now); d.ForcedPublish(ctx)
    print("Meshcat at", meshcat.web_url())
    print("Default: the mug is believed 2 cm off along world y, the fingers' closing direction. Press the button.")

    def read_delta():
        return np.array([meshcat.GetSliderValue(n) for n in ("Δx (cm)", "Δy (cm)", "Δz (cm)")]) / 100.0

    def move_to(q_target, seconds=1.5):
        nonlocal q_now
        q_from = q_now.copy()
        for k in range(int(seconds / 0.03) + 1):
            s = min(1.0, k * 0.03 / seconds); s = s * s * (3 - 2 * s)
            q_now = (1 - s) * q_from + s * q_target
            plant.SetPositions(pc, q_now); d.ForcedPublish(ctx); time.sleep(0.03)

    def run_once():
        nonlocal q_now
        delta = read_delta()
        draw_mugs(meshcat, p_WM_true, p_WM_true + delta, GHOST)
        r = execute(plant, pc, hand, p_WM_true, X_WH0.rotation(), delta, q0)
        move_to(r["path"][0], 1.2)                                       # to the pre-grasp pose above the plan
        for q_k in r["path"][1:]:                                        # the approach, stopped at contact if any
            q_now = q_k; plant.SetPositions(pc, q_now); d.ForcedPublish(ctx); time.sleep(0.04)
        if r["outcome"] != "STOPPED":
            move_to(full_q(r["q_end"][:7], r["finger"]), 0.8)             # close: on the mug, or on air
        draw_mugs(meshcat, r["mug_final"], p_WM_true + delta, GOOD if r["success"] else BAD)
        print(f"\nestimation error Δ = {delta} m  (IK {'ok' if r['ik_ok'] else 'FAILED'})")
        print("grasp point − true mug, hand axes (x pads, y closing, z depth) =", r["miss_H"])
        print(f"outcome: {r['outcome']} — {r['message']}")

    last_delta = read_delta()
    draw_mugs(meshcat, p_WM_true, p_WM_true + last_delta, GHOST)
    clicks = {"plan": 0, "back": 0}
    t_start = time.time()
    if auto:
        run_once()
        if "--html" in sys.argv:                       # a static copy of the scene, meshes embedded, for a screenshot
            out = sys.argv[sys.argv.index("--html") + 1]
            pathlib.Path(out).write_text(meshcat.StaticHtml()); print("static scene written to", out)
            meshcat.DeleteAddedControls(); return
    while meshcat.GetButtonClicks("Stop") == 0 and not (auto and time.time() - t_start > 90):
        if meshcat.GetButtonClicks("Plan on the estimate, execute on the truth") != clicks["plan"]:
            clicks["plan"] = meshcat.GetButtonClicks("Plan on the estimate, execute on the truth"); run_once()
        if meshcat.GetButtonClicks("Back to the ready pose") != clicks["back"]:
            clicks["back"] = meshcat.GetButtonClicks("Back to the ready pose")
            draw_mugs(meshcat, p_WM_true, p_WM_true + read_delta(), GHOST); move_to(q0)
        delta = read_delta()
        if np.any(delta != last_delta):                # a slider moved: the belief moves, the world does not
            last_delta = delta; draw_mugs(meshcat, p_WM_true, p_WM_true + delta, GHOST)
        time.sleep(0.05)
    meshcat.DeleteAddedControls()


def figure_mode(out, delta=np.array([0.0, 0.02, 0.0])):
    """Render the executed default case with Drake's VTK render engine (offline; no browser). The mugs are scene-graph
    visuals here, so that the renderer sees them: the truth in teal, the belief as a translucent orange ghost."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
                             MakeRenderEngineVtk, RenderCameraCore, RenderEngineVtkParams, RgbdSensor)
    d0, plant0, pc0, ctx0, panda0, hand0 = build()
    q0 = full_q(Q_READY)
    X_WH0, p_WQ0 = hand_pose(plant0, pc0, hand0, q0)
    p_WM_true = p_WQ0.copy()
    r = execute(plant0, pc0, hand0, p_WM_true, X_WH0.rotation(), delta, q0)
    q_final = full_q(r["q_end"][:7], r["finger"])
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda))
    W = plant.world_body()
    plant.RegisterVisualGeometry(W, RigidTransform(p_WM_true + np.array([0, 0, -MUG_H / 2 - 0.02])), Box(0.8, 0.6, 0.04), "table", [0.75, 0.72, 0.68, 1.0])
    plant.RegisterVisualGeometry(W, RigidTransform(r["mug_final"]), Cylinder(MUG_R, MUG_H), "mug_true", [0.2, 0.5, 0.6, 1.0])
    plant.RegisterVisualGeometry(W, RigidTransform(p_WM_true + delta), Cylinder(MUG_R * 1.01, MUG_H * 1.01), "mug_belief", [1.0, 0.6, 0.1, 0.45])
    plant.Finalize()
    eye, target = np.array([0.95, -0.75, 0.80]), p_WM_true + np.array([0, 0, 0.05])
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    X_WCam = RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye)     # Drake camera: z forward, y down
    core = RenderCameraCore("vtk", CameraInfo(1280, 960, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WCam, ColorRenderCamera(core, False), DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(ctx), q_final)
    img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(ctx))
    Image.fromarray(np.array(img.data)[:, :, :3]).save(out)
    print("rendered", out, "| error", delta, "|", r["outcome"], "—", r["message"])


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    else:
        meshcat_mode(auto="--auto" in sys.argv) if "--meshcat" in sys.argv else print_mode()
