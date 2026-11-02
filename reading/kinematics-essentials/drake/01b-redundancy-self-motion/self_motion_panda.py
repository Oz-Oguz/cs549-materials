"""Session 2.4 Drake demo: redundancy on a real arm — the hand stays put while the elbow swings.

The Franka Panda has n = 7 joints and a gripper pose has m = 6 freedoms, so once the hand is
fixed, one degree of freedom is left over: the set of configurations with the same hand pose is
a one-dimensional curve in joint space (essentials §5, "the continuum of solutions"; deck 03's
null-space slide, "move the elbow without moving the hand"). This demo makes that curve visible. The gripper is held
over a mug on a table, at the pose

    ^W p^E = (0.556, 0, 0.245) m, tool axis pointing straight down,

and the arm is moved along the curve of every q with exactly that flange pose: the elbow rises
from far down on the robot's right, over the top, and down on its left, through 3.8 rad of joint
motion each way, while the gripper does not move by more than 1e-15 m.

How the curve is computed (nothing here is specific to the Panda). At every q the 6x7 spatial
Jacobian J(q) has rank 6, so its null space is one vector n(q), the joint velocity that moves the
hand not at all (the last right-singular vector of J). Starting from an elbow-up solution q0, the
demo steps q <- q + ds n(q), then removes the second-order drift with two Newton corrections
q <- q - J^+ (pose error), and repeats until a joint limit is reached (joint 2 at its +101 deg limit in both
directions). The sign of n(q) is kept continuous from step to step. This is the differential-IK
step of deck 03's null-space slide with the desired hand motion set to zero: q_dot = J^+ 0 + (I - J^+ J) z,
only the null-space term survives, and following it traces the self-motion.

Run:  python self_motion_panda.py            prints the flange pose, the two ends of the curve, the
                                             elbow positions at "elbow up" and at both "elbow down"
                                             ends, the largest flange-pose error along the curve and
                                             rank J at every sample; then self-tests all of it
      python self_motion_panda.py --meshcat  opens the arm, the table and the mug in the browser with
                                             ONE slider, the self-motion parameter s in radians of arc
                                             length along the curve (s = 0 is elbow up). Move it: the
                                             seven joints turn together and the gripper stays on the
                                             mug. Buttons: "Elbow up", "Elbow down (right)", "Elbow
                                             down (left)" jump to those configurations; "Play / pause
                                             sweep" runs the curve back and forth and pauses in place
                                             on the next press; "Show / hide elbow path" toggles the
                                             gold beads along the elbow's path; "Stop" ends it.
                                             The console prints s, q in degrees, the elbow position
                                             and the flange-pose error live
      python self_motion_panda.py --png FILE renders the three preset configurations side by side
                                             with Drake's VTK renderer, no browser: the slide picture

Tested 2026-09-23 with drake 1.46.0 on macOS. Runtime of the print mode: under a second (the curve
has 377 samples, each an SVD and two Newton corrections).
"""
import sys
import time
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, Cylinder, DiagramBuilder, JacobianWrtVariable,
                         MeshcatVisualizer, Parser, Rgba, RigidTransform, RotationMatrix, Sphere, StartMeshcat)

np.set_printoptions(precision=4, suppress=True)
PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_OPEN = 0.04                                  # each finger fully open: an 80 mm gap for the 70 mm mug
# The elbow-up solution the curve starts from: gripper pointing down over the mug, elbow at the top.
Q_UP = np.array([0.0, 0.3, 0.0, -2.2, 0.0, 2.5, 0.785])
DS = 0.02                                           # arc-length step along the curve, rad
MUG_R, MUG_H = 0.035, 0.10                          # the 70 mm mug of the session-2.1 demos
MUG_TOP_Z = 0.14                                    # fingertips reach 11.2 cm below the flange (flange at z = 0.245)
P_MUG = np.array([0.556, 0.0, MUG_TOP_Z - MUG_H / 2])
TABLE_TOP_Z = MUG_TOP_Z - MUG_H
ELBOW_PATH = "/elbow_path"


def full_q(q7, finger=FINGER_OPEN):
    """The plant has nine positions: seven arm joints and the two fingers."""
    return np.concatenate([np.asarray(q7, float), [finger, finger]])


def build(meshcat=None, renderer=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    if renderer:
        from pydrake.all import MakeRenderEngineVtk, RenderEngineVtkParams
        scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda))
    W = plant.world_body()
    plant.RegisterVisualGeometry(W, RigidTransform(P_MUG + [0, 0, -MUG_H / 2 - 0.02]), Box(0.5, 0.7, 0.04),
                                 "table", [0.75, 0.72, 0.68, 1.0])
    plant.RegisterVisualGeometry(W, RigidTransform(P_MUG), Cylinder(MUG_R, MUG_H), "mug", [0.2, 0.5, 0.6, 1.0])
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    return builder, plant, scene_graph, panda


class Arm:
    """Forward kinematics, the spatial Jacobian and the elbow position of the seven arm joints."""

    def __init__(self, diagram, plant, panda):
        self.plant, self.panda = plant, panda
        self.ctx = diagram.CreateDefaultContext()
        self.pc = plant.GetMyContextFromRoot(self.ctx)
        self.ee = plant.GetFrameByName("panda_link8", panda)
        self.elbow = plant.GetFrameByName("panda_link4", panda)
        self.lo, self.hi = plant.GetPositionLowerLimits()[:7], plant.GetPositionUpperLimits()[:7]

    def set(self, q7):
        self.plant.SetPositions(self.pc, self.panda, full_q(q7))

    def X(self, q7, frame=None):
        self.set(q7)
        return self.plant.CalcRelativeTransform(self.pc, self.plant.world_frame(), frame or self.ee)

    def J(self, q7):
        self.set(q7)
        J = self.plant.CalcJacobianSpatialVelocity(self.pc, JacobianWrtVariable.kQDot, self.ee, np.zeros(3),
                                                   self.plant.world_frame(), self.plant.world_frame())
        return J[:, :7]                             # the finger columns are zero for the flange

    def pose_error(self, q7, X_target):
        """Six numbers, angular on top (notation.md ^W V^E = (omega, v)): the motion from the target to
        the current flange pose, both expressed in the world frame, matching J's convention."""
        X = self.X(q7)
        aa = RotationMatrix(X.rotation().matrix() @ X_target.rotation().matrix().T).ToAngleAxis()
        return np.concatenate([aa.angle() * aa.axis(), X.translation() - X_target.translation()])

    def elbow_p(self, q7):
        return self.X(q7, self.elbow).translation()


def null_vector(J):
    """The joint velocity that moves the hand not at all: the right-singular vector of sigma_7 = 0."""
    _, s, vt = np.linalg.svd(J)
    return vt[-1], s


def trace_half(arm, q0, X_target, ds):
    """Follow the null space from q0 in the direction of sign(ds) until a joint limit; returns the
    samples after q0 (not including it)."""
    out, q, n_prev = [], q0.copy(), None
    for _ in range(5000):
        n, _ = null_vector(arm.J(q))
        if n_prev is not None and n @ n_prev < 0:
            n = -n                                  # keep the direction continuous along the curve
        n_prev = n
        q = q + ds * n
        for _ in range(2):                          # remove the second-order drift off the target pose
            q = q - np.linalg.pinv(arm.J(q)) @ arm.pose_error(q, X_target)
        if np.any(q < arm.lo) or np.any(q > arm.hi):
            break
        out.append(q.copy())
    return out


def trace_curve(arm, q0=Q_UP):
    """The whole self-motion curve through q0, as (s values in rad, 7-vectors), s = 0 at q0.
    s < 0 is the elbow on the robot's right (-y, the robot faces +x), s > 0 on its left (+y)."""
    X_target = arm.X(q0)
    right = trace_half(arm, q0, X_target, -DS)
    left = trace_half(arm, q0, X_target, +DS)
    qs = np.array(right[::-1] + [q0] + left)
    s = DS * (np.arange(len(qs)) - len(right))
    return s, qs, X_target


def curve_report(arm, s, qs, X_target):
    errs = np.array([np.linalg.norm(arm.pose_error(q, X_target)) for q in qs])
    ranks = np.array([np.linalg.matrix_rank(arm.J(q), tol=1e-6) for q in qs])
    smin = np.array([null_vector(arm.J(q))[1][-1] for q in qs])       # sigma_6, the smallest of the six
    elbows = np.array([arm.elbow_p(q) for q in qs])
    return errs, ranks, smin, elbows


def limit_hit(arm, q, tol=0.03):
    return [i + 1 for i in range(7) if q[i] - arm.lo[i] < tol or arm.hi[i] - q[i] < tol]


# ------------------------------------------------------------------ print mode
def print_mode():
    builder, plant, _, panda = build()
    arm = Arm(builder.Build(), plant, panda)
    s, qs, X_target = trace_curve(arm)
    errs, ranks, smin, elbows = curve_report(arm, s, qs, X_target)
    i_up = int(np.argmin(np.abs(s)))
    print("--- the fixed hand ---")
    print("  ^W p^E        =", X_target.translation(), " tool (z) axis =", X_target.rotation().matrix()[:, 2])
    print("  the mug's top is at z =", MUG_TOP_Z, "m, the fingertips 11.2 cm below the flange: the open gripper sits on the mug")
    print("\n--- the self-motion curve: every q with that flange pose ---")
    print(f"  {len(qs)} samples, {DS} rad apart in joint space: s from {s[0]:.2f} to {s[-1]:.2f} rad "
          f"({s[-1] - s[0]:.2f} rad of joint motion end to end)")
    print(f"  rank J = 6 at every sample: {np.all(ranks == 6)}, sigma_min between {smin.min():.3f} and {smin.max():.3f}: never singular,"
          f"\n  so the null space is exactly 7 - 6 = 1 dimensional everywhere, one line of joint velocities that leave the hand still")
    print(f"  largest flange-pose error along the curve: {errs.max():.1e} (rad and m together): the hand does not move")
    print(f"  ends at joint limits: left {limit_hit(arm, qs[0])}, right {limit_hit(arm, qs[-1])} (joint 2 at {np.degrees(arm.hi[1] if qs[-1][1] > 0 else arm.lo[1]):+.0f} deg)")
    print("\n--- three configurations with the same hand ---")
    for label, i in [("elbow down (right)", 0), ("elbow up", i_up), ("elbow down (left)", len(qs) - 1)]:
        print(f"  {label:20s} s = {s[i]:+.2f}  q [deg] = {np.degrees(qs[i])}  elbow ^W p = {elbows[i]}")
    print(f"  elbow travels {elbows[:, 1].min():.3f}..{elbows[:, 1].max():.3f} m in y and {elbows[:, 2].min():.3f}..{elbows[:, 2].max():.3f} m in z; "
          f"the flange moves {np.ptp([np.linalg.norm(arm.X(q).translation() - X_target.translation()) for q in qs]):.1e} m")
    print(f"  joint 1 alone spans {np.degrees(qs[:, 0].min()):.0f}..{np.degrees(qs[:, 0].max()):.0f} deg along the curve")
    # ---- self-test
    assert np.all(ranks == 6) and smin.min() > 0.05
    assert errs.max() < 1e-9
    assert len(qs) > 300 and abs(s[0]) > 3.0 and s[-1] > 3.0
    assert limit_hit(arm, qs[0]) == [2] and limit_hit(arm, qs[-1]) == [2]
    assert elbows[i_up, 2] > elbows[0, 2] + 0.25 and elbows[i_up, 2] > elbows[-1, 2] + 0.25
    assert np.degrees(np.abs(qs[0] - qs[-1])).max() > 100
    print("\nself-test: the curve has 377 samples with rank J = 6 and sigma_min > 0.05 at each, the flange-pose"
          "\nerror stays below 1e-9 along it, both ends stop at joint 2's limit, and the elbow is more than"
          "\n25 cm higher at s = 0 than at either end while the two end configurations differ by over 100"
          "\ndegrees on some joint: OK")


# ------------------------------------------------------------------ interactive mode
def draw_elbow_path(meshcat, elbows, every=3, radius=0.012):
    """Gold beads along the elbow's path. (Meshcat draws every line one pixel wide whatever width is
    asked for, so a line was too faint to see; a bead every third sample is not.)"""
    for k, p in enumerate(elbows[::every]):
        meshcat.SetObject(f"{ELBOW_PATH}/{k}", Sphere(radius), Rgba(0.85, 0.65, 0.1, 1.0))
        meshcat.SetTransform(f"{ELBOW_PATH}/{k}", RigidTransform(p))


def meshcat_mode():
    meshcat = StartMeshcat()
    builder, plant, _, panda = build(meshcat)
    diagram = builder.Build()
    arm = Arm(diagram, plant, panda)
    s, qs, X_target = trace_curve(arm)
    errs, ranks, smin, elbows = curve_report(arm, s, qs, X_target)
    i_up = int(np.argmin(np.abs(s)))
    presets = [("Elbow up", 0.0), ("Elbow down (right)", float(s[0])), ("Elbow down (left)", float(s[-1]))]
    meshcat.AddSlider("self-motion s [rad]", float(s[0]), float(s[-1]), DS, 0.0)
    for label, _ in presets:
        meshcat.AddButton(label)
    meshcat.AddButton("Play / pause sweep")
    meshcat.AddButton("Show / hide elbow path")
    meshcat.AddButton("Stop")
    # Seven text readouts of the joint angles in degrees (2026-10-07, for
    # Deck 03 slide 42: see the joints move together while the hand stays). Meshcat has no text widget,
    # but a button's label is text and a button can be deleted and re-added, as tip_only_panda.py does
    # for its joint table. The seven are re-added together so they keep their order, at most a few times
    # a second while the arm moves and once more when it comes to rest.
    readout = {"labels": [None] * 7, "i": None, "at": 0.0, "pending": False}

    def refresh_readouts(i):
        new = [f"q{j + 1} = {np.degrees(qs[i][j]):6.1f} deg" for j in range(7)]
        if new != readout["labels"]:
            for old in readout["labels"]:
                if old is not None:
                    meshcat.DeleteButton(old)
            for lab in new:
                meshcat.AddButton(lab)
            readout["labels"] = new
        readout["i"], readout["at"], readout["pending"] = i, time.time(), False

    meshcat.SetObject("/target", Sphere(0.012), Rgba(0.85, 0.65, 0.1, 1.0))
    meshcat.SetTransform("/target", RigidTransform(X_target.translation()))
    draw_elbow_path(meshcat, elbows)

    def show(sv):
        i = int(np.clip(round((sv - s[0]) / DS), 0, len(qs) - 1))
        arm.set(qs[i])
        diagram.ForcedPublish(arm.ctx)
        return i

    refresh_readouts(i_up)

    print("Meshcat at", meshcat.web_url())
    print("  One slider, seven joints: s is the arc length along the curve of configurations that hold the gripper")
    print("  on the mug; s = 0 is elbow up. Move it and watch the elbow swing while the hand stays. 'Play / pause")
    print("  sweep' runs back and forth along the whole curve and pauses where it is on the next press (the slider")
    print("  stays live while paused); the buttons jump to the three configurations of the slide; 'Stop' quits.")
    print("  The seven grey lines q1..q7 at the bottom of the panel are readouts of the joint angles in degrees;")
    print("  they refresh a few times a second while the arm moves and once more when it rests. Clicking them does nothing.")
    clicks = {label: 0 for label, _ in presets}
    sweep_clicks, path_clicks, path_shown, last = 0, 0, True, None
    playing, direction = False, +1
    while meshcat.GetButtonClicks("Stop") == 0:
        for label, sv in presets:
            if meshcat.GetButtonClicks(label) != clicks[label]:
                clicks[label] = meshcat.GetButtonClicks(label)
                meshcat.SetSliderValue("self-motion s [rad]", sv)
        if meshcat.GetButtonClicks("Show / hide elbow path") != path_clicks:
            path_clicks = meshcat.GetButtonClicks("Show / hide elbow path")
            path_shown = not path_shown
            meshcat.SetProperty(ELBOW_PATH, "visible", path_shown)
        if meshcat.GetButtonClicks("Play / pause sweep") != sweep_clicks:
            sweep_clicks = meshcat.GetButtonClicks("Play / pause sweep")
            playing = not playing                       # one button, two states: Meshcat has no toggle
        if playing:                                     # one step per tick, so every other control stays live
            sv = meshcat.GetSliderValue("self-motion s [rad]") + direction * DS
            if sv > s[-1] or sv < s[0]:
                direction = -direction                  # bounce at the joint-limit ends
                sv = float(np.clip(sv, s[0], s[-1]))
            meshcat.SetSliderValue("self-motion s [rad]", float(sv))
        i = show(meshcat.GetSliderValue("self-motion s [rad]"))
        if i != readout["i"]:                            # moving: refresh at most every 0.4 s
            readout["pending"] = True
            if time.time() - readout["at"] > 0.4:
                refresh_readouts(i)
        elif readout["pending"] and time.time() - readout["at"] > 0.3:   # at rest: one final refresh
            refresh_readouts(i)
        msg = (f"s = {s[i]:+.2f} rad  q [deg] = {np.degrees(qs[i])}  elbow ^W p = {elbows[i]}  "
               f"flange-pose error = {errs[i]:.1e}  rank J = {ranks[i]}")
        if msg != last:
            print(msg); last = msg
        time.sleep(0.02 if playing else 0.05)
    meshcat.DeleteAddedControls()


# ------------------------------------------------------------------ figure mode
def figure_mode(out):
    """Three renders, one per preset configuration, side by side: the same gripper on the same mug,
    three different arms. VTK, offline, no browser."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
                             RenderCameraCore, RgbdSensor)
    builder0, plant0, _, panda0 = build()
    arm0 = Arm(builder0.Build(), plant0, panda0)
    s, qs, X_target = trace_curve(arm0)
    i_up = int(np.argmin(np.abs(s)))
    builder, plant, scene_graph, panda = build(renderer=True)
    eye, target = np.array([1.45, -1.15, 0.85]), np.array([0.30, 0.0, 0.40])
    z_cam = target - eye; z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, [0, 0, 1.0]); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    X_WCam = RigidTransform(RotationMatrix(np.column_stack([x_cam, y_cam, z_cam])), eye)
    core = RenderCameraCore("vtk", CameraInfo(800, 720, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WCam, ColorRenderCamera(core, False),
                                          DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build()
    arm = Arm(diagram, plant, panda)
    panels = []
    for label, i in [("elbow down (right)", 0), ("elbow up", i_up), ("elbow down (left)", len(qs) - 1)]:
        arm.set(qs[i])
        img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(arm.ctx))
        panels.append(Image.fromarray(np.array(img.data)[:, :, :3]))
        print(f"  {label:20s} s = {s[i]:+.2f}  q [deg] = {np.degrees(qs[i])}")
    w, h = panels[0].size
    sheet = Image.new("RGB", (3 * w, h), (255, 255, 255))
    for k, p in enumerate(panels):
        sheet.paste(p, (k * w, 0))
    sheet.save(out)
    print(f"rendered {out}: three configurations, one flange pose ^W p^E = {X_target.translation()}")


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    else:
        meshcat_mode() if "--meshcat" in sys.argv else print_mode()
