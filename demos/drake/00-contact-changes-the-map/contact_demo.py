"""Session 2.1 demo: contact changes the map.

Before a grasp the mug is a body fixed in the world. Moving the arm's joints does not move
it, and the Jacobian of the mug's position with respect to the arm's joint angles is exactly
zero. After the grasp the mug is rigidly attached to the gripper: it moves with the arm and
its Jacobian is the gripper's Jacobian, shifted to the mug's centre. The "grasp" is modelled
as a weld, which is what a classical solver assumes at the moment of contact; the friction,
slip, and on-off structure in between are exactly what the lecture says the map does not
capture.

Run:  python contact_demo.py            prints the numbers used in the 2.1 notes (§I.3) and slide 8,
                                        then self-tests the grasp logic of the interactive mode
      python contact_demo.py --meshcat  interactive: seven joint sliders and a "Grasp / release"
                                        button in the browser. Grasp: the fingers close on the mug
                                        and from then on the mug follows the hand; release: the
                                        fingers open and the mug drops onto the table under the
                                        point where it was let go (a rule, not physics: the demo has
                                        no dynamics). The console prints the mug's Jacobian rank
                                        after each press. The slide-8 numbers are the panda_joint1
                                        slider moved by +0.2 rad after a grasp at the ready pose.
                                        "Stop" ends the loop.
      python contact_demo.py --png FILE renders the grasped mug after the 0.2 rad base turn with
                                        Drake's VTK renderer (no browser): a check that the hand
                                        model and the finger closure look right.

Model: Drake's panda_arm_hand.urdf (arm and gripper) since 2026-09-19 — the earlier version used the
armless-flange model, so there was nothing visible between the flange and the mug. The gripper
frame used for every number is still panda_link8 (the flange; panda_hand shares its origin), so the
Jacobian table of the notes is unchanged. Tested 2026-09-05 with drake 1.46.0 on macOS, re-run
2026-09-19 (see drake/README.md). Runtime of the print mode: a few seconds plus the one-time model download.
"""
import sys
import time
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, Cylinder, DiagramBuilder,
                         JacobianWrtVariable, MeshcatVisualizer, Parser, Rgba, RigidTransform,
                         SpatialInertia, StartMeshcat, UnitInertia)

np.set_printoptions(precision=4, suppress=True)

PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
FINGER_OPEN, FINGER_ON_MUG = 0.04, 0.036   # each finger: fully open, and closed on the 70 mm mug (pads 1 mm off the mug: reads as closed, no penetration)
MUG_R, MUG_H = 0.035, 0.08                 # the mug: 70 mm across, 80 mm tall
MUG_BELOW = 0.105                          # the mug's centre this far below the flange along the hand's z axis: its top clears the
                                           # palm (which ends about 5.8 cm below the flange); the fingers reach 11.2 cm and their flat pads (9.5-11.2 cm) hold its middle
Q_START = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])   # the "bent" pose of smoke_test.py


def full_q(q7, finger=FINGER_OPEN):
    """The plant has nine positions: seven arm joints and the two fingers."""
    return np.concatenate([np.asarray(q7, float), [finger, finger]])
X_WC0 = None                                               # ^W X^M : set in main from the ready pose (MUG_BELOW under the flange)
TABLE_TOP_Z = None


def build(grasped, X_GM=None, meshcat=None):
    """Build the scene. grasped=False: mug welded to the world at X_WC0.
    grasped=True: mug welded to the gripper link at the relative pose X_GM (^G X^M)."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (Panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", Panda))
    # a table for context (visual only)
    plant.RegisterVisualGeometry(plant.world_body(), RigidTransform([X_WC0.translation()[0], X_WC0.translation()[1], TABLE_TOP_Z - 0.02]),
                                 Box(0.8, 0.6, 0.04), "table", [0.75, 0.72, 0.68, 1.0])
    # the mug: a rigid body with a cylinder visual; its inertia is irrelevant because it is welded
    inertia = SpatialInertia(mass=0.2, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia.SolidCylinder(MUG_R, MUG_H, [0, 0, 1]))
    mug = plant.AddRigidBody("mug", plant.AddModelInstance("mug_model"), inertia)
    plant.RegisterVisualGeometry(mug, RigidTransform(), Cylinder(MUG_R, MUG_H), "mug_vis", [0.2, 0.5, 0.6, 1.0])
    gripper = plant.GetFrameByName("panda_link8", Panda)
    if grasped:
        plant.WeldFrames(gripper, mug.body_frame(), X_GM)       # the map is redrawn: mug is part of the arm
    else:
        plant.WeldFrames(plant.world_frame(), mug.body_frame(), X_WC0)
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    return diagram, plant, plant.GetMyContextFromRoot(context), context, Panda, gripper, mug.body_frame()


def mug_jacobian(plant, pc, mug_frame):
    """3 x n Jacobian of the mug centre's translational velocity w.r.t. the arm's joint rates."""
    return plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, mug_frame,
                                                   np.zeros(3), plant.world_frame(), plant.world_frame())


def report(label, plant, pc, Panda, gripper, mug_frame, q, dq):
    q, dq = full_q(q), full_q(dq, 0.0)
    plant.SetPositions(pc, Panda, q)
    p_WG = plant.CalcRelativeTransform(pc, plant.world_frame(), gripper).translation()
    p_WM = plant.CalcRelativeTransform(pc, plant.world_frame(), mug_frame).translation()
    J = mug_jacobian(plant, pc, mug_frame)[:, :7]       # the seven arm joints; the finger columns are zero
    print(f"\n--- {label} ---")
    print("q                      =", q[:7])
    print("^W p^G (gripper)       =", p_WG)
    print("^W p^M (mug)           =", p_WM)
    print("J_mug = d(^W p^M)/dq   =\n", J)
    print("rank J_mug             =", np.linalg.matrix_rank(J, tol=1e-9))
    plant.SetPositions(pc, Panda, q + dq)
    p_WG2 = plant.CalcRelativeTransform(pc, plant.world_frame(), gripper).translation()
    p_WC2 = plant.CalcRelativeTransform(pc, plant.world_frame(), mug_frame).translation()
    print("after dq =", dq[:7], ":")
    print("  gripper moved by     =", p_WG2 - p_WG)
    print("  mug moved by         =", p_WC2 - p_WM, "   (first-order prediction J dq =", J @ dq[:7], ")")
    plant.SetPositions(pc, Panda, q)
    return J


def place_mug():
    """Mug MUG_BELOW under the flange of the ready pose, standing on a table whose top is at the mug's bottom."""
    global X_WC0, TABLE_TOP_Z
    d, plant, pc, ctx, iiwa, gripper, _ = build_arm_only()
    plant.SetPositions(pc, iiwa, full_q(Q_START))
    p_WG = plant.CalcRelativeTransform(pc, plant.world_frame(), gripper).translation()
    X_WC0 = RigidTransform(p_WG + np.array([0.0, 0.0, -MUG_BELOW]))
    TABLE_TOP_Z = X_WC0.translation()[2] - MUG_H / 2


def build_arm_only(meshcat=None):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    (iiwa,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", iiwa))
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext()
    return diagram, plant, plant.GetMyContextFromRoot(ctx), ctx, iiwa, plant.GetFrameByName("panda_link8", iiwa), None


def print_mode():
    place_mug()
    dq = np.array([0.2, 0, 0, 0, 0, 0, 0])   # turn the base joint by 0.2 rad
    d, plant, pc, ctx, Panda, gripper, mug_frame = build(grasped=False)
    report("BEFORE the grasp: the mug is fixed in the world", plant, pc, Panda, gripper, mug_frame, Q_START, dq)
    # the grasp: record where the mug is relative to the gripper right now, then weld it there
    plant.SetPositions(pc, Panda, full_q(Q_START))
    X_WG = plant.CalcRelativeTransform(pc, plant.world_frame(), gripper)
    X_GM = X_WG.inverse().multiply(X_WC0)                    # ^G X^M = (^W X^G)^-1 ^W X^M
    print("\n=== GRASP: weld the mug to the gripper at ^G p^M =", X_GM.translation(), "===")
    d, plant, pc, ctx, Panda, gripper, mug_frame = build(grasped=True, X_GM=X_GM)
    report("AFTER the grasp: the mug is part of the arm", plant, pc, Panda, gripper, mug_frame, Q_START, dq)
    print("\nSame q, same dq, same mug. Only the map changed.")
    selftest_grasp_logic(dq)


class Grasp:
    """The interactive mode's model of the grasp, kept outside Meshcat so that it can be self-tested.
    Not grasped: the mug is fixed in the world at X_WM. Grasped: the mug is rigidly attached to the
    gripper at the relative pose ^G X^M recorded at the moment of the grasp, so its world pose is
    ^W X^G(q) ^G X^M — the same weld as print_mode, maintained kinematically instead of rebuilt."""

    def __init__(self, X_WM):
        self.X_WM, self.X_GM, self.grasped = X_WM, None, False

    def toggle(self, X_WG):
        """Grasp: record ^G X^M. Release: the mug is let go where it is and comes to rest upright on the
        table below that point — this demo has no physics, so the fall is a rule, not a simulation.
        Returns (grasped, X_WM_released) with the pose at the moment of release, for the drop animation."""
        self.grasped = not self.grasped
        if self.grasped:
            self.X_GM = X_WG.inverse().multiply(self.X_WM)      # ^G X^M = (^W X^G)^-1 ^W X^M
            return True, None
        X_released = X_WG.multiply(self.X_GM)
        p = X_released.translation()
        self.X_WM = RigidTransform([p[0], p[1], TABLE_TOP_Z + MUG_H / 2])   # resting on the table, upright
        return False, X_released

    def mug_pose(self, X_WG):
        return X_WG.multiply(self.X_GM) if self.grasped else self.X_WM

    def jacobian(self, plant, pc, gripper):
        """3 x 7: zero while the mug sits on the table; the gripper's Jacobian shifted to the mug's centre
        while grasped — identical to the welded plant's mug Jacobian of print_mode."""
        if not self.grasped:
            return np.zeros((3, 7))
        return plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, gripper, self.X_GM.translation(),
                                                       plant.world_frame(), plant.world_frame())[:, :7]


def selftest_grasp_logic(dq):
    """Drive the interactive mode's Grasp through grasp → move → release → move without a browser and
    check it against the welded numbers above. Runs at the end of print mode."""
    d, plant, pc, ctx, panda, gripper, _ = build_arm_only()
    g = Grasp(X_WC0)
    def X_WG_at(q7):
        plant.SetPositions(pc, panda, full_q(q7)); return plant.CalcRelativeTransform(pc, plant.world_frame(), gripper)
    X0 = X_WG_at(Q_START)
    assert np.linalg.matrix_rank(g.jacobian(plant, pc, gripper)) == 0
    g.toggle(X0)                                                    # grasp at the ready pose
    J = g.jacobian(plant, pc, gripper)
    X1 = X_WG_at(Q_START + dq)
    moved = g.mug_pose(X1).translation() - X_WC0.translation()
    assert np.linalg.matrix_rank(J, tol=1e-9) == 3 and abs(J[1, 0] - 0.307) < 1e-3, J
    assert np.allclose(moved, X1.translation() - X0.translation()), moved
    _, X_rel = g.toggle(X1)                                          # release there: the mug drops to the table
    X2 = X_WG_at(Q_START)
    rest = g.mug_pose(X2).translation()
    assert np.allclose(rest[:2], X_WC0.translation()[:2] + moved[:2]) and abs(rest[2] - (TABLE_TOP_Z + MUG_H / 2)) < 1e-12
    assert np.linalg.matrix_rank(g.jacobian(plant, pc, gripper)) == 0
    print("\nself-test of the interactive grasp logic: grasp, move, release, move — the mug follows the hand only while grasped,"
          "\nand comes to rest on the table under the point where it was let go: OK")


def meshcat_mode():
    place_mug()
    meshcat = StartMeshcat()
    d, plant, pc, ctx, panda, gripper, _ = build_arm_only(meshcat)
    meshcat.SetObject("/table", Box(0.8, 0.6, 0.04), Rgba(0.75, 0.72, 0.68, 1.0))
    meshcat.SetTransform("/table", RigidTransform([X_WC0.translation()[0], X_WC0.translation()[1], TABLE_TOP_Z - 0.02]))
    meshcat.SetObject("/mug", Cylinder(MUG_R, MUG_H), Rgba(0.2, 0.5, 0.6, 1.0))
    names = [f"panda_joint{i}" for i in range(1, 8)]
    for n, q in zip(names, Q_START):
        meshcat.AddSlider(n, -2.9, 2.9, 0.01, float(q))
    meshcat.AddButton("Grasp / release")
    meshcat.AddButton("Stop")
    meshcat.SetCameraPose(np.array([1.25, -1.05, 0.95]), X_WC0.translation())
    g, clicks = Grasp(X_WC0), 0
    print("Meshcat at", meshcat.web_url())
    print("Move the sliders: the mug stays. Press 'Grasp / release': the fingers close and the mug follows the hand;")
    print("press it again: the fingers open and the mug drops onto the table (a rule, not physics: this demo has no dynamics). 'Stop' quits.")
    while meshcat.GetButtonClicks("Stop") == 0:
        q7 = np.array([meshcat.GetSliderValue(n) for n in names])
        plant.SetPositions(pc, panda, full_q(q7, FINGER_ON_MUG if g.grasped else FINGER_OPEN))
        X_WG = plant.CalcRelativeTransform(pc, plant.world_frame(), gripper)
        if meshcat.GetButtonClicks("Grasp / release") != clicks:
            clicks = meshcat.GetButtonClicks("Grasp / release")
            grasped, X_rel = g.toggle(X_WG)
            plant.SetPositions(pc, panda, full_q(q7, FINGER_ON_MUG if grasped else FINGER_OPEN))
            J = g.jacobian(plant, pc, gripper)
            print(("GRASPED — the mug is part of the arm now" if grasped else "RELEASED — the mug drops to the table (no physics: a rule)"),
                  "| rank of the mug's Jacobian:", np.linalg.matrix_rank(J, tol=1e-9), "| max |J| =", round(float(np.abs(J).max()), 4))
            if not grasped:                                            # animate the drop: free fall from the release height, upright
                d.ForcedPublish(ctx)
                z0, z1, p_rel = X_rel.translation()[2], g.X_WM.translation()[2], X_rel.translation()
                t, T = 0.0, np.sqrt(2 * max(z0 - z1, 0.0) / 9.81)
                while t < T:
                    meshcat.SetTransform("/mug", RigidTransform([p_rel[0], p_rel[1], z0 - 0.5 * 9.81 * t * t])); time.sleep(0.02); t += 0.02
        meshcat.SetTransform("/mug", g.mug_pose(X_WG))
        d.ForcedPublish(ctx)
        time.sleep(0.03)
    meshcat.DeleteAddedControls()


def figure_mode(out):
    """Offline render of the interactive mode's state after: grasp at the ready pose, turn the base joint by 0.2 rad."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera, MakeRenderEngineVtk,
                             RenderCameraCore, RenderEngineVtkParams, RgbdSensor, RotationMatrix)
    place_mug()
    d0, plant0, pc0, ctx0, panda0, gripper0, _ = build_arm_only()
    g = Grasp(X_WC0)
    plant0.SetPositions(pc0, panda0, full_q(Q_START)); g.toggle(plant0.CalcRelativeTransform(pc0, plant0.world_frame(), gripper0))
    q7 = Q_START + np.array([0.2, 0, 0, 0, 0, 0, 0])
    plant0.SetPositions(pc0, panda0, full_q(q7)); X_WM = g.mug_pose(plant0.CalcRelativeTransform(pc0, plant0.world_frame(), gripper0))
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    (panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda))
    W = plant.world_body()
    plant.RegisterVisualGeometry(W, RigidTransform([X_WC0.translation()[0], X_WC0.translation()[1], TABLE_TOP_Z - 0.02]), Box(0.8, 0.6, 0.04), "table", [0.75, 0.72, 0.68, 1.0])
    plant.RegisterVisualGeometry(W, X_WM, Cylinder(MUG_R, MUG_H), "mug", [0.2, 0.5, 0.6, 1.0])
    plant.RegisterVisualGeometry(W, X_WC0, Cylinder(MUG_R, MUG_H), "mug_before", [0.2, 0.5, 0.6, 0.25])
    plant.Finalize()
    eye, target = np.array([0.95, -0.75, 0.80]), X_WC0.translation() + np.array([0, 0, 0.05])
    z = target - eye; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
    core = RenderCameraCore("vtk", CameraInfo(1280, 960, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye),
                                          ColorRenderCamera(core, False), DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(ctx), full_q(q7, FINGER_ON_MUG))
    img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(ctx))
    Image.fromarray(np.array(img.data)[:, :, :3]).save(out)
    print("rendered", out, "— the mug in the closed hand after the base turn; its position before the turn as a faint cylinder")


if __name__ == "__main__":
    if "--png" in sys.argv:
        figure_mode(sys.argv[sys.argv.index("--png") + 1])
    else:
        meshcat_mode() if "--meshcat" in sys.argv else print_mode()
