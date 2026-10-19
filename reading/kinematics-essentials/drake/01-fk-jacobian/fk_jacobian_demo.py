"""Sessions 2.3–2.4 Drake demo: forward kinematics and the Jacobian, on the two-link arm and on a real 7-joint arm.

Part 1 builds the lecture's two-link planar arm (l1 = 0.4 m, l2 = 0.3 m) in Drake from two
revolute joints and checks the notes' numbers: the tip position from the kinematic tree and
the 2×2 position Jacobian, at q = (30°, 45°) and at the stretched singularity q2 = 0.
Part 2 loads the Franka Panda and prints its 6×7 spatial Jacobian, rank, and smallest singular
value at the all-zero configuration (rank 5, singular) and at the ready pose (rank 6). It then
prints the ready pose again as the browser's 0.01 rad sliders can hold it, because that rounding
is what the live console shows: sigma_min reads 0.2246 there and 0.2244 at the exact ready pose.
Part 3 prints the near-singular pose that the browser's second preset button sets: rank is still
6 of 6, yet sigma_min is about 4e-4, so the two diagnostics disagree and only sigma_min is
telling the truth about the joint speeds.
Part 3b sweeps joint 4 alone from the ready pose on the sliders' grid (sigma_min dips to 4.3e-4 at -0.47 rad), finds the exact
elbow singularity at q4 = -0.4670024 rad from a 6x6 minor of J (sigma_min there is 3e-13, the numerical floor), and shows
that the grid pose is 0.17 deg away from it and rank 6 at the tolerance 1e-6: a rounded near-singular pose, not the singularity.
It also prints the Panda model's own joint limits and asserts that the near-singular preset is inside them.
Part 4 prints the seven joint axes — direction, origin, and lever arm to the flange — at three
poses, and the six pairwise angles between the axes of joints 1, 3, 5 and 7. Those six angles are all zero at
the all-zero configuration: joints 1, 3 and 5 lie on the world z axis and joint 7 is parallel to
it, 8.8 cm away through the flange point. That is the rank loss of part 2 seen as geometry.

Run:  python fk_jacobian_demo.py            prints everything (notes §5 (i) and (j)), then self-tests
                                            the preset poses and the joint-axis geometry
      python fk_jacobian_demo.py --meshcat  adds a browser view of the Panda with seven joint sliders;
                                            the console prints the end-effector position, rank J,
                                            sigma_min, 0.1/sigma_min and the weakest task direction
                                            live as you move them. Three preset buttons set the
                                            sliders: "Off-plane (axes spread)" (rank 6, the seven
                                            axes point seven ways), "Near-singular (rank still 6)"
                                            (rank 6 and sigma_min about 4e-4), and "Zero configuration
                                            (singular)" (rank 5, sigma_min 0, the axes of joints
                                            1, 3, 5 and 7 all vertical, and the console naming the
                                            lost motion: a pure rotation about the world x axis).
                                            "Show / hide joint axes"
                                            toggles the seven coloured axis arrows, which are drawn by
                                            default; "Stop" ends it
      python fk_jacobian_demo.py --png FILE renders the arm and the seven axis arrows with Drake's VTK
                                            renderer, no browser; add --pose ready|offplane|zero to
                                            choose the configuration (default offplane): the slide
                                            fallback picture for the deck's slide "A real arm: the Panda's zero configuration is singular"; --wrist gives its close-up

Tested 2026-09-05 with drake 1.46.0 on macOS; the presets, the joint axes and the figure mode added
2026-09-20. Runtime of the print mode: a few seconds.
"""
import sys
import time
from pathlib import Path
import numpy as np
from pydrake.all import (AddMultibodyPlantSceneGraph, Box, Cylinder, DiagramBuilder, FixedOffsetFrame,
                         JacobianWrtVariable, Mesh, MeshcatVisualizer, Parser, RevoluteJoint, Rgba,
                         RigidTransform, RotationMatrix, SpatialInertia, StartMeshcat, UnitInertia)

np.set_printoptions(precision=4, suppress=True)
L1, L2 = 0.4, 0.3


# ------------------------------------------------------------------ part 1: two-link arm
def build_two_link():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    inst = plant.AddModelInstance("two_link")
    inertia = SpatialInertia(mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia.SolidBox(0.1, 0.1, 0.1))
    link1 = plant.AddRigidBody("link1", inst, inertia)
    link2 = plant.AddRigidBody("link2", inst, inertia)
    plant.RegisterVisualGeometry(link1, RigidTransform([L1 / 2, 0, 0]), Box(L1, 0.04, 0.04), "link1_vis", [0.6, 0.6, 0.6, 1])
    plant.RegisterVisualGeometry(link2, RigidTransform([L2 / 2, 0, 0]), Box(L2, 0.04, 0.04), "link2_vis", [0.6, 0.6, 0.6, 1])
    # joint 1: link1's frame rotates about the world z axis at the world origin
    plant.AddJoint(RevoluteJoint("q1", plant.world_frame(), link1.body_frame(), [0, 0, 1]))
    # joint 2: sits at the end of link 1, i.e. at T(l1, 0) in link1's frame; link2 rotates about z there
    end1 = plant.AddFrame(FixedOffsetFrame("end_of_link1", link1.body_frame(), RigidTransform([L1, 0, 0])))
    plant.AddJoint(RevoluteJoint("q2", end1, link2.body_frame(), [0, 0, 1]))
    # the end-effector frame E is T(l2, 0) in link2's frame
    tip = plant.AddFrame(FixedOffsetFrame("E", link2.body_frame(), RigidTransform([L2, 0, 0])))
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    return plant, plant.GetMyContextFromRoot(ctx), tip


def two_link_report(q_deg):
    plant, pc, tip = build_two_link()
    q = np.deg2rad(q_deg)
    plant.SetPositions(pc, q)
    X_WE = plant.CalcRelativeTransform(pc, plant.world_frame(), tip)
    J = plant.CalcJacobianTranslationalVelocity(pc, JacobianWrtVariable.kQDot, tip, np.zeros(3),
                                                plant.world_frame(), plant.world_frame())[:2, :]   # x and y rows
    heading = np.degrees(np.arctan2(X_WE.rotation().matrix()[1, 0], X_WE.rotation().matrix()[0, 0]))
    print(f"\n--- two-link arm in Drake, q = {q_deg} deg ---")
    print("  ^W p^E        =", X_WE.translation()[:2], f"  heading {heading:.0f} deg")
    print("  J (x, y rows) =\n", J)
    print(f"  det J = {np.linalg.det(J):.4f}, singular values = {np.linalg.svd(J, compute_uv=False)}, rank = {np.linalg.matrix_rank(J, tol=1e-9)}")
    return X_WE.translation()[:2], J


# ------------------------------------------------------------------ part 2: the Panda
PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm.urdf"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
Q_HOME = np.zeros(7)
Q_BENT = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
# The near-singular preset of the demo plan §3.1. Every entry is a multiple of the sliders' 0.01 rad
# increment, so the pose the browser shows is exactly the pose printed here: sigma_min = 0.000433,
# and a pose whose sigma_min is quoted on a slide must not move when the sliders round it.
# The plan wrote joint 6 as 0.31 rad = 17.76 deg; this file uses 0.44 rad = 25.21 deg, which is also
# inside the Panda model's joint limits (checked in print mode, Part 3b) and changes sigma_min only in
# the sixth decimal (0.000433 against 0.000434), leaving 0.1/sigma_min at 231 rad/s either way.
# Joint 4 at -0.47 rad is the rounded neighbour of the exact elbow singularity at -0.4670024 rad (Part 3b).
Q_NEAR = np.array([0.0, -0.16, 0.0, -0.47, 0.0, 0.44, 0.16])
# The off-plane preset of §3.4: the ready pose with the three roll joints turned, so that the seven
# joint axes point seven different ways before the all-zero button collapses four of them. Nothing
# quoted anywhere depends on its fourth decimal, so it is left as the exact sum rather than snapped.
# The three offsets are chosen so that the figure mode's fixed camera sees every rod from the side:
# with the plan's example offsets (0.6, 0.3, 0.4) joint 4's rod pointed within 3.5 degrees of the
# camera's optical axis and rendered as a 34-pixel blob. With these, the closest any rod comes to
# the optical axis is 45.1 degrees and the closest any two axes come to each other is 22.9 degrees.
Q_OFF = Q_BENT + np.array([-0.25, 0.0, 0.4, 0.0, 0.7, 0.0, 0.0])
PRESET_POSES = {"ready": Q_BENT, "offplane": Q_OFF, "near": Q_NEAR, "zero": Q_HOME}


def build_Panda(meshcat=None):
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


def Panda_numbers(plant, pc, Panda, ee, q):
    plant.SetPositions(pc, Panda, q)
    p = plant.CalcRelativeTransform(pc, plant.world_frame(), ee).translation()
    J = plant.CalcJacobianSpatialVelocity(pc, JacobianWrtVariable.kQDot, ee, np.zeros(3), plant.world_frame(), plant.world_frame())
    sv = np.linalg.svd(J, compute_uv=False)
    return p, J, np.linalg.matrix_rank(J, tol=1e-6), sv[-1]


TASK_COMPONENTS = ["rotation about the world x axis (omega_x)", "rotation about the world y axis (omega_y)",
                   "rotation about the world z axis (omega_z)", "translation along the world x axis (v_x)",
                   "translation along the world y axis (v_y)", "translation along the world z axis (v_z)"]


def weakest_direction(J):
    """The unit task-space direction that the arm serves worst at this pose: the left singular vector
    of J belonging to the smallest singular value. Its six entries are (omega_x, omega_y, omega_z,
    v_x, v_y, v_z), so it mixes rad/s with m/s; the caller is told what fraction of it is translation,
    because only then does "rad/s per 0.1 m/s" mean anything."""
    U, _, _ = np.linalg.svd(J)
    u = U[:, -1]
    return u, float(np.linalg.norm(u[3:]))


def name_direction(d):
    """Say in words what a task direction is when it is one of the six coordinate motions and nothing
    else, and otherwise say only that it is a mixture. The all-zero configuration is the case that
    matters: the direction lost there is (-1, 0, 0, 0, 0, 0), a pure rotation about the world x axis,
    and the console should say so in words at the moment the lecture stops on that pose."""
    alone = np.flatnonzero(np.abs(d) > 0.999)
    if alone.size == 1:
        return "a pure " + TASK_COMPONENTS[int(alone[0])]
    return "a mixture of rotation and translation"


def speed_line(J, smin, indent="    "):
    """The second console line: the joint speed that the weakest task direction costs. 0.1/sigma_min is
    the norm of the least-norm joint velocity that produces a task velocity of 0.1 units along that
    direction, so it is the number that lands in the room. The direction mixes rad/s with m/s, so the
    line names the unit the 0.1 really carries: metres per second when the direction is almost all
    translation, radians per second when it is almost all rotation, and a bare unit in between. At an
    exact singularity the speed is infinite and no joint velocity produces that direction at all, so
    the line names the lost motion instead."""
    d, trans = weakest_direction(J)
    unit = "0.1 m/s" if trans > 0.9 else ("0.1 rad/s" if trans < 0.1 else "0.1 unit")
    if smin <= 0.0:
        return (f"{indent}0.1/sigma_min = inf: the weakest task direction {d} is lost entirely — it is "
                f"{name_direction(d)}, and no joint velocity produces it")
    return (f"{indent}0.1/sigma_min = {0.1 / smin:.1f} rad/s per {unit} along the weakest task direction "
            f"{d} ({100 * trans:.1f}% translation)")


def joint_speed_for_pure_lift(J, v=0.1):
    """A unit-clean companion to 0.1/sigma_min: the least-norm joint velocity, in rad/s, that lifts the
    flange at v metres per second straight up with no rotation. This command has no mixed units at all,
    which is why it is the figure to quote when the weakest direction is not a pure translation."""
    return float(np.linalg.norm(np.linalg.pinv(J) @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, v])))


# ------------------------------------------------------------------ part 3: the seven joint axes
AXIS_LENGTH = 0.5          # full arrow spans ±0.25 m from the joint's origin
AXIS_RADIUS = 0.006        # thin, but a cylinder rather than a line: Meshcat draws lines one pixel wide
ARROWHEAD_LENGTH = 0.06
ARROWHEAD_FILE = Path(__file__).with_name("axis_arrowhead.obj")
AXIS_PATH = "/joint_axes"  # the Meshcat group that the show/hide button makes visible or invisible
AXIS_COLOURS = [(0.84, 0.19, 0.15), (0.13, 0.40, 0.75), (0.17, 0.60, 0.29), (0.55, 0.30, 0.72),
                (0.95, 0.60, 0.07), (0.10, 0.62, 0.62), (0.88, 0.25, 0.60)]


def axis_radius(i):
    """Joint i's rod is a little thinner than joint i−1's. The difference is invisible at a glance, and
    it matters at the all-zero configuration, where the rods of joints 1, 3 and 5 lie on the same line:
    a thinner rod sits inside a thicker one and the picture stays clean, where equal radii would give
    the flickering surface that two coincident faces produce in any renderer."""
    return AXIS_RADIUS * (1.0 - 0.08 * (i - 1))


def joint_axes(plant, pc, Panda, q):
    """The seven joint axes at the configuration q, in world coordinates.

    Returns a list of pairs: the unit direction z-hat_i of joint i's axis and a point on that axis,
    both in the world frame. The hat marks a unit vector, as in [u-hat]_x in the notes. The axis of a
    revolute joint is fixed in the joint's child frame, so the world direction is R_W_Ci times that
    fixed axis and the point is the child frame's origin. The axis is read from the joint object
    rather than assumed: it happens to be (0, 0, 1) for all seven Panda joints, which is what makes
    z-hat_i the third column of the child frame's rotation matrix, but that is a property of this URDF
    and the self-test below is what checks it."""
    plant.SetPositions(pc, Panda, q)
    out = []
    for name in JOINT_NAMES:
        joint = plant.GetJointByName(name, Panda)
        X_WC = plant.CalcRelativeTransform(pc, plant.world_frame(), joint.frame_on_child())
        z = X_WC.rotation().matrix() @ joint.revolute_axis()
        out.append((z / np.linalg.norm(z), X_WC.translation()))
    return out


def axis_angle_table(axes, joints=(1, 3, 5, 7)):
    """The angles between the axes of the named joints, in radians.

    This table compares the unoriented axis lines, ignoring the arrows' signs. Reversing a
    stored direction leaves its line unchanged, so the angle between two axes lies in
    [0, pi/2]. That is why the four roll axes of the Panda register as an angle of exactly zero at the
    all-zero configuration even though joint 7's stored direction is the negative of joint 1's.

    An angle of zero means parallel, which is weaker than collinear and is on its own no reason for
    the rank to fall: at the ready pose joints 1 and 7 are already parallel, and joints 2, 4 and 6 are
    parallel to one another as well, and the rank there is a full 6. What costs the rank is the seven
    directions ceasing to span three dimensions, which is what axis_span_rank below measures."""
    pairs = {}
    for a_pos, a in enumerate(joints):
        for b in joints[a_pos + 1:]:
            c = abs(float(np.dot(axes[a - 1][0], axes[b - 1][0])))
            pairs[(a, b)] = float(np.arccos(min(1.0, c)))
    return pairs


def lever_arm(z, o, p):
    """The perpendicular distance from the point p to the line through o with direction z. For the
    flange point this is |z-hat_i x (p - o)|, which is exactly the length of column i of the
    translational Jacobian: the speed the flange gains per radian per second of joint i."""
    return float(np.linalg.norm(np.cross(z, p - o)))


def axis_span_rank(axes):
    """How many dimensions the seven axis directions span, that is the rank of the 3x7 matrix whose
    columns are the z-hat_i. This is the angular half of the Jacobian's rank: the top three rows of J
    are exactly those seven directions, so a flange rotation about a world axis outside their span is
    one that no combination of joint speeds can produce. It is 3 at the ready pose and 2 at the
    all-zero configuration, where every direction lies in the y-z plane and omega_x is lost."""
    return int(np.linalg.matrix_rank(np.column_stack([z for z, o in axes]), tol=1e-9))


def axis_report(plant, pc, Panda, ee, q, label):
    """Print the seven axes at one pose, with their spread, and return them."""
    p, J, rank, smin = Panda_numbers(plant, pc, Panda, ee, q)
    axes = joint_axes(plant, pc, Panda, q)
    print(f"\n--- joint axes at the {label} pose: q = {q} ---")
    print(f"  ^W p^E = {p}   rank J = {rank} of 6   sigma_min = {smin:.6f}   "
          f"the seven axis directions span {axis_span_rank(axes)} of 3 dimensions")
    print("  joint   axis direction z-hat_i (world)  a point on that axis (world)      lever arm to the flange [m]")
    for i, (z, o) in enumerate(axes, 1):
        print(f"    {i}     {str(z):33s} {str(o):33s} {lever_arm(z, o, p):.4f}")
    angles = axis_angle_table(axes)
    print("  angles between the axes of joints 1, 3, 5, 7 (as lines, so in [0, 90] deg):")
    print("   ", "   ".join(f"({a},{b}) = {np.degrees(v):6.2f} deg" for (a, b), v in angles.items()))
    print(f"    largest {np.degrees(max(angles.values())):.2f} deg, smallest {np.degrees(min(angles.values())):.2f} deg")
    return axes


def axis_visuals(i, z, o):
    """Shared browser/VTK geometry; the cone tip marks the positive joint axis.

    The shaft starts at o−0.25z and meets the cone base at o+0.19z.
    The cone's local tip is zero, so placing it at o+0.25z gives the arrow endpoint.
    These lengths only set the drawing scale; they do not change the robot model.
    """
    X_WA = RigidTransform(RotationMatrix.MakeFromOneVector(z, 2), o)
    return [("shaft", X_WA @ RigidTransform([0., 0., -ARROWHEAD_LENGTH/2]),
             Cylinder(axis_radius(i), AXIS_LENGTH-ARROWHEAD_LENGTH)),
            ("head", X_WA @ RigidTransform([0., 0., AXIS_LENGTH/2]),
             Mesh(str(ARROWHEAD_FILE)))]


def draw_joint_axes(meshcat, axes):
    """Draw a colored shaft and solid cone at the positive end of each joint axis."""
    for i, (z, o) in enumerate(axes, 1):
        r, g, b = AXIS_COLOURS[i - 1]
        for name, pose, shape in axis_visuals(i, z, o):
            path = f"{AXIS_PATH}/joint{i}/{name}"
            meshcat.SetObject(path, shape, Rgba(r, g, b, 1.0))
            meshcat.SetTransform(path, pose)


def axis_colour_legend():
    names = ["red", "blue", "green", "purple", "orange", "teal", "pink"]
    return "joint axis colours: " + ", ".join(
        f"{i} = {n} ({r:.2f}, {g:.2f}, {b:.2f})" for i, ((r, g, b), n) in enumerate(zip(AXIS_COLOURS, names), 1))


def joint4_report(plant, pc, Panda, ee):
    """Part 3b: joint 4 alone, the elbow singularity, and the difference between a rounded near-singular
    pose and an exact singularity. The exact root is found from a 6x6 minor of J (a smooth function that
    changes sign), not from sigma_min, whose V-shaped minimum cannot be located to machine precision."""
    from scipy.optimize import brentq
    lo, hi = plant.GetPositionLowerLimits()[:7], plant.GetPositionUpperLimits()[:7]

    def jac(q): return Panda_numbers(plant, pc, Panda, ee, q)[1]
    def minors(q): J = jac(q); return np.array([np.linalg.det(np.delete(J, j, axis=1)) for j in range(7)])
    def with_q4(base, x): q = np.array(base, float); q[3] = x; return q

    print("\n=== Part 3b: joint 4 alone, the rounded near-singular pose against the exact singularity ===")
    print("  Panda-model joint limits (from the URDF Drake ships): "
          f"joint 4 [{np.degrees(lo[3]):.1f}, {np.degrees(hi[3]):.1f}] deg, joint 6 [{np.degrees(lo[5]):.1f}, {np.degrees(hi[5]):.1f}] deg;")
    inside = bool(np.all(Q_NEAR >= lo) and np.all(Q_NEAR <= hi))
    print(f"  the near-singular preset is inside them: {inside};  the all-zero configuration is: "
          f"{bool(np.all(Q_HOME >= lo) and np.all(Q_HOME <= hi))} (joint 4 must stay below {np.degrees(hi[3]):.1f} deg, joint 6 above {np.degrees(lo[5]):.1f} deg)")
    assert inside
    print("  sweep of joint 4 on the sliders' 0.01 rad grid, the other joints at the ready pose (what the live console shows):")
    print("     q4 [rad]   sigma_min    rank J at tol 1e-6   at tol 1e-3")
    grid = [-0.49, -0.48, -0.47, -0.46, -0.45]
    smins = []
    for x in grid:
        _, J, r6, smin = Panda_numbers(plant, pc, Panda, ee, with_q4(Q_BENT, x))
        smins.append(smin)
        print(f"     {x:+.2f}      {smin:.3e}         {r6}                   {np.linalg.matrix_rank(J, tol=1e-3)}")
    assert grid[int(np.argmin(smins))] == -0.47 and abs(smins[2] - 0.000433) < 1e-6
    f = lambda x: minors(with_q4(Q_BENT, x))[0]
    x_star = brentq(f, -0.50, -0.44, xtol=1e-15, rtol=1e-15)
    q_star = with_q4(Q_BENT, x_star); s = np.linalg.svd(jac(q_star), compute_uv=False)
    print(f"  exact singularity: q4* = {x_star:.10f} rad = {np.degrees(x_star):.4f} deg, the root of a 6x6 minor of J (bisection to 1e-15);")
    print(f"     there all seven 6x6 minors are below {np.abs(minors(q_star)).max():.1e} and sigma_min = {s[-1]:.1e} against sigma_max = {s[0]:.2f}: numerically zero, rank 5.")
    assert s[-1] < 1e-10 and np.abs(minors(q_star)).max() < 1e-10
    print(f"  the 0.01 rad grid pose q4 = -0.47 is {abs(-0.47 - x_star):.4f} rad = {np.degrees(abs(-0.47 - x_star)):.2f} deg away from q4*: sigma_min = {smins[2]:.2e} there, small but not zero.")
    print("     'rank 5' is a statement about a tolerance: the grid pose is rank 6 at tol 1e-6 and rank 5 at tol 1e-3, while q4* is rank 5 at both.")
    # The same root for the near-singular preset, whose joints 2, 6 and 7 differ from the ready pose's.
    x_near = brentq(lambda x: minors(with_q4(Q_NEAR, x))[0], -0.50, -0.44, xtol=1e-15, rtol=1e-15)
    print(f"  the preset's other joints (2, 6, 7) differ from the ready pose's, and its root is q4* = {x_near:.10f} rad: the same to 1e-12.")
    assert abs(x_near - x_star) < 1e-12
    rng = np.random.default_rng(0)
    worst = max(Panda_numbers(plant, pc, Panda, ee, with_q4(rng.uniform(lo, hi), x_star))[3] for _ in range(8))
    print(f"  This is not a general law: for 8 random poses inside the limits (seed 0), sigma_min at q4 = q4* reaches {worst:.1e}, so the")
    print("  location of the singularity depends on the other joints; the two poses above agree, and both have q1 = q3 = q5 = 0.")
    assert worst > 1e-4
    _, _, _, s_near = Panda_numbers(plant, pc, Panda, ee, Q_NEAR)
    print(f"  the browser preset 'Near-singular (rank still 6)' has q4 = {Q_NEAR[3]:+.2f} and sigma_min = {s_near:.6f}: a rounded neighbour of this elbow singularity, not the singularity itself.")


# ------------------------------------------------------------------ print mode and its self-test
def print_mode():
    print("=== Part 1: the lecture's two-link arm, built in Drake from two revolute joints ===")
    two_link_report([30.0, 45.0])
    two_link_report([30.0, 0.0])
    print("\n=== Part 2: Franka Panda, 7 joints, spatial Jacobian 6x7 ===")
    d, plant, pc, ctx, Panda, ee = build_Panda()
    smin_exact = 0.0
    for name, q in (("all-zero configuration (straight up)", Q_HOME), ("bent", Q_BENT)):
        p, J, rank, smin = Panda_numbers(plant, pc, Panda, ee, q)
        print(f"\n--- {name}: q = {q} ---")
        print("  ^W p^E =", p)
        print("  J (rows: omega_x omega_y omega_z v_x v_y v_z; columns: joints 1..7) =\n", J)
        print(f"  rank J = {rank} of 6,  sigma_min = {smin:.4f}")
        smin_exact = smin
    # The browser's sliders move in steps of 0.01 rad, so --meshcat cannot open at the exact ready
    # pose; it opens at the nearest pose on that grid, and the console's sigma_min there is not the
    # one printed above. Meshcat rounds a half step away from zero, so -0.785 becomes -0.79 rather
    # than numpy's -0.78; the line below uses the same rule, and the pose it prints was checked
    # against the sliders of a live Meshcat server on 2026-09-20.
    q_grid = np.sign(Q_BENT) * np.floor(np.abs(Q_BENT) / 0.01 + 0.5) * 0.01
    _, _, rank_grid, smin_grid = Panda_numbers(plant, pc, Panda, ee, q_grid)
    print(f"\n--- the ready pose as the browser's sliders can hold it: q = {q_grid} ---")
    print(f"  rank J = {rank_grid} of 6,  sigma_min = {smin_grid:.6f}, which the live console rounds to "
          f"{smin_grid:.4f}")
    print(f"  The exact ready pose above gives {smin_exact:.6f}, or {smin_exact:.4f} on screen. A quote of either")
    print("  number should say which of the two poses it means.")

    print("\n=== Part 3: the near-singular preset — rank still 6, and only sigma_min says so ===")
    p, J, rank, smin = Panda_numbers(plant, pc, Panda, ee, Q_NEAR)
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"\n--- near-singular (browser preset 'Near-singular (rank still 6)'): q = {Q_NEAR} ---")
    print("  ^W p^E =", p)
    print("  singular values =", sv)
    print(f"  rank J at tol 1e-6 = {rank} of 6,  sigma_min = {smin:.6f}")
    print(f"  rank J at tol 1e-3 = {np.linalg.matrix_rank(J, tol=1e-3)} of 6   (rank is a yes/no that a tolerance decided)")
    print(speed_line(J, smin, indent="  "))
    print(f"  least-norm joint speed for a pure 0.1 m/s lift of the flange: {joint_speed_for_pure_lift(J):.2f} rad/s")
    _, J_ready, _, smin_ready = Panda_numbers(plant, pc, Panda, ee, Q_BENT)
    print(f"  the same command at the ready pose (sigma_min = {smin_ready:.4f}): "
          f"{joint_speed_for_pure_lift(J_ready):.2f} rad/s")
    joint4_report(plant, pc, Panda, ee)

    print("\n=== Part 4: the seven joint axes, drawn in the browser and tabulated here ===")
    print(" ", axis_colour_legend())
    for label, q in (("ready", Q_BENT), ("off-plane (browser preset 'Off-plane (axes spread)')", Q_OFF),
                     ("all-zero", Q_HOME)):
        axis_report(plant, pc, Panda, ee, q, label)
    print("\nAt the all-zero configuration the four roll axes are parallel: joints 1, 3 and 5 lie on the")
    print("world z axis itself and joint 7 is parallel to it, 8.8 cm away, through the flange point. Three")
    print("lever arms are 0.088 m and joint 7's is 0, which is why v_y survives: those three joints still")
    print("carry the flange sideways. omega_x is lost for a different reason, the one the span column")
    print("reports: no axis has any component along world x, so the seven directions span only the y-z")
    print("plane and no joint can turn the flange about x. Parallel by itself would have cost nothing — at")
    print("the ready pose joints 1 and 7 are parallel, joints 2, 4 and 6 are parallel to one another, and")
    print("the rank there is a full 6. The rank falls when the directions stop spanning three dimensions.")
    selftest_presets_and_axes(plant, pc, Panda, ee)


def selftest_presets_and_axes(plant, pc, Panda, ee):
    """Check the claims that the browser's preset buttons and axis rods are there to make. Runs at the
    end of print mode, so that check_all.sh exercises every number the slides quote."""
    p_near, J_near, rank_near, smin_near = Panda_numbers(plant, pc, Panda, ee, Q_NEAR)
    assert rank_near == 6 and smin_near < 1e-3, (rank_near, smin_near)
    # The browser's sliders step by 0.01 rad, so a preset off that grid would put a sigma_min on screen
    # that no script prints. This is the assertion behind the comment at the constant.
    assert np.allclose(Q_NEAR / 0.01, np.round(Q_NEAR / 0.01)), Q_NEAR
    p_zero, J_zero, rank_zero, smin_zero = Panda_numbers(plant, pc, Panda, ee, Q_HOME)
    assert rank_zero == 5 and smin_zero < 1e-9, (rank_zero, smin_zero)
    assert max(axis_angle_table(joint_axes(plant, pc, Panda, Q_HOME)).values()) < 1e-6
    # Parallel axes alone do not cost rank; a collapsed span does. Both halves are pinned here.
    assert axis_span_rank(joint_axes(plant, pc, Panda, Q_HOME)) == 2
    assert axis_span_rank(joint_axes(plant, pc, Panda, Q_BENT)) == 3
    assert axis_angle_table(joint_axes(plant, pc, Panda, Q_BENT))[(1, 7)] < 1e-6
    # The axis directions are the third column of each joint frame's rotation matrix exactly because
    # every Panda joint turns about (0, 0, 1) in its own child frame. Check both halves of that.
    for q in (Q_BENT, Q_OFF, Q_NEAR, Q_HOME):
        plant.SetPositions(pc, Panda, q)
        _, J, _, _ = Panda_numbers(plant, pc, Panda, ee, q)
        for i, ((z, o), name) in enumerate(zip(joint_axes(plant, pc, Panda, q), JOINT_NAMES), 1):
            joint = plant.GetJointByName(name, Panda)
            assert np.allclose(joint.revolute_axis(), [0.0, 0.0, 1.0]), (name, joint.revolute_axis())
            X_WC = plant.CalcRelativeTransform(pc, plant.world_frame(), joint.frame_on_child())
            assert np.allclose(z, X_WC.rotation().matrix()[:, 2]), (name, z)
            assert np.allclose(o, X_WC.translation()), (name, o)
            assert np.allclose(J[:3, i-1], z), (name, "positive angular Jacobian column")
            _, head_pose, _ = axis_visuals(i, z, o)[1]
            assert np.allclose(head_pose.translation()-o, AXIS_LENGTH/2*z), (name, "arrow tip")
            assert np.allclose(head_pose.rotation().matrix()[:, 2], z), (name, "arrow direction")
    print("\nself-test: the near-singular preset gives rank 6 with sigma_min < 1e-3 and sits exactly on the"
          "\nsliders' 0.01 rad grid; the all-zero pose gives rank 5 with sigma_min < 1e-9, the axes of joints"
          "\n1, 3, 5 and 7 are parallel to within 1e-6 rad there, and the seven directions span 2 dimensions"
          "\nagainst 3 at the ready pose, where joints 1 and 7 are already parallel at full rank; and every"
          "\njoint axis is the third column of its child frame's rotation; each arrow points along the"
          "\npositive angular Jacobian column: OK")


# ------------------------------------------------------------------ interactive mode
def meshcat_mode():
    meshcat = StartMeshcat()
    presets = [("Off-plane (axes spread)", Q_OFF), ("Near-singular (rank still 6)", Q_NEAR),
               ("Zero configuration (singular)", Q_HOME)]
    for n, q in zip(JOINT_NAMES, Q_BENT):
        meshcat.AddSlider(n, -2.9, 2.9, 0.01, float(q))
    for label, _ in presets:
        meshcat.AddButton(label)
    meshcat.AddButton("Show / hide joint axes")
    meshcat.AddButton("Arrows point along positive joint axes")
    meshcat.AddButton("Positive rotation follows the right-hand rule")
    meshcat.AddButton("Stop")
    d, plant, pc, ctx, Panda, ee = build_Panda(meshcat)
    print("Meshcat at", meshcat.web_url(), "- move the sliders; watch rank J and sigma_min; 'Stop' to quit.")
    print(" ", axis_colour_legend())
    print("  Arrowheads indicate the positive joint axes; positive rotation follows the right-hand rule.")
    print("  At the pose it opens in, joints 1 and 7 are already parallel and joints 2, 4 and 6 are parallel to one")
    print("  another, at full rank 6: parallel is not collinear. Press the off-plane preset to spread the rods,")
    print("  then the zero configuration to collapse joints 1, 3, 5, 7 onto vertical lines.")
    print("  The sliders round to 0.01 rad, so the ready pose they open at is the ready pose rounded;")
    print("  the near-singular preset is on that grid exactly, so its sigma_min is the printed one.")
    last, clicks, axes_shown = None, {label: 0 for label, _ in presets}, True
    axes_clicks = 0
    while meshcat.GetButtonClicks("Stop") == 0:
        for label, q_preset in presets:                     # each preset owns its own click counter
            if meshcat.GetButtonClicks(label) != clicks[label]:
                clicks[label] = meshcat.GetButtonClicks(label)
                for n, value in zip(JOINT_NAMES, q_preset):
                    meshcat.SetSliderValue(n, float(value))
        if meshcat.GetButtonClicks("Show / hide joint axes") != axes_clicks:
            axes_clicks = meshcat.GetButtonClicks("Show / hide joint axes")
            axes_shown = not axes_shown                     # Meshcat has no checkbox, so the button carries the state
            meshcat.SetProperty(AXIS_PATH, "visible", axes_shown)
        q = np.array([meshcat.GetSliderValue(n) for n in JOINT_NAMES])
        p, J, rank, smin = Panda_numbers(plant, pc, Panda, ee, q)
        draw_joint_axes(meshcat, joint_axes(plant, pc, Panda, q))
        d.ForcedPublish(ctx)
        msg = (f"^W p^E = {p}   rank J = {rank}/6   sigma_min = {smin:.4f}\n" + speed_line(J, smin))
        if msg != last:
            print(msg); last = msg
        time.sleep(0.05)
    meshcat.DeleteAddedControls()


# ------------------------------------------------------------------ figure mode
def figure_mode(out, pose="offplane", wrist=False):
    """Render the arm and its seven joint axes with Drake's VTK render engine (offline; no browser).
    The axis rods are scene-graph visuals here, not Meshcat objects, so that the renderer sees them.
    One honest limitation of the all-zero picture: the axes of joints 1 and 3 run down the middle of
    the arm's own links, so their rods are hidden inside the robot. What the picture does show is that
    every rod is either vertical or along y, and that nothing at all points toward the viewer — the
    missing omega_x. In the browser the view can be orbited; here it cannot."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
                             MakeRenderEngineVtk, RenderCameraCore, RenderEngineVtkParams, RgbdSensor)
    q = PRESET_POSES[pose]
    d0, plant0, pc0, ctx0, Panda0, ee0 = build_Panda()
    p, J, rank, smin = Panda_numbers(plant0, pc0, Panda0, ee0, q)
    axes = joint_axes(plant0, pc0, Panda0, q)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    (Panda,) = Parser(plant).AddModelsFromUrl(PANDA_URL)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", Panda))
    W = plant.world_body()
    for i, (z, o) in enumerate(axes, 1):
        r, g, b = AXIS_COLOURS[i - 1]
        for name, X_Wvisual, shape in axis_visuals(i, z, o):
            plant.RegisterVisualGeometry(W, X_Wvisual, shape, f"axis{i}_{name}", [r, g, b, 1.0])
    plant.Finalize()
    # The camera looks at the middle of the arm from in front, above and to the side, far enough back
    # that the half-metre rods stay inside the picture at all three poses.
    eye, target = np.array([1.90, -1.60, 1.30]), np.array([0.05, 0.0, 0.66])
    if wrist:
        target = p + np.array([0., 0., 0.10])
        eye = target + np.array([0.80, -1.00, 0.30])
    z_cam = target - eye; z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, [0, 0, 1.0]); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    X_WCam = RigidTransform(RotationMatrix(np.column_stack([x_cam, y_cam, z_cam])), eye)   # Drake camera: z forward, y down
    core = RenderCameraCore("vtk", CameraInfo(1280, 960, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WCam, ColorRenderCamera(core, False),
                                          DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build(); ctx = diagram.CreateDefaultContext()
    plant.SetPositions(plant.GetMyContextFromRoot(ctx), q)
    img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(ctx))
    Image.fromarray(np.array(img.data)[:, :, :3]).save(out)
    print(f"rendered {out} | pose {pose}: q = {q} | rank J = {rank}/6 | sigma_min = {smin:.6f}")
    print(" ", axis_colour_legend())


if __name__ == "__main__":
    if "--png" in sys.argv:
        which = sys.argv[sys.argv.index("--pose") + 1] if "--pose" in sys.argv else "offplane"
        figure_mode(sys.argv[sys.argv.index("--png") + 1], which, wrist="--wrist" in sys.argv)
    else:
        meshcat_mode() if "--meshcat" in sys.argv else print_mode()
