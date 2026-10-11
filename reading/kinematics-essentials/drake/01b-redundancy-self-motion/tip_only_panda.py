"""Session 2.4 Drake demo, second version: only the fingertip POSITION is fixed; everything else is free.

The first demo (self_motion_panda.py) fixes the whole gripper pose, m = 6, and the Panda's n = 7
joints leave a one-dimensional curve of solutions. Here only the point between the fingertips is
fixed, m = 3, so four degrees of freedom are left over: the gripper may tilt and turn any way it
likes as long as that point stays on the mug. There is no longer one curve to slide along, so the
interaction changes: the seven joint sliders are back, but moving one joint makes the OTHER SIX
compensate so that the tip does not move. You choose one joint; the arm chooses the rest.

    tip point:  0.1034 m along the flange's z axis (Franka's tool centre point, between the fingertips)
    held at     ^W p^tip = (0.556, 0, 0.142) m, the top of the mug

How it works. J_p(q) is the 3x7 translational Jacobian of the tip point (rank 3 everywhere the arm
is not stretched, so its null space is 7 - 3 = 4 dimensional). When slider j moves to a new value,
q_j is set to it and the remaining six joints get the minimum-norm correction that puts the tip
back: dq_others = -J_others^+ (p(q) - p*), two or three times (Gauss-Newton on the position error,
essentials §7 and Appendix C.3). "Random pose, same tip" takes a 0.6 rad step in a random null-space direction and
corrects the same way; "Play / pause wander" does that continuously in small steps with a slowly
turning direction, so the arm drifts through the four-dimensional family of solutions while the
tip stays. Joint limits are respected by clipping; if a commanded joint value leaves no room for
the others to hold the tip, the console says so and reports the residual instead of pretending.

Why some joints are free and others nearly stuck (added 2026-09-23). Commanding joint j by one
radian moves the tip by the column J_j, and the other six must undo it with -J_others^+ J_j. Joint 7's
axis passes through the tool centre point, so J_7 = 0: turning it moves the tip not at all and
nothing compensates. Joint 5's axis passes 5.5 cm from the tip: almost free. Joint 2 moves the tip
0.59 m per radian, and the only joints that can undo that in the same plane are 4 and 6, at 2.5 and
4.7 rad per rad. Dragging joint 2 downward stops after about two degrees, near 15 deg, and no joint
limit is involved: the Jacobian of the six compensating joints loses rank there (their nearest limit
is 40 deg away), so they approach a stretched configuration of their own. Dragging it upward has 72
deg of room, to 89 deg, where joint 4 meets its -4 deg limit. Joint 3's column is as long as joint 2's
(0.588 m per rad) yet the others cancel it at 1.05 rad per rad and it keeps its whole range: the column
says how far the tip would go, the compensation number says how hard that is to cancel. The demo shows the columns as arrows at the tip and the three numbers per joint
as readouts in the control panel. (A version that shrank each slider to its feasible range was tried
and dropped on 2026-09-23: Meshcat draws every slider at the same width, so the range changed but
nothing visible did.)

Run:  python tip_only_panda.py            prints rank and nullity of the 3x7 tip Jacobian at the elbow-up
                                          start and the per-joint table (column norm |J_j|, the other six
                                          joints' motion per radian of j, the feasible range with the tip
                                          held), then holds the tip while joint 1 is commanded from 0 to
                                          60 deg (the other six joints' answer), then a 400-step random
                                          wander through the null space with the largest tip error, the
                                          range of gripper tilt and of each joint; then self-tests
      python tip_only_panda.py --meshcat  opens the arm with its gripper on the mug and SEVEN sliders. Drag
                                          any one: that joint follows the slider and the other six move so
                                          that the tip stays on the mug (the sliders of the six are updated
                                          to show what the arm did). Buttons: "Random pose, same tip";
                                          "Play / pause wander" (a continuous random walk through the
                                          solutions, pausing in place on the next press); "Reset (elbow up)";
                                          "Show / hide wrist trail" (teal beads where the flange has been:
                                          the wrist wanders, the fingertips do not); "Show / hide Jacobian
                                          columns" (the seven columns of the 3x7 tip Jacobian as arrows at
                                          the tip, one colour per joint as in drake/01-fk-jacobian: joint 7
                                          has none, its axis passes through the tip; joint 5's is a stub);
                                          "Stop" ends it. Below the buttons, seven READOUTS (buttons that do
                                          nothing, relabelled whenever the arm comes to rest): per joint,
                                          |J_j| in tip metres per radian, how many radians the other six
                                          must move per radian of that joint, and the range it can be
                                          commanded to from here with the tip held (joint 2: 15..89 deg at
                                          the start, joint 7: the full range). The console prints q in
                                          degrees, the tip error, the gripper's tilt from vertical and, at
                                          rest, the same ranges
      python tip_only_panda.py --png FILE renders four random solutions side by side, offline
      python tip_only_panda.py --app      serves the same live scene with a readable text/table panel;
                                          joint sliders stay in Meshcat's Open Controls menu. The panel
                                          shows the actual held-point error separately from refused
                                          commands. Add --app-port 8766 for a fixed app address.

Tested 2026-09-23 with drake 1.46.0 on macOS. Runtime of the print mode: about a second.
"""
import argparse
import atexit
import sys
import time
from pathlib import Path
import numpy as np
from pydrake.all import (Cylinder, JacobianWrtVariable, MeshcatCone, MeshcatVisualizer, Rgba, RigidTransform,
                         RotationMatrix, Sphere, StartMeshcat)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from self_motion_panda import Arm, MUG_TOP_Z, JOINT_NAMES, Q_UP, build   # noqa: E402  (same scene, same arm)

np.set_printoptions(precision=4, suppress=True)
P_TIP_E = np.array([0.0, 0.0, 0.1034])              # the tool centre point in the flange (panda_link8) frame
TRAIL = "/wrist_trail"
TRAIL_N = 240                                        # beads kept; older ones are recycled
RNG_SEED = 3
ARROWS = "/jacobian_columns"
# the joint colours of drake/01-fk-jacobian (red, blue, green, purple, orange, teal, pink), so that the
# arrow of joint j here is the colour of joint j's axis rod there
AXIS_COLOURS = [(0.84, 0.19, 0.15), (0.13, 0.40, 0.75), (0.17, 0.60, 0.29), (0.55, 0.30, 0.72),
                (0.95, 0.60, 0.07), (0.10, 0.62, 0.62), (0.88, 0.25, 0.60)]
ARROW_SCALE = 0.5                                    # metres of arrow per metre-per-radian of column


class TipArm(Arm):
    """The Arm of the first demo plus the tip point's position and 3x7 Jacobian."""

    def tip(self, q7):
        return self.X(q7).multiply(P_TIP_E)

    def Jp(self, q7):
        self.set(q7)
        J = self.plant.CalcJacobianTranslationalVelocity(self.pc, JacobianWrtVariable.kQDot, self.ee, P_TIP_E,
                                                         self.plant.world_frame(), self.plant.world_frame())
        return J[:, :7]

    def tilt_deg(self, q7):
        """Angle between the tool axis (flange z, pointing out of the gripper) and straight down."""
        z = self.X(q7).rotation().matrix()[:, 2]
        return np.degrees(np.arccos(np.clip(-z[2], -1.0, 1.0)))

    def clip(self, q7):
        return np.clip(q7, self.lo, self.hi)

    def hold_tip(self, q7, p_star, free, iters=8, tol=1e-10):
        """Gauss-Newton on the tip error, moving only the joints in `free` (a boolean mask); the
        others keep their values. Returns the corrected q and its remaining tip error in metres.
        Joint limits or failure of the local correction can leave a non-zero residual."""
        q = q7.copy()
        for _ in range(iters):
            e = self.tip(q) - p_star
            if np.linalg.norm(e) < tol:
                break
            Jf = self.Jp(q)[:, free]
            q[free] = q[free] - np.linalg.pinv(Jf) @ e
            q = self.clip(q)
        return q, np.linalg.norm(self.tip(q) - p_star)

    def null_basis(self, q7):
        """The four joint velocities that leave the tip still: the right-singular vectors of J_p
        beyond its rank."""
        _, s, vt = np.linalg.svd(self.Jp(q7))
        return vt[3:].T, s                           # 7 x 4, and the three singular values

    def null_step(self, q7, p_star, c, ds):
        """Move ds along the null-space direction with coefficients c (a 4-vector), then correct."""
        N, _ = self.null_basis(q7)
        n = N @ c
        n /= np.linalg.norm(n)
        q = self.clip(q7 + ds * n)
        return self.hold_tip(q, p_star, np.ones(7, bool))


def command_joint(arm, q, j, value, p_star):
    """Slider j moved: set that joint, let the other six put the tip back."""
    q = q.copy()
    q[j] = value
    free = np.ones(7, bool)
    free[j] = False
    return arm.hold_tip(q, p_star, free)


def wander(arm, q0, p_star, steps, ds=0.02, seed=RNG_SEED, turn=0.15):
    """A smooth random walk through the null space: the direction c turns a little each step and
    flips where a joint limit stops the motion. Returns the samples and the tip errors."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(4)
    q, qs, errs = q0.copy(), [q0.copy()], [0.0]
    for _ in range(steps):
        c = c + turn * rng.standard_normal(4)
        c /= np.linalg.norm(c)
        q_new, err = arm.null_step(q, p_star, c, ds)
        if err > 1e-8 or np.linalg.norm(q_new - q) < 0.25 * ds:   # a joint limit is in the way: pick a new way
            c = rng.standard_normal(4)
            q_new, err = arm.null_step(q, p_star, c, ds)
            if err > 1e-8:
                q_new, err = q, 0.0                                # stay put this step rather than drift
        q = q_new
        qs.append(q.copy()); errs.append(err)
    return np.array(qs), np.array(errs)


def joint_table(arm, q, p_star):
    """Per joint: how far the tip moves per radian (the column norm |J_j|) and how many radians the
    other six must move per radian of j to undo it (|J_others^+ J_j|). Joint 7's column is zero
    because the tool centre point lies on its axis; joint 5's is small for the same reason."""
    J = arm.Jp(q)
    rows = []
    for j in range(7):
        free = np.ones(7, bool); free[j] = False
        comp = np.linalg.pinv(J[:, free]) @ J[:, j]
        rows.append((np.linalg.norm(J[:, j]), np.linalg.norm(comp)))
    return J, rows


def reach(arm, q, p_star, step_deg=2.0, tol=1e-6):
    """For each joint, the interval of values it can be commanded to, from q, while the other six
    still hold the tip inside their limits: the feasible range, scanned in steps of step_deg."""
    out = []
    for j in range(7):
        ends = []
        for sgn in (-1, 1):
            qq, last = q.copy(), q[j]
            for k in range(1, int(360 / step_deg)):
                v = q[j] + sgn * np.radians(k * step_deg)
                if v < arm.lo[j] or v > arm.hi[j]:
                    break
                qq, err = command_joint(arm, qq, j, v, p_star)
                if err > tol:
                    break
                last = v
            ends.append(last)
        out.append((ends[0], ends[1]))
    return out


def print_table(arm, q, p_star):
    J, rows = joint_table(arm, q, p_star)
    rg = reach(arm, q, p_star)
    print("  joint | tip per rad |J_j| | other six per rad | feasible range with the tip held")
    for j, ((cj, comp), (a, b)) in enumerate(zip(rows, rg), 1):
        print(f"    {j}   |   {cj:5.3f} m      |   {comp:5.2f} rad      |  {np.degrees(a):7.1f} .. {np.degrees(b):7.1f} deg"
              f"  (limits {np.degrees(arm.lo[j - 1]):.0f} .. {np.degrees(arm.hi[j - 1]):.0f}, now {np.degrees(q[j - 1]):.1f})")
    return J, rows, rg


# ------------------------------------------------------------------ print mode
def print_mode():
    builder, plant, _, panda = build()
    arm = TipArm(builder.Build(), plant, panda)
    p_star = arm.tip(Q_UP)
    N, s = arm.null_basis(Q_UP)
    print("--- the fixed point ---")
    print(f"  tip ^W p = {p_star} (the mug's top is at z = {MUG_TOP_Z}); flange ^W p^E = {arm.X(Q_UP).translation()}")
    print(f"  J_p is 3 x 7 with singular values {s}: rank 3, null space of dimension 7 - 3 = {N.shape[1]}")
    print("\n--- why some joints are free and others nearly stuck (at the elbow-up start) ---")
    print("  J_p [m/rad], rows x y z, one column per joint:\n", arm.Jp(Q_UP))
    J, rows, rg = print_table(arm, Q_UP, p_star)
    print("  joint 7's column is exactly zero: the tool centre point lies on its axis, so turning it moves the tip")
    print("  not at all and nothing has to compensate. Joint 2 moves the tip 0.59 m per rad and only joints 4 and 6")
    print("  can undo that in the same plane, at several rad per rad; a downward drag of joint 2 stops after about two")
    print("  degrees, where those compensating joints approach a stretched configuration of their own and lose rank.")
    print("  No joint limit is involved there; upward, joint 4's -4 deg limit ends the motion near 89 deg.")
    print("\n--- command one joint, the other six hold the tip ---")
    q = Q_UP.copy()
    for deg in (20, 40, 60):
        q, err = command_joint(arm, q, 0, np.radians(deg), p_star)
        print(f"  joint 1 -> {deg:2d} deg:  q [deg] = {np.degrees(q)}  tip error {err:.1e} m  tilt {arm.tilt_deg(q):5.1f} deg")
    q60 = q.copy()
    print("\n--- 400 steps of random wander through the null space ---")
    qs, errs = wander(arm, Q_UP, p_star, 400)
    tilts = np.array([arm.tilt_deg(qq) for qq in qs])
    flange = np.array([arm.X(qq).translation() for qq in qs])
    print(f"  largest tip error {errs.max():.1e} m; gripper tilt from vertical {tilts.min():.1f}..{tilts.max():.1f} deg")
    print(f"  flange travels {np.ptp(flange, axis=0)} m while the tip does not")
    print("  joint ranges [deg]:", np.degrees(qs.min(axis=0)), "..", np.degrees(qs.max(axis=0)))
    assert N.shape == (7, 4) and s.min() > 0.05
    assert rows[6][0] < 1e-9 and rows[6][1] < 1e-9                       # joint 7: zero column, no compensation
    assert rows[1][1] > 5 and rows[1][0] > 0.5                            # joint 2: the expensive one
    assert np.degrees(rg[1][1] - rg[1][0]) < 80 and np.degrees(rg[6][1] - rg[6][0]) > 300
    assert err < 1e-9 and abs(q60[0] - np.radians(60)) < 1e-12
    assert errs.max() < 1e-8 and tilts.max() > 20 and np.ptp(flange, axis=0).max() > 0.05
    print("\nself-test: rank 3 and a 4-dimensional null space at the start; joint 1 commanded to 60 deg with the"
          "\ntip held to 1e-9 m by the other six; a 400-step wander keeps the tip within 1e-8 m while the gripper"
          "\ntilts by more than 20 deg and the flange moves more than 5 cm; joint 7's Jacobian column is zero and"
          "\nneeds no compensation, joint 2's costs over 5 rad per rad and its feasible range is under 80 deg"
          "\nagainst over 300 deg for joint 7: OK")


# ------------------------------------------------------------------ interactive mode
def draw_columns(meshcat, arm, q):
    """The seven columns of J_p as arrows from the tip, coloured by joint, length ARROW_SCALE |J_j|:
    where the tip goes, per radian, when that joint alone turns. Joint 7's arrow has zero length."""
    J = arm.Jp(q)
    p = arm.tip(q)
    for j in range(7):
        col = J[:, j]
        length = ARROW_SCALE * np.linalg.norm(col)
        r, g, b = AXIS_COLOURS[j]
        path = f"{ARROWS}/joint{j + 1}"
        if length < 1e-3:
            meshcat.Delete(path)
            continue
        R = RotationMatrix.MakeFromOneVector(col, 2)
        meshcat.SetObject(f"{path}/shaft", Cylinder(0.006, length), Rgba(r, g, b, 1.0))
        meshcat.SetTransform(f"{path}/shaft", RigidTransform(R, p + 0.5 * length * col / np.linalg.norm(col)))
        # MeshcatCone has its point at the origin and widens along +z, so the head is turned round
        # (z pointing back along the shaft) and its point placed one head length beyond the shaft's end;
        # before 2026-10-07 the wide end pointed outward and every arrow looked reversed.
        head = 0.03
        R_head = R @ RotationMatrix.MakeXRotation(np.pi)
        meshcat.SetObject(f"{path}/head", MeshcatCone(head, 0.015, 0.015), Rgba(r, g, b, 1.0))
        meshcat.SetTransform(f"{path}/head", RigidTransform(R_head, p + (length + head) * col / np.linalg.norm(col)))


class Readouts:
    """Live per-joint numbers in the control panel. Meshcat has no text widget, but a button's label
    is text and a button can be deleted and re-added, so seven buttons that do nothing carry, for each
    joint: |J_j| (tip metres per radian), the other six joints' motion per radian of j, and the range
    the joint can be commanded to from here while the tip is held. The readouts are refreshed when the
    arm comes to rest, not on every tick, so the panel does not flicker while a slider is dragged."""

    def __init__(self, meshcat):
        self.meshcat, self.labels = meshcat, [None] * 7

    @staticmethod
    def label(j, cj, comp, a, b):
        return (f"j{j + 1}: tip {cj:4.2f} m/rad | others {comp:4.2f} rad/rad | "
                f"range {np.degrees(a):4.0f}..{np.degrees(b):4.0f} deg")

    def update(self, arm, q, p_star):
        _, rows = joint_table(arm, q, p_star)
        rg = reach(arm, q, p_star)
        new = [self.label(j, cj, comp, a, b) for j, ((cj, comp), (a, b)) in enumerate(zip(rows, rg))]
        if new != self.labels:                        # re-add all seven together so they keep their order
            for old in self.labels:
                if old is not None:
                    self.meshcat.DeleteButton(old)
            for lab in new:
                self.meshcat.AddButton(lab)
            self.labels = new
        return rg


class PanelReadouts:
    """Cache the expensive continuation scan without adding dummy Meshcat buttons."""

    def __init__(self):
        self.ranges, self.q = None, None

    def update(self, arm, q, p_star):
        self.ranges = reach(arm, q, p_star)
        self.q = q.copy()
        arm.set(q)  # The continuation scan leaves the plant at its last trial.
        return self.ranges


def panel_snapshot(arm, q, p_star, readouts, *, playing=False, status="ready",
                   message="Move one joint and watch the others compensate.",
                   commanded_joint=None, last_action="Elbow-up start"):
    """Measure the displayed configuration, never a rejected trial configuration."""
    J, rows = joint_table(arm, q, p_star)
    position = arm.tip(q)
    rank = int(np.linalg.matrix_rank(J))
    snapshot = {
        "playing": playing, "status": status, "message": message,
        "commanded_joint": commanded_joint, "last_action": last_action,
        "tip_position_m": position.tolist(), "target_position_m": p_star.tolist(),
        "tip_error_m": float(np.linalg.norm(position - p_star)),
        "tilt_deg": float(arm.tilt_deg(q)),
        "joints_deg": np.degrees(q).tolist(),
        "joint_change_deg": np.degrees(q - Q_UP).tolist(),
        "jacobian_rank": rank, "nullity": J.shape[1] - rank,
        "jacobian_pos": J.tolist(),
        "column_norms_m_per_rad": [float(row[0]) for row in rows],
        "compensation_norms_rad_per_rad": [float(row[1]) for row in rows],
        "range_estimates_deg": (np.degrees(readouts.ranges).tolist()
                                if readouts.ranges is not None else None),
        "ranges_current": (readouts.q is not None and
                           bool(np.linalg.norm(q - readouts.q) < 1e-9)),
    }
    arm.set(q)
    return snapshot


def meshcat_mode(app=False, app_port=0, seed=None):
    meshcat = StartMeshcat()
    if app:
        meshcat.SetCameraPose([1.25, -1.0, 0.85], [0.30, 0.0, 0.35])
    builder, plant, _, panda = build(meshcat)
    diagram = builder.Build()
    arm = TipArm(diagram, plant, panda)
    p_star = arm.tip(Q_UP)
    rng = np.random.default_rng(seed)
    # The joint sliders read in degrees (the instructor's request, 2026-10-07); the arm and every
    # computation stay in radians, so the values are converted at the slider boundary.
    SLIDERS = [f"{n} [deg]" for n in JOINT_NAMES]
    for n, lo_j, hi_j, v in zip(SLIDERS, arm.lo, arm.hi, Q_UP):
        meshcat.AddSlider(n, float(np.degrees(lo_j)), float(np.degrees(hi_j)), 0.5, float(np.degrees(v)))
    for b in ("Random pose, same tip", "Play / pause wander", "Reset (elbow up)", "Show / hide wrist trail",
              "Show / hide Jacobian columns", "Stop"):
        meshcat.AddButton(b)
    meshcat.SetObject("/tip_target", Sphere(0.012), Rgba(0.85, 0.65, 0.1, 1.0))
    meshcat.SetTransform("/tip_target", RigidTransform(p_star))

    q = Q_UP.copy()
    trail_i, trail_shown = 0, True
    shown = np.array(Q_UP)                             # what the sliders read after we last wrote them

    def show(q, err):
        nonlocal trail_i, shown
        arm.set(q)
        diagram.ForcedPublish(arm.ctx)
        draw_columns(meshcat, arm, q)
        for n, v in zip(SLIDERS, q):
            meshcat.SetSliderValue(n, float(np.degrees(v)))
        shown = np.radians([meshcat.GetSliderValue(n) for n in SLIDERS])   # as clamped to the joint limits
        meshcat.SetObject(f"{TRAIL}/{trail_i % TRAIL_N}", Sphere(0.006), Rgba(0.2, 0.5, 0.6, 1.0))
        meshcat.SetTransform(f"{TRAIL}/{trail_i % TRAIL_N}", RigidTransform(arm.X(q).translation()))
        trail_i += 1

    show(q, 0.0)
    readouts = PanelReadouts() if app else Readouts(meshcat)
    readouts.update(arm, q, p_star)
    panel = None
    if app:
        from live_panel import LivePanel
        panel = LivePanel(meshcat.web_url(), port=app_port)
        panel.publish(panel_snapshot(arm, q, p_star, readouts))
        atexit.register(panel.close)
        print("Demo app at", panel.start(), flush=True)
    print("Meshcat at", meshcat.web_url())
    print("  The arrows at the tip are the seven columns of the 3 x 7 tip Jacobian, one colour per joint (the colours")
    print("  of drake/01-fk-jacobian's axis rods): where the tip goes per radian of that joint alone. Joint 7 has no")
    print("  arrow: its axis passes through the tool centre point. Per-joint values are in the app's table" if app
          else "  arrow: its axis passes through the tool centre point. The seven grey buttons are readouts.")
    print("  Tip gain is in m/rad; the norm of the other six joints' compensation is in rad/rad.")
    print("  Command ranges are estimates from a 2-degree continuation scan, refreshed at rest.")
    print_table(arm, q, p_star)
    print("  Drag any ONE slider: that joint follows it and the other six move so the fingertips stay on the mug.")
    print("  'Random pose, same tip' jumps to another solution; 'Play / pause wander' drifts through them; the")
    print("  teal beads mark where the flange has been. Only the tip's position is fixed, so the gripper tilts.")
    clicks = {b: 0 for b in ("Random pose, same tip", "Play / pause wander", "Reset (elbow up)", "Show / hide wrist trail",
                             "Show / hide Jacobian columns")}
    playing, c, last, err = False, rng.standard_normal(4), None, 0.0
    columns_shown, settled_at, moved = True, None, False   # `moved` and `settled_at` time the readout refresh
    status, message = "ready", "Move one joint and watch the others compensate."
    commanded_joint, last_action, published_at = None, "Elbow-up start", 0.0

    def pressed(b):
        if meshcat.GetButtonClicks(b) != clicks[b]:
            clicks[b] = meshcat.GetButtonClicks(b)
            return True
        return False

    while meshcat.GetButtonClicks("Stop") == 0:
        sliders = np.radians([meshcat.GetSliderValue(n) for n in SLIDERS])
        j = int(np.argmax(np.abs(sliders - shown)))
        q_before = q
        if abs(sliders[j] - shown[j]) > 1e-9 and not playing:   # the user moved slider j: command it, hold the tip
            q_try, err = command_joint(arm, q, j, sliders[j], p_star)
            commanded_joint, last_action = j + 1, f"Joint {j + 1} command"
            if err > 1e-4:                                   # refuse the step: the slider snaps back, so the failure is seen
                print(f"  joint {j + 1} at {np.degrees(sliders[j]):.1f} deg: the other six cannot hold the tip (residual {err * 1000:.1f} mm); step refused")
                status = "refused"
                message = (f"Joint {j + 1} move refused: the trial left {err * 1000:.1f} mm of position error. "
                           "The arm keeps its previous configuration.")
            else:
                q = q_try
                status, message = "ready", f"Joint {j + 1} moved; the other joints compensated."
        if pressed("Show / hide Jacobian columns"):
            columns_shown = not columns_shown
            meshcat.SetProperty(ARROWS, "visible", columns_shown)
        if pressed("Random pose, same tip"):
            # A single candidate must stay put if refused; do not also wander this tick.
            playing = False
            moved, settled_at = True, time.time()
            q_try, err = arm.null_step(q, p_star, rng.standard_normal(4), 0.6)
            commanded_joint, last_action = None, "Random pose"
            if err <= 1e-8:
                q = q_try
                status, message = "ready", "Another configuration holds the same point."
            else:
                status = "refused"
                message = (f"Random move refused: its correction left {err * 1000:.3f} mm of position error. "
                           "The arm keeps its previous configuration.")
                print(" ", message)
        if pressed("Reset (elbow up)"):
            q, err, playing = Q_UP.copy(), 0.0, False
            commanded_joint, last_action = None, "Reset to elbow-up start"
            status, message = "ready", "Initial joint angles restored; wander is paused."
            moved, settled_at = True, time.time()
        if pressed("Show / hide wrist trail"):
            trail_shown = not trail_shown
            meshcat.SetProperty(TRAIL, "visible", trail_shown)
        if pressed("Play / pause wander"):
            playing = not playing
            commanded_joint, last_action = None, "Wander" if playing else "Paused wander"
            status = "moving" if playing else "ready"
            message = ("Coordinated joint motion keeps the fingertip position fixed." if playing
                       else "Wander paused at this configuration.")
            if not playing:
                moved, settled_at = True, time.time()
        if playing:
            c = c + 0.15 * rng.standard_normal(4)
            c /= np.linalg.norm(c)
            q_new, err = arm.null_step(q, p_star, c, 0.02)
            if err > 1e-8 or np.linalg.norm(q_new - q) < 0.005:      # a joint limit is in the way: pick a new way
                c = rng.standard_normal(4)
                q_new, err = arm.null_step(q, p_star, c, 0.02)
                if err > 1e-8:
                    q_new, err = q, 0.0
            q = q_new
        show(q, err)
        if np.linalg.norm(q - q_before) > 1e-9:          # the arm moved this tick
            moved, settled_at = True, time.time()
        elif moved and not playing and time.time() - settled_at > 0.6:   # it has come to rest: refresh the readouts
            moved = False
            rg = readouts.update(arm, q, p_star)
            print("  at rest; feasible ranges with the tip held [deg]: " +
                  ", ".join(f"j{k + 1} {np.degrees(a):.0f}..{np.degrees(b):.0f}" for k, (a, b) in enumerate(rg)))
        # A failed command's error belongs to its trial, while the displayed arm has stayed put.
        err = float(np.linalg.norm(arm.tip(q) - p_star))
        if panel and time.monotonic() - published_at >= 0.1:
            panel.publish(panel_snapshot(arm, q, p_star, readouts, playing=playing,
                                         status=status, message=message,
                                         commanded_joint=commanded_joint, last_action=last_action))
            published_at = time.monotonic()
        msg = f"q [deg] = {np.degrees(q)}  tip error = {err:.1e} m  gripper tilt = {arm.tilt_deg(q):5.1f} deg"
        if msg != last:
            print(msg); last = msg
        time.sleep(0.02 if playing else 0.05)
    meshcat.DeleteAddedControls()
    if panel:
        panel.close()
        atexit.unregister(panel.close)


# ------------------------------------------------------------------ figure mode
def figure_mode(out):
    """Four solutions for the same tip point, side by side: elbow up, then three from the wander."""
    from PIL import Image
    from pydrake.all import (CameraInfo, ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
                             DiagramBuilder, RenderCameraCore, RgbdSensor, RotationMatrix)
    builder0, plant0, _, panda0 = build()
    arm0 = TipArm(builder0.Build(), plant0, panda0)
    p_star = arm0.tip(Q_UP)
    qs, _ = wander(arm0, Q_UP, p_star, 400)
    picks = [Q_UP, qs[130], qs[260], qs[400]]
    builder, plant, scene_graph, panda = build(renderer=True)
    eye, target = np.array([1.45, -1.15, 0.85]), np.array([0.30, 0.0, 0.40])
    z_cam = target - eye; z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(z_cam, [0, 0, 1.0]); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    X_WCam = RigidTransform(RotationMatrix(np.column_stack([x_cam, y_cam, z_cam])), eye)
    core = RenderCameraCore("vtk", CameraInfo(600, 720, 0.62), ClippingRange(0.05, 10.0), RigidTransform())
    sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WCam, ColorRenderCamera(core, False),
                                          DepthRenderCamera(core, DepthRange(0.05, 10.0))))
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    arm = TipArm(builder.Build(), plant, panda)
    panels = []
    for q in picks:
        arm.set(q)
        img = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(arm.ctx))
        panels.append(Image.fromarray(np.array(img.data)[:, :, :3]))
        print(f"  q [deg] = {np.degrees(q)}  tip error {np.linalg.norm(arm.tip(q) - p_star):.1e} m  tilt {arm.tilt_deg(q):5.1f} deg")
    w, h = panels[0].size
    sheet = Image.new("RGB", (len(panels) * w, h), (255, 255, 255))
    for k, p in enumerate(panels):
        sheet.paste(p, (k * w, 0))
    sheet.save(out)
    print(f"rendered {out}: four solutions, one tip point ^W p = {p_star}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meshcat", action="store_true", help="Run the original Meshcat viewer.")
    parser.add_argument("--app", action="store_true", help="Run Meshcat with a live text/table panel.")
    parser.add_argument("--app-port", type=int, default=0, help="App HTTP port; 0 selects a free port.")
    parser.add_argument("--seed", type=int, help="Optional reproducible random-motion seed.")
    parser.add_argument("--png", metavar="FILE", help="Render the existing offline figure.")
    args = parser.parse_args()
    if args.png:
        figure_mode(args.png)
    else:
        meshcat_mode(app=args.app, app_port=args.app_port, seed=args.seed) if args.meshcat or args.app else print_mode()
