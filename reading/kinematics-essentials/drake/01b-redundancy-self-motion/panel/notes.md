# Hold the fingertip position with live readouts

This test app shows several joint configurations that hold the same point between the
Panda's fingertips. Moving one joint makes the other six compensate. The panel shows
their joint angles and the remaining position error while the robot moves.

The held point lies 0.1034 m along the flange's local z axis. Its target comes from forward
kinematics at the elbow-up start. Its world position is approximately

$$
\mathbf p_*=(0.5561,\ 0,\ 0.1418)^\top\ \mathrm m.
$$

The target is above the mug. The demo imposes a point-position condition. It does not
simulate contact, grasp forces, dynamics, or collision avoidance.

## Start the test app

Run this command from the repository root:

```bash
.venv/bin/python drake/01b-redundancy-self-motion/tip_only_panda.py --app --app-port 8766
```

Open `http://127.0.0.1:8766/`. The robot starts paused at the elbow-up configuration.
The readout panel starts docked on the right. **Compact overlay** floats it over the viewer.
**Hide panel** gives the robot the full viewer width. **Joint gains and scan ranges**
and **Position Jacobian matrix** expand the detailed tables.

A newly started `scripts/demo_server.py` also lists the test at
`http://127.0.0.1:8765/run/tip-only-app`. The original `/run/tip-only` link and
`--meshcat` command still open the original viewer. The app requires the local Python
process to stay running. Press **Stop** in Meshcat or use Ctrl-C in the terminal to end it.

## Use the existing Meshcat controls

Choose **Open Controls** inside the robot view. Its seven `panda_joint` sliders use degrees in 0.5-degree steps.
The arm computations and Jacobian gains still use radians. The panel measures each
angle change from the elbow-up starting angles.

| Control | Default or units | What it shows |
|---|---|---|
| `panda_joint1` through `panda_joint7` | Degrees; the loaded Panda model supplies the limits | One commanded joint moves and the other six compensate. |
| **Random pose, same tip** | Paused; a candidate local move has length 0.6 rad | Another posture can keep the same fingertip position. A failed correction is refused. |
| **Play / pause wander** | Paused | Repeated coordinated moves change posture and gripper orientation. |
| **Reset (elbow up)** | Restores the starting configuration and pauses wander | The target remains the same. |
| **Show / hide wrist trail** | Shown | The flange moves while the held point stays nearly fixed. |
| **Show / hide Jacobian columns** | Shown | Colored arrows show the effects of individual positive joint motions. |
| **Stop** | The demo is running | The Python process and its app server end. |

## Read the values

The position Jacobian maps seven joint angular velocities into three world-coordinate
velocity components at the held point:

$$
\dot{\mathbf p}=\mathbf J_{\mathrm{pos}}(\mathbf q)\dot{\mathbf q},
\qquad \mathbf J_{\mathrm{pos}}\in\mathbb R^{3\times7}.
$$

The initial rank is three, leaving four independent joint-motion directions with zero
instantaneous position effect. The panel computes rank and nullity at the displayed pose.
It does not assume they stay constant at every possible configuration.

The **tip gain** is the length of one Jacobian column, in meters per radian. At the start,
joint 1's gain is about 0.556 m/rad. If joint 1 moved alone by 0.01 rad, the local model
would predict about 5.56 mm of fingertip displacement. The demo moves other joints to
cancel that displacement. Joint 7's gain is numerically zero because its axis passes
through the held point.

The **compensation** value is the Euclidean norm of the other six joints' local corrective
motions per radian of the selected joint. It uses rad/rad. It is neither the sum of six
angles nor the number of joints that move.

The **scan range** is an estimate found by continuing in 2-degree steps from a resting
configuration. The scan uses a 1-micrometer position-error threshold. It is not a certified
global interval, and a different path or correction method may reach other configurations.
During motion, the panel labels ranges from an earlier resting pose. It updates them
after the arm settles.

**Fingertip error** is the distance between the displayed point and its fixed target.
**Gripper tilt** is the angle between the flange's positive z axis and world down.
A refused command keeps the previous robot configuration. The status message reports
the failed trial's error separately from the actual fingertip error shown above it.

## Try the teaching sequence

1. Reset. Predict whether joint 7 can turn without moving the held point. Move its slider
   to 90 degrees. Its angle changes to about 90 degrees while the other six remain still.
2. Reset. Move joint 1 through 20, 40, and 60 degrees. Several joint angles change
   together. The gripper tilts by about 27 degrees while the point remains at its target.
3. Reset. Move joint 2 down to 10 degrees. The local correction fails to hold the
   point. The command is refused, the slider returns, and the panel keeps the actual error
   distinct from the rejected trial's error.
4. Reset. Play wander, then pause. Watch the wrist trail and tilt change. Compare the
   current and target positions. The fallback is the supplied `drake-tip-only.png` figure
   on Deck 03's fingertip-only slide.

The position correction uses a local Jacobian model and respects joint bounds through
clipping. It supplies a teaching example of redundancy. It supplies no global feasibility
or motion-safety guarantee.

## Verification limits

The implementation record is under `planning/tip-only-live-panel-2026-10-07/`.
Numerical execution and automated browser interaction are recorded separately from human
rehearsal. The panel clears readouts if the live state stops updating or the demo ends.
Deck 03 slide 43 links directly to the app through `/run/tip-only-app`.
