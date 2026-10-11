# The FR3 chain reaches a tool frame, and the mug defines a target

This example uses the manufacturer's **FR3** model. Its current tool pose comes
from seven joint angles. A separate chain combines a supplied mug pose with a
chosen grasp. The initial lesson compares these poses and constructs an approach
pose. It does not solve inverse kinematics or execute a grasp.

The live interaction lets a student change one joint and follow the downstream
frames while the object-based targets stay fixed. The earlier Panda demos remain
separate examples. This folder does not rename or replace their model.

## Run the example

From the repository root, using the existing course environment:

```bash
.venv/bin/python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py
.venv/bin/python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --meshcat
.venv/bin/python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --facts /tmp/fr3-facts.json
.venv/bin/python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --png /tmp/fr3-grasp-chain.png
.venv/bin/python drake/01c-fr3-grasp-chain/prepare_model.py
```

The default command prints every relative factor and cumulative product, checks
them against Drake, and exits. It also builds the angular Jacobian from the seven
world-axis directions, checks it against Drake at the ready and changed poses, and
checks the predicted angular velocity against a finite difference of the flange
rotation. At ready, moving only joints 1 and 2 at 0.1 and 0.2 rad/s gives world
angular velocity (0, 0.2, 0.1) rad/s. The last command only verifies model hashes and
unchanged joint/inertial entries. Runtime uses local model files and needs no
network, ROS, xacro, or model download. Tested on macOS arm64 with Drake 1.46.0.
The offline figure uses Drake's VTK renderer and Matplotlib. A restricted macOS
sandbox may need graphics access for VTK.

The launcher route is `/run/fr3-chain`. Start the launcher from the same checkout
or worktree as these slides:

```bash
.venv/bin/python scripts/demo_server.py
```

If the link says “No demo called fr3-chain”, an older launcher is serving port8765.
Stop that launcher with Ctrl-C in its terminal, then start this checkout's launcher.
An already-running Python process keeps its original demo list until restarted.

For a direct browser test, start the Meshcat command, then run the following with
its printed URL. The test clicks Stop when done.

```bash
node drake/01c-fr3-grasp-chain/check_meshcat.mjs http://localhost:7001
```

## The official model is pinned and stored locally

The source is [Franka Robotics' franka_description release 2.9.0](https://github.com/frankarobotics/franka_description/tree/7aeeddc449edf8d62b594f9e36a81da53e7796f9),
commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9`. The selected robot type is
`fr3`, not `fer`, `fr3v2`, or `fr3v2_1`. The original files are in `model/source/`.
Their complete `LICENSE` and `NOTICE` are retained. The project declares
Apache-2.0; the copied LICENSE also retains its included BSD license text.

`model/fr3-official.urdf` is the expanded manufacturer description. It enables
the Franka hand, disables extra self-collision capsule sublinks, and leaves the
base attachment to this demo. `model/fr3.urdf` retains every original joint,
origin, axis, limit, dynamics, and inertial entry. The print check compares these
entries directly. `model/provenance.json` stores the source pin, expansion
arguments, converter versions, and SHA-256 hashes.

Only geometry presentation and local mesh paths change. The official collision
shapes become the low-detail visuals. Binary STL triangles are converted to OBJ
with their coordinates and normals preserved. Finger collision boxes supply
their visuals. These shapes give a recognizable arm without requiring the large
DAE visual assets. They are explicitly identified as simplified visuals in the
figure. No manufacturer kinematic dimension is substituted.

Drake 1.46 reports that it ignores URDF safety-controller settings, joint
friction, and the finger mimic rule in a continuous plant. The demo does not
advance dynamics. It sets both finger positions to the same open value on every
update. The original fields remain in the model, but this code does not implement
a hardware controller or claim to enforce all manufacturer operating limits.
The seven position and velocity limits are checked against Drake. All joint
limit fields, including effort and position-dependent velocity metadata, are
retained in the pinned source and URDF.

To rebuild the runtime geometry from the expanded official URDF:

```bash
.venv/bin/python drake/01c-fr3-grasp-chain/prepare_model.py --rebuild
```

To also reproduce the xacro expansion without changing the course environment:

```bash
uv pip install --target /tmp/cs549-fr3-tools xacro==2.1.1 pyyaml==6.0.3
PYTHONPATH=/tmp/cs549-fr3-tools .venv/bin/python drake/01c-fr3-grasp-chain/prepare_model.py --expand
```

The converter resolves the one `franka_description` package to the vendored
source. It does not alter macro contents or require a ROS installation.

## The scene and frame contract remain explicit

`data/scene.json` is the source for the robot configuration, supplied mug pose,
chosen grasp, approach offset, dimensions, and assumptions. `facts.json` records
the verified numbers, all ten relative factors, and all cumulative world poses.
Lengths use metres, and angles use radians. Homogeneous matrices act on column
vectors. The base frame is welded to the world with the identity transform.

| Course frame | Model or scene definition |
|---|---|
| W | The fixed world and `fr3_link0` base frame coincide. |
| E | `fr3_link8` is the manufacturer's flange frame. |
| G | `fr3_hand_tcp` is the manufacturer's hand tool-center frame between the fingers. |
| M | The supplied mug frame is at its center. Its positive z axis is upright, and its positive x axis points toward the handle. |
| G* | The chosen desired pose of the same tool frame G is at the mug center, with positive z pointing down. |
| G_app* | The approach target has the grasp target's orientation and lies 0.10 m above it. |

The exact ready configuration is

$$
\mathbf q=(0,-\pi/4,0,-3\pi/4,0,\pi/2,\pi/4).
$$

The fixed manufacturer factors after the seven moving joints are a 0.107 m
translation to E, a rotation of minus 45 degrees from E to the hand frame, and
a 0.1034 m translation from that frame to G. Thus

$$
{}^E X^G=
\begin{bmatrix}R_z(-\pi/4)&(0,0,0.1034)^\top\\0&1\end{bmatrix}.
$$

The mug is 0.08 m tall, with radius 0.035 m. Its center has world coordinates

$$
{}^W\mathbf p^M=(0.50,0.10,0.10)\ \mathrm m.
$$

Its yaw is 30 degrees. Its bottom therefore meets the visual table at world
height 0.06 m. The target chain is

$$
{}^W X^{G^*}={}^W X^M{}^M X^{G^*},\qquad
{}^M X^{G^*}=\begin{bmatrix}R_x(\pi)&0\\0&1\end{bmatrix}.
$$

The approach is a displacement along negative target-local z:

$$
{}^{G^*}X^{G_{\rm app}^*}=
\begin{bmatrix}I&(0,0,-0.10)^\top\\0&1\end{bmatrix},\qquad
{}^W X^{G_{\rm app}^*}={}^W X^{G^*}{}^{G^*}X^{G_{\rm app}^*}.
$$

| Position in W | Value in metres |
|---|---|
| Current flange E | `(0.3068905666, 0, 0.5902820523)` |
| Current tool G | `(0.3068905666, 0, 0.4868820523)` |
| Requested G* | `(0.50, 0.10, 0.10)` |
| Requested G_app* | `(0.50, 0.10, 0.20)` |

## The controls reveal one part of the calculation

Open Controls in Meshcat. Collapsing the standard Scene folder gives the lesson
controls more room. The default is stage 4 at exact ready q. RGB rods indicate
x, y, and z axes. The origin colors distinguish current G, the grasp target, and
the approach target.

| Control | What changes |
|---|---|
| Stage 1 | At full opacity, the visual arm shells are hidden so that the joint-frame origins, their connecting line, and the selected frame are visible. Reduce Link opacity to show translucent shells around those frames. The model and joint angles remain the same. The line connects frame origins; it is not the physical shape of each link. |
| Joint frame k | In stage 1, select the base with 0 or a joint's child frame with 1–7. Its world origin is printed. Frames can share an origin while their axes differ. |
| Stage 2 | The arm shells return, and current E and G show the fixed tool offset. |
| Stage 3 | The selected mug-center frame appears. Use the frame-view buttons to show the supplied mug frame, the chosen grasp frame, or both. The current tool stays where FK places it. |
| Show mug frame M | Show only the supplied mug frame at the mug center. Its positive z-axis points up. The origin marker is teal. |
| Show grasp frame G* | Show only the chosen desired tool frame at the mug center. Its positive z-axis points down. The origin marker is orange. This is the default frame view. |
| Compare M and G* | Show both frames at their true shared origin. M has longer dashed axes; G* has shorter solid axes. Their positive x-axes coincide, and their y- and z-axes point in opposite directions. RGB still identifies x/y/z. Axis lengths distinguish the drawings and are not physical offsets. |
| Stage 4 | The fixed approach target appears above the grasp target. This is a reveal control, not a distance or motion command. |
| Toggle distinct link colors | Assign a different color to each physical link 0–7, or restore the original gray materials. The link-origin markers and selected origin use the same palette. The hand and fingers stay gray; the x/y/z axis rods keep their red/green/blue convention. |
| Link opacity | Set all arm, hand and finger meshes from 0 (invisible) to 1 (fully opaque). A value around 0.35 reveals internal frames. Colors and opacity are independent. Axes, origin markers and targets remain solid. The hand and fingers retain their original colors. |
| Mug opacity | Set the mug body and handle from 0 (invisible) to 1 (fully opaque). A value around 0.25 reveals the selected frame through the surface. Its axes and origin stay solid. This control is independent of link opacity and leaves the supplied mug pose and targets unchanged. |
| Seven dq sliders | Change one angle from its exact ready value. Each slider uses the official FR3 position limits and a 0.001 rad step. The supplied mug and both targets stay fixed. |
| Reset exact ready pose | Restore zero offsets, joint-frame index 7, stage 4, the original fully opaque gray links, the fully opaque mug, and grasp-only frame view. |
| Stop | End the loop and remove the controls. |

The sliders display offsets because rounding the ready angles themselves would
change the numerical FK example. The total configuration is

$$
\mathbf q=\mathbf q_{\rm ready}+\Delta\mathbf q.
$$

Readout rows use Meshcat's button styling but perform no action. The figure and
print output retain more digits than the narrow live panel.

Start in stage 3 with mug opacity around 0.25. Select **Show mug frame M** and identify
its upward z-axis. Predict which axes will change before selecting **Show grasp frame G\***.
The origin and x-axis stay fixed, while y and z reverse. **Compare M and G\*** then
shows that relationship with dashed and solid axes. The frame-view buttons change
only which axes are drawn; they do not move the mug or change the desired pose.
In stage 4, the approach frame is another target above the mug. It is separate from
the two frames that share the mug center.

The color control starts off, and both opacity sliders start at 1. When colors are enabled,
two readout rows show this legend:

| Link ID | Color |
|---|---|
| 0, the fixed base | Gray |
| 1 | Red |
| 2 | Orange |
| 3 | Green |
| 4 | Cyan |
| 5 | Blue |
| 6 | Purple |
| 7 | Pink |

These are link IDs, not zero-based joint IDs. Joint 2 has angle $q_2$ and joins
links 1 and 2. The color choice affects the Meshcat display only. The pinned model,
joint angles, pose calculations, and target frames stay the same.

## Reuse this scene in later lessons

`fr3_scene.py` provides the following contract.

| API | Result |
|---|---|
| `build_scene()` | A Drake diagram, plant, scene graph, contexts, and FR3 model instance. |
| `Scene.set_q(q)` | Update seven named arm joints and the symmetric open fingers. |
| `current_pose(scene, q=None, frame=None)` | Current world pose of G, or the requested manufacturer frame. |
| `parsed_chain(frame=None)` | Manufacturer origins, axes, types, limits, and frame names along the chain. |
| `chain_products(q, frame=None)` | Independent NumPy origin placements, joint rotations, relative products, and world products. |
| `grasp_target()` / `approach_target()` | World target poses built from the scene contract. |
| `verified_facts()` | Reproducible data after model-integrity and Drake-agreement checks. |

Later Jacobian and IK lessons should keep G as the task frame. If a lesson uses E
instead, it must name that task change and account for the tool offset. The next
step can ask for a configuration whose current G equals G*. Constraint and
planning work must add collision geometry for the mug and table, validate
reachability, and check the path. Both objects are currently **visual geometry
only**. Perception can later replace the supplied mug pose without changing the
grasp-relative transform. Contact and closing the fingers need a separate model.

The present target is a chosen pose request. No target feasibility, collision-free
approach, stable physical grasp, calibration accuracy, or hardware pick is
established by the FK check.

## Verification and remaining limits

The independent product agrees with Drake at ten frames for the exact ready
configuration and eight deterministic configurations within the position limits.
The largest absolute matrix-entry difference is $3.33\times10^{-16}$. The model
hashes, complete joint/inertial entries, and Drake position/velocity limits pass.
The live checks cover all four stages, selected origins, a joint perturbation,
unchanged targets, exact reset, WebGL initialization, and browser errors.

Evidence, browser captures, and the verification report are in
`planning/fk-grasp-continuity-2026-10-04/robot-evidence/`. The geometry is nominal
and visually coarse. Stage 1 hides opaque shells or shows translucent shells around internal
frames when link opacity is reduced. Platform coverage is currently macOS arm64; the local model
paths and pure calculations are portable, but the browser and rendering commands
have not been tested on Windows or Linux.
