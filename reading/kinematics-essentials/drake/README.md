# Drake track — cross-platform copy of the course examples (opened 2026-09-04)

**Status: experimental, instructor-only.** RAI (`robotic`) remains the assignment platform
for A1–A2 in Fall 2026; every student ran it on Ubuntu or WSL last year. Drake is being
prepared as a second copy that also runs on macOS, tested on the instructor's Mac during this
semester and integrated later. Nothing here is promised to students. The in-class demos
(tier T2 of `modules/ROADMAP.md`) already run from here.

## Robot models and task frames

The existing Panda demos load Drake's `panda_arm.urdf` or `panda_arm_hand.urdf`.
Their arm joints are `panda_joint1..7`, base is `panda_link0`, and flange is `panda_link8`.
Keep their numerical results, joint limits and singularity claims attached to the Panda
model used by each script. A flange pose and the position between the fingertips are
separate task quantities, even when both are called an end effector.

The new [`01c-fr3-grasp-chain`](01c-fr3-grasp-chain/README.md) example loads the official
**Franka Research 3 (FR3)** model from the pinned `franka_description` release 2.9.0.
Its source, license, runtime assets and hashes are local to that directory. Its own model
supplies its limits. `fr3_link8` is flange E, and `fr3_hand_tcp` is tool-center G. The hand
mount includes a −45° turn about flange z, and the tool center lies 0.1034 m along that z.
The README records model verification separately from any physical-robot calibration.

Do not substitute the old unverified FR3 limit table for values in either loaded model.
The pinned FR3 model and the Panda demos are distinct examples. Similar nominal poses do
not establish that every model property, limit or installed tool is interchangeable.

Panda demos commonly use rounded ready angles `(0, −0.785, 0, −2.356, 0, 1.571, 0.785)` rad.
The FR3 walkthrough uses exact multiples of pi. Its data file is the source of truth.
The Panda all-zero configuration is a kinematic reference outside its joint-4 limit;
print-mode limit checks and the singularity demo identify this explicitly.

## Install (macOS, Linux; Windows via WSL)

The course environment (`SETUP.md`) already has Drake; from the repository root:

```bash
uv venv .venv --python 3.12                                # once
uv pip install --python .venv/bin/python -r requirements/mac-dev.txt
.venv/bin/python drake/smoke_test.py --hold                 # opens Meshcat; Enter quits
```

The first run downloads Drake's model archive (`drake_models`, about 200 MB) once and needs
the network. Meshcat prints a URL such as `http://localhost:7000`; open it in any browser.

## Layout

```
drake/
  README.md                       this file
  smoke_test.py                   loads the Panda, prints FK and the 6x7 Jacobian at two poses, starts Meshcat
  00-contact-changes-the-map/     session 2.1: the mug's 3x7 sensitivity table is zero before a grasp, rank 3 after
  00b-when-the-map-is-wrong/      session 2.1: the stack plans on the estimated mug; a finger lands on the real mug's rim and the arm stops (2026-09-19)
  01-fk-jacobian/                 sessions 2.3-2.4: the two-link arm rebuilt from two revolute joints; the Panda's Jacobian
  01c-fr3-grasp-chain/           session 2.3: pinned official FR3, chain stages, flange/tool center, mug grasp and approach targets
  01b-redundancy-self-motion/     session 2.4: the Panda's elbow swings along the 1-D self-motion curve while the gripper stays on a mug; a second script fixes only the fingertip position (4-D null space) (2026-09-23)
  02-gauss-newton-ik/             session 3.2: the damped Gauss-Newton IK step with Drake's phi and J
  03-ik-as-nlp/                   session 4.2: IK as a nonlinear program on the Panda (SNOPT), active set and multipliers
  04-diff-ik-qp/                  (planned) differential IK as a QP with limits, plus the dragged-target demo (4.3)
  05-trajopt-rrt/                 (planned) kinematic trajectory optimization and RRT (4.2, 4.4)
  06-pointclouds/                 (planned) depth -> point cloud -> world frame; normals (5.2)
  07-grasp-selection/             (planned) antipodal grasp sampling on point clouds (6.1)
  A1/, A2/                        (planned) homework mirrors, after the RAI versions exist
```

## Running the demos (how-to)

Every script has two modes: **print mode** (no arguments; prints the numbers quoted in the
lecture notes and exits) and an **interactive mode** (`--meshcat`, or `--hold` for the smoke
test) that opens the arm in the browser. Run them from the repository root with the course
interpreter; `Ctrl-C` in the terminal always stops a script.

| Demo | Command | What you see | Interactive controls |
|---|---|---|---|
| Smoke test | `.venv/bin/python drake/smoke_test.py --hold` | `drake 1.46.0 …`, FK and Jacobian rank at the zero and ready poses, `SMOKE TEST OK`, a Meshcat URL | open the URL; the arm at the ready pose; **Enter** in the terminal quits |
| 00 · contact changes the map (2.1) | `.venv/bin/python drake/00-contact-changes-the-map/contact_demo.py --meshcat` | the arm **with its gripper** at the ready pose, a table, a mug between the open fingers (hand model since 2026-09-19) | seven joint sliders; **Grasp / release**: the fingers close on the mug and it follows the hand from then on; release opens the fingers and the mug drops onto the table under the point where it was let go (a rule, not physics: the demo has no dynamics); the console prints the rank of the mug's Jacobian after each press; **Stop** ends it. `--png FILE` renders the grasped state offline |
| 00b · when the map is wrong (2.1) | `.venv/bin/python drake/00b-when-the-map-is-wrong/map_error_demo.py --meshcat` | the same scene with Drake's hand model; the true mug in teal and the *believed* mug as an orange ghost 2 cm off; print mode tabulates eight estimation errors under three contact rules (stop at first contact on the approach; grasp with self-centring; close on air) | three sliders Δx, Δy, Δz in cm; **Plan on the estimate, execute on the truth** solves IK on the ghost, animates the descent — stopping where a finger or the palm meets the real mug — and closes the fingers only if the mug is between them; the ghost turns green (grasp) or red; **Back to the ready pose**; **Stop**. `--auto` runs the default once on start, `--png FILE` renders it offline |
| 01 · FK and the Jacobian (2.3–2.4) | `.venv/bin/python drake/01-fk-jacobian/fk_jacobian_demo.py --meshcat` | print mode first checks the two-link arm against the notes, then the Panda at the zero and ready poses, then the ready pose again as the browser's sliders can hold it (`σ_min = 0.2246` against `0.2244` at the exact ready pose), then the near-singular preset (rank 6, `σ_min = 0.000433`), then the seven joint axes at three poses with the number of dimensions their directions span, and ends with a self-test; in the browser the arm carries seven coloured arrows, one along each positive joint axis; positive rotation follows the right-hand rule | seven sliders; the console prints `^W p^E`, `rank J`, `σ_min`, `0.1/σ_min` and the weakest task direction live, with the unit the 0.1 carries — m/s when that direction is nearly all translation, rad/s when it is nearly all rotation; three presets — **Off-plane (axes spread)** (rank 6; the seven axes point seven ways, no two closer than 22.9°), **Near-singular (rank still 6)** (rank 6 but `σ_min ≈ 4·10⁻⁴`, so 231 rad/s for a task request of norm 0.1 in the demo's stated mixed angular/linear coordinates; every entry on the sliders' 0.01 rad grid and inside the loaded Panda model's joint limits), **Zero configuration (singular)** (rank 5, `σ_min = 0`, the axes of joints 1, 3, 5, 7 all vertical, and the console names the lost motion in words: a pure rotation about the world x axis); **Show / hide joint axes** toggles the complete arrows, which are on by default; **Stop** ends it. `--png FILE [--pose ready\|offplane\|zero]` renders the arm and the same positive-axis arrows offline (default `offplane`); add `--wrist` for a close-up |
| 01c · FR3 chain and grasp target (2.3) | `.venv/bin/python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --meshcat` | Print mode verifies the pinned model, each cumulative transform against an independent chain, the flange-to-tool offset, and the supplied grasp/approach targets. See the [example README](01c-fr3-grasp-chain/README.md) for controls and exact expected numbers. | Follow construction stages at fixed joints, move a joint to update the current tool pose, then reset. The desired mug-relative poses stay fixed. FK and target construction only; no IK or contact result is claimed. |
| 01b · redundancy, self-motion (2.4) | `.venv/bin/python drake/01b-redundancy-self-motion/self_motion_panda.py --meshcat` | (Meshcat: one slider `s` for the self-motion, three preset buttons, a play/pause sweep, and since 2026-10-07 seven text readouts `q1 = … deg` to `q7 = … deg` at the bottom of the panel that follow the joint angles, refreshed a few times a second while the arm moves) the arm **with its gripper** poised over a mug on a table, flange at `(0.556, 0, 0.245)` m pointing down; print mode traces the curve of every configuration with that flange pose (377 samples, joint 2's limit at both ends), prints the three preset configurations, the elbow's travel (`y` ±0.28 m, `z` 0.285–0.611 m) and the largest flange-pose error along the curve (`6e-16`), then self-tests | **one slider**, the self-motion parameter `s` in rad of arc length (`s = 0` elbow up, ±3.76 the two ends); the seven joints move together and the gripper stays; **Elbow up**, **Elbow down (right)**, **Elbow down (left)** jump to the slide's three configurations; **Sweep** plays the curve end to end; **Show / hide elbow path** toggles the gold beads along the elbow's path; **Stop**. `--png FILE` renders the three configurations side by side offline |
| 01b · tip position only (2.4) | `.venv/bin/python drake/01b-redundancy-self-motion/tip_only_panda.py --meshcat` | (Meshcat: seven joint sliders in degrees since 2026-10-07, the other joints hold the tip) the same scene, but only the point between the fingertips (Franka's tool centre point, 0.1034 m along the flange z axis) is held, at `(0.556, 0, 0.142)` m on the mug's top; print mode prints rank 3 and a 4-D null space of the 3 × 7 tip Jacobian, joint 1 commanded to 20/40/60° with the other six holding the tip (tilt 6.8/14.9/27.1°), and a 400-step null-space wander (tip error `1e-10`, tilt up to 70°, flange travel 7–9 cm), then self-tests | **seven sliders**: drag one and the other six move so the tip stays (their sliders follow); **Random pose, same tip**; **Play / pause wander** (a smooth random walk through the solutions, pausing in place on the next press); **Reset (elbow up)**; **Show / hide wrist trail** (teal beads where the flange has been); **Show / hide Jacobian columns** (the seven columns of the 3 × 7 tip Jacobian as arrows at the tip, coloured by joint as the FK demo's axis rods: joint 7 has none, joint 5 a stub); seven **readouts** at the bottom of the panel (buttons that do nothing, relabelled whenever the arm comes to rest): per joint, tip metres per radian, radians the other six must move per radian of it, and the range it can be commanded to from here with the tip held (joint 2: 15..89°, joint 7: full range, at the start); **Stop**. Print mode also prints the per-joint table: column norm, the other six joints' motion per rad, feasible range. `--png FILE` renders four solutions side by side |
| 02 · Gauss-Newton IK (3.2) | `.venv/bin/python drake/02-gauss-newton-ik/gn_ik_panda.py --meshcat` | runs 1 and 2 of the print mode animated, one iterate per second, the target as a gold sphere | none while running; **Enter** in the terminal quits at the end |
| 03 · IK as a nonlinear program (4.2) | `.venv/bin/python drake/03-ik-as-nlp/ik_nlp_panda.py --meshcat` | print mode: three runs (free target; target behind the robot with joint 2 at its limit and the price check; obstacle sphere with the wrist point exactly on its surface), the active set and the multipliers of each | three target sliders (x, y, z; the x slider's range 0.1 to 0.9 m excludes run 2's target); the sliders start at run 3's target with the sphere drawn; every move is solved from the last pose and from q_ready and the lower-cost solution is shown; the console prints the cost and the active constraints; a failed solve keeps the arm in place, turns the target grey and says that SNOPT found no feasible point (a solver failure, not a proof); **Stop** button |
| 04 · KOMO on the two-link arm, cross-checked with SNOPT (4.2) | `cd lectures/04-D1-constraints-and-komo/code && ../../../.venv/bin/python komo_two_link.py --drake` | print mode: the k = 2 trajectory program of Deck 05 slide 26 solved by the course's augmented Lagrangian from two guesses; `--drake` hands the same program to `MathematicalProgram` + SNOPT, prints its cost, duals and a KKT check, and restarts our loop at SNOPT's point (Deck 05 slide 31; about ten seconds) | none: print only (no Meshcat) |

Pitfalls seen so far: on the first run of any demo the model download takes a minute; if a
Meshcat page stays empty, reload it after the script prints the URL; two demos at once open
two Meshcat servers on different ports, which is fine. In class, start the demo before the
lecture and keep the browser tab open.

## Fingertip-only live panel test

The existing fingertip-only demo also accepts `--app`. It embeds Meshcat in a browser
page with a live table for joint angles, position error, tilt, Jacobian column gains,
compensation and approximate scanned command ranges. The panel can be docked, floated
as an overlay, or hidden. See the [app notes](01b-redundancy-self-motion/panel/notes.md)
for the starting state, units and teaching sequence.

```bash
.venv/bin/python drake/01b-redundancy-self-motion/tip_only_panda.py --app --app-port 8766
```

Open `http://127.0.0.1:8766/`. A newly started lecture launcher also offers
`/run/tip-only-app`, now used by Deck 03 slide 43. The original viewer is still available
through `/run/tip-only` or `--meshcat`. The app needs its local Python process; it is not an offline HTML simulation.
Run `node scripts/check_tip_only_app.mjs` for the focused automated browser checks.

## Smoke test result on the instructor's Mac (2026-09-06, Panda model)

```
drake 1.46.0 | python 3.12.14 | platform darwin
joints: 7

home  q = 0
  ^W p^E [m]   : [ 0.088 -0.     0.926]
  J shape/rank : (6, 7) 5
  sigma_min(J) : 0.0

ready q = (0, -0.785, 0, -2.356, 0, 1.571, 0.785)
  ^W p^E [m]   : [ 0.307  -0.      0.5903]
  J shape/rank : (6, 7) 6
  sigma_min(J) : 0.22437

Meshcat URL: http://localhost:7001
SMOKE TEST OK
```

Facts the lectures use from these runs (all reproduced by the print modes):

- **Zero configuration** (2.4): rank 5 of 6. Three roll axes (joints 1, 3, 5) coincide and joint 7's is parallel to them 8.8 cm away (through the flange, so it adds no translation),
  while the pitch axes (joints 2, 4, 6) all point along `±y`: every one of the seven axis directions
  therefore lies in the `y`–`z` plane, spanning two dimensions instead of three, which is why
  no joint combination produces `ω_x`; the 8.8 cm flange offset keeps all three translations
  available (`v_y` comes from those same joints), which is why this singularity is in rotation
  only and why position-only inverse kinematics does not get stuck there.
- **Redundancy** (2.4): with the flange pose fixed, the Panda's 6 × 7 Jacobian keeps rank 6 along the
  whole self-motion curve (`σ_min` between 0.150 and 0.219), so the null space is exactly one line; following
  it from the elbow-up solution `q = (0, 17.2°, 0, −126.1°, 0, 143.2°, 45.0°)` reaches joint 2's +101° limit
  after 3.76 rad on either side, at `q = (∓72.8°, 100.6°, ±97.1°, −119.5°, ∓95.8°, 78.7°, ·)`, with the elbow
  32.5 cm lower and 28 cm to the side, and the flange never moves by more than `6e-16` m.
  With only the fingertip **position** fixed (`m = 3`) the null space is 4-dimensional: joint 1 can be
  commanded to 60° and the other six still hold the tip (the gripper tilts 27°); a 400-step random walk in
  the null space tilts it up to 70° and moves the flange 7–9 cm while the tip stays to `1e-10` m.
  Why joint 7 is free and joint 2 nearly stuck there: joint 7's tip-Jacobian column is exactly zero (its axis
  passes through the tool centre point), joint 2's is 0.59 m/rad and undoing it costs joints 4 and 6 about
  2.5 and 4.7 rad per rad, so joint 2 can move only from 15° to 89° with the tip held against joint 7's ±165°. The lower end is
  a loss of rank among the six compensating joints (their nearest limit is 40° away); the upper end is joint 4's
  −4° limit.
- **Contact** (2.1): with the mug welded to the world, its 3 × 7 Jacobian is zero; welded to
  the gripper it has rank 3; 0.2 rad on joint 1 moves gripper and mug by `(−0.006, 0.061, 0)` m,
  and the first-order prediction `(0, 0.061, 0)` is 6 mm off.
- **Gauss-Newton IK** (3.2): `JᵀJ` is 7 × 7 of rank 3, so on a redundant arm the damping `D` (`ρI` here; `notation.md` §2.1) is
  what makes the step exist at all; from the ready pose a 10 cm target is reached in 4 steps
  (residuals `1.1e-1, 1.7e-2, 3.7e-4, 3.3e-6, 5.0e-8`); from a stretched pose a target near the
  workspace boundary converges in 18 steps at ρ = 10⁻³, the damping's own linear rate (5 steps with ρ = 10⁻⁸ or J⁺); an unreachable target wanders with steps of about 1 to 4 rad
  steps under full Gauss-Newton and settles at `0.246 m` from the target, the distance to the
  reachable set, once Armijo backtracking is applied to the same direction.

The Windows/WSL setup has not been tested here. The newer hand-equipped Panda demos and
the pinned FR3 example have their own verification records above; the older smoke-test
entry is historical. Physical calibration and hardware execution remain outside these checks.

## Deck 03 optional diagnostics

These details were moved out of the student explanation during the 2026-10-06 editorial
revision. They describe the existing print checks; this relocation is not a new live rehearsal.
The core examples remain in the [lecture notes](../lectures/02-D2-transforms-fk-jacobian/notes.md).

### Near-singular Panda preset

The FK demo's preset uses

$$\mathbf q=(0,-0.16,0,-0.47,0,0.44,0.16)^\top\ \mathrm{rad}.$$

All entries lie on the sliders' 0.01 rad grid and inside the loaded model's joint limits.
At rank tolerance $10^{-6}$ the spatial Jacobian has rank 6; at $10^{-3}$ its reported
numerical rank is 5. Its smallest singular value is about $4.3\times10^{-4}$.
A task-velocity vector of norm 0.1 in the weakest direction therefore requires a joint-speed
norm of about 230 rad/s. This uses the demo's unscaled angular/linear stack, with angular
components in rad/s and linear components in m/s. The request is mostly translation,
but is not a pure lift; changing task scaling changes the singular values.

Print-mode Part 3b sweeps joint 4 and locates a singularity using a $6\times6$ minor.
For the stated preset's other joints, the root is approximately $q_4=-0.4670024$ rad,
where the smallest singular value is about $3\times10^{-13}$. The slider value of
$-0.47$ rad is about 0.003 rad away. The ready-pose sweep finds the same root, but it
is not a configuration-independent singular joint angle.

At the ready pose, joints 1 and 7 are parallel, as are joints 2, 4 and 6, while the full
Jacobian has rank 6. Parallel axis directions alone do not establish a singularity.
At the zero configuration, axes 1, 3 and 5 coincide, and axis 7 is parallel but 8.8 cm away.
All angular columns then lie in the world y–z plane. The three position rows still have
rank 3 when orientation is left free. The joint-axis arrows show positive rotation axes;
their cones give the direction used by the right-hand rule.

### Self-motion trace and correction

The full-pose demo traces 377 samples with the spatial Jacobian's smallest singular value
between 0.150 and 0.219. Each predictor step has length 0.02 rad along the unit null direction.
A differential-IK correction then removes the small pose drift. The correction uses a local
orientation error consistent with the geometric angular block near zero error; it does not
treat that block as an exact derivative of an arbitrary finite orientation residual.
The recorded pose-error norm stays below $10^{-15}$ along this computed trace.
The endpoint configurations, elbow travel and joint-2 limits are recorded above.

The position-only demo checks how the other six joints compensate for a commanded joint.
Their first-order compensation per unit motion of joint $j$ is

$$-\mathbf J_{\mathrm{others}}^+\mathbf J_j.$$

Near the lower end of joint 2's reported 15°–89° interval, the compensating Jacobian's
smallest singular value falls to about 0.01 while its nearest joint limit remains about
40° away. The upper end is joint 4's upper limit. These diagnostic limits concern the
chosen compensating motion and current task, not a global certificate of all feasible postures.
