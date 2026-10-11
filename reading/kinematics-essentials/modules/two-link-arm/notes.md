# two-link-arm — companion notes (sessions 2.3–2.4; tier T1, single file, no install)

Open `index.html` in any browser. Nothing is loaded from the network. The page has three
parts stacked down the page: the original arm-and-ellipse view, a commanded tip velocity you
can point, and the same four transforms multiplied in the wrong order. The first part is
unchanged by the other two, down to the text of its readouts, because the lecture screenshot
on Deck 03’s “Live demo: the two-link module” and the worked Jacobian numbers were taken from it. The single exception
is the line the first part prints when `det J = 0`, which now ends by pointing at the section
below; that line does not appear in the default view, so the screenshot and every number on
those worked examples are unchanged.

## What the interaction teaches that static text cannot

That the manipulability ellipse collapses **continuously** as the elbow straightens: sliding
`q_2` from 45° toward 0° you watch the small semi-axis shrink to nothing, the two column
arrows of `J` turn until they are parallel, and the joint rates needed for a fixed tip speed with a component along the arm
grow without bound. Three static snapshots (the lecture figure) show the endpoints; the module
shows the approach, and that crossing `q_2 = 0` flips the sign of `det J` (elbow-up becomes
elbow-down) while the ellipse swings through the line.

## Default state

The worked example of the 2.3–2.4 notes §5: `l_1 = 0.4 m`, `l_2 = 0.3 m`, `q = (30°, 45°)`.
Before anyone touches a control the readouts show `^W p^E = (0.4241, 0.4898)` m, heading
75°, the Jacobian `[[−0.4898, −0.2898], [0.4241, 0.0776]]`, `det J = 0.0849`, singular values
`0.704` and `0.121`, and the joint rates `q̇ = (0.342, −0.577)` rad/s that move the tip
straight up at 0.1 m/s. The ellipse is visibly elongated: the arm moves its tip easily
across the line to the base and poorly along it.

Part 2 opens on the tangential direction, so a number is on screen before anything is
touched: `|q̇| = 0.154 rad/s` for the commanded `|ẋ| = 0.1 m/s`, together with the two
extremes over all directions, `0.142 rad/s` along the long semi-axis of the ellipse and
`0.829 rad/s` along the short one. Part 3 opens with both ghost arms drawn, so the
phenomenon — two wrong orderings of the same four factors, two wrong arms, all three with
the same heading — is visible before any click.

## Controls

| Control | Range and units | What changing it shows |
|---|---|---|
| Slider `q_1` | −180° to 180°, 1° steps; default 30° | Rotates the whole arm about the base; the tip position changes, the ellipse turns with the arm while its shape, area and `det J` do not change (they depend on `q_2` only), so `q_1` is "where", `q_2` is "how well". |
| Slider `q_2` | −180° to 180°, 1° steps; default 45° | The lesson: the small semi-axis is `l_1 l_2 |sin q_2|` divided by the large one, roughly; at `q_2 → 0` and `q_2 → ±180°` the ellipse becomes a line and the differential-IK readout warns that `J` has no inverse and sends the reader to the commanded-velocity section below, which does compute something there. The two readouts are deliberately not contradicting each other: part 1 solves the square system for one fixed command and has no solution, part 2 gives the least-norm rates for the part of the command that survives. |
| Buttons | four presets | Worked example; stretched (`q_2 = 0`); folded (`q_2 = 180°`); almost stretched (`q_2 = 10°`, where the tip speed of 0.1 m/s already costs 3–4 times the joint rates of the default). |
| Checkbox "reachable workspace" | on/off; default on | Draws the annulus between radii `|l_1 − l_2| = 0.1 m` and `l_1 + l_2 = 0.7 m`; the singular configurations are exactly the tip on its two boundary circles. |
| Checkbox "columns of J" | on/off; default on | Hides the two arrows to look at the ellipse alone. |
| Slider "direction of ẋ" (part 2) | −180° to 180°, 1° steps; default 139.1°, the tangential direction at `q = (30°, 45°)` | Turns the commanded tip velocity, whose magnitude stays 0.1 m/s. The joint rates change by a factor of `σ_max/σ_min` = 5.8 at the default configuration and without bound as `q_2 → 0`; the drawing shows the arrow, the readout the rates. Dragging the slider fixes the direction in the world frame; the thumb sits at the nearest whole degree while the readout prints the exact angle. |
| Button "radial" (part 2) | sets the direction to the base-to-tip line | The expensive direction near a singularity, and the direction that is lost at one: at `q_2 = 0` none of the 0.1 m/s can be produced. The button keeps following the arm when `q_1` or `q_2` moves. |
| Button "tangential" (part 2) | sets the direction perpendicular to the base-to-tip line; the default | The cheap direction: it is the direction of the first column of `J`, so while `J` is invertible joint 1 alone produces it and `|q̇| = 0.1/|p|` — 0.154 rad/s at the default, 0.143 rad/s at `q_2 = 10°` and again 0.143 rad/s at `q_2 = 1°`, where `|p|` has already reached 0.7000 m. Exactly at the stretched singular pose that formula stops applying: the two columns of `J` are parallel, the square system has many solutions, and the module prints the least-norm one instead, `q̇ = (0.121, 0.052)` rad/s with `|q̇| = 0.1/σ_max = 0.131` rad/s, which reaches the same tip velocity with **both** joints turning. The step from 0.143 to 0.131 as `q_2` crosses 0 is the discontinuity of `J⁺` at a rank drop, not rounding and not a bug. In every one of these poses the radial direction costs everything. |
| Checkbox "multiply on the wrong side" (part 3) | on/off; **default on** | Draws the two mis-ordered ghost arms beside the correct one. Clearing it leaves the correct arm alone, which is what the reader should *not* see first. |

## Axes

One plot: `x [m]` along the bottom, `y [m]` on the left, grid every 0.2 m, base at the
origin. Arrows and ellipse are velocities in m/s drawn at a scale of 0.5 s, so an arrow of
length 0.25 m on the drawing means 0.5 m/s.

Parts 2 and 3 have their own smaller plots, with the same labelling: `x [m]` along the bottom,
`y [m]` on the left, grid every 0.2 m from −0.8 m to 0.8 m. They are drawn at a coarser scale
than the first plot and their window is centred on the base, so that it covers a radius of
0.91 m: the whole reachable annulus (`l_1 + l_2 = 0.7 m`) plus the longest velocity arrow part 2
can draw (0.2 m). That matters because `q_1` runs over the full turn, and an earlier version
used a window covering only the upper-right corner, in which the arm left its own picture for
about two thirds of the `q_1` slider while the readouts kept printing correct numbers. Velocity
arrows in part 2 are drawn at **2 s**, not 0.5 s, so that the fixed 0.1 m/s command is 0.2 m long
on the drawing and can be seen; the caption says so, and the first plot's 0.5 s scale is
untouched.

## Companion notes (stand-alone prose)

The tip of a two-link planar arm sits at `φ(q) = (l_1 cos q_1 + l_2 cos(q_1 + q_2),
l_1 sin q_1 + l_2 sin(q_1 + q_2))`, the product of four planar transforms `R(q_1) T(l_1, 0)
R(q_2) T(l_2, 0)` read off in coordinates. Its Jacobian `J = ∂φ/∂q` is the 2 × 2 matrix whose
columns say how the tip moves per unit rate of each joint: the first column is `(−y, x)`,
perpendicular to the line from base to tip, the second is `l_2 (−sin(q_1+q_2), cos(q_1+q_2))`,
perpendicular to link 2. The tip velocity is `ṗ = J q̇`. Feeding every unit joint-velocity
vector through `J` gives an ellipse whose semi-axes are the singular values of `J`; its area is
proportional to `|det J| = l_1 l_2 |sin q_2|`. When `sin q_2 = 0` the two columns are parallel,
the ellipse is a line, and no joint motion moves the tip along the arm at first order: a
singular configuration. Away from it, the square system `J q̇ = ẋ` has the unique solution
`q̇ = J⁻¹ ẋ`, and the module shows how that solution blows up, for a request with a component
along the arm, as the singularity is approached. The derivations and the numbers are in
`lectures/02-D2-transforms-fk-jacobian/notes.md` §4–§5.

## Part 2: a commanded tip velocity you can point

**What the interaction teaches that static text cannot.** The lecture states that a
singularity is the loss of one *direction*, and the slides end the differential-inverse-kinematics
page with "singular does not mean stuck; it means violent". A table of `|q̇|` for one fixed
command cannot show either claim, because one command is one direction. Here the direction is the
control: at `q = (30°, 45°)` the same 0.1 m/s costs `0.154 rad/s` tangentially and `0.827 rad/s`
radially, and as `q_2` closes on 0 the tangential cost barely moves (0.143 at 10°, 0.143 at 1°,
0.131 at 0°) while the radial cost runs away (3.640, 36.363, and then off the scale). The last
step of that tangential row, 0.143 down to 0.131, is not rounding: away from the singularity the
module solves `J q̇ = ẋ` and joint 1 alone turns, at `|q̇| = 0.1/|p| = 0.143` rad/s, while at
`det J = 0` it switches to the least-norm solution `q̇ = J⁺ ẋ`, which spreads the same tip
velocity over both joints at `0.1/σ_max = 0.131` rad/s. `J⁺` is discontinuous where the rank
drops, and that step is what the discontinuity looks like on screen. Sliding
`q_2` with "radial" selected is the blow-up; pressing "tangential" immediately afterwards, at the
same configuration, is the other half of the sentence. The number that lands is the one at
`q_2 = 0`: the cheapest direction there costs `0.131 rad/s`, slightly *less* than the `0.142 rad/s`
it cost at the comfortable default. One direction is gone; every other direction is marginally
better than it was. The joint rates never misbehave at the singularity itself — what fails is the
task, not the actuator.

**Default state.** Tangential, `|ẋ| = 0.1 m/s`, at the default configuration, which already prints
a rate and the two extremes.

**What the readout says at `det J = 0`.** It does not refuse. Writing the singular value
decomposition of the Jacobian as `J = σ_1 u_1 v_1ᵀ + σ_2 u_2 v_2ᵀ`, a singular configuration is
`σ_2 = 0`, so the tip velocities the arm can produce are exactly the multiples of `u_1`: one line
through the tip, drawn dashed in the picture. The module reports the **least-norm solution**

    q̇ = J⁺ ẋ,   J⁺ = (1/σ_1) v_1 u_1ᵀ,

the smallest joint-rate vector among those that come closest to the command, and splits the
command into the part the arm produces,

    J J⁺ ẋ = (u_1ᵀ ẋ) u_1     (the reachable part, the projection of ẋ onto that line),

and the remainder `ẋ − J J⁺ ẋ`, which is printed as the unreachable part together with the
direction it points in. At the stretched pose `q = (30°, 0)` a radial command of 0.1 m/s gives
`q̇ = 0`, a reachable part of 0, and 0.100 m/s lost along `(0.866, 0.500)` — the direction of the
arm itself.

Two honest caveats. First, `J⁺` is previewed in session 2.4 (Deck 03’s “Any J, any arm: the pseudo-inverse (a preview)”: the closest
achievable velocity first, then the least joint speed) and derived in week 3, where the damped,
weighted `J♯` of Spring 2026 deck 03 p. 56 has it as the special case `K → ∞`, `D = I`; nothing in
sessions 2.3–2.4 derives it, and the module deliberately presents the result geometrically, as
the part of `ẋ` that lies on the reachable line and the part that does not, rather than as a
formula the reader is expected to have seen. Second, the two
buttons are *arm*-relative and the two extremes in the second readout are *ellipse*-relative, and
they are not the same directions: at `q = (30°, 45°)` the cheapest direction is
`±(−0.802, 0.597)` at 0.142 rad/s while the tangential direction is `±(−0.756, 0.655)` at
0.154 rad/s. Both are printed, each with its own label, precisely so the reader does not conflate
them. They coincide exactly at a singularity, where the radial direction *is* the lost one.

## Part 3: multiply on the wrong side

**What the interaction teaches that static text cannot.** The deck's first failure mode of the
day is writing the four factors of `^W X^E = R(q_1) T(l_1,0) R(q_2) T(l_2,0)` in the wrong order,
and the reason it is a *silent* failure is geometric: the arm is somewhere else entirely and its
end-effector frame points exactly where it should. The module draws the correct arm and two
mis-orderings of the very same four factors:

| | product | elbow | tip | heading |
|---|---|---|---|---|
| correct | `R(q_1) T(l_1,0) R(q_2) T(l_2,0)` | `(0.3464, 0.2000)` | `(0.4241, 0.4898)` | 75° |
| ghost A | `T(l_1,0) R(q_1) T(l_2,0) R(q_2)` | `(0.4000, 0.0000)`, 0.2071 m off | `(0.6598, 0.1500)`, 0.4136 m off | 75° |
| ghost B | `R(q_1) T(l_1,0) T(l_2,0) R(q_2)` | `(0.3464, 0.2000)`, **right** | `(0.6062, 0.3500)`, 0.2296 m off | 75° |

The table is the worked example; the two ghosts behave the same way at every `q_1`, but neither
of them depends on `q_2` at all. Both mis-ordered products end in `R(q_2)`, and a rotation
applied last turns the end-effector frame without moving any point, so ghost A's tip is
`(l_1 + l_2 cos q_1, l_2 sin q_1)` and ghost B's is `R(q_1)(l_1 + l_2, 0)` whatever `q_2` is.
Sliding `q_2` therefore leaves both ghost arms frozen and turns only their heading arrows, which
is why each ghost is drawn with an end-effector frame that does not line up with its own forearm
— and the module says so on screen. It has one consequence worth pointing out in class before a
student finds it alone: at the **Stretched** preset, `q_2 = 0`, ghost B's tip is the correct tip
exactly, and the readout prints `0.0000 m from the correct tip`. The wrong order happens to be
right there, because `R(0)` is the identity and the two translations are then parallel. The
module prints that explanation whenever a ghost tip coincides, rather than letting a chain
labelled "wrong order" print a zero with no comment. One degree away, at `q_2 = 1°`, ghost B is
0.0052 m off, which is too small to see but no longer zero.

Ghost A moves the first link: translating along the world `x` axis before rotating puts the elbow
at `(0.4, 0)` instead of `(0.3464, 0.2)`, 20.7 cm away, with the *same* elbow heading of 30°.
Ghost B leaves the elbow exactly right — its first link is drawn underneath the correct one — and
only the tip is wrong, which is the case a reader who checks the elbow will still miss. Ghost B's
tip is 0.7000 m from the base, that is `l_1 + l_2` exactly: it lands on the outer circle of the
reachable workspace drawn in the first part, which is the stretched singular pose the rest of the
module is about.

**The inverse pose, printed beside the correct chain.** The same readout gives `^E p^W`, the
position of the world origin as the end-effector frame sees it. At the default configuration it
is `(−0.5828, 0.2828)` m, and the readout prints the wrong guess next to it: `−(^W p^E) =
(−0.4241, −0.4898)` m. The two are different because inverting `^W X^E = (R, p)` gives
`^E X^W = (Rᵀ, −Rᵀ p)`, so the inverse rotates as well as negates, and the guess `−p` skips the
`Rᵀ`. This is the line deck 03 page 6 needs, and it is here rather than in a module of its own
because the page already has the correct pose on screen; a reader can check the arithmetic by
turning `(0.4241, 0.4898)` through −75° and changing the sign.

**Default state.** Both ghosts drawn. Clearing the checkbox is how the reader gets rid of them,
not how the reader summons them.

**Where the lesson stops.** All three chains have heading `q_1 + q_2 = 75°`, and so do all 24
orderings of the four factors: in the plane every rotation block is a rotation about the same
axis, and rotations about one axis commute. That is a **planar accident**, not a general fact,
and the module says so on screen. In three dimensions the order changes the orientation as well,
which is exactly what tab 2 of `modules/rotations-3d` (session 2.2, two days earlier) shows with
two books. Read the two modules together: `rotations-3d` is order in `SO(3)`, this part is order
in `SE(3)`, where one of the factors is a translation and the position is what moves.

## Open questions and weak points

- The module is planar and square (two joints, two task coordinates); a real arm has a 6 × 7
  Jacobian, and its singularities are not visible as a collapsed ellipse in two dimensions.
  The Drake demo `drake/01-fk-jacobian/` covers that with the Panda's rank readout.
- Tip dragging (inverse kinematics) is deliberately absent: it is week 3's topic. Part 2
  commands a tip *velocity*, which is the differential problem of session 2.4, not a tip target.
- Part 2's readout at `det J = 0` uses the pseudo-inverse, which sessions 2.3–2.4 do not derive.
  It is presented geometrically (the projection onto the one reachable line) and the formula is
  named but not proved; the alternative was a readout that refuses to compute, which argues
  against the slide it supports. Flagged for the instructor.
- Part 3 shows two mis-orderings out of the 23 wrong ones. They were chosen because between them
  they produce the two numbers the deck's failure-mode slide needs — an elbow that moves and an
  elbow that does not — and not because they are the most likely typing mistakes.
- The direction slider steps by 1° while the radial and tangential buttons set the exact angle,
  so after pressing a button the slider thumb and the printed angle can differ by up to half a
  degree. The printed angle is the one the numbers are computed from.
- Every number in all three parts is reproduced by `check_numbers.py` in this folder, which
  imports `fk` and `jacobian` from `lectures/02-D2-transforms-fk-jacobian/code/two_link.py`.
  Run it with `.venv/bin/python modules/two-link-arm/check_numbers.py`. It covers the default
  view, the header's URL presets (radial and tangential at q₂ = 45°, 10°, 1°, 0, 180°; ?v=90; ?order=wrong), and all four preset buttons, including the folded pose `q_2 = 180°`,
  which has no URL of its own and is the module's second singular configuration: there the
  0.1 m/s tangential command costs `0.316 rad/s = 0.1/σ_max` with `σ_max = 0.316`, and ghost B's
  tip is 0.6000 m from the correct one.
- URL parameters, for screenshots and for checking the readouts without clicking:
  `?q1=<deg>&q2=<deg>`, `?v=radial|tangential|<deg>`, `?order=wrong|right`. With no query string
  the page is the default view above. Useful states: `?q2=0&v=radial` (the whole command lost),
  `?q2=0&v=tangential` (0.131 rad/s, produced exactly), `?order=wrong` (the default).
- Tested by loading the file in Chrome on macOS on 2026-09-05, and parts 2 and 3 in headless
  Chrome on 2026-09-20; not on a touch device, and not yet in Firefox or Safari.
