# gauss-newton-ik — companion notes (session 3.2; tier T1, single file, no install)

Open `index.html` in any browser, or through the lecture launcher (`scripts/demo_server.py`) at
`http://127.0.0.1:8765/modules/gauss-newton-ik/`. Nothing is loaded from the network.
The address can set the starting target and method: `?y=0.5,0.2&method=gd` is Deck 04 page 14's
button, and page 29's button has no query, the default state. The joints always start at
(30°, 45°). Appending `?run` (or `run=1`) runs up to 60 iterations on load, which is used only for
the slide screenshot; `run=0` and `run=false` do not run. Unknown or invalid parameters are ignored
and named above the plot.

Chapter 02's first example opens at `?view=local-step`. It starts paused with the target
at (0.50, 0.20) m and separates the local prediction from the forward-kinematics check.
The ordinary solver view and its starting state are unchanged.

## What the interaction teaches that static text cannot

That the same update rule, `q ← q + (JᵀJ + ρI)⁻¹Jᵀ(y* − φ(q))`, is fast and tame or explosive
depending on one number and one target: with `ρ = 0` and an unreachable target you watch the
ghost arms fly off as the arm straightens and `det J → 0`; tick Armijo, or raise `ρ` by three
clicks, and the same arm settles on the boundary of its reach, pointing at the target. A table
can show the two outcomes; only stepping through the iterates shows *when* and *why* the
Gauss-Newton step goes wrong, which is the moment the linearisation is trusted at a
near-singular configuration.

## Default state

Method Gauss-Newton (`ρ = 0`), no line search, target `y* = (0.75, 0.05)` m, which is
unreachable (`|y*| = 0.752 > l₁ + l₂ = 0.7`), arm at `q⁽⁰⁾ = (30°, 45°)`. Before anyone touches a
control the readout already says "unreachable" and shows the first residual `0.55 m`. One
press of **Run** produces the phenomenon: a step of tens of radians within the first few
iterations, a residual that never settles, and a scatter of ghost arms.

## Controls

| Control | Range and units | What changing it shows |
|---|---|---|
| Method radio | Gauss-Newton / damped Gauss-Newton (fixed `ρ`; the radio was labelled Levenberg-Marquardt until 2026-09-29) / gradient descent | GN: `ρ = 0`, `J♯ = J⁻¹` for this square arm. Damped: the same with `ρ > 0`. GD: `αJᵀ(y* − φ)` with Armijo and no cap on `α`, the first-order method of 3.1: 46 iterations to `10⁻⁶` on the reachable target, against Gauss-Newton's 5; one Step from the reachable preset is Deck 04 slide 14's gradient step. |
| Slider `ρ` | `10⁻⁴` to `10¹`, quarter-decades; three clicks up from the default `10⁻³` (to `5.6 × 10⁻³`) is the first position that settles the default target inside 60 iterations (within 1 mm of the closest point from step 4; `10⁻³` still wanders at step 60) | Small `ρ`: GN speed, GN fragility. Large `ρ`: steps shrink toward gradient descent, robust and slow. Choosing the damped method automatically when moved. |
| Checkbox Armijo | on/off; default off; shown ticked and greyed out while gradient descent is selected, because that method always backtracks with Armijo (2026-10-05); the user's choice returns with the Gauss-Newton directions | Keeps the GN direction, halves the step until the cost decreases sufficiently (`c = 0.01`), and **rejects** the step when no length above `10⁻⁸` passes; the readout then says "stalled" (2026-09-29; before, the failed step was taken anyway). On the default target the arm is on the boundary by step 5, `0.4 mm` from the closest reachable point, with residual 0.0517 m, which is `|y*| − (l₁ + l₂)` to four digits, and the readout then says stalled after 9 accepted iterations, because the 10th direction is rejected; the gradient there is `3 × 10⁻⁴`, not zero, so this is a stall, not a stationary point. For Deck 04's target `(0.8, 0.1)` the module stalls after 9 accepted iterations, 9.96 mm from the closest point. That is the second row of Deck 04 slide 28's table, because this page's floor is `10⁻⁸` and the deck's code defaults to `10⁻⁶`. |
| Step / Run / Reset | one iteration; up to 60; back to `q⁽⁰⁾` | Run stops when the readout reaches solved (`10⁻⁶` m), stationary or stalled, or at its cap of 60 iterations, which the readout reports as "limit". |
| Target presets | reachable (0.5, 0.2); boundary (0.7, 0); unreachable (0.75, 0.05); unreachable (0.8, 0.1), the deck's slide-28 target | Reachable: 5 GN iterations, quadratic convergence visible on the log chart. Boundary: convergence but only linear (`q₂` halves per step, the residual drops by a factor of four) because the solution is singular. Unreachable: the default phenomenon. |
| Drag ★ | anywhere on the plot | Sets `y*`; the reachable annulus is shaded. |

## Axes

The plot is the arm's plane, `x [m]` along the bottom and `y [m]` on the left, grid every
0.2 m, with the reachable annulus (radii 0.1 m and 0.7 m) shaded. The chart is
`log₁₀ ‖φ(q) − y*‖` in metres against the iteration index, with the axis stretched to the run's length (at least 10, at most 60), so a five-step run fills the chart.

## Companion notes (stand-alone prose)

Inverse kinematics asks for `q` with `φ(q) = y*`. Written as a least-squares cost
`f(q) = ½‖φ(q) − y*‖²`, its gradient is `Jᵀ(φ(q) − y*)`. Linearising `φ` at the current `q`,
`φ(q + δ) ≈ φ(q) + Jδ`, and minimising the linearised cost plus a damping term `½ρ‖δ‖²`
gives the step `δ = (JᵀJ + ρI)⁻¹Jᵀ(y* − φ(q))`. With `ρ = 0` this is Gauss-Newton, which for a
square invertible `J` is simply `J⁻¹(y* − φ)`: it solves the linear model exactly, and near a
regular solution its error after a step is at most a constant times the square of the error
before it (quadratic convergence). With a fixed `ρ > 0` it is damped Gauss-Newton
(Levenberg–Marquardt methods adapt `ρ`): the step is shortened, most in the directions where `J` is weak, and as `ρ` grows it turns into a
gradient step with step size `1/ρ`. The set of fixed points is the same for every `ρ`, because the
update vanishes exactly when `Jᵀ(y* − φ) = 0`; `ρ` changes the path, and from some starts the
valley, never the set of answers. When the target is unreachable, the least-squares solution is
the arm stretched toward the target, a singular configuration where `JᵀJ` is not invertible:
the inverse formula of plain Gauss-Newton fails there (the least-squares subproblem still has
minimisers, and the pseudo-inverse picks the shortest), and the step explodes on approach. Damping with enough `ρ`
keeps the steps finite and walks the arm to the closest reachable point, where the gradient is
zero. A line search on the Gauss-Newton direction also keeps the steps finite, but near a singular
configuration the direction turns almost perpendicular to the gradient, and this page rejects it
when the trial step at its floor (`10⁻⁸`) fails the Armijo test: the readout says "stalled". That
stop is this page's rule, not a theorem: a smaller trial can still pass the test, and lowering the
floor only lets a few more tiny steps through while the tip stays about as far from the closest
reachable point (Deck 04 notes §5 (k) prints the trials around the floor). The readout names the
outcome after every Run, as the deck's code does: solved, stationary, stalled, or limit (the cap of
60 iterations; after single Steps it says running); "stationary" means
`|Jᵀ(φ − y*)| ≤ 10⁻⁸ ‖J‖ |φ − y*|` (`‖J‖` the Frobenius norm) while the residual is not small, the
deck's relative gradient test (not MINPACK's columnwise cosine; it can stop with task error left
along a weak direction), which is the least-squares compromise when it is a minimum and never a
proof that the target is unreachable. The
derivation with weights `K` and `D` in the form of Spring 2026 deck 03 (motion planning) p.56,
`J♯ = (JᵀKJ + D)⁻¹JᵀK`, and the numbers behind the presets are in
`lectures/03-D1-optimization-1/notes.md` §4–§5 (its examples (e)–(k)).

## Chapter 02: predict one step, then check it with forward kinematics

Open [the staged local-step view](index.html?view=local-step), or use
`http://127.0.0.1:8765/modules/gauss-newton-ik/?view=local-step` with the lecture launcher.
This view supports Chapter 02 §1.1 before students need the Gauss–Newton derivation.
It uses the existing forward-kinematics and Jacobian functions and solves one square linear
system directly. It does not run an iterative solver.

The arm has link lengths of 0.4 m and 0.3 m. It starts at joint angles of 30° and 45°,
and its target is (0.50, 0.20) m. All positions use the base frame, with the horizontal
axis pointing right and the vertical axis pointing up. The numerical calculation uses
radians for the angles. Ask students to predict whether the local model reaching the
target guarantees that the proposed joint configuration reaches it.

1. Press **1. Predict with J**. The dashed teal line joins the initial tip to the local
   prediction, which coincides with the target. The prediction solves

   $$
   \mathbf J(\mathbf q^{(0)})\boldsymbol\delta
   =\mathbf y^*-\varphi(\mathbf q^{(0)}),
   \qquad
   \boldsymbol\delta\approx(-0.920116,\ 1.293088)^\top\ \mathrm{rad}.
   $$

   The displayed joint change is about (−52.72°, 74.09°). The gray arm has not moved.
   The teal circle is a predicted tip point, not a predicted whole-arm configuration.
2. Press **2. Check with FK**. The maroon arm shows the configuration obtained by
   adding the proposed change to the initial joint angles. Forward kinematics places
   its tip at about (0.335682, 0.143665) m. The dashed maroon segment is the remaining
   tip-to-target error, whose length is 0.173707 m.
3. Compare the two rows in the cost table. The initial distance is 0.299564 m, so this
   step improves the task. The cost is half the squared distance and falls from
   0.044869 to 0.015087 m². The finite step does not attain the local model's zero error.
   Forward kinematics checks what the model predicted; the smaller cost does not establish
   that the target was reached.
4. Press **Reset** to hide the prediction and the FK check and return to the initial
   view. The target stays fixed throughout this example. The link below the panel opens
   the ordinary iterative solver from the same starting pose and target, for later sections.

The second button stays disabled until the prediction has been revealed. Neither button
runs a physical simulation or measures a robot. No damping or line search is used.
Parameters `y`, `method` and `run` are ignored and named in a notice when
`view=local-step` is selected, so even a link with `run=1` opens paused.

The interaction reveals the predicted endpoint before the FK-checked endpoint. This gives
students a chance to make a prediction before seeing that the large joint change breaks
the local approximation. The static fallback is the two-panel figure in Chapter 02 §1.1.
The focused automated check is `node scripts/demo_smoke.mjs local-step`; it checks both
stages, the numerical values, Reset, URL overrides and the original solver defaults.
It does not establish that the instructor has rehearsed the example by hand.
