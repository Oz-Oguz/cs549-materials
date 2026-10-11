# opt-explorer — inherited interactive app (status: inherited, not yet reviewed against the module template)

Copied on 2026-09-04 from `l4r_opt_tool/web_app` (github.com/Oz-Oguz/l4r_opt_tool, commit
ec53b74). Vite + React + TypeScript + Plotly. Three tabs:

- **Trajectory** — 2-D trajectory optimization with obstacles (signed distance, projection),
  a planar arm (`forwardKinematics`, `armCollides`), GD / BFGS / Gauss-Newton / LM steps,
  backtracking line search. Its augmented-Lagrangian control is disabled pending correction of
  the multiplier update's coordinate mismatch; Constrained-tab AL remains available. Seed of `komo-2d` (4.2).
- **Concepts** — scalar 2-D optimization on eight test functions (Quadratic Bowl, Elongated
  Quadratic, Rosenbrock, Himmelblau, Beale, Quartic with a Maximum at the Origin, Armijo & Wolfe 1-D, Newton
  Hessian caveat 1-D) with GD / Newton / BFGS / CG / LM, line search, adaptive step size.
  This is `descent-landscapes` (3.1). Its LM damps the **exact** Hessian, `(∇²f + λI)δ = −∇f`,
  so it is the modified Newton of deck 04 page 18, **not** the `(JᵀJ + ρI)` of page 21: nothing in
  this tab forms a residual Jacobian. Point it at page 18, never at `gauss-newton-ik` (3.2).

  Two things to know before using it in class. Its function **"Quartic with a Maximum at the
  Origin"** (`f = x⁴ + y⁴ − 2x² − 2y²`, id `quartic-saddle`, called "Quartic with Saddle Point" until
  2026-10-05) has a local maximum at the origin: `H = diag(12x² − 4, 12y² − 4)` is negative definite
  there, both eigenvalues −3.88 at the preset Near the maximum (0.1, 0.1). It is the right function
  for "Newton finds stationary points, not minima". The true saddles are at (±1, 0) and (0, ±1). The
  old preset slugs `near-saddle-0-0` and `off-saddle` still open the same starts. Since 2026-10-05
  the Diagnostics row ‖∇f‖ reports the gradient at the displayed iterate on every path, and the
  message at a small gradient reads "Approximately stationary", because a maximum or a saddle passes
  that test as well. And the app is a
  Vite project: `index.html` is the dev entry and cannot be opened from `file://`. For a lecture,
  run `npm run build` (after `npm ci` on a fresh machine, because `dist/` is gitignored) and again
  after any source change, then start `scripts/demo_server.py`: it serves the build at
  `http://127.0.0.1:8765/modules/opt-explorer/`, and without a usable build that address shows
  the build instructions instead of a blank page. Since 2026-10-04 the build uses relative asset
  paths (`base: './'` in `vite.config.ts`) so that it loads under that sub-path. The page address
  can choose the starting tab and, on the Unconstrained tab, the function, start, method, line
  search, step size and whether the step-preview arrows show (`src/linkState.ts`; Deck 04 page 13's button uses it).
  On the Constrained tab it can choose the problem, the outer method, the growth factor and the
  "Conv. tol" slider (`?tab=constrained&problem=circle-interior&method=auglag&growth=1&tol=1e-6`,
  Deck 05 page 24's button), and on the Trajectory tab it can load the named Between-knot
  collision example (`?tab=trajectory&preset=between-knot`, Deck 05 page 33's button). A link sets only
  the starting state and never steps; unknown or invalid parameters are ignored and named in a
  notice below the tab bar.
- **Constrained** — log barrier, squared penalty, augmented Lagrangian on the same problems
  (circle, linear halfspace, wedge, equality, ill-conditioned + halfspace) with a 3-D surface
  and a dual-variable chart. This is the "constrained explorer" that `syllabus-2026.md` §5
  names as the optional module for 4.2.

Run: `npm ci && npm run dev` (Node 20+), then open the printed localhost URL. Build check:
`npm run build`. `CONCEPTS_INTEGRATION_GUIDE.md` in `source-2026/opt-lectures/docs/` records
where each tab was used in last year's lectures. `vercel.json.l4r-original` is the deployment
config of the original repository (paths differ here).

To do before classroom use: rename symbols to `notation.md` (optimizer variable `z`, damping
`ρ`), check every default state shows its phenomenon (template §"Default state"), write the
companion notes proper.

## The Deck 05 Constrained-tab moment (slide 24 since the splits of 2026-10-06, slide 23 earlier that day, slide 22 before; added 2026-09-29)

Tab Constrained, problem Circle Constraint (cost $(x-2)^2 + y^2$, fence $x^2 + y^2 \le 1$), method
Augmented Lagrangian, inner solver Newton, Growth ×1, Conv. tol $10^{-6}$ (one slider for the inner gradient test and
the outer stop; the default $10^{-2}$ stops too early to show the contrast). Students first compute $\lambda^*$ from the force balance at $(1, 0)$: the pull is
$-\nabla f = (2, 0)$ and the fence's gradient is $\nabla g = (2, 0)$, so $\lambda^* = 1$. Then run: the dual chart
climbs $0, 0.716, 0.910, 0.970, 0.990$ toward 1 while $\mu$ stays at 1, in 13 outer steps; with growth 5, the
factor RAI's documentation gives, it takes 5. Row 0 of the log is the $\mu = 1$ penalty with violation
$0.358$, where the first-order rule $\lambda^*/(2\mu)$ predicts $0.5$. The optimizer code was checked in node and
reproduced in Python on 2026-09-28; on 2026-09-29 the interface was driven by script
(`scripts/harness/scenario_opt_explorer.mjs`, run by `scripts/demo_smoke.mjs`): the $\lambda$ sequence, the 13 against 5
outer steps, and the 5 against 4 at the default tolerance all reproduced. The $\mu_0$ slider stops at 10, so no penalty
sweep to 100 is possible. Rehearse once by hand before class.

## Deck 04 Armijo preview (corrected 2026-09-30)

The actual step, arrow endpoint and text preview now share one calculation and the same adaptive
initial step size. **α used** is the previous accepted step; **Next initial trial α** is the next
search's starting value; **predicted next α** is its predicted accepted value. From Elongated
Quadratic / Classic zig-zag with Armijo and the slider at 1, the first six accepted values remain
0.0625, 0.075, 0.09, 0.108, 0.1296, 0.07776. The browser check compares each preview endpoint and
alpha with the subsequent Step. `scripts/check_scalar_preview.mjs` additionally checks all five
methods, their state and Newton safeguards without a browser.

## Deck 05 Trajectory moment (slide 33 since the splits of 2026-10-06, slide 32 earlier that day, slide 30 before; corrected 2026-09-30)

Use **Between-knot collision**, a named preset independent of the tab's random initial state.
It loads four knots with first-joint angles −60°, −20°, 20°, 60° and second-joint angle 0°.
The base is (0.5, 0.5), links 0.22 and 0.15; the circular obstacle has radius 0.04 and its center
is 0.32 from the base at 5°. Distances are workspace units. It selects penalty mode, weight 100,
second-difference smoothness with weight 1, target weight 0.01, locked endpoints, no zero-velocity
endpoint constraint or bounds. Optional Gauss-Newton uses damping 1; the demonstration needs no
optimization step.

Ask: **Does zero obstacle cost at these four poses imply a collision-free motion?** The panel
shows knot clearance +0.0428 and obstacle cost 0. Twenty subdivisions of each joint-linear interval
(61 total samples) find clearance −0.0344. The red dashed arm shows a colliding configuration;
the blue dashed tip path uses FK of joint interpolation, not Cartesian chords between tip knots.
The circular-obstacle cost and diagnostic both use whole link segments. Rectangles and triangles
elsewhere in the app use bounding-circle approximations; the panel labels that distinction.

Click **Add midpoint knots**. The same joint path now has seven knots, knot clearance −0.0121
and obstacle cost 0.0147. Its 121 samples find clearance −0.0400. Adding knots exposes the missed
collision; it does not repair the path. A negative sample establishes a collision in the model;
positive samples cannot certify all intermediate configurations. This sampling lesson replaces the
unreproduced valley-jump claim; use slide 28 for the two-valley example. The Python example there
checks three points per arm, a separate approximation from this module's full-link circle check.

`scripts/check_trajectory_sampling.mjs` checks the geometry independently using trigonometry and
verifies that refinement preserves joint interpolation. The browser scenario loads the same visible
preset, checks the readouts, adds midpoints, takes a GN step and checks that reloading the preset
restores the original values. The Trajectory AL radio is disabled; its implementation remains
unrepaired. Automated checks do not replace the instructor's hand rehearsal.

## Browser verification

`node scripts/demo_smoke.mjs` builds current explorer source before serving it. A failed explorer
build still allows the four standalone module scenarios to run. `node scripts/demo_smoke.mjs
opt-explorer` selects this app only; `CAPTURE_FIGS=1` also refreshes its lecture screenshots.
Ordinary runs keep captures in a temporary folder. The 3-D WebGL panels remain outside this
headless check's coverage.
