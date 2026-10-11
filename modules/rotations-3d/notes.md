# Rotations in 3-D — companion notes for `modules/rotations-3d/index.html`

Session 2.2 (Deck 02, PDF pages 23, 28, 29). One file, three tabs. Built 2026-09-19 from
`planning/demos-deck02-2026-09-19.md`. The 2-D module `frames-monogram` shows what a change
of frame does to *coordinates*; this one shows what a rotation *is*, why the order of two
rotations matters, and that one orientation has four spellings that must not be confused.

**Needs a network connection the first time it is opened**: three.js (pinned 0.160.0) comes
from `cdn.jsdelivr.net`, then the browser caches it. If the picture stays blank with a
loading message after ten seconds, that is the reason. Everything printed in the readouts is
computed by plain JavaScript that never touches three.js, and `check_numbers.py` in this
folder reproduces every default number with NumPy.

## What the interaction teaches that static text cannot

- **Tab 1.** The columns of `R` are the body's axes written in world coordinates: drag θ
  and the first column and the red arrow move as one thing. The *same* matrix answers two
  different questions — active (the body turns, the frame stays) and passive (the body
  stays, the frame turns) — and you can only see the difference by watching what moves.
- **Tab 2.** Two books, the same two rotations, opposite order, different final poses. The
  notes say `R_z(90°) R_x(90°) ≠ R_x(90°) R_z(90°)`; the tab lets you step: after one rotation
  each book has had only its own first turn (already different, not yet surprising); after the
  second both have had the same two turns in opposite orders and still disagree. The gimbal mode shows what
  a loss of one independent angle-rate direction looks like. At pitch +90° or −90° the
  yaw and roll axes share one line. The two rates have opposite effects at +90° and the same
  effect at −90°. The body still has three rotational freedoms.
- **Tab 3.** The matrix from Rodrigues' formula, the axis–angle pair, the unit quaternion
  and the RAI `.g` line are one orientation. A quaternion written `(w, x, y, z)` and read as
  `(x, y, z, w)` is drawn as a ghost in the wrong pose: for the tray's 30° about z the ghost is
  **150° about x**, and no library raises an error.

## Default states (each already shows the phenomenon)

| Tab | Default | What is on screen before anything is touched |
|---|---|---|
| 1 | `R_z(30°)`, active | the tray's rotation from notes §5; `R = [[0.866, −0.5, 0], [0.5, 0.866, 0], [0, 0, 1]]`, column 1 coloured like the red `x_B` arrow; the corner `^B p = (0.30, 0.40, 0.06)` lands at `^W p = (0.0598, 0.4964, 0.06)` |
| 2, order | `a = R_x(90°)`, `b = R_z(90°)`, both applied | left book `R_z R_x`, first column `(0, 1, 0)`; right book `R_x R_z`, first column `(0, 0, 1)` — notes §4.5 |
| 2, gimbal | `ψ = 10°, ϑ_p = 90°, ϕ = 0°` | locked: the red roll axis lies on the blue yaw axis, the blue and green rings are coplanar, and the readout shows `R(10°, 90°, 0°)` and `R(20°, 90°, 10°)` differ by `1.1e−17` — notes §4.7, §5 (g) |
| 3 | 30° about `z`, ghost off | `r = (0.9659, 0, 0, 0.2588)` as on page 29; both RAI spellings of the tray's pose (page 30) |

## Controls

| Tab | Control | Range and units | What changing it shows |
|---|---|---|---|
| 1 | axis `x` / `y` / `z` | buttons | the three elementary rotations of page 23 |
| 1 | `θ` | −180°…180°, 1° steps | the columns of `R` moving with the drawn body axes |
| 1 | active / passive | radio | active: the book turns, the ghost shows where it was, the corner's world position changes. passive: the book stays, the RGB frame becomes a new frame `F`, the same corner gets new coordinates `^F p = Rᵀ ^W p` |
| 1 | "set θ = 90° and read the columns" | button | the check page 23 asks for: every column is now a signed unit vector |
| 1 | "break it: negate column x" | toggle | `det R = −1`: the book is mirrored (the cover text reads backwards) although `RᵀR = I` still holds — unit, perpendicular columns are not enough |
| 2 | `a`, `b` | axis menu + angle slider each, degrees | any pair; same-axis pairs commute, and special different-axis pairs can also agree |
| 2 | reset / step / apply both | buttons | stage 0 → 1 → 2, animated; the books receive different first turns at stage 1 and have received both turns at stage 2. Matrices and axis coordinates follow the currently drawn books. |
| 2 | `ψ`, `ϑ_p`, `ϕ` | −180°…180°, degrees | yaw, pitch, roll of `R = R_z(ψ) R_y(ϑ_p) R_x(ϕ)`; the rings are the gimbals |
| 2 | "go to pitch 90°" / "pitch 30°" | buttons | lock on, lock off |
| 3 | axis presets; azimuth, elevation of `û` | buttons; degrees | any unit axis; the dashed maroon line is `û` |
| 3 | `θ` | −180°…180°, degrees | Rodrigues' three terms and their sum, live |
| 3 | "hand this quaternion to a library that expects (x, y, z, w)" | checkbox | the orange ghost: what SciPy or ROS would build from RAI's or Drake's `(w, x, y, z)` |

Drag in the picture to orbit, wheel to zoom, right-drag to pan. World axes are grey, body
axes red/green/blue (`x`, `y`, `z`), the misread ghost orange. Symbols follow `notation.md`:
`θ` rotation angle, `ϑ_p` pitch (ruling R5), `R`, `r = (w, x, y, z)`, `[û]_×`.

Reset, an angle or axis edit, a mode change, and a tab change cancel the current order
animation and every pending step of “apply both.” An edit redraws the selected stage with the
new angle or axis. Returning to the order tab shows the selected stage without resuming an
old animation. Pressing “apply both” during motion starts a fresh sequence from stage 0.

## Axes

The 3-D view has no numeric axes; lengths are unitless (the book is 0.6 × 0.8 × 0.12 in
world units, the arrows are unit vectors). Every number that matters is in the readouts, in
metres for the tray position of the RAI lines and in degrees / radians as labelled.

## The mathematics behind each readout (notes §3–§4, in short)

- **Elementary rotations** (notes §3.3): `R_z(θ)` has columns `(c, s, 0)`, `(−s, c, 0)`,
  `(0, 0, 1)`; `R_x`, `R_y` by cyclic renaming. Tab 1 prints them, tab 3 uses them as the
  cross-check for Rodrigues when `û` is a coordinate axis.
- **Columns, `RᵀR = I`, `det R = 1`** (§4.2): `R e_j` is the j-th column, and it is the j-th
  body axis in world coordinates. Orthonormal columns give `RᵀR = I`; `det = +1` separates a
  rotation from a mirror. Tab 1's "break it" negates one column: `RᵀR` is unchanged, `det`
  flips, and the book is its own mirror image.
- **Active vs passive** (§4.2): active, `^W p = R ^B p` moves the point; passive, the point is
  fixed and `^F p = Rᵀ ^W p` are its coordinates in a frame turned by `R`. Robotics asks the
  passive question almost always, which is why `R` carries frame letters but no subscript.
- **Composition** (§4.5): `^A R^C = ^A R^B ^B R^C`. For rotations about world axes, the one
  applied first stands on the right. Tab 2's left book is `R_b R_a` ("a then b") after both
  steps. Stage 0 shows identity matrices, and stage 1 shows `R_a` on the left and `R_b` on the
  right. A difference after stage 1 does not establish an order effect because the books have
  received different rotations. Same-axis pairs commute. Different-axis pairs can also commute
  at special angles, including two half turns about perpendicular axes.
- **Euler angles and gimbal lock** (§4.7): `R = R_z(ψ) R_y(ϑ_p) R_x(ϕ)`. At `ϑ_p = 90°` the
  body x axis is `R_z(ψ) R_y(90°) e_x = −e_z`, so roll turns about the world −z axis while
  yaw turns about +z: only the difference `ψ − ϕ` survives. The readout moves both by +10°,
  which keeps `ψ − ϕ` fixed, and reports `max |ΔR| ≈ 1e−17`.
  At negative ninety-degree pitch, the roll axis instead points along positive world z. The
  orientation depends on the sum of yaw and roll, so increasing yaw while decreasing roll
  by the same amount leaves it unchanged. The readout selects the cancellation sign from the
  signed axis alignment. Axis dependence determines the lock status; a single paired-angle
  experiment by itself does not establish whether the angle-rate map has full rank.
- **Rodrigues** (§4.6): `R = I + sin θ [û]_× + (1 − cos θ) [û]_×²`; tab 3 prints `[û]_×`
  and the sum.
- **Quaternion** (§3.4): `r = (cos θ/2, sin θ/2 û)`; the matrix from `r` is the standard
  formula and the readout shows it agrees with Rodrigues to machine precision. Reading
  `(a, b, c, d)` as `(x, y, z, w)` makes `w′ = d` and `(x′, y′, z′) = (a, b, c)`; it is still a
  unit quaternion, so nothing complains, and it encodes a different rotation.

## Checking the module against NumPy

```
.venv/bin/python modules/rotations-3d/check_numbers.py
```
prints every default readout. The readouts themselves can be read without a screen:
open `index.html?tab=3&misread=1` (or `?tab=1`, `?tab=1&ap=passive`, `?tab=1&mirror=1`,
`?tab=2&mode=order`, `?tab=2&mode=gimbal`) in headless Chrome with `--dump-dom` and compare.
Done 2026-09-19: all six default states agree with the script to the printed precision.
The script now also checks the three order stages, same-axis and perpendicular-half-turn
commutation, and both signs of the pitch lock. The scripted browser scenario is
`scripts/harness/scenario_rotations_3d.mjs`; run it through
`node scripts/demo_smoke.mjs rotations-3d`. It checks the stage readouts, animation
interruptions, both locks, an unlocked state, and quaternion ordering. It reports whether
WebGL initialized and records the renderer details.
