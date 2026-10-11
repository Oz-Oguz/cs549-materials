# KKT explorer — the fence, the force, and the multiplier

**Session:** 4.1 (Deck 05, slides 7, 8 and 13). **File:** `index.html`, a single self-contained page (vanilla
JavaScript and SVG, no dependencies); open it in any browser, or serve the folder. The slides' **run it
live** buttons open it through `scripts/demo_server.py` with a starting state in the address:
`?mode=disc|wall|rail` picks the constraint and `?target=<z1>,<z2>` places the target (each within
±2.2; anything else keeps the defaults, the round fence and the target at (1.6, 0.6)). The button
"(0.9, 0.9), the prediction" sets the target that slide 7 asks the students to predict. **Companion
lecture notes:** `lectures/04-D1-constraints-and-komo/notes.md` §4.1, §4.3–4.5.

## What it shows

The smallest constrained problem that has every feature of the general one:

    min_z  f(z) = ½‖z − z_t‖²    subject to one of
      round fence     g(z) = ‖z‖ − 1 ≤ 0            (stay inside the unit disc)
      straight fence  g(z) = (z₁ + z₂ − 1)/√2 ≤ 0    (stay below a line; g is the signed distance)
      rail            h(z) = (z₁ + z₂ − 1)/√2 = 0    (stay on the line; an equality)

The red target `z_t` is draggable. The page draws the feasible set, the level sets of `f` (circles
around the target), the solution `z*`, and two arrows at `z*`: the downhill pull of the cost, `−∇f`,
and the push of the fence, `−λ∇g` (or `−κ∇h`). Beside the picture it prints `z*`, `f*`, `g(z*)`, the
multiplier, both gradient vectors, the four KKT conditions with live check marks, and the price of the
constraint.

Everything is computed in closed form; there is no solver in the page. Round fence:
`z* = z_t / max(1, ‖z_t‖)`, `λ* = max(0, ‖z_t‖ − 1)`. Straight fence: `z* = z_t − max(0, g(z_t)) n`,
`λ* = max(0, g(z_t))`, with `n` the unit normal. Rail: `z* = z_t − h(z_t) n`, `κ* = h(z_t)`, of
either sign.

## What the interaction teaches that a static figure cannot

- **Complementarity is a switch in the activity.** Drag the target across the fence. Inside, `λ` is
  exactly zero and `g(z*)` is negative. Outside, `g(z*)` is exactly zero and `λ = ‖z_t‖ − 1` is
  positive. `λ` itself rises continuously from zero, with no jump. What switches is which of the two
  is zero, and that on/off logic is what makes constrained optimization combinatorial.
- **The multiplier is a length.** For the round fence `λ` is the distance by which the target lies
  outside. Students can predict it before dragging (the target at `(0.9, 0.9)` on the slide, where
  `λ = 0.273`) and then check.
- **Force balance is literal.** The two arrows are always equal and opposite; the page prints their
  sum. Moving the target along the boundary rotates both arrows together.
- **Rails have signed multipliers.** Switch to the rail and cross the line: `κ` changes sign, the
  push arrow flips, and the "dual feasibility" row shows a dash instead of a check mark.
- **The multiplier is a price.** The sensitivity line says by how much `f*` would drop if the fence
  were widened by `0.1`. The drop is about `0.1 λ` only while `λ > 0.1`. Closer to the fence, the
  widened fence goes idle, the whole `½λ²` disappears, and the rule overshoots. The drop is zero when
  the fence is not touched.

## Default state

Round fence, target at `(1.6, 0.6)`: the constraint is active, `λ = 0.709`, both arrows visible, all
four KKT rows ticked. The four preset buttons put the target outside, inside, just across the fence
(`λ = 0.02`), and far on the other side.

## Controls

| Control | Effect |
|---|---|
| drag anywhere on the plot | moves the target `z_t` (clamped to the plotted square) |
| radio buttons | round fence / straight fence / rail |
| preset buttons | four target positions |
| level sets checkbox | show or hide the circles around the target and the dashed circle through `z*` |

Axes are the coordinates `z₁`, `z₂` in the same units as the fence (the unit disc has radius 1).

## Relation to the other material

Deck 05 uses the same disc problem as worked example (a) with three numeric cases; the module is
the same example with a continuous target. The inherited `opt-explorer` Constrained tab covers the
*algorithms* (log barrier, squared penalty, augmented Lagrangian) on richer problems; this page covers
only the *conditions* at the solution, on purpose. The two-link arm with an elbow limit (Deck 05
worked example (b), `code/constrained_ik.py`) is the same picture on a robot.

## Limitations and weak points

- One constraint at a time. The wedge case with two simultaneously active fences (last year's deck
  04_05 exercise, inherited explorer "Wedge (Corner Solution)") is not shown here; the cone reading
  of stationarity with two multipliers is therefore only on the slides.
- The cost is a bowl centred on the target, so the unconstrained minimum is the target itself. A
  non-quadratic cost would show that `−∇f` at `z*` is not in general directed at the unconstrained
  minimum; the page does not go there.
- Touch devices: dragging works through pointer events; the readouts are below the plot on narrow
  screens.
