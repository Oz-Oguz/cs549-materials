# frames-monogram — companion notes (session 2.2; tier T1, single file, no install)

Open `index.html` in any browser. Nothing is loaded from the network.

## What the interaction teaches that static text cannot

That `^T p^M_T` and `^T p^M_W` are one arrow with two sets of numbers: you can watch the arrow
stay put while turning the tray changes one readout and not the other.

## Default state

Exactly the worked example of the session 2.2 notes §5: tray frame `T` at `^W p^T = (0.40,
0.20)` m turned by `θ_T = 30°`, mug at `^T p^M = (0.15, 0.05)` m in the tray's axes, mug
attached to the tray, decomposition drawn along `T`'s axes, quiz mode off. Before anyone
touches a control the three readouts already show the phenomenon: `^T p^M_T = (0.150, 0.050)`,
`^T p^M_W = (0.105, 0.118)`, `^W p^M = (0.505, 0.318)`, and the same arrow length
`0.158 m` for both spellings of the tray-to-mug arrow. (The notes carry a third, vertical
component; the module is a top view and shows the first two.)

## Controls

| Control | Range and units | What changing it shows |
|---|---|---|
| Slider `θ_T` | −180° to 180°, 1° steps; default 30° | With the mug attached, `^T p^M_T` stays fixed while `^T p^M_W` and `^W p^M` change: the tray's own description of the mug does not care how the tray is placed. `^W R^T` updates live. |
| Drag `T` (its origin) | inside the 0.8 m × 0.6 m table | Translates the tray; with the mug attached, `^W p^M` shifts by the same vector and the two `^T p^M` readouts are unchanged. |
| Drag `M` | inside the table | Changes `^T p^M`; the readout shows the inverse computation `^T p^M = (^W R^T)ᵀ (^W p^M − ^W p^T)` happening (notes eq. 13). |
| Radio "decompose along T's axes / W's axes" | two states; default T | Draws the dashed staircase of the tray-to-mug arrow along one frame's axes or the other's. The two staircases have different legs and the same hypotenuse. |
| Checkbox "mug attached to the tray" | on/off; default on | Off: the mug stays where it is in the world while the tray moves, so `^T p^M` must change. This is the "camera sees the mug, where is it on the tray?" situation. |
| Checkbox "quiz mode" | on/off; default off | Hides the numbers of the tray-to-mug arrow so the instructor can ask the room for them (slide 20's exercise) and then reveal. |
| Button "Reset" | — | Back to the worked example. |

## Axes

One plot, the top view of the table: `x [m] (world frame W)` horizontal, `y [m] (world
frame W)` vertical, grid every 0.1 m, tick labels in metres. Frame axes are drawn 0.10 m
long (0.06 m for the mug) and labelled `x_W, y_W, x_T, y_T, x_C, y_C`.

## Companion notes (stand-alone prose)

A frame is an origin with two perpendicular unit axes (three in space; the module is
planar). `^A p^B_C` is the arrow from the origin of frame `A` to the origin of frame `B`,
with its two numbers written in the axes of frame `C`; when `C` is omitted it equals `A`.
`^A R^B` is the rotation matrix whose columns are `B`'s axes written in `A`'s coordinates;
for a planar rotation by `θ` it is `[[cos θ, −sin θ], [sin θ, cos θ]]`. The one equation of
the module is

    ^W p^M = ^W p^T + ^W R^T ^T p^M,

read as: to get from the world origin to the mug, go to the tray's origin, then along the
tray-to-mug arrow, whose numbers must first be converted from the tray's axes into the
world's by multiplying with `^W R^T`. Its inverse, used when you drag the mug, is
`^T p^M = (^W R^T)ᵀ (^W p^M − ^W p^T)`: subtract the offset, then rotate back, and no matrix
inverse is needed because the inverse of a rotation is its transpose. The full derivation,
the three-dimensional version, and the worked numbers are in
`lectures/02-D1-learning-overlay-and-frames/notes.md` §3–§5.

## Open questions and weak points

- Planar rotations commute, so the module cannot show that the order of rotations matters
  in three dimensions; that lesson is a numeric example in the notes (§4.5) and tab 2 of
  `modules/rotations-3d`, shown on Deck 02's composition slide.
- Dragging on a touch screen works through pointer events but has not been tested on a
  tablet; tested in Chrome and Safari on macOS on 2026-09-04 by loading the file, not by a
  user study.
- The module uses `θ_T` for the tray's angle, consistent with the notes; `θ` also names
  learned parameters in later weeks (collision recorded in the lecture notes).
