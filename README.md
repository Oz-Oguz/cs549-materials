# CS 549 Learning for Robotics (Bilkent, Fall 2026): course materials

This repository holds the slides, the kinematics and optimization readings and the interactive demonstrations of CS 549.
Everything else about the course (assignments, announcements, grades) is on Moodle.

## Slides

- `slides/01-D1-intro-and-the-stack.pdf`
- `slides/02-D1-learning-overlay-and-frames.pdf`
- `slides/02-D2-transforms-fk-jacobian.pdf`
- `slides/03-D1-optimization-1.pdf`
- `slides/04-D1-constraints-and-komo.pdf`
- `slides/04-D2-motion-planning.pdf`

The slides' "run it live" buttons open the demonstrations. A button for a browser module opens that module on
the published site, already set to the slide's example. A button for a Drake demonstration opens the guide in
`demos/`, which says how to install and run it on your own computer.

## The kinematics reading

`reading/kinematics-essentials/` is the reading on kinematic foundations with its companion files.
Open `chapter-01.pdf` from that folder, because its links open files beside it, and read its
`README.html` first if a link does not open. The Drake demonstrations in it are optional enrichment.

## The optimization reading

`reading/optimization-essentials/` is Chapter 2, on optimization and constrained inverse kinematics, with its
companion files. Open `chapter-02-essentials.pdf`, or `chapter-02-essentials.html` in a browser; read the folder's
`README.html` first. The reading's interactive examples need a local web server from a downloaded copy
(`python3 launch.py` in that folder), and they open directly on the published site,
https://oz-oguz.github.io/cs549-materials/reading/optimization-essentials/chapter-02-essentials.html . Its Drake demonstrations are optional.

## Interactive modules

Each folder under `modules/` holds one browser demonstration, `index.html`, which explains its controls on screen.
The easiest way to use them is the published site, https://oz-oguz.github.io/cs549-materials/ , where every module and PDF of this
repository opens in the browser (for example https://oz-oguz.github.io/cs549-materials/modules/two-link-arm/ or
https://oz-oguz.github.io/cs549-materials/modules/opt-explorer/). From a downloaded copy, open a module's `index.html` in any current
browser; the rotations module needs WebGL and an internet connection the first time, and the
optimization explorer needs a local web server: run `python3 -m http.server` in this repository's
folder and open `http://localhost:8000/modules/opt-explorer/`.

## Drake demonstrations

`demos/` holds the ten demonstrations that run a seven-joint robot arm in the Drake robotics toolbox, with
`demos/README.md` as their setup guide (also at https://oz-oguz.github.io/cs549-materials/demos/). They run in Python on Linux, on Apple-silicon
Macs and under WSL on Windows, and they are optional: the slides carry their results.

Updated 2026-10-26. © 2026 Özgür S. Öğüz, Bilkent University. Licensed under CC BY 4.0: use and adapt these materials freely, but credit the source as described in `LICENSE.md`. The FR3 robot model inside the reading package keeps its own Apache 2.0 license and notice.
