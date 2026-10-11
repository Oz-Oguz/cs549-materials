# CS 549 Learning for Robotics (Bilkent, Fall 2026): course materials

This repository holds the slides, the lecture notes and the interactive demonstrations of CS 549.
Everything else about the course (assignments, announcements, grades) is on Moodle.

## Lectures

- `lectures/01-D1-intro-and-the-stack/`: `slides.pdf` and `notes.pdf`
- `lectures/02-D1-learning-overlay-and-frames/`: `slides.pdf` and `notes.pdf`
- `lectures/02-D2-transforms-fk-jacobian/`: `slides.pdf` and `notes.pdf`
- `lectures/03-D1-optimization-1/`: `slides.pdf` and `notes.pdf`
- `lectures/04-D1-constraints-and-komo/`: `slides.pdf` and `notes.pdf`

The slides carry "run it live" buttons that work only on the lecturer's computer during the lecture;
the demonstrations behind them are the modules below.

## The kinematics reading

`reading/kinematics-essentials/` is the reading on kinematic foundations with its companion files.
Open `essentials.pdf` from that folder, because its links open files beside it, and read its
`README.html` first if a link does not open. The Drake demonstrations in it are optional enrichment.

## Interactive modules

Each folder under `modules/` holds one browser demonstration, `index.html`, and its guide, `notes.md`.
The easiest way to use them is the published site, https://oz-oguz.github.io/cs549-materials/ , where every module and PDF of this
repository opens in the browser (for example https://oz-oguz.github.io/cs549-materials/modules/two-link-arm/ or
https://oz-oguz.github.io/cs549-materials/modules/opt-explorer/). From a downloaded copy, open a module's `index.html` in any current
browser; the rotations module needs WebGL and an internet connection the first time, and the
optimization explorer needs a local web server: run `python3 -m http.server` in this repository's
folder and open `http://localhost:8000/modules/opt-explorer/`.

Updated 2026-10-11. © 2026 Özgür S. Öğüz, Bilkent University. Licensed under CC BY 4.0: use and adapt these materials freely, but credit the source as described in `LICENSE.md`. The FR3 robot model inside the reading package keeps its own Apache 2.0 license and notice.
