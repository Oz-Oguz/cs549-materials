# Chapter 3: motion planning

Extract the complete zip into a folder, then open `README.html`. Keep the folder
together. The reading retains all four appendices. A PDF copied out of this
folder does not carry its companion files.

- [Read the essentials PDF](chapter-03-essentials.pdf).
- [Read the same essentials in a browser](chapter-03-essentials.html).

Some PDF viewers block local links or drop query parameters. Use the matching
links below in `README.html` when that happens. The browser reading includes
its mathematical renderer and fonts, so it does not need an internet connection.

## Start the browser apps

From this extracted folder run:

```bash
python3 launch.py
```

On Windows use `py launch.py`. Open the address printed by the command if a
browser does not open. This serves only this extracted folder on your computer.
It needs Python 3 and no Python packages. Keep the terminal open while using
the apps. Stop it with Ctrl+C. Use `--port 8766` if port 8765 is already in use.

The sampling app also opens directly as a local HTML file. The optimization
explorer, which has the Motion refinement tab, uses JavaScript modules and
needs the local server. It is already built; you do not need Node.js or npm.

| Link in the reading | File in this folder |
|---|---|
| figure generator | [`lectures/04-D2-motion-planning/code/make_figures.py`](lectures/04-D2-motion-planning/code/make_figures.py) |
| sampling reference | [`modules/motion-planning/rrt_two_link.py`](modules/motion-planning/rrt_two_link.py) |
| edge demonstration | [`modules/motion-planning/index.html?mode=edge`](modules/motion-planning/index.html?mode=edge) |
| cost comparison | [`lectures/04-reading-motion-planning/code/cost_comparison.py`](lectures/04-reading-motion-planning/code/cost_comparison.py) |
| section check | [`lectures/04-reading-motion-planning/code/check_section2.py`](lectures/04-reading-motion-planning/code/check_section2.py) |
| whole-link implementation | [`modules/motion-planning/staged_two_link.py`](modules/motion-planning/staged_two_link.py) |
| rai_nlp.py | [`lectures/04-D2-motion-planning/code/rai_nlp.py`](lectures/04-D2-motion-planning/code/rai_nlp.py) |
| stage4_rai.json | [`modules/motion-planning/fixtures/stage4_rai.json`](modules/motion-planning/fixtures/stage4_rai.json) |
| native RAI program | [`modules/motion-planning/stage4_komo_rai.py`](modules/motion-planning/stage4_komo_rai.py) |
| archived run records | [`lectures/05-motion-planning/sampling-demo/evidence/rai-stage4/run4-before-defaults.txt`](lectures/05-motion-planning/sampling-demo/evidence/rai-stage4/run4-before-defaults.txt) |
| seed-study record | [`lectures/05-motion-planning/sampling-demo/evidence/rai-stage4/seed-run.txt`](lectures/05-motion-planning/sampling-demo/evidence/rai-stage4/seed-run.txt) |
| Python reference | [`modules/motion-planning/rai_bidir_rrt.py`](modules/motion-planning/rai_bidir_rrt.py) |
| chapter check | [`lectures/04-reading-motion-planning/code/check_continuation.py`](lectures/04-reading-motion-planning/code/check_continuation.py) |
| rrtstar_reference.py | [`modules/motion-planning/rrtstar_reference.py`](modules/motion-planning/rrtstar_reference.py) |
| refine_figures.py | [`modules/motion-planning/refine_figures.py`](modules/motion-planning/refine_figures.py) |
| refine_rai.json | [`modules/opt-explorer/fixtures/refine_rai.json`](modules/opt-explorer/fixtures/refine_rai.json) |
| sampling views | [`modules/motion-planning/index.html`](modules/motion-planning/index.html) |
| optimization explorer | [`modules/opt-explorer/index.html`](modules/opt-explorer/index.html) |

## Python for the numerical examples {#python}

The appendices say to run their commands from the course repository root.
In this package, run them from this folder instead. The supplied `lectures/`
and `modules/` paths are part of the package. Create its environment on
macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -m venv .venv`, then `.venv\Scripts\activate`. After activation,
use `python` in place of `.venv/bin/python` in the appendices' Bash examples.
Set `PYTHONUTF8=1` if a Windows terminal cannot print a mathematical character.
The numerical scripts need NumPy; figure generators also need Matplotlib.
No robotics package is needed for the two-link calculations.

Each Bash block in the appendices is a separate example. Run each from this
folder. The commands print their results. They write no figures unless you
pass an option that asks for them. Comments and usage lines in some scripts
name files of the course repository that are not in this folder; the commands
in the reading do not need them.

## Native RAI runs {#rai}

The native RAI command in Appendix B.3 is optional. Its recorded results are
included under `modules/motion-planning/fixtures/` and
`modules/opt-explorer/fixtures/`, and the NumPy scripts compare against them.
The course ran it with `robotic` 0.4.1 on Linux x86_64. In the course
repository, `scripts/rai_run.sh` starts that environment in a container.
In this package, `scripts/rai_run.sh` runs the file with this folder's
`.venv/bin/python`, or with `python3` when there is no `.venv`. To use it,
install `robotic==0.4.1` in that environment yourself. The package does not
include RAI or a container image. Its `RAI_NOVNC` setting has no effect here.

## Course records that are not included {#course-records}

Two links in the reading point to course records that are not part of this
package. The module notes are written for the instructor, and their replay
evidence is summarized in Appendix C.1. The Python reference
`modules/motion-planning/rai_bidir_rrt.py` contains the replay code itself.
The lecture whose figures Appendix A reuses is Deck 06; its
[slides](https://oz-oguz.github.io/cs549-materials/slides/04-D2-motion-planning.pdf)
on the course site carry the same figures.

## License

See [LICENSE.md](LICENSE.md) for the course-material license and `licenses/`
for the notices of bundled third-party browser dependencies.
