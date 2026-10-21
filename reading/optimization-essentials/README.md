# Chapter 2: optimization and constrained inverse kinematics

Extract the complete zip into a folder, then open `README.html`. Keep the folder
together. The reading retains all three appendices. A PDF copied out of this
folder does not carry its companion files.

- [Read the essentials PDF](chapter-02-essentials.pdf).
- [Read the same essentials in a browser](chapter-02-essentials.html).
- [Read the full detailed reference](chapter-02-detailed.html).
- [Review the Chapter 1 prerequisites](prerequisites.html).
- [Open the demonstration guide](demos.html).

Some PDF viewers block local links or drop query parameters. Use the matching
links below in `README.html` when that happens. The browser readings include
their mathematical renderer and fonts, so they do not need an internet connection.

## Start the browser apps

From this extracted folder run:

```bash
python3 launch.py
```

On Windows use `py launch.py`. Open the address printed by the command if a
browser does not open. This serves only this extracted folder on your computer.
It needs Python 3 and no Python packages. Keep the terminal open while using
the apps. Stop it with Ctrl+C. Use `--port 8766` if port 8765 is already in use.

The two-link and constraint apps also open directly as local HTML files.
The optimization explorer uses JavaScript modules and needs the local server.
It is already built; students do not need Node.js or npm. The server is also
a convenient way to open all the readings and preserve their app query strings.

| Reading | Companion |
|---|---|
| Essentials | [detailed reference](chapter-02-detailed.html) |
| Essentials | [detailed reference](chapter-02-detailed.html#penalize-a-large-proposed-step) |
| Essentials | [environment guide](README.html#python) |
| Essentials | [local_step.py](lectures/03-reading-optimization-planning-and-control/code/local_step.py) |
| Essentials | [jacobian_transpose.py](lectures/03-reading-optimization-planning-and-control/code/jacobian_transpose.py) |
| Essentials | [two-link IK calculation](lectures/03-D1-optimization-1/code/gauss_newton_ik.py) |
| Essentials | [history-direction calculation](lectures/03-D1-optimization-1/code/history_directions.py) |
| Essentials | [line-search figure script](lectures/03-D1-optimization-1/code/make_solver_overview.py) |
| Essentials | [constrained_ik.py](lectures/04-D1-constraints-and-komo/code/constrained_ik.py) |
| Essentials | [solver checks](lectures/04-D1-constraints-and-komo/code/check_solvers.py) |
| Essentials | [first program](drake/02-gauss-newton-ik/gn_ik_panda.py) |
| Essentials | [figure script](lectures/03-reading-optimization-planning-and-control/code/panda_runs_figure.py) |
| Essentials | [second](drake/03-ik-as-nlp/ik_nlp_panda.py) |
| Essentials | [third](drake/05-ik-nlp-al/ik_nlp_al_panda.py) |
| Essentials | [demonstration guide](demos.html) |
| Essentials | [position-update view](modules/gauss-newton-ik/index.html?view=local-step) |
| Essentials | [gradient-descent view](modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gd) |
| Essentials | [step-size view](modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=off&alpha=0.08&arrows=off) |
| Essentials | [Armijo view](modules/opt-explorer/index.html?tab=concepts&fn=elongated&preset=classic-zig-zag&method=gd&ls=armijo&alpha=1&arrows=on) |
| Essentials | [Newton view](modules/opt-explorer/index.html?tab=concepts&fn=quartic-saddle&preset=near-the-maximum&method=newton&ls=off&arrows=off) |
| Essentials | [lecture-arm view](modules/opt-explorer/index.html?tab=trajectory&preset=ik) |
| Essentials | [reach-boundary view](modules/opt-explorer/index.html?tab=trajectory&preset=ik&y=0.7,0) |
| Essentials | [elbow-clipping view](modules/opt-explorer/index.html?tab=trajectory&preset=ik&elbow=60) |
| Essentials | [L-BFGS view](modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=lbfgs&ls=exact&arrows=off) |
| Essentials | [BFGS view](modules/opt-explorer/index.html?tab=concepts&fn=rosenbrock&preset=classic-start&method=bfgs&ls=strong-wolfe&arrows=off) |
| Essentials | [line-search view](modules/opt-explorer/index.html?tab=concepts&fn=armijo-wolfe&preset=start-x-2&method=gd&ls=strong-wolfe&arrows=off) |
| Essentials | [disc view](modules/kkt-explorer/index.html?mode=disc&target=0.5,0.3) |
| Essentials | [line view](modules/kkt-explorer/index.html?mode=wall&target=1.6,0.6) |
| Essentials | [detailed reference’s demonstration appendix](chapter-02-detailed.html#appendix-b-demonstrations) |
| Essentials | [1966 paper](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-p.pdf) |
| Detailed reference | [Chapter 01 §5.2](prerequisites.html#how-many-answers-the-planar-geometry) |
| Detailed reference | [bowl_derivatives.py](lectures/03-reading-optimization-planning-and-control/code/bowl_derivatives.py) |
| Detailed reference | [descent.py](lectures/03-D1-optimization-1/code/descent.py) |
| Detailed reference | [make_vocabulary_figures.py](lectures/03-D1-optimization-1/code/make_vocabulary_figures.py) |
| Detailed reference | [gradient_descent.py](lectures/03-reading-optimization-planning-and-control/code/gradient_descent.py) |
| Detailed reference | [armijo.py](lectures/03-reading-optimization-planning-and-control/code/armijo.py) |
| Detailed reference | [opening prediction figure](lectures/03-reading-optimization-planning-and-control/fig/local_prediction_vs_fk.png) |
| Detailed reference | [Drake guide](README.html#drake) |
| Detailed reference | [constrained_ik_panda.py](drake/04-constrained-ik-al/constrained_ik_panda.py) |
| Detailed reference | [two-link Gauss–Newton module](modules/gauss-newton-ik/index.html?y=0.5,0.2&method=gn) |
| Detailed reference | [paused Panda replay](demos.html#gn-ik-replay) |
| Detailed reference | [existing replay plots](lectures/03-D1-optimization-1/fig/drake-gn-replay.png) |
| Detailed reference | [Panda constrained-IK demonstration](demos.html#ik-nlp) |
| Detailed reference | [one-limit Panda replay](demos.html#constrained-ik-replay) |
| Detailed reference | [hard-target Panda replay](demos.html#ik-nlp-al-replay) |
| Detailed reference | [kinematics essentials](prerequisites.html#use-a-local-model-to-propose-a-joint-position-update) |
| Detailed reference | [the corresponding detailed derivations](chapter-02-detailed.html#optimization-fundamentals) |
| Detailed reference | [the corresponding detailed derivations](chapter-02-detailed.html#constraints-and-constrained-inverse-kinematics) |

## Python for the numerical examples {#python}

The commands in Appendix B run from this folder. The supplied `lectures/` and
`drake/` paths are part of the package. Create its environment on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows use `py -m venv .venv`, then `.venv\Scripts\activate`. After activation,
use `python` in place of `.venv/bin/python` in the appendix's Bash examples.
Set `PYTHONUTF8=1` if a Windows terminal cannot print a mathematical character.
The small numerical scripts need NumPy; figure generators also need Matplotlib.
No robotics package is needed for the two-link calculations.

Each Bash block in the detailed reference is a separate example. Run each
from this folder; a block that changes directories does not change the starting
directory of the next example. Script outputs may use lecture or source-file
names to identify the shared calculations.

## Panda demonstrations with Drake {#drake}

The Panda scripts are optional. Their static observations remain in the reading.
They need a Python environment in which Drake can be installed. The course's
tested environment uses Python 3.12 and Drake 1.46.0. To match it:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-drake.txt
```

If you already created `.venv`, install the Drake requirements in that environment
only if its Python supports the pinned Drake release. The package does not
include Python, Drake or the Panda model. Drake's first model load may need an
internet connection and a model download. Browser readings and the bundled
browser apps work without those downloads.

Use the commands in Appendix B.5 for print modes. For the paused web replays,
activate the environment and run one command from the demonstration guide.
Open the replay address printed in that terminal. Those scripts provide their
own Meshcat viewer and controls; the course checkout and its classroom launcher
are not needed. Run one replay at a time, and stop it with Ctrl+C before starting
another.

The code keeps the source solvers and numerical parameters. The rail replay's
reference helper and saved reference events are included under `support/`.
Its provenance/reference paths are adapted during packaging; its numerical
functions are unchanged. The package includes the neighboring helper modules,
HTML panels and regression baselines required by the linked scripts.
Comments and usage lines in some scripts name files of the course repository,
such as its planning records, that are not in this folder; the scripts do not
need them.

## What the detailed reference adds

The detailed browser reading retains its full derivations and appendices.
Its course-launcher links lead to the portable demonstration guide. The two
Chapter 1 references lead to included extracts of those sections. References
to the two lecture-note discussions lead to the corresponding derivations in
this detailed reading. These delivery adaptations preserve the mathematical
content while keeping the package independent of the course checkout.

The examples solve local numerical problems. Feasible waypoints do not certify
safe motion between them. No physical robot execution or realistic contact
simulation is included in this package.

## License

See [LICENSE.md](LICENSE.md) for the course-material license and `licenses/`
for the notices of bundled third-party browser dependencies. Drake and its
downloaded robot models keep their own licenses and are not redistributed here.
