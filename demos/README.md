# Running the lecture demonstrations

Most "run it live" buttons in the slides open a browser module on the course site and need no setup.
Ten buttons belong to demonstrations that run a seven-joint robot arm in the Drake robotics toolbox. These run
in Python on your own computer and show the robot in your browser. They are optional: every number they show
is also in the slides' tables and figures. In the slide PDFs, those ten buttons open this page.

## One-time setup

Drake installs from PyPI on these systems only:

| Your computer | Python |
|---|---|
| Linux on x86_64 (Ubuntu 22.04 or newer) | 3.10 to 3.13 |
| Mac with Apple silicon (M1 or newer), macOS 15 or newer | 3.12 or 3.13 |
| Windows | use WSL 2 with Ubuntu 22.04 or newer, then follow the Linux steps |

An Intel Mac, or a Mac on macOS 14 or older, has no Drake package; use the browser modules instead.

Download this repository (the green "Code" button on GitHub, then "Download ZIP", or `git clone`). In a terminal,
from the repository's `demos` folder:

```bash
python3 --version                      # must be one of the versions above
python3 -m venv .venv                  # a private Python environment for the demos
source .venv/bin/activate              # run this again in every new terminal
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python drake/smoke_test.py             # prints the arm's position and Jacobian, then exits
```

The first Drake run downloads Drake's robot models once, about 240 MB, which took under a minute on a fast
connection; later runs work offline.
`python drake/smoke_test.py --hold` also starts the viewer: open the address it prints in a browser and
press Enter in the terminal to quit.

## Running a demonstration

Activate the environment, then run the command listed below from the `demos` folder. Each command
prints an address such as `http://localhost:7000` or `http://127.0.0.1:52341/`; open it in a browser. The demo
waits for you and never moves by itself. End it with its "Stop" button, or press Ctrl+C in the terminal. After
Ctrl+C, Python prints a few lines ending in `KeyboardInterrupt`; that is how it reports the stop, not an error.
Running a script without its flag prints the numbers that appear on the slide, then exits.

### Contact changes the map

Slides: `slides/02-D1-learning-overlay-and-frames.pdf`, page 8 (Computed: how much the mug moves per joint).

```bash
python drake/00-contact-changes-the-map/contact_demo.py --meshcat
```

Seven joint sliders and a "Grasp / release" button in the browser. Before the grasp, moving the joints leaves the mug where it is; after the grasp, the mug moves with the hand. The terminal prints the rank of the mug's Jacobian after each press.

### When the map is wrong

Slides: `slides/02-D1-learning-overlay-and-frames.pdf`, page 9 (Challenge 2: state estimation).

```bash
python drake/00b-when-the-map-is-wrong/map_error_demo.py --meshcat
```

Sliders set the error of the estimated mug position in centimetres. "Plan on the estimate, execute on the truth" plans for the orange ghost mug and executes against the real teal mug, so a small error lands a finger on the rim. "Back to the ready pose" resets the arm.

### The FR3 transform chain

Slides: `slides/02-D2-transforms-fk-jacobian.pdf`, page 19 (The FR3 uses the same transform chain).

```bash
python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --meshcat
```

The Franka Research 3 model in its ready pose. Buttons step through the chain from the base to the flange, the tool center and the mug grasp; sliders change the link and mug opacity and nudge each joint.

### Forward kinematics and the Jacobian on the Panda

Slides: `slides/02-D2-transforms-fk-jacobian.pdf`, page 25 (Live demo: the Panda's 7-joint arm in Drake); `slides/02-D2-transforms-fk-jacobian.pdf`, page 57 (Backup: the Panda's zero configuration is singular).

```bash
python drake/01-fk-jacobian/fk_jacobian_demo.py --meshcat
```

The Panda with seven joint sliders. The terminal prints the end-effector position, the rank of the Jacobian and its smallest singular value as you move them. Three buttons set a spread pose, a near-singular pose and the singular zero configuration.

### Redundancy: the elbow swings, the hand stays

Slides: `slides/02-D2-transforms-fk-jacobian.pdf`, page 42 (Live demo: redundancy, the elbow swings, the hand stays).

```bash
python drake/01b-redundancy-self-motion/self_motion_panda.py --meshcat
```

The gripper stays on the mug while one slider moves the elbow along the curve of configurations that keep the hand still. Buttons jump to elbow up and the two elbow-down poses, and play a sweep.

### Only the fingertip position fixed

Slides: `slides/02-D2-transforms-fk-jacobian.pdf`, page 43 (Same demo, only the fingertip position fixed).

```bash
python drake/01b-redundancy-self-motion/tip_only_panda.py --app
```

A web page with the scene and a readable panel. Only the fingertip position is fixed, so the seven joint sliders (in the scene's Open Controls menu) move the arm in four free directions.

### Gauss-Newton inverse kinematics, step by step

Slides: `slides/03-D1-optimization-1.pdf`, page 31 (A real arm: the Panda in Drake).

```bash
python drake/02-gauss-newton-ik/gn_ik_panda.py --replay --app
```

A web page, paused at the start. Step, Back and the slider walk through the Gauss-Newton iterates; the predicted and the actual flange positions are drawn at each step, and four runs can be selected.

### Inverse kinematics as a nonlinear program

Slides: `slides/04-D1-constraints-and-komo.pdf`, page 18 (Live demo: IK as a nonlinear program on the Panda).

```bash
python drake/03-ik-as-nlp/ik_nlp_panda.py --app
```

Target sliders in the browser. After each move, the arm re-solves inverse kinematics as a nonlinear program, and the terminal prints which constraints are active.

### Constrained inverse kinematics: the multiplier is learned

Slides: `slides/04-D1-constraints-and-komo.pdf`, page 24 (The augmented Lagrangian learns the multiplier).

```bash
python drake/04-constrained-ik-al/constrained_ik_panda.py --replay --app
```

A web page, paused at the start. Step through the augmented-Lagrangian loop with one elbow stop: joint updates move the arm, and multiplier updates change the multiplier with the arm fixed.

### The target is a constraint; posture is the cost

Slides: `slides/04-D1-constraints-and-komo.pdf`, page 27 (The target is a constraint; posture is the cost).

```bash
python drake/05-ik-nlp-al/ik_nlp_al_panda.py --replay --app
```

A web page, paused at the start. The target is a constraint and the posture is the cost; step through joint updates and multiplier updates for two targets, and watch a joint bound become active in the second.

## If something does not work

- **`pip` finds no matching distribution for `drake`.** Your Python version or system is not in the table above.
  Check `python --version` inside the activated environment.
- **The address does not open.** Copy it exactly from the terminal, including the port number. Try
  `127.0.0.1` in place of `localhost`. Under WSL, open it in your Windows browser.
- **A demo says the address is already in use.** An earlier demo is still running. Stop it, or close its terminal.
- **The first run stops while downloading.** The model download needs the internet once. Run the command again.

The course tested these steps on macOS 26 with Python 3.12 and Drake 1.46.0. Linux and WSL follow Drake's own
packages, which the course did not test separately.
