# The Drake demonstrations of the reading

These scripts run a robot arm in the Drake robotics toolbox and show it in a browser. They are optional
enrichment. Install Drake once, as the course's demonstrations guide describes: https://oz-oguz.github.io/cs549-materials/demos/ (in the
course materials repository, `demos/README.md`). Then, with that environment active, run a script from the
reading's folder:

| Demonstration | Command | What it shows |
|---|---|---|
| Panda forward kinematics and Jacobian | `python drake/01-fk-jacobian/fk_jacobian_demo.py --meshcat` | The Panda with seven joint sliders. The terminal prints the end-effector position, the rank of the Jacobian and its smallest singular value as you move them. Three buttons set a spread pose, a near-singular pose and the singular zero configuration. |
| Self-motion with the full pose held | `python drake/01b-redundancy-self-motion/self_motion_panda.py --meshcat` | The gripper stays on the mug while one slider moves the elbow along the curve of configurations that keep the hand still. Buttons jump to elbow up and the two elbow-down poses, and play a sweep. |
| Fingertip-only app | `python drake/01b-redundancy-self-motion/tip_only_panda.py --app` | A web page with the scene and a readable panel. Only the fingertip position is fixed, so the seven joint sliders (in the scene's Open Controls menu) move the arm in four free directions. |
| FR3 chain and grasp target | `python drake/01c-fr3-grasp-chain/fr3_grasp_chain.py --meshcat` | The Franka Research 3 model in its ready pose. Buttons step through the chain from the base to the flange, the tool center and the mug grasp; sliders change the link and mug opacity and nudge each joint. |

Each command prints an address; open it in a browser. End a demonstration with its "Stop" button, or with
Ctrl+C in the terminal. Without its flag, a script prints its numbers and exits.
