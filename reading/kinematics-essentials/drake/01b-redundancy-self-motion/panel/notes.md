# The fingertip-only app

`tip_only_panda.py --app` serves this panel beside the robot view. Only the point between the fingertips is held
fixed, so the seven joint sliders (in the scene's Open Controls menu) move the arm in four free directions while
the panel shows the held point's error. From the reading's folder, run
`python drake/01b-redundancy-self-motion/tip_only_panda.py --app` and open the address printed after
"Demo app at". The setup is in `../../README.md`.
