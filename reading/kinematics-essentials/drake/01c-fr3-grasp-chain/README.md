# The FR3 chain and grasp target

`fr3_grasp_chain.py` traces the forward kinematics of the Franka Research 3 arm from its base to the flange,
the tool center and a mug grasp target. Run it with `--meshcat` and open the address it prints (setup:
`../README.md`). Without the flag, it prints and checks every factor of the chain.

Drake reports that it ignores some fields of the robot description: the safety-controller settings, joint
friction and the finger mimic rule. This is expected. The demonstration does not simulate dynamics, and it sets
both fingers to the same opening on every update.

The robot model comes from Franka Robotics' `franka_description`, release 2.9.0, and keeps its Apache 2.0
license: see `model/source/LICENSE` and `model/source/NOTICE`. `model/provenance.json` records its source and
file hashes.
