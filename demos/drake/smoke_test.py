"""Drake smoke test (2026-09-04).

Loads the Franka Panda model that ships with drake_models, welds its base to the world,
prints forward kinematics and the 6x7 spatial Jacobian of the end-effector at two joint
configurations, and starts a Meshcat server (prints its URL; open it in a browser to see the
arm). Run:  python drake/smoke_test.py          (exits at once)
       python drake/smoke_test.py --hold   (keeps Meshcat alive until you press Enter)
"""
import sys
from importlib.metadata import version as pkg_version
import numpy as np

import pydrake
from pydrake.all import (AddMultibodyPlantSceneGraph, DiagramBuilder, JacobianWrtVariable,
                         MeshcatVisualizer, Parser, RigidTransform, StartMeshcat)

print("drake", pkg_version("drake"), "| python", sys.version.split()[0], "| platform", sys.platform)

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
(Panda,) = parser.AddModelsFromUrl(
    "package://drake_models/franka_description/urdf/panda_arm.urdf")
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", Panda))
plant.Finalize()

meshcat = StartMeshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

ee = plant.GetFrameByName("panda_link8", Panda)
n = plant.num_positions()
print("joints:", n)

for name, q in [("home  q = 0", np.zeros(n)),
                ("ready q = (0, -0.785, 0, -2.356, 0, 1.571, 0.785)", np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))]:
    plant.SetPositions(plant_context, Panda, q)
    X_WE = plant.CalcRelativeTransform(plant_context, plant.world_frame(), ee)   # ^W X^E
    J = plant.CalcJacobianSpatialVelocity(plant_context, JacobianWrtVariable.kQDot,
                                          ee, [0, 0, 0], plant.world_frame(), plant.world_frame())
    print(f"\n{name}")
    print("  ^W p^E [m]   :", np.round(X_WE.translation(), 4))
    print("  ^W R^E rows  :", np.round(X_WE.rotation().matrix(), 3).tolist())
    print("  J shape/rank :", J.shape, np.linalg.matrix_rank(J, tol=1e-6))
    print("  sigma_min(J) :", round(float(np.linalg.svd(J, compute_uv=False)[-1]), 5))

diagram.ForcedPublish(context)
print("\nMeshcat URL:", meshcat.web_url())
print("SMOKE TEST OK")
if "--hold" in sys.argv:
    input("Meshcat is live at the URL above; open it in a browser. Press Enter to quit. ")
