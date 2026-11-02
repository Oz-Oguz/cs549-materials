"""Reusable FR3 model and known-mug scene, with an independent NumPy FK chain.

Public contract: SCENE/Q_READY, build_scene(), current_pose(), grasp_target(),
approach_target(), parsed_chain(), chain_products(), and verified_facts().
Matrices are homogeneous transforms with column vectors; X_AB maps B to A.
The independent chain reads URDF origins and axes, never Drake FK results.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model" / "fr3.urdf"
SCENE = json.loads((HERE / "data/scene.json").read_text())
Q_READY = np.pi * np.array(SCENE["q_ready_pi_fractions"])
JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r, p, y = rpy
    X = np.eye(4)
    X[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    X[:3, 3] = xyz
    return X


def scene_transform(key):
    entry = SCENE[key]
    rpy = np.pi*np.array(entry["rpy_pi_fractions"]) if "rpy_pi_fractions" in entry else entry["rpy"]
    return transform(entry["xyz"], rpy)


def grasp_target():
    """Return X_WG*: supplied object pose times a chosen object-relative grasp."""
    return scene_transform("world_to_mug") @ scene_transform("mug_to_grasp_target")


def approach_target():
    """Return X_WGapp: retreat 10 cm along negative target-local z."""
    return grasp_target() @ scene_transform("grasp_to_approach")


def _numbers(element, attribute, default):
    return np.fromstring(element.get(attribute, default) if element is not None else default, sep=" ")


def parsed_chain(frame=None):
    """Read the base-to-frame path, including all manufacturer fixed factors."""
    frame = frame or SCENE["grasp_frame"]
    by_child = {j.find("child").get("link"): j for j in ET.parse(MODEL).getroot().findall("joint")}
    path = []
    while frame != SCENE["base_frame"]:
        joint = by_child[frame]
        origin = joint.find("origin")
        limit = joint.find("limit")
        path.append({"name": joint.get("name"), "type": joint.get("type"),
                     "parent": joint.find("parent").get("link"), "child": frame,
                     "xyz": _numbers(origin, "xyz", "0 0 0"),
                     "rpy": _numbers(origin, "rpy", "0 0 0"),
                     "axis": _numbers(joint.find("axis"), "xyz", "1 0 0"),
                     "limit": {k: float(v) for k, v in limit.attrib.items()} if limit is not None else None})
        frame = path[-1]["parent"]
    return list(reversed(path))


def chain_products(q=Q_READY, frame=None):
    """Return the relative factor and cumulative world pose at every step."""
    angles = dict(zip(JOINT_NAMES, np.asarray(q, float), strict=True))
    X = scene_transform("world_to_base")
    products = []
    for step in parsed_chain(frame):
        O = transform(step["xyz"], step["rpy"])
        Q = np.eye(4)
        angle = angles.get(step["name"], 0.)
        if step["type"] == "revolute":
            Q[:3, :3] = rotation(step["axis"], angle)
        elif step["type"] != "fixed":
            raise ValueError("This public arm/TCP chain contains only revolute and fixed joints")
        F = O @ Q
        X = X @ F
        products.append({**step, "q_rad": angle, "origin_transform": O, "joint_motion": Q,
                         "relative_transform": F, "world_transform": X.copy()})
    return products


@dataclass
class Scene:
    diagram: object
    plant: object
    scene_graph: object
    context: object
    plant_context: object
    robot: object

    def set_q(self, q):
        q = np.asarray(q, float)
        if q.shape != (7,):
            raise ValueError("Expected seven arm angles in radians")
        for name, value in zip(JOINT_NAMES, q):
            self.plant.GetJointByName(name, self.robot).set_angle(self.plant_context, value)
        for i in (1, 2):
            self.plant.GetJointByName(f"fr3_finger_joint{i}", self.robot).set_translation(
                self.plant_context, SCENE["finger_opening_each_m"])


def build_scene(meshcat=None, *, scene_geometry=True, renderer=None):
    """Create a local Drake scene. Nothing is fetched or simulated.

    renderer=(name, engine) optionally installs a renderer before building.
    The Meshcat demo publishes static configurations from this same API.
    """
    from pydrake.all import (AddMultibodyPlantSceneGraph, Box, Cylinder, DiagramBuilder,
                            MeshcatVisualizer, Parser, RigidTransform)
    builder = DiagramBuilder()
    plant, graph = AddMultibodyPlantSceneGraph(builder, time_step=0.)
    if renderer:
        graph.AddRenderer(*renderer)
    (robot,) = Parser(plant).AddModels(str(MODEL))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(SCENE["base_frame"], robot),
                     RigidTransform(scene_transform("world_to_base")))
    if scene_geometry:
        table, mug = SCENE["table"], SCENE["mug"]
        size = table["size_xyz_m"]
        p = [*table["center_xy_m"], table["top_z_m"]-size[2]/2]
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(p), Box(*size),
                                     "table", [0.80, 0.76, 0.68, 1])
        X_WM = scene_transform("world_to_mug")
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(X_WM),
                                     Cylinder(mug["radius_m"], mug["height_m"]),
                                     "known_mug", [0.12, 0.55, 0.62, 1])
        # Three small bars make the +x_M handle identifiable; this is visual only.
        for name, xyz, dims in [("outer", [0.060,0,0], [0.009,0.012,0.05]),
                                ("top", [0.047,0,0.024], [0.032,0.012,0.008]),
                                ("bottom", [0.047,0,-0.024], [0.032,0.012,0.008])]:
            plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(X_WM @ transform(xyz)),
                                         Box(*dims), "mug_handle_"+name, [0.12,0.55,0.62,1])
    plant.Finalize()
    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    result = Scene(diagram, plant, graph, context, plant.GetMyContextFromRoot(context), robot)
    result.set_q(Q_READY)
    return result


def current_pose(scene, q=None, frame=None):
    """Return X_WG(q), or X_WE(q) when frame='fr3_link8'."""
    if q is not None:
        scene.set_q(q)
    target = scene.plant.GetFrameByName(frame or SCENE["grasp_frame"], scene.robot)
    return scene.plant.CalcRelativeTransform(scene.plant_context, scene.plant.world_frame(), target).GetAsMatrix4()


def _signature(element):
    return (element.tag, sorted(element.attrib.items()), (element.text or "").strip(),
            [_signature(child) for child in element])


def verify_model():
    provenance = json.loads((HERE / "model/provenance.json").read_text())
    for name, expected in provenance["files"].items():
        assert hashlib.sha256((HERE/"model"/name).read_bytes()).hexdigest() == expected, name
    official = ET.parse(HERE / "model/fr3-official.urdf").getroot()
    local = ET.parse(MODEL).getroot()
    for tag in ["joint", "link"]:
        originals = {e.get("name"): e for e in official.findall(tag)}
        for entry in local.findall(tag):
            original = originals.pop(entry.get("name"))
            if tag == "joint":
                assert _signature(entry) == _signature(original), entry.get("name")
            else:
                a, b = entry.find("inertial"), original.find("inertial")
                assert (a is None and b is None) or _signature(a) == _signature(b)
        assert not originals
    return provenance


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def verified_facts(scene=None):
    provenance = verify_model()
    scene = scene or build_scene()
    factors = chain_products()
    max_error = 0.
    rng = np.random.default_rng(549)
    limits = [s["limit"] for s in factors if s["type"] == "revolute"]
    lower = np.array([v["lower"] for v in limits]); upper = np.array([v["upper"] for v in limits])
    assert np.all((Q_READY >= lower) & (Q_READY <= upper))
    for i, name in enumerate(JOINT_NAMES):
        joint = scene.plant.GetJointByName(name, scene.robot)
        assert joint.position_lower_limits()[0] == lower[i]
        assert joint.position_upper_limits()[0] == upper[i]
        assert joint.velocity_upper_limits()[0] == limits[i]["velocity"]
    configurations = [Q_READY, *rng.uniform(lower, upper, size=(8, 7))]
    for q in configurations:
        scene.set_q(q)
        for step in chain_products(q):
            actual = current_pose(scene, frame=step["child"])
            error = float(np.max(np.abs(step["world_transform"]-actual)))
            max_error = max(max_error, error)
            assert error < 3e-14, (step["child"], error)
    scene.set_q(Q_READY)
    X_WE = current_pose(scene, frame=SCENE["flange_frame"])
    X_WG = current_pose(scene)
    X_EG = np.linalg.inv(X_WE) @ X_WG
    expected_EG = transform([0,0,0.1034], [0,0,-np.pi/4])
    assert np.allclose(X_EG, expected_EG, atol=1e-14)
    assert np.allclose(approach_target()[:3,3]-grasp_target()[:3,3], [0,0,0.10])
    assert abs(SCENE["world_to_mug"]["xyz"][2]-SCENE["mug"]["height_m"]/2-SCENE["table"]["top_z_m"]) < 1e-14
    facts = {"schema_version": 1, "provenance": {k: provenance[k] for k in ["repository","release","commit","robot_type","modification"]},
             "drake_version": importlib.metadata.version("drake"), "scene": SCENE,
             "q_rad": Q_READY, "q_deg": np.rad2deg(Q_READY), "arm_joint_limits": limits,
             "X_W_E": X_WE, "X_E_G": X_EG, "X_W_G": X_WG,
             "X_W_M": scene_transform("world_to_mug"), "X_M_Gstar": scene_transform("mug_to_grasp_target"),
             "X_W_Gstar": grasp_target(), "X_Gstar_Gapproach": scene_transform("grasp_to_approach"),
             "X_W_Gapproach": approach_target(), "chain": factors,
             "current_to_grasp_translation_world_m": grasp_target()[:3,3]-X_WG[:3,3],
             "verification": {"configurations": len(configurations), "frame_comparisons": len(configurations)*len(factors),
                              "max_absolute_matrix_error": max_error, "official_joint_and_inertial_entries_preserved": True,
                              "joint_position_and_velocity_limits_match_drake": True,
                              "target_reachability_or_collision_free_motion_verified": False}}
    return _jsonable(facts)
