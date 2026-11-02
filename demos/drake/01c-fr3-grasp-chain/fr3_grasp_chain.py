"""Trace actual FR3 forward kinematics and construct a known-mug grasp target.

Default: print and verify every chain factor. --meshcat: staged static-FK demo.
--png FILE: offline Drake/VTK figure. --facts FILE: machine-readable verified data.
This lesson neither solves IK nor simulates contact.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from fr3_scene import (HERE, SCENE, Q_READY, JOINT_NAMES, build_scene, current_pose,
                       parsed_chain, chain_products, transform, scene_transform,
                       grasp_target, approach_target, verified_facts)

STAGES = ["1. Trace the joint chain", "2. Add the flange and tool", "3. Compare the grasp target", "4. Add the approach target"]
COLORS = {"current": (0.08,0.40,0.85), "grasp": (0.93,0.54,0.04),
          "approach": (0.56,0.20,0.73), "mug": (0.12,0.55,0.62)}
# Link IDs are 0–7; actuated joint IDs are 1–7. Axis rods keep their RGB convention.
LINK_COLORS = [
    ("gray", (0.35,0.38,0.42)),
    ("red", (0.85,0.20,0.16)),
    ("orange", (0.95,0.55,0.05)),
    ("green", (0.16,0.60,0.26)),
    ("cyan", (0.05,0.68,0.74)),
    ("blue", (0.14,0.33,0.85)),
    ("purple", (0.53,0.27,0.74)),
    ("pink", (0.90,0.34,0.62)),
]
COLOR_BUTTON = "Toggle distinct link colors"
FRAME_BUTTONS = {"Show mug frame M": "mug", "Show grasp frame G*": "grasp",
                 "Compare M and G*": "both"}


class LinkColorizer:
    """Color arm links and fade all robot meshes, including the hand and fingers."""
    def __init__(self, meshcat, scene):
        self.meshcat = meshcat
        self.enabled = False
        self.opacity = 1.0
        self.visuals = []
        inspector = scene.scene_graph.model_inspector()
        for body_index in scene.plant.GetBodyIndices(scene.robot):
            body = scene.plant.get_body(body_index)
            link_id = next((i for i in range(8) if body.name() == f"fr3_link{i}"), None)
            for geometry in scene.plant.GetVisualGeometriesForBody(body):
                name = inspector.GetName(geometry).replace("::", "/")
                path = f"/drake/visualizer/fr3/{body.name()}/{name}"
                original = inspector.GetIllustrationProperties(geometry).GetProperty("phong", "diffuse")
                self.visuals.append((link_id, path, inspector.GetShape(geometry), original))

    def set_appearance(self, enabled, opacity):
        if enabled == self.enabled and opacity == self.opacity:
            return
        from pydrake.all import Rgba
        for i, path, shape, original in self.visuals:
            if not self.meshcat.HasPath(path):
                raise RuntimeError(f"Missing published FR3 visual: {path}")
            # SetObject creates a separate material for each color. Editing a shared
            # gray material in the browser could otherwise recolor every link at once.
            rgb = LINK_COLORS[i][1] if enabled and i is not None else (original.r(), original.g(), original.b())
            color = Rgba(*rgb, original.a()*opacity)
            self.meshcat.SetObject(path, shape, color)
        self.enabled = enabled
        self.opacity = opacity


class MugAppearance:
    """Change the mug body and handle materials while keeping its frame solid."""
    def __init__(self, meshcat, scene):
        self.meshcat = meshcat
        self.opacity = 1.0
        self.visuals = []
        inspector = scene.scene_graph.model_inspector()
        for geometry in scene.plant.GetVisualGeometriesForBody(scene.plant.world_body()):
            name = inspector.GetName(geometry)
            if name == "known_mug" or name.startswith("mug_handle_"):
                path = f"/drake/visualizer/{name}"
                original = inspector.GetIllustrationProperties(geometry).GetProperty("phong", "diffuse")
                self.visuals.append((path, inspector.GetShape(geometry), original))
        if len(self.visuals) != 4:
            raise RuntimeError("Expected the mug body and three handle visuals")

    def set_opacity(self, opacity):
        if opacity == self.opacity:
            return
        from pydrake.all import Rgba
        for path, shape, original in self.visuals:
            if not self.meshcat.HasPath(path):
                raise RuntimeError(f"Missing published mug visual: {path}")
            self.meshcat.SetObject(path, shape, Rgba(original.r(), original.g(), original.b(), original.a()*opacity))
        self.opacity = opacity


class LessonState:
    """Controls change q or visibility only. Targets always come from the scene."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.q = Q_READY.copy()
        self.stage = 4
        self.joint_step = 7
        self.link_colors = False
        self.link_opacity = 1.0
        self.mug_opacity = 1.0
        self.target_frames = "grasp"

    def set_target_frames(self, value):
        if value not in FRAME_BUTTONS.values():
            raise ValueError("Choose mug, grasp or both target frames")
        self.target_frames = value

    def set_link_opacity(self, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Link opacity must be between 0 and 1")
        self.link_opacity = float(value)

    def set_mug_opacity(self, value):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Mug opacity must be between 0 and 1")
        self.mug_opacity = float(value)

    def set_stage(self, value):
        if value not in range(1,5):
            raise ValueError("Stage must be 1–4")
        self.stage = value

    def set_q(self, q):
        limits = [j["limit"] for j in parsed_chain() if j["type"] == "revolute"]
        q = np.asarray(q, float)
        if q.shape != (7,) or any(not lim["lower"]-1e-12 <= angle <= lim["upper"]+1e-12 for angle,lim in zip(q,limits)):
            raise ValueError("The sliders use the official FR3 position limits")
        self.q = np.clip(q,[v["lower"] for v in limits],[v["upper"] for v in limits])

    def snapshot(self):
        path = chain_products(self.q)
        return {"stage": self.stage, "q": self.q.copy(), "joint_step": self.joint_step,
                "link_colors": self.link_colors,
                "link_opacity": self.link_opacity,
                "mug_opacity": self.mug_opacity,
                "target_frames": self.target_frames,
                "joint_origin": np.zeros(3) if self.joint_step == 0 else path[self.joint_step-1]["world_transform"][:3,3],
                "X_W_E": path[7]["world_transform"], "X_W_G": path[-1]["world_transform"],
                "X_W_Gstar": grasp_target(), "X_W_Gapproach": approach_target()}


def check_controls():
    state = LessonState()
    ready = state.snapshot()
    assert state.stage == 4 and np.array_equal(state.q,Q_READY)
    state.link_colors = True
    state.set_link_opacity(0.35)
    state.set_mug_opacity(0.25)
    colored = state.snapshot()
    for key in ["X_W_E", "X_W_G", "X_W_Gstar", "X_W_Gapproach"]:
        assert np.array_equal(colored[key], ready[key])
    for mode in FRAME_BUTTONS.values():
        state.set_target_frames(mode)
        for key in ["X_W_E", "X_W_G", "X_W_Gstar", "X_W_Gapproach"]:
            assert np.array_equal(state.snapshot()[key], ready[key])
    for stage in range(1,5):
        state.set_stage(stage)
        assert state.snapshot()["stage"] == stage
    q = state.q.copy()
    q[0] += 0.2
    state.set_q(q)
    moved = state.snapshot()
    assert not np.allclose(moved["X_W_G"],ready["X_W_G"])
    assert np.array_equal(moved["X_W_Gstar"],ready["X_W_Gstar"])
    assert np.array_equal(moved["X_W_Gapproach"],ready["X_W_Gapproach"])
    for step in range(8):
        state.joint_step = step
        assert state.snapshot()["joint_origin"].shape == (3,)
    limits = [j["limit"] for j in parsed_chain() if j["type"] == "revolute"]
    state.set_q([v["lower"] for v in limits])
    state.set_q([v["upper"] for v in limits])
    state.reset()
    assert np.array_equal(state.q,Q_READY) and state.stage == 4 and not state.link_colors and state.link_opacity == 1 and state.mug_opacity == 1
    assert state.target_frames == "grasp"
    for value in [0, 1]:
        state.set_link_opacity(value)
    for value in [-0.1, 1.1, float("nan")]:
        try:
            state.set_link_opacity(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid link opacity accepted")
    state.reset()
    for value in [0, 0.35, 1]:
        state.set_mug_opacity(value)
    for value in [-0.1, 1.1, float("nan")]:
        try:
            state.set_mug_opacity(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid mug opacity accepted")
    state.reset()
    try:
        state.set_q(np.zeros(7))
    except ValueError:
        pass
    else:
        raise AssertionError("All-zero q violates FR3 limits at joints 4 and 6")
    return "Stages 1–4, joint steps 0–7, target invariance, exact reset and limit rejection passed."


def check_angular_block():
    """Build world-axis columns from the independent chain and compare with Drake."""
    from pydrake.all import JacobianWrtVariable
    scene = build_scene(scene_geometry=False)
    world = scene.plant.world_frame()
    flange = scene.plant.GetBodyByName("fr3_link8", scene.robot).body_frame()
    columns = [scene.plant.GetJointByName(name, scene.robot).position_start()
               for name in JOINT_NAMES]
    qdot = np.array([0.1, 0.2, 0, 0, 0, 0, 0])
    ready_J = None
    for q in [Q_READY, Q_READY + np.array([0.2, 0, 0, 0, 0, 0, 0])]:
        moving = [step for step in chain_products(q) if step["type"] == "revolute"]
        J_ang = np.column_stack([step["world_transform"][:3, :3] @ step["axis"]
                                 for step in moving])
        scene.set_q(q)
        drake_J = scene.plant.CalcJacobianAngularVelocity(
            scene.plant_context, JacobianWrtVariable.kQDot, flange, world, world)
        assert np.allclose(J_ang, drake_J[:, columns], atol=1e-14, rtol=0)
        # Check the predicted turning against a time derivative of the FK rotation.
        h = 1e-6
        R = chain_products(q)[7]["world_transform"][:3, :3]
        Rp = chain_products(q + h*qdot)[7]["world_transform"][:3, :3]
        Rm = chain_products(q - h*qdot)[7]["world_transform"][:3, :3]
        skew = ((Rp - Rm)/(2*h)) @ R.T
        omega_fd = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        assert np.allclose(J_ang @ qdot, omega_fd, atol=1e-9, rtol=0)
        if ready_J is None:
            ready_J = J_ang
    assert np.allclose(ready_J[:, :2], [[0, 0], [0, 1], [1, 0]], atol=1e-14, rtol=0)
    assert np.allclose(ready_J @ qdot, [0, 0.2, 0.1], atol=1e-14, rtol=0)
    print("\nAngular Jacobian from world joint-axis directions, ready pose =\n", ready_J)
    print("First two angular columns =\n", ready_J[:, :2])
    print("qdot [rad/s] =", qdot)
    print("omega in world axes [rad/s] =", ready_J @ qdot)
    print("Angular columns match Drake at ready and changed poses; FK time derivatives agree.")


def print_mode(facts):
    np.set_printoptions(precision=6,suppress=True,linewidth=105)
    print("Actual FR3 | official franka_description 2.9.0 | Drake",facts["drake_version"])
    print("q [rad] =",Q_READY)
    print("q [deg] =",np.rad2deg(Q_READY))
    print("W is fixed to fr3_link0; E is fr3_link8; G is fr3_hand_tcp.")
    print("Each relative factor is URDF origin placement times its local joint rotation.")
    for step in chain_products():
        print(f"\n{step['name']}: {step['parent']} -> {step['child']} ({step['type']})")
        print("fixed origin [xyz m; rpy rad]:",step["xyz"],step["rpy"])
        if step["type"] == "revolute":
            print("local axis:",step["axis"],"angle [rad]:",step["q_rad"])
        print("relative factor =\n",step["relative_transform"])
        print("cumulative world pose =\n",step["world_transform"])
    for key in ["X_W_E","X_E_G","X_W_G","X_W_M","X_M_Gstar","X_W_Gstar","X_Gstar_Gapproach","X_W_Gapproach"]:
        print(f"\n{key} =\n",np.array(facts[key]))
    print("\nCurrent G to requested G* displacement in W [m]:",facts["current_to_grasp_translation_world_m"])
    print(f"M is the mug center; the table is at z={SCENE['table']['top_z_m']:.2f} m; the mug is {SCENE['mug']['height_m']:.2f} m high.")
    print("Target-local approach displacement [m]:",scene_transform("grasp_to_approach")[:3,3])
    print("World approach displacement [m]:",approach_target()[:3,3]-grasp_target()[:3,3])
    print("These are current FK and two requested poses. No IK, contact, or motion feasibility is solved.")
    print("Checks:",facts["verification"])
    print(check_controls())
    check_angular_block()
    print("FR3 CHAIN CHECKS PASSED")


def _meshcat_frame(meshcat,path,X,length=0.07,origin_color=(0.15,0.15,0.15),radius=0.004,dashed=None):
    from pydrake.all import Cylinder,Rgba,RigidTransform,RotationMatrix,Sphere
    meshcat.SetObject(path+"/origin",Sphere(radius*1.8),Rgba(*origin_color,1))
    meshcat.SetTransform(path+"/origin",RigidTransform(X[:3,3]))
    for i,color in enumerate([(0.85,0.13,0.13),(0.12,0.65,0.18),(0.15,0.28,0.85)]):
        direction = X[:3,i]
        pose = RigidTransform(RotationMatrix.MakeFromOneVector(direction,2),X[:3,3]+length*direction/2)
        meshcat.SetObject(path+f"/axis{i}",Cylinder(radius,length),Rgba(*color,1))
        meshcat.SetTransform(path+f"/axis{i}",pose)
        if dashed is not None:
            meshcat.SetProperty(path+f"/axis{i}","visible",not dashed)
            # Cylinder dashes have visible thickness even when WebGL clamps line width.
            # Keep both styles at the same physical origin; no display offset is used.
            dash_length = length/16
            for j,start in enumerate(np.linspace(0,length,8,endpoint=False)):
                dash_path = path+f"/dashed_axis{i}/segment{j}"
                meshcat.SetObject(dash_path,Cylinder(0.003,dash_length),Rgba(*color,1))
                meshcat.SetTransform(dash_path,RigidTransform(
                    RotationMatrix.MakeFromOneVector(direction,2),
                    X[:3,3]+direction*(start+dash_length/2)))
            meshcat.SetProperty(path+f"/dashed_axis{i}","visible",dashed)


def update_meshcat(meshcat,scene,state,colorizer=None,mug_appearance=None):
    from pydrake.all import Rgba,RigidTransform,Sphere
    scene.set_q(state.q)
    scene.diagram.ForcedPublish(scene.context)
    if colorizer is not None:
        colorizer.set_appearance(state.link_colors, state.link_opacity)
    if mug_appearance is not None:
        mug_appearance.set_opacity(state.mug_opacity)
    values = state.snapshot()
    factors = chain_products(state.q)
    # The line joins frame origins to teach the chain. It is not a link skeleton.
    origins = np.column_stack([np.zeros(3),*[f["world_transform"][:3,3] for f in factors[:7]]])
    meshcat.SetLine("/lesson/joint_chain",origins,line_width=3,rgba=Rgba(0.10,0.45,0.85,0.85))
    for i,point in enumerate(origins.T):
        color = LINK_COLORS[i][1] if state.link_colors else COLORS["current"]
        meshcat.SetObject(f"/lesson/origins/{i}",Sphere(0.009),Rgba(*color,1))
        meshcat.SetTransform(f"/lesson/origins/{i}",RigidTransform(point))
    selected = np.eye(4) if state.joint_step == 0 else factors[state.joint_step-1]["world_transform"]
    selected_color = LINK_COLORS[state.joint_step][1] if state.link_colors else (0.15,0.15,0.15)
    _meshcat_frame(meshcat,"/lesson/selected_joint",selected,0.09,selected_color,radius=0.003)
    _meshcat_frame(meshcat,"/lesson/world",np.eye(4),0.12,radius=0.003)
    _meshcat_frame(meshcat,"/lesson/current/E",values["X_W_E"],0.05,COLORS["current"],0.002)
    _meshcat_frame(meshcat,"/lesson/current/G",values["X_W_G"],0.075,COLORS["current"])
    meshcat.SetLine("/lesson/current/tool_offset",np.column_stack([values["X_W_E"][:3,3],values["X_W_G"][:3,3]]),3,Rgba(*COLORS["current"],1))
    _meshcat_frame(meshcat,"/lesson/target/M",scene_transform("world_to_mug"),0.10,COLORS["mug"],0.002,dashed=state.target_frames == "both")
    _meshcat_frame(meshcat,"/lesson/target/Gstar",grasp_target(),0.065,COLORS["grasp"],0.003)
    _meshcat_frame(meshcat,"/lesson/approach/Gapp",approach_target(),0.075,COLORS["approach"],0.003)
    meshcat.SetLine("/lesson/approach/retreat",np.column_stack([grasp_target()[:3,3],approach_target()[:3,3]]),3,Rgba(*COLORS["approach"],1))
    meshcat.SetProperty("/lesson/current","visible",state.stage >= 2)
    meshcat.SetProperty("/lesson/target","visible",state.stage >= 3)
    meshcat.SetProperty("/lesson/target/M","visible",state.target_frames != "grasp")
    meshcat.SetProperty("/lesson/target/Gstar","visible",state.target_frames != "mug")
    meshcat.SetProperty("/lesson/approach","visible",state.stage >= 4)
    meshcat.SetProperty("/lesson/selected_joint","visible",state.stage == 1)
    meshcat.SetProperty("/lesson/origins","visible",state.stage == 1)
    meshcat.SetProperty("/lesson/joint_chain","visible",state.stage == 1)
    # Opaque shells hide internal frame origins. In stage 1 keep the original
    # hidden-shell view at full opacity, or show translucent shells for context.
    meshcat.SetProperty("/drake/visualizer/fr3","visible",state.stage != 1 or state.link_opacity < 1)
    return values


def meshcat_mode(seconds=None):
    from pydrake.all import StartMeshcat
    meshcat = StartMeshcat()
    scene = build_scene(meshcat)
    state = LessonState()
    colorizer = LinkColorizer(meshcat, scene)
    mug_appearance = MugAppearance(meshcat, scene)
    for label in STAGES:
        meshcat.AddButton(label)
    meshcat.AddButton(COLOR_BUTTON)
    meshcat.AddSlider("Link opacity",0,1,0.05,1)
    meshcat.AddSlider("Mug opacity",0,1,0.05,1)
    for label in FRAME_BUTTONS:
        meshcat.AddButton(label)
    meshcat.AddSlider("Joint frame k",0,7,1,7)
    joints = [j for j in parsed_chain() if j["type"] == "revolute"]
    for i,(name,value,entry) in enumerate(zip(JOINT_NAMES,Q_READY,joints),1):
        limit=entry["limit"]
        # The sliders report offsets from the exact ready angles. Zero therefore
        # reproduces exact pi fractions rather than rounded 0.001-rad values.
        meshcat.AddSlider(f"dq{i} [rad]",limit["lower"]-value,limit["upper"]-value,0.001,0.)
    meshcat.AddButton("Reset exact ready pose")
    meshcat.AddButton("Stop")
    meshcat.SetCameraPose(np.array([0.95,-0.85,0.90]),np.array([0.27,0.02,0.33]))
    meshcat.SetProperty("/Grid","visible",False)
    meshcat.SetProperty("/Axes","visible",False)
    readouts=[]
    def update_readouts(values):
        nonlocal readouts
        for label in readouts:
            meshcat.DeleteButton(label)
        p=values["X_W_G"][:3,3]
        selected=values["joint_origin"]
        position = lambda X: ", ".join(f"{x:.3f}" for x in X[:3,3])
        readouts=[f"Stage {state.stage}; q = ready q + dq",
                  "G [m]: " + position(values["X_W_G"]),
                  "G* [m]: " + position(values["X_W_Gstar"]),
                  "G_app* [m]: " + position(values["X_W_Gapproach"]),
                  f"Frame {state.joint_step} [m]: " + ", ".join(f"{x:.3f}" for x in selected),
                  ("Shells hidden; frame chain shown" if state.link_opacity == 1 else "Transparent links; frame chain shown") if state.stage == 1 else "G blue; G* orange; approach purple",
                  "Known mug; no IK/contact solve"]
        readouts.append("Link opacity: arm, hand and fingers")
        readouts.append("Mug opacity: lower it to see the axes")
        if state.stage >= 3:
            readouts.append({"mug": "Showing M: supplied mug; +z up",
                             "grasp": "Showing G*: chosen grasp; +z down",
                             "both": "M dashed; G* solid; same origin"}[state.target_frames])
            if state.target_frames == "both":
                readouts.append("Shared +x; opposite y and z axes")
        else:
            readouts.append("M and G* appear in stages 3–4")
        if state.link_colors:
            readouts.extend(["Links 0–3: gray, red, orange, green",
                             "Links 4–7: cyan, blue, purple, pink"])
        else:
            readouts.append("Link colors off; original gray")
        for label in readouts:
            meshcat.AddButton(label)
    update_readouts(update_meshcat(meshcat,scene,state,colorizer,mug_appearance))
    print("Meshcat at",meshcat.web_url(),flush=True)
    print("Start at stage 4, then select stages 1–4. RGB rods are x/y/z axes.",flush=True)
    print("Change one joint to move the current pose. Both object-based targets stay fixed.",flush=True)
    print("Slider angles are offsets from exact ready q; every reset returns exactly to it.",flush=True)
    print("Toggle distinct link colors to identify links 0–7; the hand stays gray.",flush=True)
    print("Link opacity affects the arm, hand and fingers: 1 is opaque; 0 is invisible. Frames remain solid.",flush=True)
    print("Mug opacity: reduce it to see the selected mug-center frame through the surface. Axes remain solid.",flush=True)
    print("M and G* share an origin: show either alone, or compare dashed M with solid G*. Their y/z axes are opposite.",flush=True)
    clicks={label:0 for label in [*STAGES,*FRAME_BUTTONS,COLOR_BUTTON,"Reset exact ready pose"]}
    started=time.monotonic()
    try:
        while meshcat.GetButtonClicks("Stop") == 0 and (seconds is None or time.monotonic()-started < seconds):
            dirty=False
            for i,label in enumerate(STAGES,1):
                now=meshcat.GetButtonClicks(label)
                if now != clicks[label]:
                    state.set_stage(i);clicks[label]=now;dirty=True
            for label,mode in FRAME_BUTTONS.items():
                now=meshcat.GetButtonClicks(label)
                if now != clicks[label]:
                    state.set_target_frames(mode);clicks[label]=now;dirty=True
            colors=meshcat.GetButtonClicks(COLOR_BUTTON)
            if colors != clicks[COLOR_BUTTON]:
                if (colors-clicks[COLOR_BUTTON]) % 2:
                    state.link_colors = not state.link_colors
                clicks[COLOR_BUTTON]=colors;dirty=True
            reset=meshcat.GetButtonClicks("Reset exact ready pose")
            if reset != clicks["Reset exact ready pose"]:
                clicks["Reset exact ready pose"]=reset
                state.reset()
                for i in range(1,8):
                    meshcat.SetSliderValue(f"dq{i} [rad]",0.)
                meshcat.SetSliderValue("Joint frame k",7)
                meshcat.SetSliderValue("Link opacity",1)
                meshcat.SetSliderValue("Mug opacity",1)
                dirty=True
            opacity=meshcat.GetSliderValue("Link opacity")
            if opacity != state.link_opacity:
                state.set_link_opacity(opacity);dirty=True
            mug_opacity=meshcat.GetSliderValue("Mug opacity")
            if mug_opacity != state.mug_opacity:
                state.set_mug_opacity(mug_opacity);dirty=True
            q=Q_READY+np.array([meshcat.GetSliderValue(f"dq{i} [rad]") for i in range(1,8)])
            step=int(meshcat.GetSliderValue("Joint frame k"))
            if not np.array_equal(q,state.q) or step != state.joint_step:
                state.set_q(q);state.joint_step=step;dirty=True
            if dirty:
                values=update_meshcat(meshcat,scene,state,colorizer,mug_appearance)
                update_readouts(values)
                print(f"stage {state.stage}, q={state.q}, current G={values['X_W_G'][:3,3]}",flush=True)
            time.sleep(0.05)
    finally:
        meshcat.DeleteAddedControls()


def figure_mode(out):
    """Render official low-detail geometry with VTK; annotate true projected frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pydrake.all import (CameraInfo,ClippingRange,ColorRenderCamera,MakeRenderEngineVtk,
                            RenderCameraCore,RenderEngineVtkParams,RigidTransform,RotationMatrix)
    params=RenderEngineVtkParams()
    params.default_clear_color=np.ones(3)
    scene=build_scene(renderer=("vtk",MakeRenderEngineVtk(params)))
    width,height=1500,1100
    eye=np.array([1.45,-1.50,1.10]);target=np.array([0.28,0.025,0.34])
    z=target-eye;z/=np.linalg.norm(z)
    x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    R=np.column_stack([x,y,z]);X=RigidTransform(RotationMatrix(R),eye)
    camera=CameraInfo(width,height,0.62)
    core=RenderCameraCore("vtk",camera,ClippingRange(0.05,10),RigidTransform())
    query=scene.scene_graph.get_query_output_port().Eval(scene.scene_graph.GetMyContextFromRoot(scene.context))
    pixels=np.array(query.RenderColorImage(ColorRenderCamera(core,False),scene.scene_graph.world_frame_id(),X).data)
    fig,ax=plt.subplots(figsize=(12,8.8),dpi=170)
    ax.imshow(pixels);ax.axis("off")
    def project(p):
        pc=R.T@(np.asarray(p)-eye)
        return np.array([camera.focal_x()*pc[0]/pc[2]+camera.center_x(),camera.focal_y()*pc[1]/pc[2]+camera.center_y()])
    def frame(X,color,length=0.085):
        p=project(X[:3,3]);ax.scatter(*p,c=[color],s=38,zorder=5,edgecolors="white",linewidths=0.8)
        for i,c in enumerate(["#c93232","#279640","#285ad0"]):
            tip=project(X[:3,3]+length*X[:3,i])
            ax.annotate("",xy=tip,xytext=p,arrowprops={"arrowstyle":"->","color":c,"lw":1.8})
        return p
    XE=current_pose(scene,frame=SCENE["flange_frame"]);XG=current_pose(scene)
    pW=frame(np.eye(4),"#444444",0.13)
    pE=frame(XE,COLORS["current"],0.055);pG=frame(XG,COLORS["current"])
    pT=frame(grasp_target(),COLORS["grasp"],0.085);pA=frame(approach_target(),COLORS["approach"],0.085)
    ax.plot([pE[0],pG[0]],[pE[1],pG[1]],color=COLORS["current"],lw=2)
    ax.plot([pT[0],pA[0]],[pT[1],pA[1]],color=COLORS["approach"],ls="--",lw=2)
    for label,p,offset,color in [("W: fixed base",pW,(-20,55),"#333333"),
                                ("E: current flange",pE,(-160,-40),COLORS["current"]),
                                ("G: current hand TCP",pG,(-205,10),COLORS["current"]),
                                ("M center and requested G*",pT,(80,-30),COLORS["grasp"]),
                                ("Requested approach G_app*",pA,(85,45),COLORS["approach"])]:
        ax.annotate(label,xy=p,xytext=offset,textcoords="offset pixels",fontsize=10.5,color=color,
                    bbox={"facecolor":"white","alpha":0.90,"edgecolor":"none","pad":2},
                    arrowprops={"arrowstyle":"-","color":color,"lw":1.1})
    ax.text(0.015,0.975,"FR3 model at the exact ready configuration",transform=ax.transAxes,va="top",fontsize=15,weight="bold",color="#9b1b4b")
    ax.text(0.015,0.923,"Current pose comes from the joints. The mug pose and grasp choice supply the targets.",transform=ax.transAxes,va="top",fontsize=10.5)
    position = lambda X: "(" + ", ".join(f"{v:.4f}" for v in X[:3,3]) + ") m"
    ax.text(0.015,0.055,"G now: " + position(XG) + "    G*: " + position(grasp_target()) + "    G_app*: " + position(approach_target()),transform=ax.transAxes,fontsize=10)
    ax.text(0.015,0.020,"Axes: x red, y green, z blue. Official collision shapes are used as low-detail visuals. No IK or contact is solved.",transform=ax.transAxes,fontsize=8.5,color="#555555")
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,facecolor="white");plt.close(fig)
    print("Rendered",out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshcat",action="store_true")
    parser.add_argument("--seconds",type=float,help="Optional bounded Meshcat smoke-test duration")
    parser.add_argument("--png",type=Path)
    parser.add_argument("--facts",type=Path)
    args=parser.parse_args()
    if args.meshcat:
        meshcat_mode(args.seconds)
    elif args.png:
        figure_mode(args.png)
    else:
        facts=verified_facts()
        if args.facts:
            args.facts.parent.mkdir(parents=True,exist_ok=True)
            args.facts.write_text(json.dumps(facts,indent=2)+"\n")
        print_mode(facts)


if __name__ == "__main__":
    main()
