"""Rebuild offline visual assets from the pinned manufacturer description.

The checked-in runtime model needs no converter or network. Rebuilding the
expanded URDF additionally needs xacro==2.1.1 and PyYAML==6.0.3; these can be
installed in a temporary --target directory and selected with PYTHONPATH.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model"
SOURCE = MODEL / "source"
COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
MAPPINGS = {"hand": "true", "with_sc": "false",
            "include_self_collision_geometry": "false", "connected_to": ""}


def expand():
    import xacro
    import xacro.substitution_args
    def find(package):
        if package != "franka_description":
            raise ValueError(f"Unexpected package: {package}")
        return str(SOURCE)
    xacro.substitution_args._eval_find = find
    source = SOURCE / "robots/fr3/fr3.urdf.xacro"
    # xacro appends default arguments to the supplied dict. Keep the recorded
    # explicit overrides stable across --rebuild and --expand invocations.
    xml = xacro.process_file(str(source), mappings=dict(MAPPINGS)).toprettyxml(indent="  ")
    xml = xml.replace(str(SOURCE) + "/", "model/source/")
    (MODEL / "fr3-official.urdf").write_text(xml)


def stl_to_obj(source, target):
    """Convert binary STL triangles without altering their coordinates."""
    data = source.read_bytes()
    count, = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * count:
        raise ValueError(f"Expected binary STL: {source}")
    lines = [f"# Official Franka collision triangles; source commit {COMMIT}.",
             "# Converted to OBJ for Drake visuals; no geometric simplification."]
    for i in range(count):
        normal = struct.unpack_from("<3f", data, 84 + 50*i)
        xyz = struct.unpack_from("<9f", data, 84 + 50*i + 12)
        for j in range(3):
            lines.append("v " + " ".join(format(x, ".9g") for x in xyz[3*j:3*j+3]))
        lines.append("vn " + " ".join(format(x, ".9g") for x in normal))
        lines.append(f"f {3*i+1}//{i+1} {3*i+2}//{i+1} {3*i+3}//{i+1}")
    target.write_text("\n".join(lines) + "\n")


def build():
    meshes = MODEL / "meshes"
    meshes.mkdir(exist_ok=True)
    for source in sorted(SOURCE.rglob("*.stl")):
        stl_to_obj(source, meshes / (source.stem + ".obj"))
    root = ET.parse(MODEL / "fr3-official.urdf").getroot()
    for link in root.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        # Use the official collision shape as a low-detail visual. All joint,
        # frame, inertia, limit, and collision-origin entries remain unchanged.
        for i, collision in enumerate(link.findall("collision")):
            visual = copy.deepcopy(collision)
            visual.tag = "visual"
            visual.set("name", link.get("name") + f"_teaching_visual_{i}")
            material = ET.SubElement(visual, "material", name="teaching_grey")
            ET.SubElement(material, "color", rgba="0.78 0.81 0.85 1")
            link.append(visual)
    for mesh in root.iter("mesh"):
        mesh.set("filename", "meshes/" + Path(mesh.get("filename")).stem + ".obj")
    ET.indent(root, space="  ")
    notice = ("<!-- Modified for CS 549 on 2026-10-04. Original kinematics and limits retained.\n"
              "     Official collision geometry replaces detailed visual meshes.\n"
              "     See model/source/LICENSE, model/source/NOTICE and model/provenance.json. -->\n")
    (MODEL / "fr3.urdf").write_text('<?xml version="1.0"?>\n' + notice + ET.tostring(root, encoding="unicode") + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"repository": "https://github.com/frankarobotics/franka_description",
                "release": "2.9.0", "commit": COMMIT, "robot_type": "fr3",
                "license": "Apache-2.0; retain the complete upstream LICENSE and NOTICE",
                "xacro_version": "2.1.1", "pyyaml_version": "6.0.3", "mappings": MAPPINGS,
                "modification": "Only visual geometry and local mesh paths change. Visuals use official collision shapes; STL triangles are converted to OBJ.",
                "files": {str(p.relative_to(MODEL)): sha(p) for p in sorted(MODEL.rglob("*")) if p.is_file() and p.name != "provenance.json"}}
    (MODEL / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Rebuilt", MODEL / "fr3.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expand", action="store_true", help="First rerun xacro against vendored source")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild runtime URDF/OBJ from the expanded official URDF")
    args = parser.parse_args()
    if args.expand or args.rebuild:
        if args.expand:
            expand()
        build()
    else:
        from fr3_scene import verify_model
        manifest = verify_model()
        print("FR3 model hashes and unchanged joint/inertial entries verified:", manifest["commit"])
