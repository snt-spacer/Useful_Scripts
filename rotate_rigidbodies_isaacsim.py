"""
Rotate all rigid bodies in pingu.usd by 45 degrees around the Z axis
with respect to world space, then save as rot_pingu.usd.

Run this inside the Isaac Sim scripting window (Window > Script Editor).

Only root-level rigid bodies (those without a rigid body ancestor) are
rotated directly — children move with them, avoiding double-rotation
in articulated hierarchies.

Adjust INPUT_PATH, OUTPUT_PATH, DEGREES, and AXIS as needed.
"""

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH  = "/home/ricard/pingu_dev/Isaaclab/source/isaaclab_assets/data/Robots/SpaceR-TheDreamLab/Pingu/pingu.usd"
OUTPUT_PATH = "/home/ricard/pingu_dev/Isaaclab/source/isaaclab_assets/data/Robots/SpaceR-TheDreamLab/Pingu/rot_pingu.usd"
DEGREES     = 45.0
AXIS        = "Z"   # "X", "Y", or "Z"
# ──────────────────────────────────────────────────────────────────────────────


def has_rigidbody_ancestor(prim: Usd.Prim) -> bool:
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        parent = parent.GetParent()
    return False


def build_rotation_matrix(degrees: float, axis: str) -> Gf.Matrix4d:
    half = math.radians(degrees) / 2.0
    c, s = math.cos(half), math.sin(half)
    axis_vec = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[axis.upper()]
    quat = Gf.Quatd(c, axis_vec * s)
    mat = Gf.Matrix4d()
    mat.SetRotate(quat)
    return mat


def rotate_rigidbodies(input_path: str, output_path: str, degrees: float, axis: str) -> None:
    # Open in a scratch stage so the live viewport is not touched
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] Could not open: {input_path}")
        return

    rot_world  = build_rotation_matrix(degrees, axis)
    xform_cache = UsdGeom.XformCache()

    root_rb_prims = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not has_rigidbody_ancestor(prim)
    ]

    if not root_rb_prims:
        print("[WARN] No rigid body prims found — nothing to rotate.")
        return

    # Phase 1 — snapshot all transforms before any edits
    prim_data = []
    for prim in root_rb_prims:
        world_xform  = xform_cache.GetLocalToWorldTransform(prim)
        parent       = prim.GetParent()
        parent_world = (
            xform_cache.GetLocalToWorldTransform(parent)
            if parent and parent.IsValid() and not parent.IsPseudoRoot()
            else Gf.Matrix4d(1)
        )
        prim_data.append((prim, world_xform, parent_world))

    # Phase 2 — apply rotation and write new local transforms
    for prim, world_xform, parent_world in prim_data:
        new_world = rot_world * world_xform
        new_local = new_world * parent_world.GetInverse()

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(new_local)

        print(f"  Rotated: {prim.GetPath()}")

    print(f"\nRotated {len(prim_data)} root rigid body prim(s) by {degrees}° around {axis.upper()}.")

    stage.Export(output_path)
    print(f"Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
rotate_rigidbodies(INPUT_PATH, OUTPUT_PATH, DEGREES, AXIS)
