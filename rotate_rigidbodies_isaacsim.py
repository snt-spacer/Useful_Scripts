"""
Apply a world-space rotation (X/Y/Z Euler angles) and translation offset
to the articulation root Xform in a USD file, then save the result.

Run this inside the Isaac Sim scripting window (Window > Script Editor).

Targets the ArticulationRoot prim (or the stage default prim if none is found).
This is the correct prim to rotate for articulations: PhysX owns the rigid body
transforms inside and overwrites them on sim init, so rotations baked on rigid
body prims are silently discarded.

Adjust the Configuration block at the top as needed.
"""

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH  = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\Cubo\\cubo.usd"
OUTPUT_PATH = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\Cubo\\cubo_rot.usd"

# Rotation in degrees applied around each world axis (order: X → Y → Z)
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = 45.0

# Translation offset added in world space (same units as the USD stage)
TRANS_X = 0.0
TRANS_Y = 0.0
TRANS_Z = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def build_transform_matrix(rot_x: float, rot_y: float, rot_z: float,
                            trans_x: float, trans_y: float, trans_z: float) -> Gf.Matrix4d:
    """Build a 4×4 world-space transform: rotation (X→Y→Z Euler) then translation."""
    def axis_quat(degrees: float, axis: Gf.Vec3d) -> Gf.Quatd:
        half = math.radians(degrees) / 2.0
        return Gf.Quatd(math.cos(half), axis * math.sin(half))

    qx = axis_quat(rot_x, Gf.Vec3d(1, 0, 0))
    qy = axis_quat(rot_y, Gf.Vec3d(0, 1, 0))
    qz = axis_quat(rot_z, Gf.Vec3d(0, 0, 1))

    combined_quat = qx * qy * qz

    rot_mat = Gf.Matrix4d()
    rot_mat.SetRotate(combined_quat)

    trans_mat = Gf.Matrix4d()
    trans_mat.SetTranslate(Gf.Vec3d(trans_x, trans_y, trans_z))

    return rot_mat * trans_mat


def apply_transform(input_path: str, output_path: str,
                    rot_x: float, rot_y: float, rot_z: float,
                    trans_x: float, trans_y: float, trans_z: float) -> None:
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] Could not open: {input_path}")
        return

    # Rotate the direct children of each rigid body prim.
    # PhysX owns the rigid body's world transform but preserves the local
    # transform of children (visuals/collisions) relative to their parent
    # rigid body — so this is the correct place to bake a visual rotation.
    world_delta = build_transform_matrix(rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
    xform_cache = UsdGeom.XformCache()

    rb_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]

    if not rb_prims:
        print("[WARN] No rigid body prims found — nothing to transform.")
        return

    # Collect all direct children of rigid bodies that are Xformable
    target_prims = []
    for rb in rb_prims:
        for child in rb.GetChildren():
            if UsdGeom.Xformable(child):
                target_prims.append(child)

    if not target_prims:
        print("[WARN] No Xformable children of rigid bodies found — nothing to transform.")
        return

    # Phase 1 — snapshot all transforms before any edits
    prim_data = []
    for prim in target_prims:
        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        parent = prim.GetParent()
        parent_world = (
            xform_cache.GetLocalToWorldTransform(parent)
            if parent and parent.IsValid() and not parent.IsPseudoRoot()
            else Gf.Matrix4d(1)
        )
        prim_data.append((prim, world_xform, parent_world))

    # Phase 2 — apply delta and write new local transforms
    for prim, world_xform, parent_world in prim_data:
        new_world = world_xform * world_delta
        new_local = new_world * parent_world.GetInverse()

        decomposed = Gf.Transform(new_local)
        translation = decomposed.GetTranslation()
        orient = decomposed.GetRotation().GetQuat()
        scale = decomposed.GetScale()

        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translation)
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orient)
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(scale)

        print(f"  Transformed: {prim.GetPath()}")

    print(
        f"\nApplied to {len(prim_data)} child prim(s) of rigid bodies:\n"
        f"  Rotation  X={rot_x}°  Y={rot_y}°  Z={rot_z}°\n"
        f"  Translation  X={trans_x}  Y={trans_y}  Z={trans_z}"
    )

    stage.Export(output_path)
    print(f"Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
apply_transform(INPUT_PATH, OUTPUT_PATH, ROT_X, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z)
