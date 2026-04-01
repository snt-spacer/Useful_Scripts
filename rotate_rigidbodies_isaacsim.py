"""
Apply a rotation (X/Y/Z Euler angles) to an articulated USD robot, keeping
physics and visuals consistent.

What gets rotated and why
─────────────────────────
1. Xformable children of every rigid body prim
   → visuals/collision shapes live here as local offsets; PhysX preserves them.

2. Joint local frames (localPos0/1, localRot0/1) for every joint
   → these express the joint attachment point in the parent/child body frame.
     If we rotate the visual frame but not these, thrusters fire from the
     old (pre-rotation) positions.

What is intentionally NOT rotated
──────────────────────────────────
• Rigid body prims themselves — PhysX overwrites their world transforms on
  sim init; editing their xformOps has no effect.
• The ArticulationRoot xform — use init_state.rot in the Isaac Lab config
  to control spawn/reset placement.

Run inside the Isaac Sim Script Editor (Window > Script Editor).
Adjust the Configuration block at the top as needed.
"""

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH  = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\Cubo\\cubo.usd"
OUTPUT_PATH = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\UniluFP\\deep_cubo_rot.usd"

# Rotation in degrees applied around each axis (order: X → Y → Z)
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = 45.0

# Translation offset (same units as the USD stage)
TRANS_X = 0.0
TRANS_Y = 0.0
TRANS_Z = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def build_rotation_quat(rot_x: float, rot_y: float, rot_z: float) -> Gf.Quatd:
    """Compose a quaternion from X→Y→Z Euler angles (degrees)."""
    def axis_quat(degrees: float, axis: Gf.Vec3d) -> Gf.Quatd:
        half = math.radians(degrees) / 2.0
        return Gf.Quatd(math.cos(half), axis * math.sin(half))

    qx = axis_quat(rot_x, Gf.Vec3d(1, 0, 0))
    qy = axis_quat(rot_y, Gf.Vec3d(0, 1, 0))
    qz = axis_quat(rot_z, Gf.Vec3d(0, 0, 1))
    return qx * qy * qz


def build_transform_matrix(rot_quat: Gf.Quatd, trans_x: float, trans_y: float, trans_z: float) -> Gf.Matrix4d:
    rot_mat = Gf.Matrix4d()
    rot_mat.SetRotate(rot_quat)
    trans_mat = Gf.Matrix4d()
    trans_mat.SetTranslate(Gf.Vec3d(trans_x, trans_y, trans_z))
    return rot_mat * trans_mat


def rotate_xformable_children(rb_prims, world_delta: Gf.Matrix4d) -> int:
    """Rotate the Xformable direct children of each rigid body prim."""
    xform_cache = UsdGeom.XformCache()

    target_prims = []
    for rb in rb_prims:
        for child in rb.GetChildren():
            if UsdGeom.Xformable(child):
                target_prims.append(child)

    if not target_prims:
        print("[WARN] No Xformable children of rigid bodies found.")
        return 0

    # Snapshot all transforms before any edits
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

    for prim, world_xform, parent_world in prim_data:
        new_world = world_xform * world_delta
        new_local = new_world * parent_world.GetInverse()

        decomposed = Gf.Transform(new_local)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetTranslation())
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetRotation().GetQuat())
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetScale())
        print(f"  [visual] {prim.GetPath()}")

    return len(prim_data)


def rotate_joint_frames(stage: Usd.Stage, rot_quat: Gf.Quatd) -> int:
    """
    Rotate all joint local frames by rot_quat.

    localPos0/1 are positions expressed in the parent/child body frame —
    they must be rotated by the same delta applied to the visual children.
    localRot0/1 are orientations of the joint frame in the body frame —
    they must be pre-composed with rot_quat.

    Joints are detected by the presence of physics:localPos0, since
    UsdPhysics.JointAPI does not exist — joints are typed schemas, not API schemas.
    """
    rotation = Gf.Rotation(rot_quat)

    joint_count = 0
    for prim in stage.Traverse():
        # Detect joints by the presence of their defining attribute
        if not prim.GetAttribute("physics:localPos0"):
            continue

        # Rotate localPos0 (Vec3f, position in body0 frame)
        pos0_attr = prim.GetAttribute("physics:localPos0")
        if pos0_attr.HasAuthoredValue():
            p0 = Gf.Vec3d(pos0_attr.Get())
            pos0_attr.Set(Gf.Vec3f(rotation.TransformDir(p0)))

        # Rotate localPos1 (Vec3f, position in body1 frame)
        pos1_attr = prim.GetAttribute("physics:localPos1")
        if pos1_attr and pos1_attr.HasAuthoredValue():
            p1 = Gf.Vec3d(pos1_attr.Get())
            pos1_attr.Set(Gf.Vec3f(rotation.TransformDir(p1)))

        # Compose localRot0 (Quatf, orientation of joint frame in body0)
        rot0_attr = prim.GetAttribute("physics:localRot0")
        if rot0_attr and rot0_attr.HasAuthoredValue():
            r0 = Gf.Quatd(rot0_attr.Get())
            rot0_attr.Set(Gf.Quatf(rot_quat * r0))

        # Compose localRot1 (Quatf, orientation of joint frame in body1)
        rot1_attr = prim.GetAttribute("physics:localRot1")
        if rot1_attr and rot1_attr.HasAuthoredValue():
            r1 = Gf.Quatd(rot1_attr.Get())
            rot1_attr.Set(Gf.Quatf(rot_quat * r1))

        print(f"  [joint]  {prim.GetPath()}")
        joint_count += 1

    return joint_count


def apply_transform(input_path: str, output_path: str,
                    rot_x: float, rot_y: float, rot_z: float,
                    trans_x: float, trans_y: float, trans_z: float) -> None:
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] Could not open: {input_path}")
        return

    rot_quat = build_rotation_quat(rot_x, rot_y, rot_z)
    world_delta = build_transform_matrix(rot_quat, trans_x, trans_y, trans_z)

    rb_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not rb_prims:
        print("[WARN] No rigid body prims found — nothing to transform.")
        return

    n_visual = rotate_xformable_children(rb_prims, world_delta)
    n_joints = rotate_joint_frames(stage, rot_quat)

    print(
        f"\nDone.\n"
        f"  Visual prims rotated : {n_visual}\n"
        f"  Joint frames rotated : {n_joints}\n"
        f"  Rotation  X={rot_x}°  Y={rot_y}°  Z={rot_z}°\n"
        f"  Translation  X={trans_x}  Y={trans_y}  Z={trans_z}"
    )

    stage.Export(output_path)
    print(f"Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
apply_transform(INPUT_PATH, OUTPUT_PATH, ROT_X, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z)
